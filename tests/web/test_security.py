"""Security audit: escaping, no external resources, secret hygiene."""

import re
from pathlib import Path
from typing import Any, cast

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from papyrus_chat.web.application import load_app
from tests.chat.mock_provider_server import MockProviderServer

TEST_ENV = {
    "LLM_BASE_URL": "https://x.example/v1",
    "LLM_MODEL": "test-model",
    "LLM_API_KEY": "sk-audit-secret-9876",
}

RESOURCE_TAGS = ("script", "link", "img", "iframe", "source")
URL_ATTRS = ("src", "href")


def client_for(corpus_artifact: Path, env: dict[str, str] | None = TEST_ENV) -> TestClient:
    return TestClient(load_app(corpus_artifact, env=env))


def all_pages(client: TestClient) -> list[str]:
    with MockProviderServer(content="Answer [1].") as mock:
        env = dict(TEST_ENV)
        env["LLM_BASE_URL"] = mock.base_url
        client2 = TestClient(load_app(_artifact_of(client), env=env))
        return [
            client2.get("/").text,
            client2.get("/search", params={"query": "ἔτους"}).text,
            client2.get("/documents/dclp:DCLP/23/23944.xml").text,
            client2.post("/chat", data={"query": "ἔτους", "document_id": ""}).text,
        ]


def _artifact_of(client: TestClient) -> Path:
    app = cast(Any, client.app)
    return app.state.artifact


def test_no_api_key_in_any_response(corpus_artifact: Path) -> None:
    for html in all_pages(client_for(corpus_artifact)):
        assert TEST_ENV["LLM_API_KEY"] not in html


def test_no_external_resources_are_loaded(corpus_artifact: Path) -> None:
    client = client_for(corpus_artifact)
    pages = [client.get("/").text, client.get("/search?query=ἔτους").text]

    for html in pages:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(RESOURCE_TAGS):
            for attr in URL_ATTRS:
                url = tag.get(attr)
                if url:
                    assert str(url).startswith("/"), (
                        f"external/local-unexpected resource: {tag.name} {url}"
                    )


def test_no_telemetry_scripts(corpus_artifact: Path) -> None:
    client = client_for(corpus_artifact)

    for html in [client.get("/").text, client.get("/search?query=x").text]:
        assert "analytics" not in html.lower()
        assert "telemetry" not in html.lower()
        scripts = BeautifulSoup(html, "html.parser").find_all("script")
        for script in scripts:
            src = str(script.get("src") or "")
            assert not re.match(r"https?://", src), f"remote script: {src}"


def test_corpus_content_is_escaped_in_search(corpus_artifact: Path) -> None:
    client = client_for(corpus_artifact)

    html = client.get("/search", params={"query": "<img src=x onerror=alert(1)>"}).text

    assert "<img src=x" not in html
