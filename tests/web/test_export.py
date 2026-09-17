"""Downloads preserve the shareable browser transcript without running the agent."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from lxml import html
from pydantic_ai.models.test import TestModel
from starlette.testclient import TestClient

from papyrus_chat.web.application import load_app
from papyrus_chat.web.exporting import PINNED_UI_URL


@pytest.fixture
def export_client(corpus_artifact: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    loader = AsyncMock(
        return_value=b'<!doctype html><html><body><div id="root"></div></body></html>'
    )
    monkeypatch.setattr("pydantic_ai.ui._web.app._get_ui_html", loader)
    app = load_app(
        corpus_artifact,
        env={"LLM_BASE_URL": "https://provider.example/v1", "LLM_MODEL": "research-model"},
        model=TestModel(call_tools=[], custom_output_text="Unused"),
    )
    app.state.test_html_loader = loader
    return TestClient(app, base_url="http://localhost")


@pytest.fixture
def snapshot() -> dict:
    return {
        "id": "/thread-1",
        "title": "Investigate πάπυρος",
        "messages": [
            {"id": "q1", "role": "user", "parts": [{"type": "text", "text": "πάπυρος?"}]},
            {
                "id": "a1",
                "role": "assistant",
                "metadata": {"internal": "PRIVATE_METADATA"},
                "parts": [
                    {"type": "step-start"},
                    {
                        "type": "reasoning",
                        "text": "I will inspect the corpus.",
                        "providerMetadata": {"signature": "OPAQUE_SIGNATURE"},
                    },
                    {
                        "type": "tool-search_documents",
                        "toolCallId": "search-1",
                        "state": "output-available",
                        "input": {"query": "πάπυρος"},
                        "output": {"text": "πάπυρος " * 10_000, "signature": "evidence"},
                        "callProviderMetadata": {"internal": "PRIVATE_METADATA"},
                    },
                    {
                        "type": "source-url",
                        "sourceId": "source-1",
                        "url": "https://papyri.info/ddbdp/example",
                        "title": "Source",
                    },
                    {"type": "text", "text": "Corpus **evidence** [source](https://papyri.info)."},
                ],
            },
            {"id": "q2", "role": "user", "parts": [{"type": "text", "text": "Continue."}]},
            {
                "id": "a2",
                "role": "assistant",
                "parts": [
                    {
                        "type": "dynamic-tool",
                        "toolName": "inspect_documents",
                        "toolCallId": "inspect-1",
                        "state": "output-error",
                        "input": {},
                        "errorText": "Record not found",
                    },
                    {
                        "type": "tool-read_document_passages",
                        "toolCallId": "read-1",
                        "state": "input-streaming",
                        "input": {"id": "partial"},
                    },
                    {"type": "data-research", "data": {"note": "Unfamiliar visible content"}},
                ],
            },
        ],
    }


def test_json_export_preserves_content_without_transport_metadata(export_client, snapshot):
    response = export_client.post("/api/export", json={"format": "json", "conversation": snapshot})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.headers["content-disposition"].startswith('attachment; filename="papyrus-chat-')
    assert response.headers["cache-control"] == "no-store"
    document = response.json()
    assert document["schema_version"] == 1
    assert document["kind"] == "papyrus-chat-browser-conversation"
    assert document["exported_at"]
    assert document["conversation"]["id"] == snapshot["id"]
    parts = document["conversation"]["messages"][1]["parts"]
    assert parts[2]["output"] == snapshot["messages"][1]["parts"][2]["output"]
    assert parts[1] == {"type": "reasoning", "text": "I will inspect the corpus."}
    assert document["conversation"]["messages"][3] == snapshot["messages"][3]
    assert "OPAQUE_SIGNATURE" not in response.text
    assert "PRIVATE_METADATA" not in response.text
    assert "metadata" not in document["conversation"]["messages"][1]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"format": "pdf", "conversation": {}},
        {"format": "json", "conversation": {"id": "/empty", "title": "Empty", "messages": []}},
        {
            "format": "json",
            "conversation": {
                "id": "/bad",
                "title": "Bad",
                "messages": [
                    {"id": "m", "role": "assistant", "parts": [{"type": "text", "text": {}}]},
                ],
            },
        },
    ],
)
def test_export_rejects_invalid_snapshots(export_client, payload):
    assert export_client.post("/api/export", json=payload).status_code == 422


def test_export_requires_json_and_handles_malformed_json(export_client):
    assert export_client.post("/api/export", content="hello").status_code == 415
    response = export_client.post(
        "/api/export", content="{", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_exports_are_independent_and_filenames_are_safe(export_client, snapshot):
    export_client.post("/api/export", json={"format": "json", "conversation": snapshot})
    snapshot["id"] = '/../../evil"\r\nheader'
    snapshot["messages"] = snapshot["messages"][:1]
    response = export_client.post("/api/export", json={"format": "json", "conversation": snapshot})
    assert len(response.json()["conversation"]["messages"]) == 1
    assert "\r" not in response.headers["content-disposition"]
    assert "../" not in response.headers["content-disposition"]


def test_export_and_assets_keep_host_protection(export_client, snapshot):
    response = export_client.post(
        "/api/export",
        json={"format": "json", "conversation": snapshot},
        headers={"host": "attacker.example"},
    )
    assert response.status_code == 421
    assert (
        export_client.get(
            "/papyrus-assets/export.js", headers={"host": "attacker.example"}
        ).status_code
        == 421
    )


def test_default_shell_pins_ui_and_serves_export_assets(export_client):
    response = export_client.get("/thread-1")
    assert response.status_code == 200
    export_client.app.state.test_html_loader.assert_awaited_once_with(PINNED_UI_URL)
    assert "@pydantic/ai-chat-ui@2.1.0/offline/" in PINNED_UI_URL
    assert "/papyrus-assets/export.js" in response.text
    for asset in ("export.js", "export-storage.js", "export.css", "shell.css"):
        assert export_client.get(f"/papyrus-assets/{asset}").status_code == 200
    assert json.loads(export_client.get("/api/health").text)


def test_html_export_contains_complete_readable_transcript(export_client, snapshot):
    response = export_client.post("/api/export", json={"format": "html", "conversation": snapshot})
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["content-disposition"].endswith('.html"')
    tree = html.fromstring(response.text)
    assert tree.xpath("//title/text()") == [snapshot["title"]]
    assert len(tree.findall(".//article")) == 4
    assert tree.xpath("//strong/text()") == ["evidence"]
    text = str(tree.xpath("string()"))
    assert "I will inspect the corpus." in text
    assert "reasoning" in str(tree.xpath("string((//summary)[1])"))
    outputs = [json.loads(block.text or "") for block in tree.findall(".//pre[@data-json]")]
    assert snapshot["messages"][1]["parts"][2]["output"] in outputs
    assert "Record not found" in text
    assert "partial input" in text
    assert "Unfamiliar visible content" in text
    assert "https://papyri.info/ddbdp/example" in [link.get("href") for link in tree.iter("a")]
    assert "PRIVATE_METADATA" not in response.text
    assert "OPAQUE_SIGNATURE" not in response.text
    assert not tree.xpath("//script | //link | //img | //iframe")
    assert tree.xpath('//meta[@http-equiv="Content-Security-Policy"]')


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "file:///etc/passwd",
        "//attacker.example/track",
        "https://[invalid",
    ],
)
def test_html_escapes_content_and_rejects_unsafe_links(export_client, snapshot, unsafe_url):
    attack = '</pre><script>alert("unsafe")</script><img src="https://attacker.example/track">'
    snapshot["title"] = attack
    snapshot["messages"][0]["parts"][0]["text"] = (
        attack + f"\n[unsafe]({unsafe_url})\n![tracking](https://attacker.example/track)"
    )
    snapshot["messages"][1]["parts"][2]["output"] = {"text": attack}
    snapshot["messages"][1]["parts"][3]["url"] = unsafe_url
    response = export_client.post("/api/export", json={"format": "html", "conversation": snapshot})
    assert response.status_code == 200
    tree = html.fromstring(response.text)
    assert not tree.xpath("//script | //img | //iframe | //object | //embed | //form")
    assert not tree.xpath("//@onerror | //@onclick")
    assert unsafe_url not in [link.get("href") for link in tree.iter("a")]
    assert attack in str(tree.xpath("string()"))


@pytest.mark.parametrize("kind", ["text", "reasoning"])
def test_html_links_bare_record_urls(export_client, snapshot, kind):
    url = "https://papyri.info/ddbdp/p.fouad;1;86"
    query_url = "https://papyri.info/search?q=flax&start=1"
    text = (
        f"Record ({url}).\n\n- {url}\n\n"
        f"| Record |\n|---|\n| {url} |\n\n"
        f"Search {query_url}.\n\n[Named record]({url})\n\n"
        f"`{url}`\n\n```text\n{url}\n```\n\n"
        "p.fouad;1;86 user@example.org ftp://example.org //example.org "
        "javascript:alert(1) file:///etc/passwd"
    )
    snapshot["messages"] = [
        {"id": "a", "role": "assistant", "parts": [{"type": kind, "text": text}]}
    ]
    response = export_client.post("/api/export", json={"format": "html", "conversation": snapshot})
    assert response.status_code == 200
    tree = html.fromstring(response.text)
    assert tree.xpath("//a/@href") == [url, url, url, query_url, url]
    assert tree.xpath("//a/text()") == [url, url, url, query_url, "Named record"]
    assert not tree.xpath("//code//a | //a//a | //script")


def test_html_renders_tables_without_external_assets(export_client, snapshot):
    snapshot["messages"][0]["parts"][0]["text"] = "| Record | Text |\n|---|---|\n| 1 | πάπυρος |"
    response = export_client.post("/api/export", json={"format": "html", "conversation": snapshot})
    assert response.status_code == 200
    tree = html.fromstring(response.text)
    assert tree.xpath("//td/text()") == ["1", "πάπυρος"]
    assert not tree.xpath("//*[@src] | //link")
