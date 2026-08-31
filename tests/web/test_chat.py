"""Chat panel and the Evidence used section (SPEC 9.3, 10)."""

from pathlib import Path

from fastapi.testclient import TestClient

from papyrus_chat.web.application import load_app
from tests.chat.mock_provider_server import MockProviderServer

MOCK_REPLY = "According to [1] the horoscope dates to year 6 of Claudius."


def client_for(corpus_artifact: Path, env: dict[str, str]) -> TestClient:
    return TestClient(load_app(corpus_artifact, env=env))


def env_with(mock: MockProviderServer) -> dict[str, str]:
    return {"LLM_BASE_URL": mock.base_url, "LLM_MODEL": "test-model"}


class TestChatPanel:
    def test_get_chat_renders_empty_form(self, corpus_artifact: Path) -> None:
        client = client_for(
            corpus_artifact, {"LLM_BASE_URL": "http://127.0.0.1:9", "LLM_MODEL": "m"}
        )

        response = client.get("/chat")
        html = response.text

        assert response.status_code == 200
        assert '<form method="post" action="/chat"' in html
        assert "Ask about the corpus" in html

    def test_answer_renders_with_evidence_used(self, corpus_artifact: Path) -> None:
        with MockProviderServer(content=MOCK_REPLY) as mock:
            client = client_for(corpus_artifact, env_with(mock))

            response = client.post("/chat", data={"query": "ἔτους", "document_id": ""})

        html = response.text
        assert response.status_code == 200
        assert "the horoscope dates to year 6" in html
        assert "Evidence used" in html
        assert "Horoscope" in html, "evidence items must show their titles"
        assert "model-generated" in html.lower()

    def test_evidence_section_lists_exact_evidence(self, corpus_artifact: Path) -> None:
        with MockProviderServer(content=MOCK_REPLY) as mock:
            client = client_for(corpus_artifact, env_with(mock))

            html = client.post("/chat", data={"query": "sovereigns decree", "document_id": ""}).text

        assert "Evidence used" in html
        assert "source translation" in html.lower(), (
            "translation evidence must be labelled as a source translation"
        )

    def test_evidence_markers_carry_citation_labels(self, corpus_artifact: Path) -> None:
        with MockProviderServer(content=MOCK_REPLY) as mock:
            client = client_for(corpus_artifact, env_with(mock))

            html = client.post("/chat", data={"query": "ἔτους", "document_id": ""}).text

        assert "[1]" in html
        assert "TM 23944" in html or "23944" in html

    def test_evidence_links_to_papyri_info(self, corpus_artifact: Path) -> None:
        with MockProviderServer(content=MOCK_REPLY) as mock:
            client = client_for(corpus_artifact, env_with(mock))

            html = client.post("/chat", data={"query": "ἔτους", "document_id": ""}).text

        assert "papyri.info/current/23944" in html

    def test_document_scope_restricts_evidence(self, corpus_artifact: Path) -> None:
        with MockProviderServer(content=MOCK_REPLY) as mock:
            client = client_for(corpus_artifact, env_with(mock))

            html = client.post(
                "/chat",
                data={
                    "query": "sovereigns",
                    "document_id": "dclp:DCLP/23/23944.xml",
                },
            ).text

        assert "3643-1" not in html

    def test_insufficient_evidence_renders_guidance(self, corpus_artifact: Path) -> None:
        with MockProviderServer() as mock:
            client = client_for(corpus_artifact, env_with(mock))

            html = client.post("/chat", data={"query": "zzzqqqxyyz absent", "document_id": ""}).text

        assert "could not find" in html.lower()
        assert "mock answer" not in html

    def test_provider_failure_renders_actionable_message(self, corpus_artifact: Path) -> None:
        client = client_for(
            corpus_artifact,
            {"LLM_BASE_URL": "http://127.0.0.1:9", "LLM_MODEL": "m"},
        )

        html = client.post("/chat", data={"query": "ἔτους", "document_id": ""}).text

        assert "reach" in html.lower() or "connect" in html.lower()

    def test_missing_provider_config_renders_guidance(self, corpus_artifact: Path) -> None:
        client = client_for(corpus_artifact, env={})

        html = client.post("/chat", data={"query": "ἔτους", "document_id": ""}).text

        assert "LLM_BASE_URL" in html

    def test_api_key_never_appears_in_responses(self, corpus_artifact: Path) -> None:
        secret = "sk-ultra-secret-key-xyz"
        env = {"LLM_BASE_URL": "http://127.0.0.1:9", "LLM_MODEL": "m", "LLM_API_KEY": secret}
        client = client_for(corpus_artifact, env)

        pages = [
            client.get("/").text,
            client.post("/chat", data={"query": "ἔτους", "document_id": ""}).text,
            client.get("/search", params={"query": "ἔτους"}).text,
        ]

        for page in pages:
            assert secret not in page

    def test_empty_model_answer_renders_error(self, corpus_artifact: Path) -> None:
        with MockProviderServer(response_body={"choices": [{"message": {"content": ""}}]}) as mock:
            client = client_for(corpus_artifact, env_with(mock))

            html = client.post("/chat", data={"query": "ἔτους", "document_id": ""}).text

        assert "empty answer" in html.lower()
        assert '<span class="label">model-generated</span>' not in html
