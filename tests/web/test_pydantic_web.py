"""Integration contract for the stock Pydantic AI web chat application."""

from pathlib import Path

from pydantic_ai.models.test import TestModel
from starlette.testclient import TestClient

from papyrus_chat.web.application import load_app

TEST_ENV = {
    "LLM_BASE_URL": "https://provider.example/v1",
    "LLM_MODEL": "research-model",
}


def test_load_app_serves_stock_chat_routes_with_local_html(
    corpus_artifact: Path, tmp_path: Path
) -> None:
    html = tmp_path / "chat.html"
    html.write_text(
        "<!doctype html><html><body><main id='chat'>Papyrologist chat</main></body></html>",
        encoding="utf-8",
    )

    app = load_app(
        corpus_artifact,
        env=TEST_ENV,
        model=TestModel(custom_output_text="Model-supplied background."),
        html_source=html,
    )
    client = TestClient(app, base_url="http://localhost")

    assert client.get("/").status_code == 200
    assert "Papyrologist chat" in client.get("/").text
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/configure").status_code in {200, 405}
    assert client.get("/api/chat").status_code == 405
    for legacy_path in ("/search", "/documents/dclp:DCLP/23/23944.xml", "/chat"):
        legacy_response = client.get(legacy_path)
        assert legacy_response.status_code in {200, 404}
        if legacy_response.status_code == 200:
            assert "Papyrologist chat" in legacy_response.text


def test_loaded_app_retains_artifact_and_agent_dependencies(
    corpus_artifact: Path, tmp_path: Path
) -> None:
    html = tmp_path / "chat.html"
    html.write_text("<main>Chat</main>", encoding="utf-8")

    app = load_app(
        corpus_artifact,
        env=TEST_ENV,
        model=TestModel(custom_output_text="Ready."),
        html_source=html,
    )

    assert app.state.artifact == corpus_artifact
    assert app.state.agent is not None
    assert app.state.tool_service is not None


def test_stock_chat_api_streams_a_deterministic_response(
    corpus_artifact: Path, tmp_path: Path
) -> None:
    html = tmp_path / "chat.html"
    html.write_text("<main>Chat</main>", encoding="utf-8")
    app = load_app(
        corpus_artifact,
        env=TEST_ENV,
        model=TestModel(call_tools=[], custom_output_text="Model-supplied background."),
        html_source=html,
    )
    client = TestClient(app, base_url="http://localhost")

    response = client.post(
        "/api/chat",
        json={
            "trigger": "submit-message",
            "id": "thread-1",
            "messages": [
                {
                    "id": "message-1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "What can be concluded?"}],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "text-delta" in response.text
    assert "Model-supplied" in response.text
