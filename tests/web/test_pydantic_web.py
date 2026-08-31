"""Integration contract for the stock Pydantic AI web chat application."""

from pathlib import Path

from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from starlette.testclient import TestClient

from papyrus_chat.web.application import load_app

TEST_ENV = {
    "LLM_BASE_URL": "https://provider.example/v1",
    "LLM_MODEL": "research-model",
}


def _chat(client: TestClient):
    return client.post(
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


def _text_stream_model(*outputs: str) -> FunctionModel:
    attempts = iter(outputs)

    async def stream(messages, info):
        del messages, info
        yield next(attempts)

    return FunctionModel(stream_function=stream)


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

    response = _chat(client)

    assert response.status_code == 200
    assert "text-delta" in response.text
    assert "Model-supplied" in response.text


def test_chat_emits_only_the_answer_that_passes_output_validation(
    corpus_artifact: Path, tmp_path: Path
) -> None:
    app = load_app(
        corpus_artifact,
        env=TEST_ENV,
        model=_text_stream_model(
            "Corpus evidence: https://papyri.info/ddbdp/not-returned-by-a-tool.",
            "Scope and method: no corpus evidence matched the request.",
        ),
        html_source=tmp_path / "unused.html",
    )

    response = _chat(TestClient(app, base_url="http://localhost"))

    assert response.status_code == 200
    assert "not-returned-by-a-tool" not in response.text
    assert response.text.count("no corpus evidence matched") == 1


def test_chat_does_not_leak_text_when_output_validation_exhausts_retries(
    corpus_artifact: Path, tmp_path: Path
) -> None:
    invalid = "Corpus evidence: https://papyri.info/ddbdp/not-returned-by-a-tool."
    app = load_app(
        corpus_artifact,
        env=TEST_ENV,
        model=_text_stream_model(invalid, invalid),
        html_source=tmp_path / "unused.html",
    )

    response = _chat(TestClient(app, base_url="http://localhost"))

    assert response.status_code == 200
    assert "not-returned-by-a-tool" not in response.text
    assert '"type":"error"' in response.text


def test_chat_streams_tool_activity_before_buffered_final_text(
    corpus_artifact: Path, tmp_path: Path
) -> None:
    async def scripted_model(messages, info):
        del info
        tool_returned = any(
            isinstance(part, ToolReturnPart) for message in messages for part in message.parts
        )
        if not tool_returned:
            yield {
                0: DeltaToolCall(
                    name="describe_corpus", json_args="{}", tool_call_id="describe-corpus"
                )
            }
            return
        yield "Model-supplied background after corpus inspection."

    app = load_app(
        corpus_artifact,
        env=TEST_ENV,
        model=FunctionModel(stream_function=scripted_model),
        html_source=tmp_path / "unused.html",
    )

    response = _chat(TestClient(app, base_url="http://localhost"))

    assert response.status_code == 200
    assert response.text.index("tool-output-available") < response.text.index(
        "Model-supplied background"
    )
