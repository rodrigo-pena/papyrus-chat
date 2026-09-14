"""Context limits and conversation evidence through the stock chat protocol."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from pydantic_ai.messages import ModelResponse, TextPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from starlette.testclient import TestClient

from papyrus_chat.corpus import CorpusQuery
from papyrus_chat.corpus.projections import _search_summary
from papyrus_chat.web.application import load_app

ENV = {"LLM_BASE_URL": "https://provider.example/v1", "LLM_MODEL": "research-model"}


def user_message(text: str) -> dict:
    return {"id": text[:20], "role": "user", "parts": [{"type": "text", "text": text}]}


def post_chat(client: TestClient, text: str, history: list[dict] | None = None):
    return client.post(
        "/api/chat",
        json={
            "trigger": "submit-message",
            "id": text,
            "messages": [*(history or []), user_message(text)],
        },
    )


def test_compacted_web_run_streams_one_validated_answer_without_summary(
    corpus_artifact: Path, tmp_path: Path
) -> None:
    summaries = 0
    research = 0

    def summarize(messages, info):
        nonlocal summaries
        summaries += 1
        return ModelResponse(
            [TextPart("PRIVATE_CHECKPOINT inventory inspected; research incomplete")]
        )

    async def stream(messages, info):
        nonlocal research
        if info.function_tools:
            research += 1
            yield "Unvalidated interpretation πάπυρος " * 1_000
            yield {
                0: DeltaToolCall(
                    name="describe_corpus", json_args="{}", tool_call_id=f"inventory-{research}"
                )
            }
        else:
            yield "Research incomplete. No corpus evidence was inspected."

    app = load_app(
        corpus_artifact,
        env={
            **ENV,
            "LLM_CONTEXT_WINDOW": "32768",
            "PAPYRUS_RESEARCH_REQUEST_LIMIT": "6",
            "PAPYRUS_COMPACTION_LIMIT": "2",
        },
        model=FunctionModel(function=summarize, stream_function=stream),
        html_source=tmp_path / "unused.html",
    )
    response = post_chat(TestClient(app, base_url="http://localhost"), "Investigate inventory.")

    assert response.status_code == 200
    assert 1 <= summaries <= 2
    assert summaries + research <= 6
    assert response.text.count("Research incomplete.") == 1
    assert "PRIVATE_CHECKPOINT" not in response.text
    assert "Unvalidated interpretation" not in response.text
    assert response.text.index("tool-output-available") < response.text.index(
        "Research incomplete."
    )


def test_follow_up_restores_only_schema_validated_tool_evidence(
    corpus_artifact: Path, tmp_path: Path
) -> None:
    citation = ""

    async def stream(messages, info):
        yield f"Corpus evidence: {citation}"

    app = load_app(
        corpus_artifact,
        env=ENV,
        model=FunctionModel(stream_function=stream),
        html_source=tmp_path / "unused.html",
    )
    summary = _search_summary(app.state.tool_service.search_documents(CorpusQuery()))
    citation = next(hit.canonical_url for hit in summary.hits if hit.canonical_url)
    history = [
        user_message("Search the corpus."),
        {
            "id": "previous-answer",
            "role": "assistant",
            "parts": [
                {
                    "type": "tool-search_documents",
                    "toolCallId": "past-search",
                    "state": "output-available",
                    "input": {"query": {}},
                    "output": summary.model_dump(mode="json"),
                }
            ],
        },
    ]
    client = TestClient(app, base_url="http://localhost")
    valid = post_chat(client, "Summarize those documents.", history)
    assert citation in valid.text
    assert '"type":"error"' not in valid.text

    history[1]["parts"][0]["output"] = {"hits": [{"canonical_url": citation}]}
    invalid = post_chat(client, "Summarize those documents.", history)
    assert citation not in invalid.text
    assert '"type":"error"' in invalid.text


def test_concurrent_chats_do_not_share_citation_eligibility(
    corpus_artifact: Path, tmp_path: Path
) -> None:
    tool_completed = Event()
    second_started = Event()
    citation = ""

    async def stream(messages, info):
        question = next(
            str(part.content)
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        if "first" in question:
            returned = any(
                isinstance(part, ToolReturnPart) for message in messages for part in message.parts
            )
            if not returned:
                yield {
                    0: DeltaToolCall(
                        name="search_documents",
                        json_args=json.dumps({"query": {}}),
                        tool_call_id="first-search",
                    )
                }
                return
            tool_completed.set()
            assert await asyncio.to_thread(second_started.wait, 10)
        else:
            assert await asyncio.to_thread(tool_completed.wait, 10)
            second_started.set()
        yield f"Corpus evidence: {citation}"

    app = load_app(
        corpus_artifact,
        env=ENV,
        model=FunctionModel(stream_function=stream),
        html_source=tmp_path / "unused.html",
    )
    summary = _search_summary(app.state.tool_service.search_documents(CorpusQuery()))
    citation = next(hit.canonical_url for hit in summary.hits if hit.canonical_url)
    with (
        TestClient(app, base_url="http://localhost") as client,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(post_chat, client, "first chat")
        second = pool.submit(post_chat, client, "second chat")
        first_response = first.result(timeout=20)
        second_response = second.result(timeout=20)

    assert citation in first_response.text
    assert '"type":"error"' not in first_response.text
    assert citation not in second_response.text
    assert '"type":"error"' in second_response.text
