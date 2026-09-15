"""Exercise actual Chat Completions payloads against a strict, in-memory endpoint."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.testclient import TestClient

import papyrus_chat.agent.runtime as runtime
from papyrus_chat.agent.context.compaction import SUMMARY_INSTRUCTIONS
from papyrus_chat.agent.context.runtime import FINAL_INSTRUCTIONS
from papyrus_chat.agent.runtime import RESEARCH_INSTRUCTIONS
from papyrus_chat.web.application import load_app


# Capture the real adapters before conftest disables network model calls. This
# subclass is only used with MockTransport, which cannot contact a real provider.
class TransportChatModel(OpenAIChatModel):
    request = OpenAIChatModel.request
    request_stream = OpenAIChatModel.request_stream


class StrictChatEndpoint:
    def __init__(self, *, large_results: bool, fail_summary: bool) -> None:
        self.large_results = large_results
        self.fail_summary = fail_summary
        self.requests: list[dict[str, Any]] = []
        self.rejected_roles: list[list[str]] = []
        self.research_calls = 0
        self.summary_calls = 0
        self.final_calls = 0

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        self.requests.append(body)
        roles = [message["role"] for message in body["messages"]]
        if roles[0] != "system" or "system" in roles[1:]:
            self.rejected_roles.append(roles)
            return httpx2.Response(
                400,
                json={
                    "error": {
                        "message": "System message must be at the beginning.",
                        "type": "BadRequestError",
                    }
                },
            )
        instructions = body["messages"][0]["content"]
        if SUMMARY_INSTRUCTIONS in instructions:
            self.summary_calls += 1
            assert not body.get("tools")
            if self.fail_summary:
                return httpx2.Response(
                    400,
                    json={
                        "error": {
                            "message": "Summary unavailable",
                            "type": "BadRequestError",
                        }
                    },
                )
            return httpx2.Response(
                200,
                json={
                    "id": "summary",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "PRIVATE_CHECKPOINT inventory checked.",
                            },
                        }
                    ],
                },
            )
        assert body["stream"] is True
        if body.get("tools"):
            self.research_calls += 1
            assert RESEARCH_INSTRUCTIONS in instructions
            assert FINAL_INSTRUCTIONS not in instructions
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"inventory-{self.research_calls}",
                        "type": "function",
                        "function": {"name": "describe_corpus", "arguments": "{}"},
                    }
                ],
            }
            if self.large_results:
                delta["content"] = "UNVALIDATED πάπυρος " * 3000
            reason = "tool_calls"
        else:
            self.final_calls += 1
            assert RESEARCH_INSTRUCTIONS in instructions
            assert FINAL_INSTRUCTIONS in instructions
            if self.final_calls == 1:
                # The retry must retain both instructions in the single system message.
                content = "Corpus evidence: https://papyri.info/ddbdp/invented;99;1"
            else:
                content = "No corpus evidence was inspected; only inventory was checked."
            delta = {"role": "assistant", "content": content}
            reason = "stop"
        chunks = [
            {
                "id": "response",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": "response",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            },
        ]
        data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx2.Response(200, text=data, headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize(
    "large_results,fail_summary", [(False, False), (True, False), (True, True)]
)
def test_strict_chat_endpoint_accepts_research_compaction_and_finalization(
    corpus_artifact: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    large_results: bool,
    fail_summary: bool,
) -> None:
    endpoint = StrictChatEndpoint(large_results=large_results, fail_summary=fail_summary)
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(endpoint))

    def provider(**kwargs):
        return OpenAIProvider(**kwargs, http_client=http_client)

    monkeypatch.setattr(runtime, "OpenAIProvider", provider)
    monkeypatch.setattr(runtime, "OpenAIChatModel", TransportChatModel)
    try:
        app = load_app(
            corpus_artifact,
            env={
                "LLM_BASE_URL": "https://strict.invalid/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "Qwen3.8-Flash-Next-FP8",
                "LLM_CONTEXT_WINDOW": "32768",
                "PAPYRUS_RESEARCH_REQUEST_LIMIT": "3",
                "PAPYRUS_COMPACTION_LIMIT": "1",
            },
            html_source=tmp_path / "unused.html",
        )
        with TestClient(app, base_url="http://localhost") as client:
            response = client.post(
                "/api/chat",
                json={
                    "trigger": "submit-message",
                    "id": "handwriting",
                    "messages": [
                        {
                            "id": "question",
                            "role": "user",
                            "parts": [
                                {
                                    "type": "text",
                                    "text": "Show how handwriting was supposed to look.",
                                }
                            ],
                        }
                    ],
                },
            )
        assert not endpoint.rejected_roles, endpoint.rejected_roles
        assert response.status_code == 200
        assert '"type":"error"' not in response.text
        assert "only inventory was checked" in response.text
        assert "incomplete" in response.text
        assert "invented;99;1" not in response.text
        assert "PRIVATE_CHECKPOINT" not in response.text
        assert "UNVALIDATED" not in response.text
        assert endpoint.final_calls == 2
        assert endpoint.research_calls + endpoint.summary_calls <= 3
        assert endpoint.summary_calls == int(large_results)
    finally:
        asyncio.run(http_client.aclose())
