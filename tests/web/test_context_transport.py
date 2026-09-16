"""Actual Chat Completions payloads against a strict, in-memory endpoint."""

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
from papyrus_chat.agent.runtime import RESEARCH_INSTRUCTIONS
from papyrus_chat.web.application import load_app


class TransportChatModel(OpenAIChatModel):
    # Capture real adapters before the network-blocking fixture patches them.
    # Every request uses MockTransport, so no provider can be contacted.
    request = OpenAIChatModel.request
    request_stream = OpenAIChatModel.request_stream


class StrictChatEndpoint:
    def __init__(self, *, large_results=False, fail_summary=False, exhaust_at=None):
        self.large_results = large_results
        self.fail_summary = fail_summary
        self.exhaust_at = exhaust_at
        self.requests: list[dict[str, Any]] = []
        self.research_calls = 0
        self.summary_calls = 0
        self.repair_calls = 0
        self.recovery_calls = 0
        self.pending_recovery = False

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        self.requests.append(body)
        roles = [message["role"] for message in body["messages"]]
        assert roles[0] == "system"
        assert "system" not in roles[1:], "Adapter must preserve one leading system message"
        instructions = body["messages"][0]["content"]
        if SUMMARY_INSTRUCTIONS in instructions:
            self.summary_calls += 1
            assert not body.get("tools")
            assert not body.get("stream")
            if self.fail_summary:
                return httpx2.Response(
                    400,
                    json={"error": {"message": "Summary unavailable", "type": "BadRequestError"}},
                )
            return self.completion(
                body,
                {"role": "assistant", "content": "PRIVATE_CHECKPOINT inventory checked."},
                "stop",
            )

        assert RESEARCH_INSTRUCTIONS in instructions
        if not body.get("stream"):
            assert self.pending_recovery
            self.pending_recovery = False
            self.recovery_calls += 1
            message, reason = self.research_message()
            return self.completion(body, message, reason)

        if body.get("tools"):
            self.research_calls += 1
            assert self.research_calls <= 3, "Model should naturally answer on its third request"
            message, reason = self.research_message()
            if self.research_calls == self.exhaust_at:
                self.pending_recovery = True
                message = {"role": "assistant", "content": "UNFINISHED_DRAFT"}
                if self.research_calls == 1:
                    message["tool_calls"] = [
                        {
                            "id": "truncated",
                            "type": "function",
                            "function": {"name": "describe_corpus", "arguments": '{"unfinished":'},
                        }
                    ]
                reason = "length"
        else:
            self.repair_calls += 1
            assert self.repair_calls == 1
            message = {
                "role": "assistant",
                "content": "No corpus evidence was inspected; only inventory was checked.",
            }
            reason = "stop"
        return self.stream(body, message, reason)

    def research_message(self):
        if self.research_calls == 3:
            return {
                "role": "assistant",
                "content": "Corpus evidence: https://papyri.info/ddbdp/invented;99;1",
            }, "stop"
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"inventory-{self.research_calls}",
                    "type": "function",
                    "function": {"name": "describe_corpus", "arguments": "{}"},
                }
            ],
        }
        if self.large_results:
            message["content"] = "UNVALIDATED πάπυρος " * 3000
        return message, "tool_calls"

    @staticmethod
    def completion(body, message, reason):
        return httpx2.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": reason, "message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    @staticmethod
    def stream(body, message, reason):
        if "tool_calls" in message:
            message["tool_calls"] = [
                {"index": i, **call} for i, call in enumerate(message["tool_calls"])
            ]
        chunks = [
            {
                "id": "response",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            for delta, finish in [(message, None), ({}, reason)]
        ]
        chunks.append(
            {
                "id": "response",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }
        )
        data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx2.Response(200, text=data, headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize(
    "large_results,fail_summary,exhaust_at,output_env",
    [
        (False, False, None, {}),
        (True, False, None, {}),
        (True, True, None, {}),
        (False, False, 1, {}),
        (False, False, 3, {}),
        (True, False, None, {"LLM_MAX_TOKENS": "18000", "PAPYRUS_SUMMARY_MAX_TOKENS": "12000"}),
    ],
)
def test_strict_chat_endpoint_accepts_continuous_research_and_repair(
    corpus_artifact: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    large_results,
    fail_summary,
    exhaust_at,
    output_env,
):
    endpoint = StrictChatEndpoint(
        large_results=large_results, fail_summary=fail_summary, exhaust_at=exhaust_at
    )
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(endpoint))
    monkeypatch.setattr(
        runtime,
        "OpenAIProvider",
        lambda **kwargs: OpenAIProvider(**kwargs, http_client=http_client),
    )
    monkeypatch.setattr(runtime, "OpenAIChatModel", TransportChatModel)
    try:
        app = load_app(
            corpus_artifact,
            env={
                "LLM_BASE_URL": "https://strict.invalid/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "Qwen3.8-Flash-Next-FP8",
                "LLM_CONTEXT_WINDOW": "32768",
                **output_env,
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
        assert response.status_code == 200
        assert '"type":"error"' not in response.text
        assert response.text.count("only inventory was checked") == 1
        for private_text in (
            "invented;99;1",
            "PRIVATE_CHECKPOINT",
            "UNVALIDATED",
            "UNFINISHED_DRAFT",
        ):
            assert private_text not in response.text
        assert endpoint.research_calls == 3
        assert endpoint.repair_calls == 1
        assert endpoint.recovery_calls == int(exhaust_at is not None)
        assert bool(endpoint.summary_calls) == large_results
        for body in endpoint.requests:
            summary = SUMMARY_INSTRUCTIONS in body["messages"][0]["content"]
            expected = output_env.get("PAPYRUS_SUMMARY_MAX_TOKENS" if summary else "LLM_MAX_TOKENS")
            actual = body.get("max_completion_tokens", body.get("max_tokens"))
            assert actual == (int(expected) if expected is not None else None)
    finally:
        asyncio.run(http_client.aclose())
