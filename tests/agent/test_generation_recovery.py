"""Recovery through real adapters with local simulated endpoints."""

import asyncio
import json
from typing import Any

import httpx2
import pytest
from pydantic_ai import AgentRunResultEvent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage

from papyrus_chat.agent.context import ResearchPolicy
from papyrus_chat.agent.context.responses import RecoverableResponsesModel
from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.corpus import CorpusService
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


class ChatAdapter(OpenAIChatModel):
    request = OpenAIChatModel.request
    request_stream = OpenAIChatModel.request_stream


class ResponsesAdapter(RecoverableResponsesModel):
    request = OpenAIResponsesModel.request
    request_stream = OpenAIResponsesModel.request_stream


def response_payload(body, text, exhausted):
    return {
        "id": "resp-1",
        "object": "response",
        "created_at": 1,
        "model": body["model"],
        "status": "incomplete" if exhausted else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if exhausted else None,
        "output": [
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "incomplete" if exhausted else "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 30},
        },
    }


def endpoint_response(body, api, kind, exhausted):
    text = "PRIVATE_DRAFT" if exhausted else "Model-supplied background: complete."
    if api == "chat":
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if exhausted and kind == "thinking":
            message = {"role": "assistant", "reasoning_content": "PRIVATE_THINKING"}
        elif exhausted and kind == "tool":
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "broken",
                        "type": "function",
                        "function": {"name": "describe_corpus", "arguments": '{"broken":'},
                    }
                ],
            }
        reason = "length" if exhausted else "stop"
        base = {"id": "chat-1", "model": body["model"], "created": 1}
        usage = {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}
        if not body.get("stream"):
            return httpx2.Response(
                200,
                json={
                    **base,
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                    "usage": usage,
                },
            )
        if "tool_calls" in message:
            message["tool_calls"] = [
                {**call, "index": i} for i, call in enumerate(message["tool_calls"])
            ]
        events = [
            {
                **base,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": message, "finish_reason": reason}],
            },
            {**base, "object": "chat.completion.chunk", "choices": [], "usage": usage},
        ]
    else:
        payload = response_payload(body, text, exhausted)
        if not body.get("stream"):
            return httpx2.Response(200, json=payload)
        item = payload["output"][0]
        events = [
            {
                "type": "response.created",
                "response": {**payload, "status": "in_progress", "output": []},
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**item, "content": []},
            },
            {
                "type": "response.content_part.added",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            },
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {
                "type": "response.incomplete" if exhausted else "response.completed",
                "response": payload,
            },
        ]
    data = (
        "".join(
            f"data: {json.dumps({**event, 'sequence_number': i})}\n\n"
            for i, event in enumerate(events)
        )
        + "data: [DONE]\n\n"
    )
    return httpx2.Response(200, text=data, headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize(
    "api,kind", [("chat", "text"), ("chat", "thinking"), ("chat", "tool"), ("responses", "text")]
)
@pytest.mark.parametrize("fail_twice", [False, True])
def test_exhaustion_is_discarded_before_execution_and_usage_is_preserved(
    corpus_artifact, api, kind, fail_twice
):
    async def scenario():
        requests = []

        def endpoint(request):
            body = json.loads(request.content)
            requests.append(body)
            assert len(requests) <= 2
            assert bool(body.get("stream")) == (len(requests) == 1)
            assert body.get("tools")
            key = "max_completion_tokens" if api == "chat" else "max_output_tokens"
            assert body[key] == 20000
            return endpoint_response(body, api, kind, len(requests) == 1 or fail_twice)

        async with httpx2.AsyncClient(transport=httpx2.MockTransport(endpoint)) as client:
            cls = ChatAdapter if api == "chat" else ResponsesAdapter
            model = cls(
                "gpt-5.2",
                provider=OpenAIProvider(api_key="test", http_client=client),
                profile=OpenAIModelProfile(
                    supports_thinking=True,
                    openai_supports_reasoning_effort_none=True,
                    openai_chat_supports_multiple_system_messages=False,
                ),
            )
            service = CorpusService(StructuredCorpusSearch(corpus_artifact / "corpus.sqlite"))
            deps = CorpusToolDeps(service)
            usage = RunUsage()
            agent = create_research_agent(
                ProviderConfig(model="gpt-5.2", base_url="https://example.invalid/v1"),
                service,
                model=model,
                policy=ResearchPolicy(context_window=131072, max_tokens=20000),
            )
            events = []

            async def run():
                async with agent.run_stream_events(
                    "Answer briefly.", deps=deps, usage=usage
                ) as stream:
                    async for event in stream:
                        events.append(event)

            try:
                if fail_twice:
                    with pytest.raises(UnexpectedModelBehavior, match="after one recovery"):
                        await run()
                else:
                    await run()
                    result = next(e.result for e in events if isinstance(e, AgentRunResultEvent))
                    assert result.output == "Model-supplied background: complete."
                    assert "PRIVATE_DRAFT" not in str(result.all_messages())
                assert usage.requests == 2
                assert usage.input_tokens == 200
                assert usage.output_tokens == 80
                assert deps.research_state.citation_repairs == 0
                assert not deps.research_state.ledger.records
                recovery = requests[-1]
                if api == "chat":
                    assert recovery["reasoning_effort"] == "none"
                else:
                    assert recovery["reasoning"]["effort"] == "none"
            finally:
                service.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_cancellation_during_direct_recovery_stops_requests(corpus_artifact, api):
    async def scenario():
        recovering = asyncio.Event()
        calls = 0

        async def endpoint(request):
            nonlocal calls
            calls += 1
            body = json.loads(request.content)
            if calls == 1:
                return endpoint_response(body, api, "text", True)
            assert calls == 2 and not body.get("stream")
            recovering.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled recovery resumed")

        async with httpx2.AsyncClient(transport=httpx2.MockTransport(endpoint)) as client:
            cls = ChatAdapter if api == "chat" else ResponsesAdapter
            model = cls("gpt-5.2", provider=OpenAIProvider(api_key="test", http_client=client))
            service = CorpusService(StructuredCorpusSearch(corpus_artifact / "corpus.sqlite"))
            agent = create_research_agent(
                ProviderConfig(base_url="https://example.invalid/v1", model="gpt-5.2"),
                service,
                model=model,
            )

            async def run():
                async with agent.run_stream_events(
                    "Answer.", deps=CorpusToolDeps(service)
                ) as events:
                    async for _ in events:
                        pass

            task = asyncio.create_task(run())
            try:
                await asyncio.wait_for(recovering.wait(), 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert calls == 2
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                service.close()

    asyncio.run(scenario())
