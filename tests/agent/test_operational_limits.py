"""Explicit limits stop work, without triggering a research-ending answer."""

import asyncio
from decimal import Decimal

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from papyrus_chat.agent.context import ResearchPolicy
from papyrus_chat.agent.context.compaction import SUMMARY_INSTRUCTIONS
from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.corpus import CorpusService


def test_cost_limit_covers_recovery_and_rejects_over_budget_text(corpus_artifact):
    service = CorpusService.open(corpus_artifact)
    calls = 0

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        assert calls <= 2
        return ModelResponse(
            [TextPart("PRIVATE_DRAFT")],
            finish_reason="length" if calls == 1 else "stop",
            usage=RequestUsage(input_tokens=10, output_tokens=20, cost=Decimal("0.06")),
        )

    agent = create_research_agent(
        ProviderConfig(base_url="https://example.invalid", model="gpt-5.2"),
        service,
        model=FunctionModel(dialogue),
        policy=ResearchPolicy(cost_limit_usd=Decimal("0.1")),
    )
    try:
        with pytest.raises(UsageLimitExceeded, match="cost_limit"):
            agent.run_sync("Answer.", deps=CorpusToolDeps(service))
        assert calls == 2
    finally:
        service.close()


def test_timeout_applies_during_summarization(corpus_artifact):
    async def scenario():
        summaries = research = 0

        async def dialogue(messages, info):
            nonlocal summaries, research
            if SUMMARY_INSTRUCTIONS in (info.instructions or ""):
                summaries += 1
                await asyncio.Event().wait()
            research += 1
            assert research == 1
            return ModelResponse(
                [TextPart("πάπυρος " * 10000), ToolCallPart("describe_corpus", {}, "1")]
            )

        service = CorpusService.open(corpus_artifact)
        agent = create_research_agent(
            ProviderConfig(base_url="https://example.invalid", model="test"),
            service,
            model=FunctionModel(dialogue),
            policy=ResearchPolicy(run_timeout_seconds=0.1),
        )
        try:
            with pytest.raises(UsageLimitExceeded, match="elapsed-time"):
                await agent.run("Research.", deps=CorpusToolDeps(service))
            assert research == summaries == 1
        finally:
            service.close()

    asyncio.run(scenario())
