"""Continuous research, explicit limits, and bounded citation repair."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai import WebSearchTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from papyrus_chat.agent.context import ResearchPolicy
from papyrus_chat.agent.context.compaction import SUMMARY_INSTRUCTIONS, ContextBudgetExceeded
from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps, CorpusToolService
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


@pytest.fixture()
def service(corpus_artifact: Path) -> CorpusToolService:
    return CorpusToolService(StructuredCorpusSearch(corpus_artifact / "corpus.sqlite"))


def make_agent(service, dialogue, policy=None):
    return create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=policy or ResearchPolicy(),
    )


def test_explicit_request_cap_raises_without_inventing_a_partial_answer(service):
    calls = 0

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        assert calls <= 3
        assert info.function_tools
        return ModelResponse([ToolCallPart("describe_corpus", {}, f"inventory-{calls}")])

    agent = make_agent(
        service, dialogue, ResearchPolicy(context_window=131072, research_request_limit=3)
    )
    with pytest.raises(UsageLimitExceeded):
        agent.run_sync("Investigate.", deps=CorpusToolDeps(service))
    assert calls == 3


@pytest.mark.parametrize("valid_repair", [True, False])
def test_natural_answer_gets_one_tool_free_citation_repair(service, valid_repair):
    calls = []
    answer = "No corpus evidence was inspected."

    def dialogue(messages, info):
        calls.append(info.model_request_parameters)
        assert len(calls) <= 2
        if len(calls) == 1:
            assert info.function_tools
            assert info.model_request_parameters.native_tools
        else:
            assert not info.function_tools
            assert not info.model_request_parameters.native_tools
            if valid_repair:
                return ModelResponse([TextPart(answer)])
        return ModelResponse(
            [TextPart("Corpus evidence: https://papyri.info/ddbdp/invented;1;999")]
        )

    agent = make_agent(service, dialogue)
    kwargs = {"deps": CorpusToolDeps(service), "capabilities": [NativeTool(WebSearchTool())]}
    if valid_repair:
        result = agent.run_sync("Investigate.", **kwargs)
        assert result.output.split("\n\nCoverage:")[0] == answer
    else:
        with pytest.raises(UnexpectedModelBehavior):
            agent.run_sync("Investigate.", **kwargs)
    assert len(calls) == 2


def test_oversized_current_question_is_rejected_before_request(service):
    def dialogue(messages, info):
        pytest.fail("Oversized user input must not reach the provider")

    agent = make_agent(service, dialogue)
    with pytest.raises(
        ContextBudgetExceeded, match="(?i)(prompt|question).*(large|fit|context|exceed)"
    ):
        agent.run_sync("πάπυρος " * 100000, deps=CorpusToolDeps(service))


@pytest.mark.parametrize("summary_fails", [False, True])
def test_long_tool_run_compacts_and_naturally_answers_even_if_summary_fails(service, summary_fails):
    research = 0
    summaries = 0
    answer = "The inventory has been examined. No corpus evidence was inspected."

    def dialogue(messages, info):
        nonlocal research, summaries
        if SUMMARY_INSTRUCTIONS in (info.instructions or ""):
            summaries += 1
            if summary_fails:
                raise RuntimeError("Summary endpoint unavailable")
            return ModelResponse(
                [TextPart("Inventory examined; textual research remains possible.")]
            )
        research += 1
        assert info.function_tools
        assert research <= 7
        if research == 7:
            assert "Investigate inventory." in str(messages)
            assert "documents" in str(messages)
            return ModelResponse([TextPart(answer)])
        return ModelResponse(
            [
                TextPart("πάπυρος " * 10000),
                ToolCallPart("describe_corpus", {}, f"inventory-{research}"),
            ]
        )

    result = make_agent(service, dialogue).run_sync(
        "Investigate inventory.", deps=CorpusToolDeps(service)
    )
    assert research == 7
    assert summaries >= 1
    assert result.output.split("\n\nCoverage:")[0] == answer


@pytest.mark.parametrize(
    "policy_limit,caller_limit,expected",
    [
        (None, None, None),
        (20000, None, 20000),
        (20000, 18000, 18000),
        (18000, 20000, 20000),
    ],
)
def test_generation_settings_preserve_caller_precedence(
    service, policy_limit, caller_limit, expected
):
    def dialogue(messages, info):
        settings = info.model_settings or {}
        if expected is None:
            assert "max_tokens" not in settings
        else:
            assert settings["max_tokens"] == expected
        return ModelResponse([TextPart("Model-supplied background: ready.")])

    agent = make_agent(
        service, dialogue, ResearchPolicy(context_window=131072, max_tokens=policy_limit)
    )
    result = agent.run_sync(
        "Hello.",
        deps=CorpusToolDeps(service),
        model_settings={"max_tokens": caller_limit} if caller_limit is not None else None,
    )
    assert result.output == "Model-supplied background: ready."


def test_followup_resets_explicit_request_budget_with_reused_dependencies(service):
    calls = 0

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse([TextPart("Model-supplied background: ready.")])

    agent = make_agent(service, dialogue, ResearchPolicy(research_request_limit=1))
    deps = CorpusToolDeps(service)
    first = agent.run_sync("Hello.", deps=deps)
    second = agent.run_sync("Continue.", deps=deps, message_history=first.all_messages())
    assert calls == 2
    assert second.output == first.output


def test_cancellation_during_summary_does_not_continue_research(service):
    async def scenario():
        started = asyncio.Event()
        research = 0

        async def dialogue(messages, info):
            nonlocal research
            if SUMMARY_INSTRUCTIONS in (info.instructions or ""):
                started.set()
                await asyncio.Event().wait()
            research += 1
            assert research == 1
            return ModelResponse(
                [
                    TextPart("πάπυρος " * 10000),
                    ToolCallPart("describe_corpus", {}, "inventory"),
                ]
            )

        agent = make_agent(service, dialogue)
        task = asyncio.create_task(agent.run("Investigate.", deps=CorpusToolDeps(service)))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert research == 1
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
