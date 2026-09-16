"""Continuous research, explicit limits, and bounded citation repair."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai import AgentRunResultEvent, UsageLimits, WebSearchTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from papyrus_chat.agent.context import ResearchPolicy
from papyrus_chat.agent.context.accounting import RequestAccounting
from papyrus_chat.agent.context.compaction import SUMMARY_INSTRUCTIONS, ContextBudgetExceeded
from papyrus_chat.agent.context.limits import (
    BUDGET_INSTRUCTION_NAME,
    BUDGET_REMAINING_PREFIX,
    FINAL_ANSWER_INSTRUCTIONS,
    FINAL_ANSWER_PROMPT,
)
from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps
from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource
from papyrus_chat.chat.provider import ProviderConfig
from papyrus_chat.corpus import CorpusService
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


@pytest.fixture()
def service(corpus_artifact: Path) -> CorpusService:
    return CorpusService(StructuredCorpusSearch(corpus_artifact / "corpus.sqlite"))


def make_agent(service, dialogue, policy=None):
    return create_research_agent(
        ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
        service,
        model=FunctionModel(dialogue),
        policy=policy or ResearchPolicy(),
    )


def budget_note(info) -> str:
    """The one dynamic budget instruction, so wording edits cannot break tests."""
    parts = [
        part.content
        for part in info.model_request_parameters.instruction_parts or []
        if getattr(part, "name", None) == BUDGET_INSTRUCTION_NAME
    ]
    assert len(parts) == 1
    return parts[0]


@pytest.mark.parametrize("limit,answer_request", [(1, 1), (2, 2), (3, 2)])
def test_explicit_request_cap_reserves_an_evidence_based_final_answer(
    service, limit, answer_request
):
    calls = 0

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        assert calls <= limit
        if calls == answer_request:
            assert not info.function_tools
            assert not info.model_request_parameters.native_tools
            assert FINAL_ANSWER_INSTRUCTIONS in (info.instructions or "")
            return ModelResponse([TextPart("No corpus evidence was inspected.")])
        assert info.function_tools
        return ModelResponse([ToolCallPart("describe_corpus", {}, f"inventory-{calls}")])

    agent = make_agent(
        service, dialogue, ResearchPolicy(context_window=131072, research_request_limit=limit)
    )
    result = agent.run_sync(
        "Investigate.",
        deps=CorpusToolDeps(service),
        capabilities=[NativeTool(WebSearchTool())],
    )
    assert result.output.startswith("No corpus evidence was inspected.")
    assert calls == result.usage.requests == answer_request


@pytest.mark.parametrize("failed_output", ["empty", "citation", "tool"])
def test_budget_synthesis_reserves_one_output_retry(service, failed_output):
    calls = 0

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        assert calls <= 4
        if calls <= 2:
            assert info.function_tools
            return ModelResponse([ToolCallPart("describe_corpus", {}, f"inventory-{calls}")])
        assert not info.function_tools
        assert not info.model_request_parameters.native_tools
        if calls == 3:
            if failed_output == "empty":
                return ModelResponse([TextPart("")], finish_reason="stop")
            if failed_output == "citation":
                return ModelResponse(
                    [TextPart("Corpus evidence: https://papyri.info/ddbdp/invented;1;999")]
                )
            return ModelResponse([ToolCallPart("describe_corpus", {}, "forbidden-research")])
        return ModelResponse([TextPart("No corpus evidence was inspected.")])

    deps = CorpusToolDeps(service)
    result = make_agent(service, dialogue, ResearchPolicy(research_request_limit=4)).run_sync(
        "Investigate.", deps=deps, capabilities=[NativeTool(WebSearchTool())]
    )
    assert result.output.startswith("No corpus evidence was inspected.")
    assert calls == result.usage.requests == 4
    assert len(deps.research_state.ledger.executions) == 2


def test_request_budget_is_visible_before_research_ends(service):
    calls = 0

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        if calls <= 2:
            assert f"{BUDGET_REMAINING_PREFIX} {3 - calls}." in budget_note(info)
            return ModelResponse([ToolCallPart("describe_corpus", {}, f"inventory-{calls}")])
        assert calls == 3
        assert not info.function_tools
        return ModelResponse([TextPart("No corpus evidence was inspected.")])

    result = make_agent(service, dialogue, ResearchPolicy(research_request_limit=4)).run_sync(
        "Investigate.", deps=CorpusToolDeps(service)
    )
    assert calls == result.usage.requests == 3


def test_compaction_cannot_spend_the_final_answer_request(service):
    calls = 0
    policy = ResearchPolicy(research_request_limit=2)

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        assert SUMMARY_INSTRUCTIONS not in (info.instructions or "")
        if calls == 1:
            return ModelResponse(
                [TextPart("πάπυρος " * 10000), ToolCallPart("describe_corpus", {}, "inventory")]
            )
        assert calls == 2
        assert not info.function_tools
        assert "documents" in str(messages)
        serialized = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
        assert serialized[-1]["parts"][-1]["part_kind"] == "user-prompt"
        assert FINAL_ANSWER_PROMPT in serialized[-1]["parts"][-1]["content"]
        assert (
            RequestAccounting().estimate(messages, info.model_request_parameters)
            <= policy.input_limit
        )
        return ModelResponse([TextPart("No corpus evidence was inspected.")])

    deps = CorpusToolDeps(service)
    result = make_agent(service, dialogue, policy).run_sync("Investigate.", deps=deps)
    assert result.output.startswith("No corpus evidence was inspected.")
    assert deps.research_state.compactions == 1
    assert calls == result.usage.requests == 2


def test_explicit_request_cap_still_bounds_a_failed_final_answer(service):
    calls = 0

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        assert calls == 1
        assert not info.function_tools
        return ModelResponse(
            [TextPart("Corpus evidence: https://papyri.info/ddbdp/invented;1;999")]
        )

    agent = make_agent(service, dialogue, ResearchPolicy(research_request_limit=1))
    with pytest.raises(UsageLimitExceeded, match="request limit"):
        agent.run_sync("Investigate.", deps=CorpusToolDeps(service))
    assert calls == 1


@pytest.mark.parametrize("repair_citation", [False, True])
def test_streaming_request_budget_returns_inspected_evidence(
    tmp_path, fixture_git_repo, repair_citation
):
    artifact = tmp_path / "corpus"
    build_artifact(
        ["ddbdp"],
        output=artifact,
        source=LocalGitSource(fixture_git_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
    )
    service = CorpusService.open(artifact)

    async def scenario():
        calls = 0
        citation = "https://papyri.info/ddbdp/p.mich;8;480"
        answer = f"Dissimilar handwriting appears in [P.Mich. 8.480]({citation})."

        async def dialogue(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield {
                    0: DeltaToolCall(
                        name="inspect_documents",
                        json_args=json.dumps({"document_ids": ["ddbdp:DDbDP/27/27093.xml"]}),
                        tool_call_id="inspection",
                    )
                }
                return
            assert calls <= (3 if repair_citation else 2)
            assert not info.function_tools
            assert citation in str(messages)
            assert "ἀνόμοιά" in str(messages)
            if calls == 2:
                serialized = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
                assert serialized[-1]["parts"][-1]["part_kind"] == "user-prompt"
                assert FINAL_ANSWER_PROMPT in serialized[-1]["parts"][-1]["content"]
            if repair_citation and calls == 2:
                yield "Corpus evidence: https://papyri.info/ddbdp/invented;1;999"
                return
            yield answer

        agent = create_research_agent(
            ProviderConfig(base_url="https://provider.example/v1", model="research-model"),
            service,
            model=FunctionModel(stream_function=dialogue),
            policy=ResearchPolicy(research_request_limit=3 if repair_citation else 2),
        )
        async with agent.run_stream_events(
            "Find evidence about handwriting.", deps=CorpusToolDeps(service)
        ) as stream:
            events = [event async for event in stream]
        result = next(event.result for event in events if isinstance(event, AgentRunResultEvent))
        assert result.output.startswith(answer)
        assert calls == result.usage.requests == (3 if repair_citation else 2)

    try:
        asyncio.run(scenario())
    finally:
        service.close()


@pytest.mark.parametrize("repair_citation", [False, True])
def test_summary_requests_count_toward_the_final_answer_budget(service, repair_citation):
    calls = 0

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        if SUMMARY_INSTRUCTIONS in (info.instructions or ""):
            assert calls == 2
            return ModelResponse([TextPart("Inventory retrieved; answer with available evidence.")])
        if calls >= 4:
            assert calls <= 5
            assert not info.function_tools
            if repair_citation and calls == 4:
                return ModelResponse(
                    [TextPart("Corpus evidence: https://papyri.info/ddbdp/invented;1;999")]
                )
            return ModelResponse([TextPart("No corpus evidence was inspected.")])
        assert calls in {1, 3}
        assert info.function_tools
        assert f"{BUDGET_REMAINING_PREFIX} {4 - calls}." in budget_note(info)
        return ModelResponse(
            [
                TextPart("πάπυρος " * 10000),
                ToolCallPart("describe_corpus", {}, f"inventory-{calls}"),
            ]
        )

    deps = CorpusToolDeps(service)
    result = make_agent(service, dialogue, ResearchPolicy(research_request_limit=5)).run_sync(
        "Investigate.", deps=deps
    )
    assert result.output.startswith("No corpus evidence was inspected.")
    assert deps.research_state.summary_requests == 1
    assert calls == result.usage.requests == (5 if repair_citation else 4)


def test_history_below_the_trigger_does_not_compact(service):
    text = "πάπυρος " * 3000
    calls = 0

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([TextPart(text), ToolCallPart("describe_corpus", {}, "inventory")])
        # The runtime trigger comparison adds instruction overhead on top of this
        # history-only measure; the run-level assertions below prove the outcome.
        size = RequestAccounting().estimate(messages, info.model_request_parameters)
        assert policy.trigger_tokens / 2 < size < policy.trigger_tokens
        assert "Research checkpoint" not in str(messages)
        return ModelResponse([TextPart("No corpus evidence was inspected.")])

    policy = ResearchPolicy(context_window=65536)
    deps = CorpusToolDeps(service)
    result = make_agent(service, dialogue, policy).run_sync("Investigate.", deps=deps)
    assert result.output.startswith("No corpus evidence was inspected.")
    assert calls == result.usage.requests == 2
    assert deps.research_state.compactions == 0
    assert deps.research_state.summary_disabled is False


def test_failed_summary_latches_mechanical_compaction(service):
    summaries = 0
    research = 0
    answer = "No corpus evidence was inspected."

    def dialogue(messages, info):
        nonlocal summaries, research
        if SUMMARY_INSTRUCTIONS in (info.instructions or ""):
            summaries += 1
            raise RuntimeError("Summary endpoint unavailable")
        research += 1
        assert research <= 3
        if research == 3:
            return ModelResponse([TextPart(answer)])
        return ModelResponse(
            [
                TextPart("πάπυρος " * 12000),
                ToolCallPart("describe_corpus", {}, f"inventory-{research}"),
            ]
        )

    deps = CorpusToolDeps(service)
    result = make_agent(service, dialogue, ResearchPolicy()).run_sync("Investigate.", deps=deps)
    assert result.output.startswith(answer)
    assert summaries == 1, "A failed summary must latch and not be retried in the same run."
    assert deps.research_state.summary_disabled is True
    assert deps.research_state.compactions >= 2
    assert deps.research_state.summary_requests == 1


def test_caller_request_limit_remains_a_hard_cap(service):
    calls = 0

    def dialogue(messages, info):
        nonlocal calls
        calls += 1
        assert calls == 1
        return ModelResponse([ToolCallPart("describe_corpus", {}, "inventory")])

    agent = make_agent(service, dialogue, ResearchPolicy(research_request_limit=3))
    with pytest.raises(UsageLimitExceeded, match="request_limit"):
        agent.run_sync(
            "Investigate.", deps=CorpusToolDeps(service), usage_limits=UsageLimits(request_limit=1)
        )
    assert calls == 1


@pytest.mark.parametrize("valid_repair", [True, False])
@pytest.mark.parametrize("request_limit", [None, 3])
def test_natural_answer_gets_one_tool_free_citation_repair(service, valid_repair, request_limit):
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

    agent = make_agent(service, dialogue, ResearchPolicy(research_request_limit=request_limit))
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
