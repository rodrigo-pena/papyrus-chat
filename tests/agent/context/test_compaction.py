import asyncio
import json

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RunUsage

from papyrus_chat.agent.context import ResearchPolicy, ResearchRunState
from papyrus_chat.agent.context.accounting import RequestAccounting
from papyrus_chat.agent.context.compaction import bounded_history, compact_history, complete_blocks
from papyrus_chat.agent.context.evidence import EvidenceRecord


def test_summary_fits_serialized_payload_and_preserves_latest_research_decision():
    policy = ResearchPolicy()
    decision = "Searches are sufficient; synthesize the authorship evidence already inspected."
    seen = []

    def summarizer(messages, info):
        seen.append(messages)
        assert decision in str(messages)
        assert (
            RequestAccounting().estimate(messages, info.model_request_parameters)
            <= policy.summary_input_limit
        )
        return ModelResponse([TextPart(decision)])

    model = FunctionModel(summarizer)
    question = ModelRequest(parts=[UserPromptPart("Find handwriting evidence.")])
    state = ResearchRunState(question=question)
    # Real tool records are JSON nested inside a serialized user message. Quotes
    # and backslashes increase that final payload beyond the inner history size.
    state.ledger.records = [
        EvidenceRecord("describe_corpus", str(i), {}, {"text": '"a" \\ x ' * 150})
        for i in range(20)
    ]
    request = ModelRequestContext(
        model=model,
        messages=[question, ModelResponse([TextPart(decision)]), ModelRequest(parts=[])],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    context = RunContext(deps=None, model=model, usage=RunUsage())
    compacted = asyncio.run(compact_history(context, request, state, policy))
    assert seen, "A full ledger must not silently disable summarization."
    assert compacted is not None
    assert state.summary == decision
    assert state.summary_requests == 1


@pytest.mark.parametrize("summary_fails", [False, True])
def test_recent_complete_exchange_has_priority_over_old_records(summary_fails):
    question = ModelRequest(parts=[UserPromptPart("Find handwriting evidence.")])
    state = ResearchRunState(question=question)
    state.ledger.records = [
        EvidenceRecord("describe_corpus", str(i), {}, {"text": "older material " * 100})
        for i in range(20)
    ]
    decision = "Conclude using inspected authorship evidence. " * 30
    latest = [
        ModelResponse([TextPart(decision), ToolCallPart("describe_corpus", {}, "latest")]),
        ModelRequest([ToolReturnPart("describe_corpus", {"collections": []}, "latest")]),
    ]

    def summarizer(messages, info):
        if summary_fails:
            raise RuntimeError("Summary unavailable")
        return ModelResponse([TextPart("The investigation is ready for synthesis.")])

    model = FunctionModel(summarizer)
    request = ModelRequestContext(
        model=model,
        messages=[question, *latest],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    context = RunContext(deps=None, model=model, usage=RunUsage())
    policy = ResearchPolicy(context_window=8192)
    compacted = asyncio.run(compact_history(context, request, state, policy))
    if compacted is None:
        compacted = bounded_history(request.messages, state, ModelRequestParameters(), 3000)
    assert latest[0] in compacted
    assert latest[1] in compacted
    assert RequestAccounting().estimate(compacted, ModelRequestParameters()) <= policy.target_tokens


def test_oversized_newest_exchange_is_not_replaced_by_older_ones():
    question = ModelRequest(parts=[UserPromptPart("Find handwriting evidence.")])
    state = ResearchRunState(question=question)
    older = [
        ModelResponse(
            [TextPart("small old decision"), ToolCallPart("inspect_documents", {}, "old")]
        ),
        ModelRequest([ToolReturnPart("inspect_documents", {"text": "old small text"}, "old")]),
    ]
    oversized = [
        ModelResponse(
            [TextPart("latest decision " * 1200), ToolCallPart("inspect_documents", {}, "new")]
        ),
        ModelRequest([ToolReturnPart("inspect_documents", {"text": "huge return " * 1200}, "new")]),
    ]
    messages = [question, *older, *oversized]
    parameters = ModelRequestParameters()
    accounting = RequestAccounting()
    base = accounting.estimate([question], parameters)
    notice_only = accounting.estimate(
        [question, ModelRequest(parts=[UserPromptPart(bounded_notice(state, parameters))])],
        parameters,
    )
    # A budget that fits only base + notice: the newest exchange cannot fit, the
    # older one would. Recency priority must not substitute the older exchange.
    budget = notice_only + 100
    assert budget > base
    compacted = bounded_history(messages, state, parameters, budget)
    assert all(message not in compacted for message in older)
    assert all(message not in compacted for message in oversized)
    assert accounting.estimate(compacted, parameters) <= budget


def bounded_notice(state, parameters):
    from papyrus_chat.agent.context.compaction import CHECKPOINT_NOTICE
    from papyrus_chat.agent.context.progress import progress_overview, research_progress

    return (
        CHECKPOINT_NOTICE
        + "\nRecorded progress: "
        + json.dumps(progress_overview(research_progress(state.ledger.records)))
        + "\n"
    )


def test_parallel_tool_pairs_are_indivisible():
    call = ModelResponse(parts=[ToolCallPart("a", {}, "a"), ToolCallPart("b", {}, "b")])
    one = ModelRequest(parts=[ToolReturnPart("a", "one", "a")])
    two = ModelRequest(parts=[ToolReturnPart("b", "two", "b")])
    assert complete_blocks([call, one, two]) == [[call, one, two]]


def test_summary_is_bounded_and_charged_to_parent_usage():
    seen = []

    def summarizer(messages, info):
        assert not info.function_tools
        assert not info.model_request_parameters.native_tools
        seen.append(messages)
        return ModelResponse(parts=[TextPart("Earlier investigation remains inconclusive.")])

    model = FunctionModel(summarizer)
    question = ModelRequest(parts=[UserPromptPart("What evidence exists?")])
    messages = [
        question,
        ModelResponse(parts=[TextPart("πάπυρος " * 10000)]),
        ModelRequest(parts=[]),
    ]
    state = ResearchRunState(question=question)
    context = RunContext(deps=None, model=model, usage=RunUsage())
    request = ModelRequestContext(
        model=model,
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    policy = ResearchPolicy()
    compacted = asyncio.run(compact_history(context, request, state, policy))
    assert compacted is not None
    assert state.summary_requests == 1
    assert state.research_requests == 1
    assert context.usage.requests == 1
    assert seen
    assert "What evidence exists?" in str(compacted)
    assert "omitted" in str(compacted)
    assert isinstance(compacted[-1], ModelRequest)
    assert (
        state.accounting.estimate(compacted, request.model_request_parameters)
        <= policy.target_tokens
    )


def test_failed_summary_preserves_last_checkpoint_and_counts_request():
    def fail(messages, info):
        raise RuntimeError("unavailable")

    model = FunctionModel(fail)
    question = ModelRequest(parts=[UserPromptPart("Question")])
    state = ResearchRunState(question=question, summary="Last valid summary")
    context = RunContext(deps=None, model=model, usage=RunUsage())
    request = ModelRequestContext(
        model=model,
        messages=[question],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    assert asyncio.run(compact_history(context, request, state, ResearchPolicy())) is None
    assert state.summary == "Last valid summary"
    assert state.summary_requests == state.research_requests == 1


def test_larger_generation_allowance_does_not_allow_unbounded_summary_text():
    def verbose_summary(messages, info):
        return ModelResponse(parts=[TextPart("πάπυρος " * 1000)])

    model = FunctionModel(verbose_summary)
    question = ModelRequest(parts=[UserPromptPart("Question")])
    state = ResearchRunState(question=question, summary="Last valid summary")
    context = RunContext(deps=None, model=model, usage=RunUsage())
    request = ModelRequestContext(
        model=model,
        messages=[question],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    assert asyncio.run(compact_history(context, request, state, ResearchPolicy())) is None
    assert state.summary == "Last valid summary"
    assert state.summary_requests == state.research_requests == 1


def test_checkpoint_preserves_exact_scoped_counts_and_inspected_lines():
    from papyrus_chat.agent.context.compaction import bounded_history
    from papyrus_chat.agent.context.evidence import EvidenceRecord

    question = ModelRequest(parts=[UserPromptPart("What evidence exists?")])
    state = ResearchRunState(question=question, summary="Tentative: the cohort may be relevant.")
    state.ledger.records = [
        EvidenceRecord(
            "search_documents",
            "search",
            {"query": {"collections": ["ddbdp"]}},
            {"candidate_count": 7, "query": {"collections": ["ddbdp"]}},
        ),
        EvidenceRecord(
            "inspect_documents",
            "inspect",
            {"document_ids": ["doc-1"]},
            {
                "inspections": [
                    {
                        "document_id": "doc-1",
                        "canonical_url": "https://papyri.info/ddbdp/p.mich;8;480",
                        "passages": [{"line_reference": "3-4", "excerpt": "δραχμὰς δέκα"}],
                    }
                ],
            },
        ),
        EvidenceRecord("describe_corpus", "oversized", {}, {"huge": "OMIT_ME" * 10000}),
    ]
    checkpoint = bounded_history(
        [question], state, ModelRequestParameters(), 5000, keep_recent=False
    )
    text = str(checkpoint)
    assert '"candidate_count": 7' in text
    assert '"collections": ["ddbdp"]' in text
    assert '"line_reference": "3-4"' in text
    assert "δραχμὰς δέκα" in text
    assert "https://papyri.info/ddbdp/p.mich;8;480" in text
    assert "Whole evidence records omitted: 1" in text
    assert "OMIT_ME" not in text
