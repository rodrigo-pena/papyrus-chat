import asyncio

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
from papyrus_chat.agent.context.compaction import compact_history, complete_blocks


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
                "document_id": "doc-1",
                "canonical_url": "https://papyri.info/ddbdp/p.mich;8;480",
                "passages": [{"line_reference": "3-4", "excerpt": "δραχμὰς δέκα"}],
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
