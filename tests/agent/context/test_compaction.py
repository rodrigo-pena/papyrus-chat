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
