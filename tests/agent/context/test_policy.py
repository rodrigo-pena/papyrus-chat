import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from papyrus_chat.agent.context import ResearchPolicy, load_research_policy
from papyrus_chat.agent.context.accounting import RequestAccounting, estimate_text


def test_policy_defaults_and_explicit_override():
    policy = load_research_policy("unknown-local-model", {})
    assert policy.context_window == 32768
    assert policy.capacity_source == "fallback"
    assert policy.research_request_limit == 16
    assert policy.hard_request_limit == 18
    override = load_research_policy("gpt-5.2", {"LLM_CONTEXT_WINDOW": "8192"})
    assert override.context_window == 8192
    assert override.capacity_source == "explicit"
    assert override.output_tokens == 2048
    assert override.summary_output_tokens == 1024


def test_known_model_capacity():
    policy = load_research_policy("openai-responses:gpt-5.2", {})
    assert policy.capacity_source == "registry"
    assert policy.context_window > 32768


@pytest.mark.parametrize(
    "env",
    [
        {"LLM_CONTEXT_WINDOW": "0"},
        {"LLM_CONTEXT_WINDOW": "not a number"},
        {"PAPYRUS_RESEARCH_REQUEST_LIMIT": "0"},
        {"PAPYRUS_COMPACTION_LIMIT": "-1"},
    ],
)
def test_invalid_environment(env):
    with pytest.raises(ValueError):
        load_research_policy("unknown", env)


def test_small_window_rejected():
    with pytest.raises(ValidationError):
        ResearchPolicy(context_window=100)


def test_accounting_anchors_usage_but_rebaselines_changed_history():
    accounting = RequestAccounting()
    first = ModelRequest(parts=[UserPromptPart("first question")])
    params = ModelRequestParameters()
    baseline = accounting.estimate([first], params)
    accounting.anchor([first], params, 2000)
    assert accounting.estimate([first], params) >= 2000
    second = ModelRequest(parts=[UserPromptPart("δεύτερη ερώτηση" * 100)])
    assert accounting.estimate([first, second], params) > 2000
    assert accounting.estimate([second], params) < 2000
    assert baseline < 2000
    assert estimate_text("α" * 100) > estimate_text("a" * 100)


def test_accounting_counts_instructions_once_and_excludes_return_schemas():
    from pydantic_ai.messages import InstructionPart
    from pydantic_ai.tools import ToolDefinition

    instructions = "Research instructions. " * 1000
    params = ModelRequestParameters(instruction_parts=[InstructionPart(instructions)])
    plain = ModelRequest(parts=[UserPromptPart("Question")])
    recorded = ModelRequest(parts=plain.parts, instructions=instructions)
    accounting = RequestAccounting()
    assert accounting.estimate([recorded], params) == accounting.estimate([plain], params)
    tool = ToolDefinition(name="search", parameters_json_schema={})
    with_tool = ModelRequestParameters(function_tools=[tool])
    before = accounting.estimate([plain], with_tool)
    tool.return_schema = {"description": "metadata only" * 10000}
    assert accounting.estimate([plain], with_tool) == before
