import pytest
from pydantic_ai.models import ModelRequestParameters

from papyrus_chat.agent.context.limits import resolve_request_phase
from papyrus_chat.agent.context.state import ResearchRunState


def test_research_phase_adds_budget_instructions():
    state = ResearchRunState(research_requests=1)
    policy = load_policy_with_limit(10)
    phase = resolve_request_phase(state, policy, ModelRequestParameters())
    assert not phase.reserved_final_answer
    assert phase.final_prompt is None
    assert any(
        part.content.startswith("Research requests remaining:")
        for part in phase.parameters.instruction_parts
    )


def load_policy_with_limit(limit):
    from papyrus_chat.agent.context.policy import ResearchPolicy

    return ResearchPolicy(context_window=32768, research_request_limit=limit)


@pytest.mark.parametrize("starting_phase", ["research", "synthesis"])
def test_due_phase_reserves_the_final_answer(starting_phase):
    state = ResearchRunState(phase=starting_phase, research_requests=9)
    policy = load_policy_with_limit(10)
    phase = resolve_request_phase(state, policy, ModelRequestParameters())
    assert state.phase == "synthesis"
    assert phase.reserved_final_answer
    assert phase.final_prompt is not None
    assert not phase.parameters.function_tools
    assert any("answer" in part.content for part in phase.parameters.instruction_parts)


def test_repair_phase_passes_parameters_through():
    state = ResearchRunState(phase="repair", research_requests=1)
    policy = load_policy_with_limit(10)
    parameters = ModelRequestParameters()
    phase = resolve_request_phase(state, policy, parameters)
    assert state.phase == "repair"
    assert phase.parameters is parameters
    assert phase.final_prompt is None
    assert not phase.reserved_final_answer
