"""Opt-in operational limits shared by research, summarization, and recovery."""

from dataclasses import replace
from typing import Any

from pydantic_ai import RunContext, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import InstructionPart, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from .policy import ResearchPolicy
from .state import ResearchRunState

FINAL_ANSWER_INSTRUCTIONS = """The configured request budget has reached the answer stage.
Stop research and answer the user's question now using only original evidence already retrieved.
Do not call tools or propose more searches. Summarize supported findings with their known citations,
state unresolved questions and incomplete coverage, and do not claim exhaustive discovery.
If no relevant evidence was retrieved, say so explicitly rather than inventing an answer.
Explain briefly that the configured request budget ended the research."""
_BUDGET_INSTRUCTION_NAME = "research_request_budget"
FINAL_ANSWER_PROMPT = (
    "Research is finished for this request. Return the actual answer now: findings and limitations "
    "supported by evidence already retrieved, with known citations. If evidence is insufficient, "
    "say what remains unresolved. Do not announce further searches, describe your next steps, "
    "or call tools."
)


def final_answer_request() -> ModelRequest:
    """Put the answer directive after the evidence, where long histories cannot bury it."""
    return ModelRequest(parts=[UserPromptPart(FINAL_ANSWER_PROMPT)])


def answer_request_reserve(policy: ResearchPolicy) -> int:
    """Allow one answer retry while preserving a research call for a two-call cap."""
    return 2 if (policy.research_request_limit or 0) >= 3 else 1


def final_request_due(state: ResearchRunState, policy: ResearchPolicy) -> bool:
    return (
        policy.research_request_limit is not None
        and state.research_requests
        >= policy.research_request_limit - answer_request_reserve(policy)
    )


def request_budget_parameters(
    parameters: ModelRequestParameters, state: ResearchRunState, policy: ResearchPolicy
) -> ModelRequestParameters:
    if policy.research_request_limit is None:
        return parameters
    reserve = answer_request_reserve(policy)
    remaining = policy.research_request_limit - state.research_requests - reserve
    note = (
        f"Research requests remaining: {remaining}. This includes the current request; "
        f"{reserve} further requests are reserved for answering and any needed correction. "
        "Summary and recovery requests also consume this budget. "
        "Prioritize inspecting the strongest candidates with inspect_documents or "
        "read_document_passages before the research allowance ends. Avoid spending the "
        "remaining requests on new search variations when useful candidates are available. "
        "Answer sooner when the evidence is sufficient."
    )
    return replace(
        parameters,
        instruction_parts=[
            *[
                part
                for part in parameters.instruction_parts or []
                if part.name != _BUDGET_INSTRUCTION_NAME
            ],
            InstructionPart(note, dynamic=True, name=_BUDGET_INSTRUCTION_NAME),
        ],
    )


def final_answer_parameters(parameters: ModelRequestParameters) -> ModelRequestParameters:
    return replace(
        parameters,
        function_tools=[],
        native_tools=[],
        tool_visibility={},
        instruction_parts=[
            *[
                part
                for part in parameters.instruction_parts or []
                if part.name != _BUDGET_INSTRUCTION_NAME
            ],
            InstructionPart(FINAL_ANSWER_INSTRUCTIONS),
        ],
    )


def check_request_limit(ctx: RunContext[Any], policy: ResearchPolicy) -> None:
    if (
        policy.research_request_limit is not None
        and ctx.deps.research_state.research_requests >= policy.research_request_limit
    ):
        raise UsageLimitExceeded("Configured research request limit reached.")
    if policy.cost_limit_usd is not None:
        UsageLimits(request_limit=None, cost_limit=policy.cost_limit_usd).check_cost(
            ctx.usage, warn_if_cost_unavailable=False
        )
    if ctx.usage_limits:
        ctx.usage_limits.check_before_request(ctx.usage)


def check_response_cost(
    ctx: RunContext[Any], response: ModelResponse, policy: ResearchPolicy
) -> None:
    if policy.cost_limit_usd is None:
        return
    if response.usage.cost is None:
        try:
            response.usage.cost = response.cost().total_price
        except (AssertionError, LookupError, ValueError) as error:
            raise UsageLimitExceeded(
                "Configured estimated-cost limit cannot be enforced: "
                "response pricing is unavailable."
            ) from error
    projected = ctx.usage + response.usage
    UsageLimits(request_limit=None, cost_limit=policy.cost_limit_usd).check_cost(projected)
