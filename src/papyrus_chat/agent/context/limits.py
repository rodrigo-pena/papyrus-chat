"""Opt-in operational limits shared by research, summarization, and recovery."""

from typing import Any

from pydantic_ai import RunContext, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse

from .policy import ResearchPolicy


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
