"""Opt-in operational limits shared by research, summarization, and recovery."""

from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.exceptions import UsageLimitExceeded

from .policy import ResearchPolicy


def check_request_limit(ctx: RunContext[Any], policy: ResearchPolicy) -> None:
    if (
        policy.research_request_limit is not None
        and ctx.deps.research_state.research_requests >= policy.research_request_limit
    ):
        raise UsageLimitExceeded("Configured research request limit reached.")
    if ctx.usage_limits:
        ctx.usage_limits.check_before_request(ctx.usage)
