"""Agent with opt-in usage limits, including for direct Python callers."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.run import AgentRun

from papyrus_chat.agent.tools import CorpusToolDeps

from .policy import ResearchPolicy

LOGGER = logging.getLogger(__name__)


class ContinuousResearchAgent(Agent[CorpusToolDeps, str]):
    def __init__(self, *args: Any, policy: ResearchPolicy, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.research_policy = policy

    @asynccontextmanager
    async def iter(
        self, *args: Any, usage_limits: UsageLimits | None = None, **kwargs: Any
    ) -> AsyncIterator[AgentRun[CorpusToolDeps, str]]:
        timeout = asyncio.timeout(self.research_policy.run_timeout_seconds)
        try:
            async with timeout:
                async with super().iter(
                    *args,
                    usage_limits=usage_limits
                    if usage_limits is not None
                    else UsageLimits(request_limit=None),
                    **kwargs,
                ) as run:
                    yield run
        except TimeoutError as error:
            if timeout.expired():
                LOGGER.warning(
                    "Research elapsed-time limit reached",
                    extra={
                        "event": "research_operational_limit",
                        "stop_reason": "elapsed_time",
                        "seconds": self.research_policy.run_timeout_seconds,
                    },
                )
                raise UsageLimitExceeded(
                    "Configured research elapsed-time limit reached "
                    f"({self.research_policy.run_timeout_seconds} seconds)."
                ) from error
            raise
