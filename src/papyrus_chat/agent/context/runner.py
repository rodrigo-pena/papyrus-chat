"""Agent with opt-in usage limits, including for direct Python callers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.run import AgentRun

from papyrus_chat.agent.tools import CorpusToolDeps


class ContinuousResearchAgent(Agent[CorpusToolDeps, str]):
    @asynccontextmanager
    async def iter(
        self, *args: Any, usage_limits: UsageLimits | None = None, **kwargs: Any
    ) -> AsyncIterator[AgentRun[CorpusToolDeps, str]]:
        async with super().iter(
            *args,
            usage_limits=usage_limits
            if usage_limits is not None
            else UsageLimits(request_limit=None),
            **kwargs,
        ) as run:
            yield run
