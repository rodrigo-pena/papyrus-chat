"""Capture evidence before any capability changes the history."""

from typing import TYPE_CHECKING

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext

from .state import ResearchRunState

if TYPE_CHECKING:
    from papyrus_chat.agent.tools import CorpusToolDeps


class EvidenceTracking(AbstractCapability["CorpusToolDeps"]):
    async def before_model_request(
        self, ctx: RunContext["CorpusToolDeps"], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        state = ctx.deps.research_state
        if state.run_id != ctx.run_id:
            # The web route supplies fresh dependencies. Also reset counters when
            # Python callers intentionally reuse dependencies for a follow-up run.
            state = ResearchRunState(run_id=ctx.run_id)
            ctx.deps.research_state = state
        state.ledger.ingest(request_context.messages)
        ctx.deps.known_corpus_urls.update(state.ledger.corpus_urls)
        return request_context
