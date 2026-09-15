"""Capture evidence before any capability changes the history."""

import logging
from typing import TYPE_CHECKING

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext

from .progress import progress_overview
from .state import ResearchRunState

LOGGER = logging.getLogger(__name__)

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
            ctx.deps.known_corpus_urls.clear()
        before = len(state.ledger.executions)
        state.ledger.ingest(request_context.messages)
        if len(state.ledger.executions) != before:
            LOGGER.info(
                "Research coverage updated",
                extra={
                    "event": "research_coverage_updated",
                    "run_id": ctx.run_id,
                    "evidence_records": len(state.ledger.records),
                    **progress_overview(state.ledger.progress()),
                },
            )
        ctx.deps.known_corpus_urls.update(state.ledger.corpus_urls)
        return request_context
