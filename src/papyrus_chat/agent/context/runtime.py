"""Enforce research and answer phases around every model request."""

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Never

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    InstructionPart,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from .accounting import RequestAccounting
from .compaction import ContextBudgetExceeded, bounded_history, compact_history, required_messages
from .policy import ResearchPolicy
from .state import ResearchRunState

if TYPE_CHECKING:
    from papyrus_chat.agent.tools import CorpusToolDeps

LOGGER = logging.getLogger(__name__)
FINAL_INSTRUCTIONS = """
The research phase is finished for this turn. Do not perform or request any more
corpus or web research. Answer the user's question now using only the retained
evidence. State that research is incomplete and identify missing coverage or
unresolved questions. Distinguish inspected findings from tentative interpretation.
A checkpoint summary is secondhand: quote only original inspected excerpts, and
pair exact counts with their search scope. If evidence is insufficient, say so
and give a useful partial answer without inventing evidence or claiming absence.
Citation corrections must remove or correct unsupported claims without new research.
""".strip()


def final_parameters(parameters: ModelRequestParameters) -> ModelRequestParameters:
    return replace(
        parameters,
        function_tools=[],
        native_tools=[],
        tool_visibility={},
        instruction_parts=[
            *(parameters.instruction_parts or []),
            InstructionPart(FINAL_INSTRUCTIONS),
        ],
    )


def start_finalizing(state: ResearchRunState, reason: str) -> None:
    if state.phase == "research":
        state.phase = "finalize"
        state.reason = reason
        LOGGER.info(
            "Research finished: %s",
            reason,
            extra={
                "event": "research_finalization_started",
                "run_id": state.run_id,
                "reason": reason,
                "research_requests": state.research_requests,
                "summary_requests": state.summary_requests,
            },
        )


@dataclass
class BoundedResearch(AbstractCapability["CorpusToolDeps"]):
    policy: ResearchPolicy

    async def before_model_request(
        self,
        ctx: RunContext["CorpusToolDeps"],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        state = ctx.deps.research_state
        messages = request_context.messages
        params = request_context.model_request_parameters
        if state.question is None:
            if ctx.prompt is not None:
                state.question = ModelRequest(parts=[UserPromptPart(ctx.prompt)])
            else:
                for message in reversed(messages):
                    if isinstance(message, ModelRequest):
                        parts = [part for part in message.parts if isinstance(part, UserPromptPart)]
                        if parts:
                            state.question = replace(message, parts=parts, instructions=None)
                            break
            if state.question is None:
                raise ContextBudgetExceeded("A current user prompt is required for research.")
            minimum = required_messages(messages, state)
            if (
                RequestAccounting().estimate(minimum, final_parameters(params))
                > self.policy.input_limit
            ):
                raise ContextBudgetExceeded(
                    "The current user prompt is too large for the configured context window."
                )

        # Honor a tighter caller request limit while reserving its remaining answer slots.
        research_limit = self.policy.research_request_limit
        if ctx.usage_limits and ctx.usage_limits.request_limit is not None:
            research_limit = min(research_limit, max(0, ctx.usage_limits.request_limit - 2))
        if state.research_requests >= research_limit:
            start_finalizing(state, "research request budget reached")

        size = state.accounting.estimate(messages, params)
        if state.phase == "research" and size >= self.policy.trigger_tokens:
            if state.summary_requests >= self.policy.compaction_limit:
                start_finalizing(state, "compaction budget reached")
            else:
                try:
                    compacted = await compact_history(ctx, request_context, state, self.policy)
                except ContextBudgetExceeded:
                    compacted = None
                if compacted is None:
                    start_finalizing(state, "compaction failed or could not fit the context")
                else:
                    after = RequestAccounting().estimate(compacted, params)
                    LOGGER.info(
                        "Research context compacted: %d -> %d estimated tokens",
                        size,
                        after,
                        extra={
                            "event": "research_context_compacted",
                            "run_id": ctx.run_id,
                            "before_tokens": size,
                            "after_tokens": after,
                        },
                    )
                    messages = compacted
                    if after >= size or after > self.policy.target_tokens:
                        start_finalizing(state, "compaction reclaimed insufficient context")
                if state.research_requests >= research_limit:
                    start_finalizing(state, "research request budget reached")

        if state.phase == "finalize":
            if state.final_requests >= 2:
                raise UnexpectedModelBehavior(
                    "Final answer could not be validated within two attempts."
                )
            params = final_parameters(params)
            messages = bounded_history(
                messages, state, params, self.policy.input_limit, keep_recent=False
            )
            state.final_requests += 1
        else:
            state.research_requests += 1
        if state.research_requests + state.final_requests > self.policy.hard_request_limit:
            raise UsageLimitExceeded("Research and final-answer request budget exhausted.")
        settings = {**(request_context.model_settings or {})}
        settings["max_tokens"] = min(
            settings.get("max_tokens") or self.policy.output_tokens, self.policy.output_tokens
        )
        return replace(
            request_context,
            messages=messages,
            model_request_parameters=params,
            model_settings=settings,
        )

    async def after_model_request(
        self,
        ctx: RunContext["CorpusToolDeps"],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        ctx.deps.research_state.accounting.anchor(
            request_context.messages,
            request_context.model_request_parameters,
            response.usage.input_tokens,
        )
        return response

    async def on_run_error(
        self,
        ctx: RunContext["CorpusToolDeps"],
        *,
        error: BaseException,
    ) -> Never:
        state = ctx.deps.research_state
        LOGGER.warning(
            "Research run ended with %s",
            type(error).__name__,
            extra={
                "event": "research_run_failed",
                "run_id": ctx.run_id,
                "error_type": type(error).__name__,
                "phase": state.phase,
                "research_requests": state.research_requests,
                "final_requests": state.final_requests,
            },
        )
        raise error

    async def before_tool_execute(
        self,
        ctx: RunContext["CorpusToolDeps"],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        # A non-compliant endpoint may emit a historical tool despite an empty tool list.
        if ctx.deps.research_state.phase == "finalize":
            raise ModelRetry("Research is finished. Produce the final answer without tools.")
        return args
