"""Compact between model requests without deciding when research must finish."""

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Never

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
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
from .limits import check_request_limit
from .policy import ResearchPolicy
from .recovery import recover_generation

if TYPE_CHECKING:
    from papyrus_chat.agent.tools import CorpusToolDeps

LOGGER = logging.getLogger(__name__)
REPAIR_INSTRUCTIONS = """Correct or remove unsupported citations in the previous answer.
Do not perform new research or call tools. Use only original evidence already retrieved.
Return a complete corrected answer; do not claim exhaustive thematic coverage."""


def repair_parameters(parameters: ModelRequestParameters) -> ModelRequestParameters:
    return replace(
        parameters,
        function_tools=[],
        native_tools=[],
        tool_visibility={},
        instruction_parts=[
            *(parameters.instruction_parts or []),
            InstructionPart(REPAIR_INSTRUCTIONS),
        ],
    )


@dataclass
class BoundedResearch(AbstractCapability["CorpusToolDeps"]):
    policy: ResearchPolicy

    async def before_model_request(
        self, ctx: RunContext["CorpusToolDeps"], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        state = ctx.deps.research_state
        check_request_limit(ctx, self.policy)
        messages = request_context.messages
        params = request_context.model_request_parameters
        settings = {**(request_context.model_settings or {})}
        if "max_tokens" not in settings and self.policy.max_tokens is not None:
            settings["max_tokens"] = self.policy.max_tokens
        effective = self.policy.model_copy(update={"max_tokens": settings.get("max_tokens")})
        if state.phase == "repair":
            params = repair_parameters(params)
        if state.question is None:
            if ctx.prompt is not None:
                state.question = ModelRequest(parts=[UserPromptPart(ctx.prompt)])
            else:
                for message in reversed(messages):
                    if isinstance(message, ModelRequest):
                        parts = [p for p in message.parts if isinstance(p, UserPromptPart)]
                        if parts:
                            state.question = replace(message, parts=parts, instructions=None)
                            break
            if state.question is None:
                raise ContextBudgetExceeded("A current user prompt is required for research.")
        if (
            RequestAccounting().estimate(required_messages(messages, state), params)
            > effective.input_limit
        ):
            raise ContextBudgetExceeded(
                "The current user prompt and required instructions are too large "
                "for the context window."
            )
        size = state.accounting.estimate(messages, params)
        if size >= effective.trigger_tokens:
            request = replace(
                request_context, model_request_parameters=params, model_settings=settings
            )
            compacted = None
            if not state.summary_disabled:
                try:
                    compacted = await compact_history(ctx, request, state, effective)
                except ContextBudgetExceeded:
                    compacted = None
            if compacted is None:
                state.summary_disabled = True
                try:
                    compacted = bounded_history(
                        messages, state, params, effective.target_tokens, keep_recent=False
                    )
                except ContextBudgetExceeded:
                    compacted = bounded_history(
                        messages, state, params, effective.input_limit, keep_recent=False
                    )
            messages = compacted
            after = RequestAccounting().estimate(messages, params)
            if after > effective.input_limit:
                raise ContextBudgetExceeded(
                    "Essential research context cannot fit after compaction."
                )
            if after >= size:
                state.summary_disabled = True
            state.compactions += 1
            LOGGER.info(
                "Research context compacted: %d -> %d estimated tokens",
                size,
                after,
                extra={
                    "event": "research_context_compacted",
                    "run_id": ctx.run_id,
                    "before_tokens": size,
                    "after_tokens": after,
                    "mechanical": state.summary_disabled,
                },
            )
        check_request_limit(ctx, effective)
        state.research_requests += 1
        return replace(
            request_context,
            messages=messages,
            model_request_parameters=params,
            model_settings=settings or None,
        )

    async def after_model_request(
        self,
        ctx: RunContext["CorpusToolDeps"],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if response.finish_reason == "length":
            return await recover_generation(ctx, request_context, response, self.policy)
        ctx.deps.research_state.accounting.anchor(
            request_context.messages,
            request_context.model_request_parameters,
            response.usage.input_tokens,
        )
        return response

    async def on_run_error(
        self, ctx: RunContext["CorpusToolDeps"], *, error: BaseException
    ) -> Never:
        LOGGER.warning(
            "Research run ended with %s",
            type(error).__name__,
            extra={
                "event": "research_run_failed",
                "run_id": ctx.run_id,
                "error_type": type(error).__name__,
                "research_requests": ctx.deps.research_state.research_requests,
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
        if ctx.deps.research_state.phase == "repair":
            raise ModelRetry(
                "Citation repair cannot start new research. Return the corrected answer."
            )
        return args
