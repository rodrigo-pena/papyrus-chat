"""Compact between requests and reserve a final answer within an explicit budget."""

import asyncio
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

from papyrus_chat.chat.profiles import DeploymentProfile

from .accounting import RequestAccounting, accounting_text, estimate_text
from .compaction import ContextBudgetExceeded, bounded_history, compact_history, required_messages
from .limits import (
    answer_request_reserve,
    check_request_limit,
    check_response_cost,
    final_answer_parameters,
    final_answer_request,
    final_request_due,
    request_budget_parameters,
)
from .policy import ResearchPolicy, load_research_policy
from .reasoning import active_deployment_profile
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
    deployment_profile: DeploymentProfile | None = None

    def policy_for_request(self, request: ModelRequestContext) -> ResearchPolicy:
        if (
            self.policy.capacity_source == "profile"
            and active_deployment_profile(request.model, self.deployment_profile) is None
        ):
            fallback = load_research_policy(request.model.model_name, {})
            return ResearchPolicy.model_validate(
                {
                    **self.policy.model_dump(),
                    "context_window": fallback.context_window,
                    "capacity_source": fallback.capacity_source,
                }
            )
        return self.policy

    async def before_model_request(
        self, ctx: RunContext["CorpusToolDeps"], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        state = ctx.deps.research_state
        policy = self.policy_for_request(request_context)
        check_request_limit(ctx, policy)
        messages = request_context.messages
        params = request_context.model_request_parameters
        settings = {**(request_context.model_settings or {})}
        if "max_tokens" not in settings and policy.max_tokens is not None:
            settings["max_tokens"] = policy.max_tokens
        effective = policy.model_copy(update={"max_tokens": settings.get("max_tokens")})
        if state.phase == "repair":
            params = repair_parameters(params)
        elif state.phase == "synthesis" or final_request_due(state, policy):
            state.phase = "synthesis"
            params = final_answer_parameters(params)
            LOGGER.info(
                "Research request budget reserved for final answer",
                extra={
                    "event": "research_final_answer",
                    "run_id": ctx.run_id,
                    "research_requests": state.research_requests,
                    "request_limit": policy.research_request_limit,
                },
            )
        else:
            params = request_budget_parameters(params, state, policy)
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
        final_prompt = final_answer_request() if state.phase == "synthesis" else None
        prompt_tokens = estimate_text(accounting_text(final_prompt)) if final_prompt else 0
        if (
            RequestAccounting().estimate(required_messages(messages, state), params)
            + prompt_tokens
            > effective.input_limit
        ):
            raise ContextBudgetExceeded(
                "The current user prompt and required instructions are too large "
                "for the context window."
            )
        size = state.accounting.estimate(messages, params) + prompt_tokens
        LOGGER.info(
            "Research context measured",
            extra={
                "event": "research_context_measured",
                "run_id": ctx.run_id,
                "input_tokens": size,
                "trigger_tokens": effective.trigger_tokens,
                "request_count": state.research_requests,
            },
        )
        if size >= effective.trigger_tokens:
            request = replace(
                request_context, model_request_parameters=params, model_settings=settings
            )
            compacted = None
            # Leave room for this research request, the answer, and its retry.
            can_summarize = (
                policy.research_request_limit is None
                or state.research_requests + 1 + answer_request_reserve(policy)
                < policy.research_request_limit
            )
            if not state.summary_disabled and can_summarize:
                try:
                    compacted = await compact_history(
                        ctx, request, state, effective, self.deployment_profile
                    )
                except ContextBudgetExceeded:
                    compacted = None
            if compacted is None:
                state.summary_disabled = True
                try:
                    compacted = bounded_history(
                        messages, state, params, effective.target_tokens - prompt_tokens
                    )
                except ContextBudgetExceeded:
                    compacted = bounded_history(
                        messages, state, params, effective.input_limit - prompt_tokens
                    )
            messages = compacted
            after = RequestAccounting().estimate(messages, params) + prompt_tokens
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
        if state.phase == "research":
            # A summary may have consumed a request since the initial estimate.
            params = request_budget_parameters(params, state, policy)
        if final_prompt is not None:
            messages = [*messages, final_prompt]
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
            response = await recover_generation(
                ctx,
                request_context,
                response,
                self.policy_for_request(request_context),
                self.deployment_profile,
            )
        else:
            ctx.deps.research_state.accounting.anchor(
                request_context.messages,
                request_context.model_request_parameters,
                response.usage.input_tokens,
            )
        check_response_cost(ctx, response, self.policy)
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
                "stop_reason": "cancelled"
                if isinstance(error, asyncio.CancelledError)
                else type(error).__name__,
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
        if ctx.deps.research_state.phase != "research":
            raise ModelRetry(
                "Research has ended. Return the answer using only evidence already retrieved."
            )
        return args
