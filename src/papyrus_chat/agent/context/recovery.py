"""Discard exhausted generations and retry once without consuming citation repair."""

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from pydantic_ai import RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import InstructionPart, ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.settings import ModelSettings

from papyrus_chat.chat.profiles import DeploymentProfile

from .accounting import RequestAccounting, accounting_text, estimate_text
from .compaction import bounded_history, required_messages
from .limits import (
    RequestPhase,
    check_request_limit,
    resolve_request_phase,
)
from .policy import ResearchPolicy
from .reasoning import reasoning_settings

if TYPE_CHECKING:
    from papyrus_chat.agent.tools import CorpusToolDeps

LOGGER = logging.getLogger(__name__)
RECOVERY_INSTRUCTIONS = (
    "The previous generation exhausted its output allowance and was discarded. "
    "Keep reasoning brief. Produce a concise complete response or the next useful tool call. "
    "The original question and research permissions still apply."
)


def charge_discarded_response(ctx: RunContext["CorpusToolDeps"], response: ModelResponse) -> None:
    """The agent graph only charges the accepted response; account for discarded work here."""
    if response.usage.cost is None and response.model_name:
        try:
            response.usage.cost = response.cost().total_price
        except (LookupError, ValueError):
            pass
    ctx.usage.incr(response.usage)


async def recover_generation(
    ctx: RunContext["CorpusToolDeps"],
    request: ModelRequestContext,
    response: ModelResponse,
    policy: ResearchPolicy,
    deployment_profile: DeploymentProfile | None = None,
) -> ModelResponse:
    state = ctx.deps.research_state
    charge_discarded_response(ctx, response)
    check_request_limit(ctx, policy)
    settings: ModelSettings = {**(request.model_settings or {})}
    settings = reasoning_settings(
        request.model,
        settings,
        purpose="recovery",
        deployment_profile=deployment_profile,
        thinking=request.model_request_parameters.thinking,
    )
    parameters = replace(
        request.model_request_parameters,
        instruction_parts=[
            *(request.model_request_parameters.instruction_parts or []),
            InstructionPart(RECOVERY_INSTRUCTIONS),
        ],
    )
    if state.phase == "repair":
        # Retry the in-flight repair request unchanged; it already carries the
        # repair instructions and must not spend the reserved answer slot.
        phase = RequestPhase(parameters, None, False)
    else:
        phase = resolve_request_phase(state, policy, parameters)
        parameters = phase.parameters
    effective = policy.model_copy(update={"max_tokens": settings.get("max_tokens")})
    final_prompt = phase.final_prompt
    prompt_tokens = estimate_text(accounting_text(final_prompt)) if final_prompt else 0
    accounting = RequestAccounting()
    essential = accounting.estimate(required_messages(request.messages, state), parameters)
    budget = min(
        effective.input_limit - prompt_tokens,
        max(
            essential + 1024,
            min(
                effective.target_tokens - prompt_tokens,
                int(accounting.estimate(request.messages, parameters) * 0.75),
            ),
        ),
    )
    messages = bounded_history(request.messages, state, parameters, budget)
    if final_prompt is not None:
        messages = [*messages, final_prompt]
    state.recovery_requests += 1
    state.research_requests += 1
    ctx.usage.requests += 1
    LOGGER.info(
        "Retrying an exhausted generation once",
        extra={
            "event": "research_generation_recovery",
            "run_id": ctx.run_id,
            "recovery_requests": state.recovery_requests,
            "input_tokens": accounting.estimate(messages, parameters),
        },
    )
    # This call bypasses hooks deliberately: no recursive recovery and no output-retry charge.
    recovered = await request.model.request(messages, settings or None, parameters)
    if recovered.finish_reason == "length":
        charge_discarded_response(ctx, recovered)
        raise UnexpectedModelBehavior(
            "Generation exhausted its output allowance again after one recovery attempt. "
            "No partial answer or truncated tool call was accepted. Configure the model server's "
            "generation/reasoning limits or an explicit LLM_MAX_TOKENS override."
        )
    state.accounting.anchor(messages, parameters, recovered.usage.input_tokens)
    return recovered
