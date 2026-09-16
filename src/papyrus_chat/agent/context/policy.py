"""Validated budgets for a single research turn."""

import logging
import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Literal, Self

from genai_prices.data_snapshot import get_snapshot
from pydantic import BaseModel, ConfigDict, Field, model_validator

from papyrus_chat.chat.profiles import DeploymentProfile

LOGGER = logging.getLogger(__name__)


class ResearchPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    context_window: int = Field(default=32768, ge=4096)
    capacity_source: Literal["explicit", "profile", "registry", "fallback"] = "explicit"
    research_request_limit: int | None = Field(default=None, ge=1)
    run_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    cost_limit_usd: Decimal | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, ge=1)
    summary_max_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_generation_space(self) -> Self:
        for setting, input_limit in (
            ("LLM_MAX_TOKENS", self.input_limit),
            ("PAPYRUS_SUMMARY_MAX_TOKENS", self.summary_input_limit),
        ):
            if input_limit < 1024:
                raise ValueError(
                    f"{setting} must leave at least 1,024 input tokens and the safety margin "
                    "within LLM_CONTEXT_WINDOW."
                )
        return self

    @property
    def summary_output_tokens(self) -> int | None:
        return self.summary_max_tokens if self.summary_max_tokens is not None else self.max_tokens

    @property
    def summary_text_tokens(self) -> int:
        # More room to think must not turn a checkpoint into another long transcript.
        return min(2048, self.context_window // 8)

    @property
    def trigger_tokens(self) -> int:
        return min(int(self.context_window * 0.65), self.input_limit)

    @property
    def target_tokens(self) -> int:
        return min(int(self.context_window * 0.45), int(self.input_limit * 0.75))

    @property
    def safety_tokens(self) -> int:
        return max(512, self.context_window // 20)

    @property
    def input_limit(self) -> int:
        # Leave additional space for provider framing and estimation errors.
        return (
            self.context_window
            - (self.max_tokens or int(self.context_window * 0.30))
            - self.safety_tokens
        )

    @property
    def summary_input_limit(self) -> int:
        return (
            self.context_window
            - (self.summary_output_tokens or int(self.context_window * 0.30))
            - self.safety_tokens
        )


def _parse_int_setting(name: str, environment: Mapping[str, str]) -> int | None:
    raw = environment.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a whole number, got {raw!r}.") from error


def load_research_policy(
    model_name: str,
    env: Mapping[str, str] | None = None,
    *,
    deployment_profile: DeploymentProfile | None = None,
) -> ResearchPolicy:
    environment = os.environ if env is None else env
    raw_window = environment.get("LLM_CONTEXT_WINDOW", "").strip()
    capacity_source: Literal["explicit", "profile", "registry", "fallback"] = "explicit"
    if raw_window:
        try:
            window = int(raw_window)
        except ValueError as error:
            raise ValueError(
                f"LLM_CONTEXT_WINDOW must be a whole number of tokens, got {raw_window!r}."
            ) from error
    elif deployment_profile is not None and deployment_profile.context_window is not None:
        window = deployment_profile.context_window
        capacity_source = "profile"
    else:
        try:
            _, model = get_snapshot().find_provider_model(
                model_name.removeprefix("openai-responses:"), None, None, None
            )
            window = model.context_window or 32768
            capacity_source = "registry" if model.context_window else "fallback"
        except LookupError:
            window = 32768
            capacity_source = "fallback"
    try:
        raw_cost = environment.get("PAPYRUS_RUN_COST_LIMIT_USD", "").strip()
        cost = Decimal(raw_cost) if raw_cost else None
    except InvalidOperation as error:
        raise ValueError("PAPYRUS_RUN_COST_LIMIT_USD must be a positive decimal number.") from error
    raw_timeout = environment.get("PAPYRUS_RUN_TIMEOUT_SECONDS", "").strip()
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except ValueError as error:
            raise ValueError(
                f"PAPYRUS_RUN_TIMEOUT_SECONDS must be a number of seconds, got {raw_timeout!r}."
            ) from error
    else:
        timeout = None
    policy = ResearchPolicy(
        run_timeout_seconds=timeout,
        cost_limit_usd=cost,
        context_window=window,
        capacity_source=capacity_source,
        research_request_limit=(_parse_int_setting("PAPYRUS_RESEARCH_REQUEST_LIMIT", environment)),
        max_tokens=_parse_int_setting("LLM_MAX_TOKENS", environment),
        summary_max_tokens=_parse_int_setting("PAPYRUS_SUMMARY_MAX_TOKENS", environment),
    )
    validate_pricing(policy, model_name)
    LOGGER.info(
        "Research context capacity: %d tokens (%s); generation limits: %s research, %s summary",
        window,
        capacity_source,
        policy.max_tokens or "server default",
        policy.summary_output_tokens or "server default",
        extra={
            "event": "research_policy_resolved",
            "context_window": window,
            "capacity_source": capacity_source,
            "max_tokens": policy.max_tokens,
            "summary_max_tokens": policy.summary_output_tokens,
            "input_limit": policy.input_limit,
        },
    )
    return policy


def validate_pricing(policy: ResearchPolicy, model_name: str) -> None:
    if policy.cost_limit_usd is None:
        return
    try:
        _, model = get_snapshot().find_provider_model(
            model_name.removeprefix("openai-responses:"), None, None, None
        )
        if not model.prices:
            raise LookupError("No pricing metadata")
    except LookupError as error:
        raise ValueError(
            "PAPYRUS_RUN_COST_LIMIT_USD requires supported pricing metadata "
            "for the configured model."
        ) from error
