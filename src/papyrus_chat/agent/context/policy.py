"""Validated budgets for a single research turn."""

import logging
import os
from collections.abc import Mapping
from typing import Literal, Self

from genai_prices.data_snapshot import get_snapshot
from pydantic import BaseModel, ConfigDict, Field, model_validator

LOGGER = logging.getLogger(__name__)


class ResearchPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    context_window: int = Field(default=32768, ge=4096)
    capacity_source: Literal["explicit", "registry", "fallback"] = "explicit"
    research_request_limit: int | None = Field(default=None, ge=1)
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
    def hard_request_limit(self) -> int | None:
        return self.research_request_limit

    @property
    def output_tokens(self) -> int | None:
        return self.max_tokens

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
            - (self.output_tokens or int(self.context_window * 0.30))
            - self.safety_tokens
        )

    @property
    def summary_input_limit(self) -> int:
        return (
            self.context_window
            - (self.summary_output_tokens or int(self.context_window * 0.30))
            - self.safety_tokens
        )


def load_research_policy(model_name: str, env: Mapping[str, str] | None = None) -> ResearchPolicy:
    environment = os.environ if env is None else env
    raw_window = environment.get("LLM_CONTEXT_WINDOW", "").strip()
    capacity_source: Literal["explicit", "registry", "fallback"] = "explicit"
    if raw_window:
        window = int(raw_window)
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
    if "PAPYRUS_COMPACTION_LIMIT" in environment:
        LOGGER.warning(
            "PAPYRUS_COMPACTION_LIMIT is deprecated and ignored; compaction continues as needed."
        )
    policy = ResearchPolicy(
        context_window=window,
        capacity_source=capacity_source,
        research_request_limit=(
            int(value)
            if (value := environment.get("PAPYRUS_RESEARCH_REQUEST_LIMIT", "").strip())
            else None
        ),
        max_tokens=(
            int(value) if (value := environment.get("LLM_MAX_TOKENS", "").strip()) else None
        ),
        summary_max_tokens=(
            int(value)
            if (value := environment.get("PAPYRUS_SUMMARY_MAX_TOKENS", "").strip())
            else None
        ),
    )
    LOGGER.info(
        "Research context capacity: %d tokens (%s); generation limits: %s research, %s summary",
        window,
        capacity_source,
        policy.output_tokens or "server default",
        policy.summary_output_tokens or "server default",
        extra={
            "event": "research_policy_resolved",
            "context_window": window,
            "capacity_source": capacity_source,
            "max_tokens": policy.output_tokens,
            "summary_max_tokens": policy.summary_output_tokens,
            "input_limit": policy.input_limit,
        },
    )
    return policy
