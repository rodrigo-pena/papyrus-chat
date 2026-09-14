"""Validated budgets for a single research turn."""

import logging
import os
from collections.abc import Mapping
from typing import Literal

from genai_prices.data_snapshot import get_snapshot
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)


class ResearchPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    context_window: int = Field(default=32768, ge=4096)
    capacity_source: Literal["explicit", "registry", "fallback"] = "explicit"
    research_request_limit: int = Field(default=16, ge=1)
    compaction_limit: int = Field(default=3, ge=0)

    @property
    def hard_request_limit(self) -> int:
        return self.research_request_limit + 2

    @property
    def output_tokens(self) -> int:
        return min(4096, self.context_window // 4)

    @property
    def summary_output_tokens(self) -> int:
        return min(2048, self.context_window // 8)

    @property
    def trigger_tokens(self) -> int:
        return int(self.context_window * 0.65)

    @property
    def target_tokens(self) -> int:
        return int(self.context_window * 0.45)

    @property
    def input_limit(self) -> int:
        # Leave additional space for provider framing and estimation errors.
        return self.context_window - self.output_tokens - max(512, self.context_window // 20)


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
    policy = ResearchPolicy(
        context_window=window,
        capacity_source=capacity_source,
        research_request_limit=int(environment.get("PAPYRUS_RESEARCH_REQUEST_LIMIT", "16")),
        compaction_limit=int(environment.get("PAPYRUS_COMPACTION_LIMIT", "3")),
    )
    LOGGER.info(
        "Research context capacity: %d tokens (%s)",
        window,
        capacity_source,
        extra={
            "event": "research_policy_resolved",
            "context_window": window,
            "capacity_source": capacity_source,
        },
    )
    return policy
