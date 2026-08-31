"""OpenAI-compatible provider configuration for the Pydantic AI runtime.

Configuration comes from the environment: ``LLM_BASE_URL`` and ``LLM_MODEL``
are required by the CLI, while ``LLM_API_KEY`` is optional for unauthenticated
local providers. The secret is held in ``SecretStr`` and never included in
configuration representations or startup errors.
"""

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator


class ProviderError(Exception):
    """A user-facing provider configuration failure."""


class ProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str
    model: str
    api_key: SecretStr | None = None

    @field_validator("base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    def __str__(self) -> str:
        return f"ProviderConfig(base_url={self.base_url!r}, model={self.model!r})"


def load_provider_config(
    env: Mapping[str, str] | None = None, *, required: bool = True
) -> ProviderConfig:
    """Read provider settings and explain exactly which variable is missing."""
    environment = env if env is not None else os.environ

    base_url = environment.get("LLM_BASE_URL", "").strip().rstrip("/")
    model = environment.get("LLM_MODEL", "").strip()
    api_key = environment.get("LLM_API_KEY", "").strip() or None

    missing = []
    if not base_url:
        missing.append("LLM_BASE_URL")
    if not model:
        missing.append("LLM_MODEL")
    if missing and required:
        raise ProviderError(
            "LLM configuration incomplete. Set "
            + ", ".join(missing)
            + " before starting papyrus-chat. Example:\n"
            '  export LLM_BASE_URL="https://provider.example/v1"\n'
            '  export LLM_MODEL="model-name"\n'
            '  export LLM_API_KEY="..."   # optional for local servers'
        )
    return ProviderConfig(base_url=base_url, model=model, api_key=api_key)
