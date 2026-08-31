"""OpenAI-compatible Chat Completions provider configuration and client (SPEC 9.2).

Configuration comes from the environment: LLM_BASE_URL (required),
LLM_MODEL (required), LLM_API_KEY (optional, so unauthenticated local
servers work). The API key stays on the Python server: it is sent only in
the Authorization header and never appears in error messages or logs.
"""

import os
from collections.abc import Mapping

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

DEFAULT_TIMEOUT_SECONDS = 60.0


class ProviderError(Exception):
    """A user-facing provider configuration or request failure."""


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
    """Read provider settings; explain exactly which variable is missing."""
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
    if not base_url and not model:
        return ProviderConfig(base_url="", model="", api_key=None)

    return ProviderConfig(base_url=base_url, model=model, api_key=api_key)


class ProviderClient:
    def __init__(
        self,
        config: ProviderConfig,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, messages: list[dict[str, str]], *, temperature: float | None = None) -> str:
        """Send a chat completion request; map failures to actionable errors."""
        url = f"{self._config.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._config.api_key is not None:
            headers["Authorization"] = f"Bearer {self._config.api_key.get_secret_value()}"

        payload: dict[str, object] = {"model": self._config.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            response = self._client.post(url, json=payload, headers=headers)
        except httpx.ConnectError as error:
            raise ProviderError(
                f"Cannot reach the LLM endpoint at {self._config.base_url}. "
                "Check LLM_BASE_URL and that the server is running."
            ) from error
        except httpx.TimeoutException as error:
            raise ProviderError(
                f"The LLM endpoint at {self._config.base_url} did not respond in time. "
                "The provider may be overloaded; try again."
            ) from error

        if response.status_code in (401, 403):
            raise ProviderError(
                f"The LLM endpoint rejected authentication (HTTP {response.status_code}). "
                "Check LLM_API_KEY."
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"The LLM provider rejected the request (HTTP {response.status_code}). "
                f"Details: {_brief(response.text)}"
            )

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as error:
            raise ProviderError(
                "The LLM endpoint returned an unexpected response format. "
                "Ensure it is an OpenAI-compatible chat/completions endpoint."
            ) from error

        if not isinstance(content, str):
            raise ProviderError("The LLM endpoint returned an unexpected response format.")
        return content


def _brief(text: str, limit: int = 300) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:limit] + ("…" if len(cleaned) > limit else "")
