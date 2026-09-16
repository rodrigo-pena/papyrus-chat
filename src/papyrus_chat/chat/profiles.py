"""Private deployment settings, bound to an endpoint and model together."""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .provider import ProviderConfig


def normalize_endpoint(value: str) -> str:
    url = urlsplit(value)
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise ValueError("Profile base_url must be an HTTP(S) API endpoint")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("Profile base_url cannot contain credentials, query, or fragment")
    _ = url.port  # Validate malformed ports without changing the API path.
    return urlunsplit((url.scheme, url.netloc.lower(), url.path.rstrip("/"), "", ""))


class DeploymentProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    base_url: str = Field(repr=False)
    model: str = Field(min_length=1)
    context_window: int | None = Field(default=None, ge=4096, strict=True)
    reasoning_adapter: Literal["auto", "qwen-chat-template", "none"] = "auto"

    @field_validator("base_url")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return normalize_endpoint(value)

    @model_validator(mode="after")
    def validate_transport(self) -> Self:
        if self.reasoning_adapter == "qwen-chat-template" and self.model.startswith(
            "openai-responses:"
        ):
            raise ValueError("qwen-chat-template requires Chat Completions")
        return self

    def matches(self, base_url: str, model: str) -> bool:
        try:
            return self.base_url == normalize_endpoint(base_url) and self.model == model
        except ValueError:
            return False


class DeploymentProfiles(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    profiles: list[DeploymentProfile] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_matches(self) -> Self:
        keys = [(profile.base_url, profile.model) for profile in self.profiles]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate deployment profile matches")
        return self


def load_deployment_profile(
    provider: ProviderConfig, env: Mapping[str, str] | None = None
) -> DeploymentProfile | None:
    environment = os.environ if env is None else env
    configured = environment.get("PAPYRUS_MODEL_PROFILES", "").strip()
    path = Path(configured or "conf/model-profiles.toml").expanduser()
    if not configured and not path.exists():
        return None
    try:
        with path.open("rb") as source:
            profiles = DeploymentProfiles.model_validate(tomllib.load(source))
    except (OSError, ValueError, ValidationError):
        # TOML and validation exceptions can contain private configuration values.
        raise ValueError(
            "Cannot load deployment profiles: check file access and profile syntax."
        ) from None
    return next(
        (
            profile
            for profile in profiles.profiles
            if profile.matches(provider.base_url, provider.model)
        ),
        None,
    )
