"""Reasoning adjustments limited to SDK capabilities or an exact deployment match."""

import logging
from copy import deepcopy
from typing import Any, Literal, cast

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.settings import ModelSettings

from papyrus_chat.chat.profiles import DeploymentProfile

LOGGER = logging.getLogger(__name__)


def active_deployment_profile(
    model: Model, profile: DeploymentProfile | None
) -> DeploymentProfile | None:
    if profile is None:
        return None
    name = model.model_name
    if isinstance(model, OpenAIResponsesModel):
        name = "openai-responses:" + name
    if profile.matches(str(model.base_url or ""), name):
        return profile
    return None


def reasoning_settings(
    model: Model,
    settings: ModelSettings | None,
    *,
    purpose: Literal["summary", "recovery"],
    deployment_profile: DeploymentProfile | None = None,
    thinking: Any = None,
) -> ModelSettings:
    """Return isolated settings; unknown models keep their original settings."""
    result: dict[str, Any] = deepcopy(dict(settings or {}))
    profile = active_deployment_profile(model, deployment_profile)
    adapter = profile.reasoning_adapter if profile else "auto"
    strategy = "unchanged"
    extra = result.get("extra_body")
    body = extra if isinstance(extra, dict) else {}
    template = body.get("chat_template_kwargs")
    template = template if isinstance(template, dict) else {}
    current = result.get("thinking", thinking)
    effort = body.get("reasoning_effort", result.get("openai_reasoning_effort"))
    if adapter == "qwen-chat-template" and isinstance(model, OpenAIChatModel):
        result.pop("thinking", None)
        result.pop("openai_reasoning_effort", None)
        body.pop("reasoning_effort", None)
        if purpose == "summary" or current is False or template.get("enable_thinking") is False:
            template["enable_thinking"] = False
            body["chat_template_kwargs"] = template
            result["extra_body"] = body
            strategy = "qwen-thinking-off"
        else:
            result["openai_reasoning_effort"] = "low"
            strategy = "qwen-reasoning-low"
    elif adapter == "auto" and (
        model.profile.get("supports_thinking", False)
        or model.profile.get("thinking_always_enabled", False)
    ):
        always_on = model.profile.get("thinking_always_enabled", False)
        target = "low" if purpose == "recovery" or always_on else False
        if purpose == "recovery":
            if current is False and not always_on:
                target = False
            elif current == "minimal":
                target = "minimal"
            if isinstance(model, (OpenAIChatModel, OpenAIResponsesModel)):
                if effort == "none" and not always_on:
                    target = False
                elif effort == "minimal" and target is not False:
                    target = "minimal"
        if isinstance(model, (OpenAIChatModel, OpenAIResponsesModel)):
            # Provider-specific effort overrides the SDK's unified thinking setting.
            result.pop("openai_reasoning_effort", None)
            body.pop("reasoning_effort", None)
        result["thinking"] = target
        strategy = "thinking-off" if target is False else f"reasoning-{target}"
    LOGGER.info(
        "Resolved %s reasoning strategy: %s",
        purpose,
        strategy,
        extra={"event": "research_reasoning_resolved", "purpose": purpose, "strategy": strategy},
    )
    return cast(ModelSettings, result)
