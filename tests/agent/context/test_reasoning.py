from copy import deepcopy
from typing import Any

import pytest
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from papyrus_chat.agent.context.reasoning import reasoning_settings
from papyrus_chat.chat.profiles import DeploymentProfile


def model(name="custom", *, responses=False, settings=None, **profile):
    cls = OpenAIResponsesModel if responses else OpenAIChatModel
    return cls(
        name,
        provider=OpenAIProvider(base_url="https://example.invalid/v1", api_key="test"),
        profile=OpenAIModelProfile(**profile),
        settings=settings,
    )


def deployment(**updates):
    return DeploymentProfile.model_validate(
        {
            "base_url": "https://example.invalid/v1",
            "model": "custom",
            "reasoning_adapter": "qwen-chat-template",
            **updates,
        }
    )


@pytest.mark.parametrize("purpose", ["summary", "recovery"])
def test_unknown_and_disabled_controls_preserve_settings(purpose):
    settings: Any = {"max_tokens": 123, "extra_body": {"options": [1]}}
    assert reasoning_settings(model(), settings, purpose=purpose) == settings
    assert (
        reasoning_settings(
            model(supports_thinking=True),
            settings,
            purpose=purpose,
            deployment_profile=deployment(reasoning_adapter="none"),
        )
        == settings
    )


@pytest.mark.parametrize("responses", [False, True])
def test_supported_controls_override_conflicting_effort_and_preserve_low_settings(responses):
    selected = model(responses=responses, supports_thinking=True)
    settings: Any = {
        "openai_reasoning_effort": "high",
        "extra_body": {"reasoning_effort": "high", "other": 1},
    }
    original = deepcopy(settings)
    summary: Any = reasoning_settings(selected, settings, purpose="summary")
    assert summary["thinking"] is False
    assert "openai_reasoning_effort" not in summary
    assert summary["extra_body"] == {"other": 1}
    assert settings == original
    assert reasoning_settings(selected, settings, purpose="recovery")["thinking"] == "low"
    for current in (False, "minimal", "low"):
        assert (
            reasoning_settings(selected, {"thinking": current}, purpose="recovery")["thinking"]
            == current
        )
    assert (
        reasoning_settings(selected, None, purpose="recovery", thinking=False)["thinking"] is False
    )
    assert (
        reasoning_settings(model(thinking_always_enabled=True), None, purpose="summary")["thinking"]
        == "low"
    )


def test_qwen_only_for_exact_active_deployment_and_isolated_settings():
    settings: Any = {"max_tokens": 123, "extra_body": {"chat_template_kwargs": {"other": 1}}}
    original = deepcopy(settings)
    summary: Any = reasoning_settings(
        model(), settings, purpose="summary", deployment_profile=deployment()
    )
    assert summary["extra_body"]["chat_template_kwargs"] == {"other": 1, "enable_thinking": False}
    assert summary["max_tokens"] == 123
    recovery: Any = reasoning_settings(
        model(), settings, purpose="recovery", deployment_profile=deployment()
    )
    assert recovery["openai_reasoning_effort"] == "low"
    assert settings == original
    for selected, profile in (
        (model("unrecognized-model"), deployment()),
        (model(responses=True), deployment()),
        (model(), deployment(base_url="https://other.invalid/v1")),
    ):
        assert (
            reasoning_settings(selected, settings, purpose="summary", deployment_profile=profile)
            == settings
        )
    disabled: Any = reasoning_settings(
        model(), summary, purpose="recovery", deployment_profile=deployment()
    )
    assert disabled["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "openai_reasoning_effort" not in disabled


def test_active_request_model_switch_rebaselines_profile_capacity():
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters

    from papyrus_chat.agent.context.policy import ResearchPolicy
    from papyrus_chat.agent.context.runtime import BoundedResearch

    capability = BoundedResearch(
        ResearchPolicy(context_window=262144, capacity_source="profile", research_request_limit=80),
        deployment(),
    )
    for selected, expected in ((model(), 262144), (model("unrecognized-model"), 32768)):
        request = ModelRequestContext(
            model=selected,
            messages=[],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )
        policy = capability.policy_for_request(request)
        assert policy.context_window == expected
        assert policy.research_request_limit == 80


def test_model_defaults_cannot_restore_high_reasoning():
    selected = model(
        supports_thinking=True,
        settings={"openai_reasoning_effort": "high", "extra_body": {"other": 1}},
    )
    settings = reasoning_settings(selected, None, purpose="summary")
    assert settings.get("openai_reasoning_effort") is None
    assert settings["thinking"] is False
    assert settings["extra_body"] == {"other": 1}
    assert selected.settings.get("openai_reasoning_effort") == "high"
