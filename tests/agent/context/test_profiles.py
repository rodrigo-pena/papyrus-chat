import pytest

from papyrus_chat.agent.context.policy import load_research_policy
from papyrus_chat.chat.profiles import (
    DeploymentProfile,
    DeploymentProfiles,
    load_deployment_profile,
)
from papyrus_chat.chat.provider import ProviderConfig


def test_private_profile_matching_and_capacity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = ProviderConfig(base_url="https://EXAMPLE.invalid/v1/", model="custom")
    assert load_deployment_profile(provider, {}) is None
    path = tmp_path / "profiles.toml"
    path.write_text("""[[profiles]]
base_url = "https://example.invalid/v1"
model = "custom"
context_window = 262144
reasoning_adapter = "qwen-chat-template"
""")
    env = {"PAPYRUS_MODEL_PROFILES": str(path)}
    profile = load_deployment_profile(provider, env)
    assert profile is not None
    assert load_research_policy("custom", {}, deployment_profile=profile).context_window == 262144
    policy = load_research_policy(
        "custom", {"LLM_CONTEXT_WINDOW": "8192"}, deployment_profile=profile
    )
    assert policy.context_window == 8192
    assert policy.capacity_source == "explicit"
    for updates in (
        {"model": "another"},
        {"model": "openai-responses:custom"},
        {"base_url": "https://other.invalid/v1"},
        {"base_url": "https://example.invalid/v2"},
        {"base_url": "http://example.invalid/v1"},
    ):
        assert load_deployment_profile(provider.model_copy(update=updates), env) is None
    assert "example.invalid" not in repr(profile)


def test_invalid_private_files_do_not_disclose_contents(tmp_path):
    path = tmp_path / "profiles.toml"
    provider = ProviderConfig(base_url="https://example.invalid/v1", model="custom")
    for content in (None, "private endpoint invalid TOML", '[[profiles]]\nsecret="PRIVATE"'):
        if content is not None:
            path.write_text(content)
        with pytest.raises(ValueError, match="Cannot load deployment profiles") as exc:
            load_deployment_profile(provider, {"PAPYRUS_MODEL_PROFILES": str(path)})
        assert "PRIVATE" not in str(exc.value)


@pytest.mark.parametrize(
    "updates",
    [
        {"context_window": 0},
        {"context_window": "8192"},
        {"reasoning_adapter": "guess"},
        {"base_url": "ftp://example.invalid"},
        {"base_url": "https://user:secret@example.invalid"},
        {"model": "openai-responses:custom", "reasoning_adapter": "qwen-chat-template"},
        {"unknown": True},
    ],
)
def test_profile_validation(updates):
    with pytest.raises(ValueError):
        DeploymentProfile.model_validate(
            {"base_url": "https://example.invalid/v1", "model": "custom", **updates}
        )


def test_duplicate_normalized_matches_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        DeploymentProfiles.model_validate(
            {
                "profiles": [
                    {"base_url": "https://EXAMPLE.invalid/v1/", "model": "custom"},
                    {"base_url": "https://example.invalid/v1", "model": "custom"},
                ]
            }
        )


def test_web_loads_profile_and_model_switch_drops_capacity(corpus_artifact, tmp_path):
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models.function import FunctionModel

    from papyrus_chat.web.application import load_app

    path = tmp_path / "profiles.toml"
    path.write_text("""[[profiles]]
base_url = "https://example.invalid/v1"
model = "custom"
context_window = 262144
reasoning_adapter = "qwen-chat-template"
""")
    env = {
        "LLM_BASE_URL": "https://example.invalid/v1",
        "LLM_MODEL": "custom",
        "LLM_API_KEY": "test",
        "PAPYRUS_MODEL_PROFILES": str(path),
    }
    app = load_app(corpus_artifact, env, html_source="<html></html>")
    try:
        assert app.state.research_policy.context_window == 262144
        assert app.state.research_policy.capacity_source == "profile"
    finally:
        app.state.tool_service.close()
    switched = load_app(
        corpus_artifact,
        env,
        model=FunctionModel(lambda m, i: ModelResponse(parts=[])),
        html_source="<html></html>",
    )
    try:
        assert switched.state.research_policy.context_window == 32768
        assert switched.state.research_policy.capacity_source == "fallback"
    finally:
        switched.state.tool_service.close()
