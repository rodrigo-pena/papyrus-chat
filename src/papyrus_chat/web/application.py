"""Starlette application using Pydantic AI's stock web chat UI."""

from pathlib import Path
from typing import Any

from starlette.applications import Starlette

from papyrus_chat.agent.runtime import create_research_agent
from papyrus_chat.agent.tools import CorpusToolDeps, CorpusToolService
from papyrus_chat.artifact.manifest import load_manifest
from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.chat.provider import ProviderError, load_provider_config
from papyrus_chat.retrieval.structured import StructuredCorpusSearch
from papyrus_chat.web.streaming import install_validated_chat_route


class StartupError(Exception):
    """The artifact or configuration cannot be used to start the application."""


def validate_startup(
    artifact: Path, env: dict[str, str] | None = None, *, require_provider: bool = True
) -> None:
    """Validate manifest, schema, files, integrity, and provider config."""
    if not artifact.is_dir():
        raise StartupError(
            f"Artifact directory not found: {artifact}. "
            "Expected a directory containing manifest.json, corpus.sqlite, "
            "and ATTRIBUTION.md (created by papyrus-corpus-build)."
        )
    try:
        validate_artifact(artifact)
        load_manifest(artifact / "manifest.json")
    except Exception as error:
        raise StartupError(f"The corpus artifact {artifact} is not usable: {error}") from error

    try:
        load_provider_config(env, required=require_provider)
    except ProviderError as error:
        raise StartupError(str(error)) from error


def load_app(
    artifact: Path,
    env: dict[str, str] | None = None,
    *,
    model: Any | None = None,
    html_source: str | Path | None = None,
) -> Starlette:
    """Build the stock Pydantic AI web app with artifact-backed dependencies.

    ``html_source`` is injectable for offline tests; production defaults to
    Pydantic AI's CDN-and-cache delivery. ``model`` is injectable for
    deterministic tests and is otherwise constructed from the existing
    ``LLM_BASE_URL``, ``LLM_MODEL``, and optional ``LLM_API_KEY`` settings.
    """
    validate_startup(artifact, env=env, require_provider=False)
    manifest = load_manifest(artifact / "manifest.json")
    provider_config = load_provider_config(env, required=False)
    search = StructuredCorpusSearch(artifact / "corpus.sqlite")
    tool_service = CorpusToolService(search)
    agent = create_research_agent(provider_config, tool_service, model=model)
    deps = CorpusToolDeps(service=tool_service)
    app = agent.to_web(
        deps=deps,
        html_source=html_source,
    )
    install_validated_chat_route(app, agent, deps)
    app.state.artifact = artifact
    app.state.manifest = manifest
    app.state.search = search
    app.state.tool_service = tool_service
    app.state.agent = agent
    app.state.provider_config = provider_config
    return app
