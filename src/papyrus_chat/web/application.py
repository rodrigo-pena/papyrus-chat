"""FastAPI application for the local web interface (SPEC 9.1, 10)."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from papyrus_chat.artifact.manifest import load_manifest
from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.chat.provider import load_provider_config
from papyrus_chat.retrieval.search import CorpusSearch

TEMPLATES_DIR = Path(__file__).parent / "templates"


class StartupError(Exception):
    """The artifact or configuration cannot be used to start the application."""


def validate_startup(artifact: Path, env: dict[str, str] | None = None) -> None:
    """Validate manifest, schema, files, integrity, and provider config (SPEC 9.1)."""
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

    load_provider_config(env, required=True)


def load_app(artifact: Path, env: dict[str, str] | None = None) -> FastAPI:
    """Validate and build the FastAPI application for a corpus artifact."""
    validate_startup(artifact, env=env)
    manifest = load_manifest(artifact / "manifest.json")

    app = FastAPI(title="Papyrus Chat", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.artifact = artifact
    app.state.manifest = manifest
    app.state.search = CorpusSearch(artifact / "corpus.sqlite")
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "static")),
        name="static",
    )
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return app.state.templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"manifest": app.state.manifest},
        )

    return app
