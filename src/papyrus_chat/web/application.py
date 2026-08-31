"""FastAPI application for the local web interface (SPEC 9.1, 10)."""

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from papyrus_chat.artifact.manifest import load_manifest
from papyrus_chat.artifact.schema import ArtifactReader
from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.chat.conversation import Conversation
from papyrus_chat.chat.provider import (
    ProviderClient,
    ProviderError,
    load_provider_config,
)
from papyrus_chat.retrieval.evidence import EvidencePacket
from papyrus_chat.retrieval.search import CorpusSearch, SearchFilters
from papyrus_chat.web.links import papyri_info_url
from papyrus_chat.web.urlsafe import document_url

TEMPLATES_DIR = Path(__file__).parent / "templates"


class StartupError(Exception):
    """The artifact or configuration cannot be used to start the application."""


def validate_startup(
    artifact: Path, env: dict[str, str] | None = None, *, require_provider: bool = True
) -> None:
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

    load_provider_config(env, required=require_provider)


def load_app(artifact: Path, env: dict[str, str] | None = None) -> FastAPI:
    """Validate and build the FastAPI application for a corpus artifact.

    Provider configuration is not required here: search works without an
    LLM (SPEC 15). The chat panel surfaces configuration errors when used,
    and the papyrus-chat CLI validates the provider before launch.
    """
    validate_startup(artifact, env=env, require_provider=False)
    manifest = load_manifest(artifact / "manifest.json")

    app = FastAPI(title="Papyrus Chat", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.artifact = artifact
    app.state.manifest = manifest
    app.state.search = CorpusSearch(artifact / "corpus.sqlite")
    app.state.provider_config = load_provider_config(env, required=False)
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "static")),
        name="static",
    )

    def render(request: Request, name: str, context: dict, status: int = 200):
        return app.state.templates.TemplateResponse(
            request=request, name=name, status_code=status, context=context
        )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return render(request, "index.html", {"manifest": app.state.manifest})

    @app.get("/search", response_class=HTMLResponse)
    async def search(
        request: Request,
        query: str = "",
        collection: str = "",
        kind: str = "",
    ) -> HTMLResponse:
        packet: EvidencePacket | None = None
        if query.strip():
            filters = SearchFilters(collection=collection or None, kind=kind or None)
            packet = app.state.search.search(query, filters, limit=25)

        return render(
            request,
            "search.html",
            {
                "query": query,
                "packet": packet,
                "selected_collection": collection,
                "selected_kind": kind,
                "doc_url": document_url,
            },
        )

    @app.get("/documents/{document_id:path}", response_class=HTMLResponse)
    async def document(request: Request, document_id: str) -> HTMLResponse:
        reader = ArtifactReader(app.state.artifact / "corpus.sqlite")
        record = reader.get_document(document_id)
        if record is None:
            reader.close()
            return render(
                request,
                "not_found.html",
                {"message": "No document with that identifier in this corpus."},
                status=404,
            )

        passages = reader.get_passages(document_id)
        identifiers = reader.get_identifiers(document_id)
        reader.close()

        preferred = next(
            (i for i in identifiers if i.namespace.lower() == "tm"),
            identifiers[0] if identifiers else None,
        )
        citation = (
            f"{preferred.namespace} {preferred.value} ({record.title})"
            if preferred
            else record.title
        )

        return render(
            request,
            "document.html",
            {
                "doc": record,
                "passages": passages,
                "identifiers": identifiers,
                "citation": citation,
                "canonical_url": papyri_info_url((i.namespace, i.value) for i in identifiers),
            },
        )

    @app.get("/chat", response_class=HTMLResponse)
    @app.post("/chat", response_class=HTMLResponse)
    async def chat(
        request: Request,
        query: str = Form(""),
        document_id: str = Form(""),
    ) -> HTMLResponse:
        answer = None
        error: str | None = None
        if query.strip():
            try:
                if app.state.provider_config.base_url:
                    conversation = Conversation(
                        app.state.search,
                        ProviderClient(app.state.provider_config),
                    )
                    answer = conversation.ask(query, document_id=document_id or None)
                else:
                    error = (
                        "LLM configuration incomplete. Set LLM_BASE_URL and "
                        "LLM_MODEL (and optionally LLM_API_KEY) before asking "
                        "questions. Searching still works."
                    )
            except ProviderError as provider_error:
                error = str(provider_error)

        return render(
            request,
            "chat.html",
            {
                "query": query,
                "document_id": document_id,
                "answer": answer,
                "error": error,
                "doc_url": document_url,
            },
        )

    return app
