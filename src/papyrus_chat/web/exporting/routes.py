"""Download endpoint and a small add-on to a pinned, otherwise unmodified stock UI."""

import json
import re
from pathlib import Path

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .models import ExportRequest, export_document
from .rendering import render_html

# Unlike dist/index.html, the offline build does not refer to unversioned CDN assets.
PINNED_UI_URL = "https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@2.1.0/offline/index.html"
ASSETS = Path(__file__).parent / "static"
LAUNCHER = (
    '<link rel="stylesheet" href="/papyrus-assets/shell.css">'
    '<script type="module" src="/papyrus-assets/export.js"></script>'
)


async def post_export(request: Request) -> Response:
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        return JSONResponse({"error": "Expected Content-Type: application/json"}, status_code=415)
    try:
        payload = ExportRequest.model_validate_json(await request.body())
    except ValidationError:
        return JSONResponse(
            {"error": "Invalid conversation snapshot or export format. Reload the chat and retry."},
            status_code=422,
        )
    document = export_document(payload.conversation)
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", payload.conversation.id)[:64] or "conversation"
    stamp = re.sub(r"[^0-9]", "", document["exported_at"])[:14]
    filename = f"papyrus-chat-{safe_id}-{stamp}.{payload.format}"
    is_html = payload.format == "html"
    return Response(
        render_html(document) if is_html else json.dumps(document, ensure_ascii=False, indent=2),
        media_type="text/html" if is_html else "application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def install_export_routes(app: Starlette) -> None:
    api_mount = next(
        route for route in app.routes if isinstance(route, Mount) and route.path == "/api"
    )
    if not isinstance(api_mount.app, Starlette):
        raise RuntimeError("Expected the stock Pydantic AI API application")
    api_mount.app.router.routes.append(Route("/export", post_export, methods=["POST"]))

    # Wrap only the stock index endpoint, retaining its CDN/cache and html_source behavior.
    index_route = next(
        route for route in app.routes if isinstance(route, Route) and route.path == "/"
    )
    original_index = index_route.endpoint

    async def index(request: Request) -> Response:
        response = await original_index(request)
        html = bytes(response.body).decode("utf-8")
        before, closing, after = html.rpartition("</body>")
        html = before + LAUNCHER + closing + after if closing else html + LAUNCHER
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    for position, route in enumerate(app.routes):
        if isinstance(route, Route) and route.path in {"/", "/{id}"}:
            app.router.routes[position] = Route(route.path, index, methods=["GET"])
    app.router.routes.insert(0, Mount("/papyrus-assets", app=StaticFiles(directory=ASSETS)))
