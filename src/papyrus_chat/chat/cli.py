"""Command-line interface for the local chat application."""

import logging
import threading
import webbrowser
from pathlib import Path

import typer
import uvicorn

from papyrus_chat.chat.provider import ProviderError
from papyrus_chat.cli_logging import configure_cli_logging
from papyrus_chat.web.application import StartupError, load_app, validate_startup

LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Search and chat with a built corpus artifact in your browser.",
)


@app.command()
def serve(
    context: typer.Context,
    artifact: Path = typer.Option(
        ...,
        "--artifact",
        exists=False,
        help="Corpus artifact directory (manifest.json, corpus.sqlite, ATTRIBUTION.md).",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Local bind address."),
    port: int = typer.Option(8000, "--port", help="Local HTTP port."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open the default browser."),
    web_search: bool = typer.Option(
        False,
        "--web-search",
        help=(
            "Enable optional historical/contextual web search (never corpus evidence or counts)."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Include detailed diagnostic and server logging.",
    ),
) -> None:
    context.call_on_close(configure_cli_logging(verbose=verbose))
    LOGGER.info(
        "Starting papyrus-chat: artifact=%s host=%s port=%d",
        artifact,
        host,
        port,
        extra={
            "event": "chat_startup_started",
            "artifact": str(artifact),
            "host": host,
            "port": port,
        },
    )
    LOGGER.info(
        "Validating corpus artifact and provider configuration",
        extra={"event": "chat_startup_validation_started"},
    )
    try:
        validate_startup(artifact)
    except (StartupError, ProviderError) as error:
        LOGGER.error("papyrus-chat startup failed: %s", error)
        raise typer.Exit(code=2) from None

    LOGGER.info(
        "Loading chat application",
        extra={"event": "chat_application_load_started"},
    )
    chat_app = load_app(artifact, enable_web_search=True) if web_search else load_app(artifact)

    if not no_open:
        LOGGER.info(
            "Opening the chat UI in the default browser",
            extra={"event": "chat_browser_open_scheduled"},
        )
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    else:
        LOGGER.debug(
            "Automatic browser opening is disabled",
            extra={"event": "chat_browser_open_disabled"},
        )

    LOGGER.info(
        "Serving papyrus-chat at http://%s:%d/ (Ctrl+C to stop)",
        host,
        port,
        extra={"event": "chat_server_starting", "host": host, "port": port},
    )
    uvicorn.run(chat_app, host=host, port=port, log_level="debug" if verbose else "info")


if __name__ == "__main__":
    app()
