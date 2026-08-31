"""Command-line interface for the local chat application (SPEC 9.1)."""

import threading
import webbrowser
from pathlib import Path

import typer
import uvicorn

from papyrus_chat.chat.provider import ProviderError
from papyrus_chat.web.application import StartupError, load_app, validate_startup

app = typer.Typer(
    add_completion=False,
    help="Search and chat with a built corpus artifact in your browser.",
)


@app.command()
def serve(
    artifact: Path = typer.Option(
        ...,
        "--artifact",
        exists=False,
        help="Corpus artifact directory (manifest.json, corpus.sqlite, ATTRIBUTION.md).",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Local bind address."),
    port: int = typer.Option(8000, "--port", help="Local HTTP port."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open the default browser."),
) -> None:
    try:
        validate_startup(artifact)
    except (StartupError, ProviderError) as error:
        typer.secho(str(error), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from None

    fastapi_app = load_app(artifact)

    if not no_open:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()

    typer.echo(f"Serving papyrus-chat at http://{host}:{port}/ (Ctrl+C to stop)")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
