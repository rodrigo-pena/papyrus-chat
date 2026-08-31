"""Command-line interface for building corpus artifacts."""

import logging
from pathlib import Path

import typer

from papyrus_chat.builder.errors import BuildError
from papyrus_chat.builder.pipeline import BUILDER_VERSION, SUPPORTED_COLLECTIONS, build_artifact
from papyrus_chat.builder.source import LocalGitSource, RemoteGitSource
from papyrus_chat.cli_logging import configure_cli_logging

LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Build a searchable corpus artifact from selected idp.data collections.",
)

DEFAULT_SOURCE = "https://github.com/papyri/idp.data.git"


@app.command()
def build(
    context: typer.Context,
    collections: list[str] | None = typer.Argument(
        None,
        metavar="COLLECTION...",
        help="One or more collections to include (see --list-collections).",
    ),
    output: str = typer.Option(
        "./papyrus-corpus",
        "--output",
        "-o",
        help="Destination artifact directory.",
    ),
    source: str = typer.Option(
        DEFAULT_SOURCE,
        "--source",
        help="Git URL or local idp.data checkout directory.",
    ),
    ref: str = typer.Option("master", "--ref", help="Branch, tag, or commit to record."),
    force: bool = typer.Option(
        False, "--force", help="Explicitly allow replacement of the destination artifact."
    ),
    list_collections: bool = typer.Option(
        False, "--list-collections", help="Print supported collection names and exit."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Include detailed diagnostic logging.",
    ),
) -> None:
    if list_collections:
        for name in sorted(SUPPORTED_COLLECTIONS):
            typer.echo(name)
        return

    if not collections:
        typer.secho(
            "No collection specified. Supported collections: "
            + ", ".join(sorted(SUPPORTED_COLLECTIONS)),
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    unknown = [c for c in collections if c.lower() not in SUPPORTED_COLLECTIONS]
    if unknown:
        typer.secho(
            f"Unknown collection: {', '.join(unknown)}. Supported collections: "
            + ", ".join(sorted(SUPPORTED_COLLECTIONS)),
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    context.call_on_close(configure_cli_logging(verbose=verbose))
    try:
        result = build_artifact(
            collections,
            output=_output_path(output),
            source=_source(source),
            source_url=source,
            requested_ref=ref,
            force=force,
        )
    except BuildError as error:
        LOGGER.error("Corpus build failed: %s", error)
        raise typer.Exit(code=2) from None

    typer.echo(f"papyrus-corpus-build {BUILDER_VERSION}: artifact created")
    typer.echo(f"Artifact path: {result.output_dir}")
    typer.echo(f"Collections: {', '.join(result.collections)}")
    typer.echo(f"Source commit: {result.resolved_commit}")
    typer.echo(f"Logical content hash: {result.logical_content_hash}")
    typer.echo(f"documents: {result.documents}")
    typer.echo(f"passages: {result.passages}")
    typer.echo(f"parse errors: {result.parse_errors}")
    for warning in result.warnings:
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW)
    typer.echo(f"Artifact size: {result.size_bytes} bytes")
    typer.echo(f"Elapsed: {result.elapsed_seconds:.2f}s")


def _output_path(output: str) -> Path:
    return Path(output).expanduser()


def _source(source: str) -> LocalGitSource | RemoteGitSource:
    path = Path(source).expanduser()
    if path.exists():
        return LocalGitSource(path)
    if source.startswith(("http://", "https://", "git@", "file://")) or source.endswith(".git"):
        return RemoteGitSource(source)
    raise BuildError(f"Source not found: {source}. Pass a Git checkout path or a remote Git URL.")


if __name__ == "__main__":
    app()
