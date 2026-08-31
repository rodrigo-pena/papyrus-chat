"""Command-line interface for building corpus artifacts (SPEC 6.1)."""

import typer

from papyrus_chat.builder.pipeline import (
    BUILDER_VERSION,
    SUPPORTED_COLLECTIONS,
    BuildError,
    build_artifact,
    resolve_commit,
)

app = typer.Typer(
    add_completion=False,
    help="Build a searchable corpus artifact from selected idp.data collections.",
)

DEFAULT_SOURCE = "https://github.com/papyri/idp.data.git"


@app.command()
def build(
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
    list_collections: bool = typer.Option(
        False, "--list-collections", help="Print supported collection names and exit."
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

    try:
        source_dir = _local_source_dir(source)
        resolved_commit = resolve_commit(source_dir, ref)
        result = build_artifact(
            collections,
            output=_output_path(output),
            source_dir=source_dir,
            source_url=source,
            requested_ref=ref,
            resolved_commit=resolved_commit,
        )
    except BuildError as error:
        typer.secho(str(error), err=True, fg=typer.colors.RED)
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


def _output_path(output: str):
    from pathlib import Path

    return Path(output).expanduser()


def _local_source_dir(source: str):
    from pathlib import Path

    path = Path(source).expanduser()
    if source.startswith(("http://", "https://", "git@")):
        raise BuildError(
            "Remote source acquisition is not implemented in this build. "
            "Pass --source pointing to a local idp.data checkout or fixture directory."
        )
    if not path.is_dir():
        raise BuildError(f"Source directory not found: {path}")
    return path


if __name__ == "__main__":
    app()
