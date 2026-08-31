"""Command-line interface for building corpus artifacts."""

import typer

app = typer.Typer(
    add_completion=False, help="Build a searchable corpus artifact from idp.data collections."
)

SUPPORTED_COLLECTIONS = ("dclp", "translations")


@app.command()
def build(
    collections: list[str] | None = typer.Argument(
        None,
        metavar="COLLECTION...",
        help="One or more collections to include (see --list-collections).",
    ),
    list_collections: bool = typer.Option(
        False,
        "--list-collections",
        help="Print supported collection names and exit.",
    ),
) -> None:
    if list_collections:
        for name in SUPPORTED_COLLECTIONS:
            typer.echo(name)
        return

    if not collections:
        typer.secho(
            "No collection specified. Supported collections: " + ", ".join(SUPPORTED_COLLECTIONS),
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    unknown = [c for c in collections if c.lower() not in SUPPORTED_COLLECTIONS]
    if unknown:
        typer.secho(
            f"Unknown collection: {', '.join(unknown)}. Supported collections: "
            + ", ".join(SUPPORTED_COLLECTIONS),
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    typer.secho("Corpus building is not implemented yet.", err=True, fg=typer.colors.RED)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
