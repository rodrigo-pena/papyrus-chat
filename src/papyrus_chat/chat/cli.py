"""Command-line interface for the local chat application."""

import typer

app = typer.Typer(add_completion=False, help="Search and chat with a built corpus artifact.")


@app.callback(invoke_without_command=True)
def serve(
    ctx: typer.Context,
    artifact: str | None = typer.Option(None, "--artifact", help="Corpus artifact directory."),
) -> None:
    if ctx.invoked_subcommand is None:
        typer.secho(
            "The chat application is not implemented yet.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
