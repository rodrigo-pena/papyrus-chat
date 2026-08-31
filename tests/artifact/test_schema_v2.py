"""Coverage for the documentary corpus artifact schema and build boundary."""

import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from papyrus_chat.artifact.schema import ArtifactReader
from papyrus_chat.builder.cli import app
from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource

runner = CliRunner()


def test_ddbdp_build_persists_linked_components_and_distinct_document_fts(
    tmp_path: Path, fixture_git_repo: Path
) -> None:
    result = build_artifact(
        ["ddbdp"],
        output=tmp_path / "corpus",
        source=LocalGitSource(fixture_git_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
    )

    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_schema_version"] == 2
    assert manifest["collections"] == ["ddbdp"]
    assert manifest["statistics"]["components"] == 2
    assert manifest["statistics"]["links"] == 1

    connection = sqlite3.connect(result.output_dir / "corpus.sqlite")
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        )
    }
    assert {
        "components",
        "component_identifiers",
        "metadata",
        "dates",
        "languages",
        "component_links",
        "documents_fts",
        "passages_fts",
    } <= tables

    components = connection.execute(
        "SELECT component_id, kind, title FROM components ORDER BY component_id"
    ).fetchall()
    assert [row[1] for row in components] == ["ddbdp", "hgv"]
    assert components[1][2] == "Terentianus to Tiberianus"

    links = connection.execute("SELECT * FROM component_links").fetchall()
    assert links == [("ddbdp:DDbDP/27/27093.xml", "hgv:HGV_meta_EpiDoc/HGV28/27093.xml")]

    assert connection.execute(
        "SELECT namespace, value FROM component_identifiers"
        " WHERE component_id = 'hgv:HGV_meta_EpiDoc/HGV28/27093.xml'"
        " AND namespace = 'HGV'"
    ).fetchone() == ("HGV", "27093")
    assert set(
        connection.execute(
            "SELECT value FROM metadata WHERE component_id LIKE 'hgv:%' AND key = 'subject'"
        ).fetchall()
    ) == {
        ("Brief (privat)",),
        ("Terentianus an Tiberianus (Vater)",),
        ("Schwierigkeiten bei Registrierung von Urkunden",),
        ("Geld",),
        ("erneute Petition",),
    }
    assert connection.execute(
        "SELECT not_before, not_after, text FROM dates WHERE component_id LIKE 'hgv:%'"
    ).fetchone() == ("0101", "0125", "frühes II")
    assert connection.execute(
        "SELECT language, role FROM languages WHERE component_id LIKE 'ddbdp:%'"
    ).fetchall() == [("grc", "edition")]

    document_fts_count = connection.execute("SELECT count(*) FROM documents_fts").fetchone()[0]
    assert document_fts_count == 1
    assert (
        connection.execute(
            "SELECT count(*) FROM documents_fts WHERE documents_fts MATCH 'Terentianus'"
        ).fetchone()[0]
        == 1
    )
    connection.close()

    reader = ArtifactReader(result.output_dir / "corpus.sqlite")
    components = reader.get_components("ddbdp:DDbDP/27/27093.xml")
    assert {component.kind for component in components} == {"ddbdp", "hgv"}
    reader.close()


def test_cli_lists_ddbdp_as_a_supported_collection() -> None:
    result = runner.invoke(app, ["--list-collections"])

    assert result.exit_code == 0
    assert "ddbdp" in result.output.splitlines()
