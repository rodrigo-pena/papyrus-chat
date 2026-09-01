"""Coverage for the documentary corpus artifact schema and build boundary."""

import json
import shutil
import sqlite3
import subprocess
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
        "passage_languages",
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
    assert (
        connection.execute(
            "SELECT count(*) FROM documents_fts WHERE documents_fts MATCH 'Geld'"
        ).fetchone()[0]
        == 1
    )
    connection.close()

    reader = ArtifactReader(result.output_dir / "corpus.sqlite")
    components = reader.get_components("ddbdp:DDbDP/27/27093.xml")
    assert {component.kind for component in components} == {"ddbdp", "hgv"}
    hgv_component = next(component for component in components if component.kind == "hgv")
    assert hgv_component.document_id is None
    assert reader.get_passages("ddbdp:DDbDP/27/27093.xml")[0].language == "grc"
    reader.close()


def test_shared_hgv_component_is_linked_to_every_ddbdp_document(
    tmp_path: Path, fixture_git_repo: Path
) -> None:
    source_repo = tmp_path / "source"
    subprocess.run(["git", "clone", "--quiet", str(fixture_git_repo), str(source_repo)], check=True)
    duplicate = source_repo / "DDbDP" / "27" / "27094.xml"
    shutil.copy2(source_repo / "DDbDP" / "27" / "27093.xml", duplicate)
    subprocess.run(["git", "add", str(duplicate)], cwd=source_repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "add shared HGV fixture",
        ],
        cwd=source_repo,
        check=True,
    )
    result = build_artifact(
        ["ddbdp"],
        output=tmp_path / "corpus",
        source=LocalGitSource(source_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="HEAD",
    )

    connection = sqlite3.connect(result.output_dir / "corpus.sqlite")
    assert connection.execute(
        "SELECT document_id FROM components WHERE kind = 'hgv'"
    ).fetchone() == (None,)
    assert connection.execute("SELECT count(*) FROM component_links").fetchone() == (2,)
    assert connection.execute(
        "SELECT count(*) FROM documents_fts WHERE documents_fts MATCH 'Terentianus'"
    ).fetchone() == (2,)
    connection.close()

    reader = ArtifactReader(result.output_dir / "corpus.sqlite")
    assert {component.kind for component in reader.get_components("ddbdp:DDbDP/27/27093.xml")} == {
        "ddbdp",
        "hgv",
    }
    assert {component.kind for component in reader.get_components("ddbdp:DDbDP/27/27094.xml")} == {
        "ddbdp",
        "hgv",
    }
    reader.close()


def test_cli_lists_ddbdp_as_a_supported_collection() -> None:
    result = runner.invoke(app, ["--list-collections"])

    assert result.exit_code == 0
    assert "ddbdp" in result.output.splitlines()
