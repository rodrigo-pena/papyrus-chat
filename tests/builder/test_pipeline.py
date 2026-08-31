"""End-to-end tests for the corpus build pipeline and CLI (SPEC 6.1, 6.4, 6.5, 7.x)."""

import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.builder.cli import app
from papyrus_chat.builder.pipeline import BuildResult, build_artifact

FIXTURES = Path(__file__).parent.parent / "fixtures" / "idp.data"
PINNED_COMMIT = "04568cb5ea3775d8113bb6e7edfd9c7168cf7e88"

runner = CliRunner()


def build_fixture(tmp_path: Path, output: str = "corpus") -> BuildResult:
    return build_artifact(
        collections=["dclp"],
        output=tmp_path / output,
        source_dir=FIXTURES,
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
        resolved_commit=PINNED_COMMIT,
    )


class TestPipeline:
    def test_builds_valid_artifact_with_documented_files(self, tmp_path: Path) -> None:
        result = build_fixture(tmp_path)

        assert result.output_dir.is_dir()
        assert sorted(p.name for p in result.output_dir.iterdir()) == [
            "ATTRIBUTION.md",
            "corpus.sqlite",
            "manifest.json",
        ]
        validate_artifact(result.output_dir)

    def test_manifest_records_provenance_and_statistics(self, tmp_path: Path) -> None:
        result = build_fixture(tmp_path)
        manifest = json.loads((result.output_dir / "manifest.json").read_text())

        assert manifest["artifact_schema_version"] == 1
        assert manifest["source"]["resolved_commit"] == PINNED_COMMIT
        assert manifest["source"]["requested_ref"] == "master"
        assert manifest["collections"] == ["dclp"]
        assert manifest["statistics"]["documents"] == 2
        assert manifest["statistics"]["passages"] >= 1
        assert manifest["statistics"]["parse_errors"] == 0
        assert manifest["logical_content_hash"].startswith("sha256:")
        assert manifest["builder"]["name"] == "papyrus-corpus-build"

    def test_documents_and_passages_are_queryable(self, tmp_path: Path) -> None:
        result = build_fixture(tmp_path)

        connection = sqlite3.connect(result.output_dir / "corpus.sqlite")
        documents = connection.execute(
            "SELECT collection, title FROM documents ORDER BY collection, title"
        ).fetchall()
        passages = connection.execute(
            "SELECT kind, display_text FROM passages ORDER BY sequence"
        ).fetchall()
        identifiers = connection.execute(
            "SELECT namespace, value FROM identifiers WHERE namespace = 'TM'"
        ).fetchall()
        connection.close()

        assert len(documents) == 2
        assert any(row[0] == "dclp" for row in documents)
        assert any(row[1] == "Sb. 20 14258" for row in documents)
        assert any(row[0] == "edition" for row in passages)
        assert ("TM", "23702") in identifiers or ("TM", "23944") in identifiers

    def test_metadata_only_record_has_no_passages(self, tmp_path: Path) -> None:
        result = build_fixture(tmp_path)

        connection = sqlite3.connect(result.output_dir / "corpus.sqlite")
        counts = connection.execute(
            "SELECT d.title, count(p.passage_id) FROM documents d"
            " LEFT JOIN passages p ON p.document_id = d.document_id"
            " GROUP BY d.document_id ORDER BY d.title"
        ).fetchall()
        connection.close()

        by_title = dict(counts)
        assert by_title["Sb. 20 14258"] == 0

    def test_attribution_states_license_and_disclaimer(self, tmp_path: Path) -> None:
        result = build_fixture(tmp_path)
        attribution = (result.output_dir / "ATTRIBUTION.md").read_text(encoding="utf-8")

        assert "CC BY 3.0" in attribution
        assert "github.com/papyri/idp.data" in attribution
        assert PINNED_COMMIT in attribution
        assert "model-generated" in attribution.lower() or "model generated" in attribution.lower()


class TestBuildCli:
    def test_build_from_fixture_source_end_to_end(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "dclp",
                "--output",
                str(tmp_path / "corpus"),
                "--source",
                str(FIXTURES),
                "--ref",
                PINNED_COMMIT,
            ],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "corpus" / "manifest.json").is_file()
        assert "papyrus-corpus" in result.output
        assert "documents" in result.output.lower()

    def test_report_includes_counts_size_and_elapsed(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "dclp",
                "--output",
                str(tmp_path / "corpus"),
                "--source",
                str(FIXTURES),
                "--ref",
                PINNED_COMMIT,
            ],
        )

        assert "documents: 2" in result.output
        assert "size" in result.output.lower()
        assert "elapsed" in result.output.lower()


class TestMultiCollectionCli:
    def test_mixed_case_collections_are_canonicalized(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "DCLP",
                "Translations",
                "--output",
                str(tmp_path / "corpus"),
                "--source",
                str(FIXTURES),
                "--ref",
                PINNED_COMMIT,
            ],
        )

        assert result.exit_code == 0, result.output
        manifest = json.loads((tmp_path / "corpus" / "manifest.json").read_text())
        assert manifest["collections"] == ["dclp", "translations"]
