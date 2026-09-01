"""End-to-end tests for the corpus build pipeline and CLI."""

import json
import sqlite3
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.builder.cli import app
from papyrus_chat.builder.pipeline import BuildResult, build_artifact
from papyrus_chat.builder.source import LocalGitSource

runner = CliRunner()


def repo_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "master"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def build_fixture(tmp_path: Path, output: str = "corpus", repo: Path | None = None) -> BuildResult:
    assert repo is not None, "pass the fixture_git_repo fixture"
    return build_artifact(
        collections=["dclp"],
        output=tmp_path / output,
        source=LocalGitSource(repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
    )


class TestPipeline:
    def test_builds_valid_artifact_with_documented_files(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        result = build_fixture(tmp_path, repo=fixture_git_repo)

        assert result.output_dir.is_dir()
        assert sorted(p.name for p in result.output_dir.iterdir()) == [
            "ATTRIBUTION.md",
            "corpus.sqlite",
            "manifest.json",
        ]
        validate_artifact(result.output_dir)

    def test_manifest_records_provenance_and_statistics(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        result = build_fixture(tmp_path, repo=fixture_git_repo)
        manifest = json.loads((result.output_dir / "manifest.json").read_text())

        assert manifest["artifact_schema_version"] == 2
        assert manifest["source"]["resolved_commit"] == repo_sha(fixture_git_repo)
        assert manifest["source"]["requested_ref"] == "master"
        assert manifest["collections"] == ["dclp"]
        assert manifest["statistics"]["documents"] == 2
        assert manifest["statistics"]["passages"] >= 1
        assert manifest["statistics"]["parse_errors"] == 0
        assert manifest["logical_content_hash"].startswith("sha256:")
        assert manifest["builder"]["name"] == "papyrus-corpus-build"

    def test_documents_and_passages_are_queryable(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        result = build_fixture(tmp_path, repo=fixture_git_repo)

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

    def test_metadata_only_record_has_no_passages(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        result = build_fixture(tmp_path, repo=fixture_git_repo)

        connection = sqlite3.connect(result.output_dir / "corpus.sqlite")
        counts = connection.execute(
            "SELECT d.title, count(p.passage_id) FROM documents d"
            " LEFT JOIN passages p ON p.document_id = d.document_id"
            " GROUP BY d.document_id ORDER BY d.title"
        ).fetchall()
        connection.close()

        by_title = dict(counts)
        assert by_title["Sb. 20 14258"] == 0

    def test_attribution_states_license_and_disclaimer(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        result = build_fixture(tmp_path, repo=fixture_git_repo)
        attribution = (result.output_dir / "ATTRIBUTION.md").read_text(encoding="utf-8")

        assert "CC BY 3.0" in attribution
        assert "github.com/papyri/idp.data" in attribution
        assert repo_sha(fixture_git_repo) in attribution
        assert "model-generated" in attribution.lower() or "model generated" in attribution.lower()


class TestBuildCli:
    def cli_build(self, tmp_path: Path, repo: Path, *extra: str):
        return runner.invoke(
            app,
            [
                "dclp",
                "--output",
                str(tmp_path / "corpus"),
                "--source",
                str(repo),
                "--ref",
                "master",
                *extra,
            ],
        )

    def test_build_from_git_source_end_to_end(self, tmp_path: Path, fixture_git_repo: Path) -> None:
        result = self.cli_build(tmp_path, fixture_git_repo)

        assert result.exit_code == 0, result.output
        assert (tmp_path / "corpus" / "manifest.json").is_file()
        assert "papyrus-corpus" in result.output
        assert "documents" in result.output.lower()

    def test_report_includes_counts_size_and_elapsed(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        result = self.cli_build(tmp_path, fixture_git_repo)

        assert "documents: 2" in result.output
        assert "size" in result.output.lower()
        assert "elapsed" in result.output.lower()

    def test_reports_build_stages_and_parsing_progress(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        result = self.cli_build(tmp_path, fixture_git_repo)

        assert result.exit_code == 0, result.output
        assert "Starting corpus build" in result.output
        assert "Resolving source ref" in result.output
        assert "Parsing dclp collection (2 XML records)" in result.output
        assert "Parsed dclp records: 1/2 (50%)" in result.output
        assert "Validating parsed corpus integrity" in result.output
        assert "Writing corpus database" in result.output
        assert "Validating staged artifact" in result.output
        assert "Corpus build completed" in result.output


class TestMultiCollectionCli:
    def test_mixed_case_collections_are_canonicalized(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "DCLP",
                "Translations",
                "--output",
                str(tmp_path / "corpus"),
                "--source",
                str(fixture_git_repo),
                "--ref",
                "master",
            ],
        )

        assert result.exit_code == 0, result.output
        manifest = json.loads((tmp_path / "corpus" / "manifest.json").read_text())
        assert manifest["collections"] == ["dclp", "translations"]
