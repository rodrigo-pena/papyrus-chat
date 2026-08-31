"""Failure, replacement, and atomic assembly behavior."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import papyrus_chat.builder.pipeline as pipeline_module
from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.builder.cli import app
from papyrus_chat.builder.pipeline import BuildError, build_artifact
from papyrus_chat.builder.source import LocalGitSource

runner = CliRunner()


def build(tmp_path: Path, output: str, repo: Path, **kwargs) -> None:
    build_artifact(
        ["dclp"],
        output=tmp_path / output,
        source=LocalGitSource(repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
        **kwargs,
    )


class TestReplacement:
    def test_existing_output_fails_before_building_without_force(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        build(tmp_path, "corpus", repo=fixture_git_repo)
        before = (tmp_path / "corpus" / "manifest.json").read_text()

        with pytest.raises(BuildError, match="--force"):
            build(tmp_path, "corpus", repo=fixture_git_repo)

        assert (tmp_path / "corpus" / "manifest.json").read_text() == before
        assert list((tmp_path / "corpus").iterdir()) and "No source" not in str(
            (tmp_path / "corpus").iterdir()
        )

    def test_force_replaces_exactly_the_named_directory(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        build(tmp_path, "corpus", repo=fixture_git_repo)
        sibling = tmp_path / "sibling-artifact"
        sibling.mkdir()
        (sibling / "keep.txt").write_text("keep", encoding="utf-8")

        build(tmp_path, "corpus", repo=fixture_git_repo, force=True)

        validate_artifact(tmp_path / "corpus")
        assert (sibling / "keep.txt").read_text() == "keep"


class TestFailureSafety:
    def test_malformed_record_fails_build_and_preserves_previous_artifact(
        self, tmp_path: Path, fixture_git_repo: Path, invalid_git_repo: Path
    ) -> None:
        build(tmp_path, "corpus", repo=fixture_git_repo)
        previous_hash = json.loads((tmp_path / "corpus" / "manifest.json").read_text())[
            "logical_content_hash"
        ]

        with pytest.raises(BuildError) as excinfo:
            build(tmp_path, "corpus2", repo=invalid_git_repo)

        message = str(excinfo.value)
        assert "dclp" in message
        assert "broken-record.xml" in message
        assert (
            json.loads((tmp_path / "corpus" / "manifest.json").read_text())["logical_content_hash"]
            == previous_hash
        )
        assert not (tmp_path / "corpus2").exists()

    def test_failed_validation_leaves_previous_artifact_intact(
        self, tmp_path: Path, fixture_git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        build(tmp_path, "corpus", repo=fixture_git_repo)
        previous = (tmp_path / "corpus" / "manifest.json").read_text()

        calls = {"count": 0}
        real_validate = pipeline_module.validate_artifact

        def flaky_validate(root: Path) -> None:
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("simulated validation failure")
            real_validate(root)

        monkeypatch.setattr(pipeline_module, "validate_artifact", flaky_validate)

        with pytest.raises(RuntimeError):
            build(tmp_path, "corpus", repo=fixture_git_repo, force=True)

        assert (tmp_path / "corpus" / "manifest.json").read_text() == previous

    def test_no_llm_environment_is_read_during_build(
        self, tmp_path: Path, fixture_git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_API_KEY", "sk-super-secret")
        monkeypatch.setenv("LLM_BASE_URL", "https://secret.example")

        build(tmp_path, "corpus", repo=fixture_git_repo)

        for name in ("manifest.json", "ATTRIBUTION.md"):
            content = (tmp_path / "corpus" / name).read_text(encoding="utf-8")
            assert "sk-super-secret" not in content
            assert "secret.example" not in content


class TestForceCli:
    def test_cli_existing_output_without_force_fails(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        args = [
            "dclp",
            "--output",
            str(tmp_path / "corpus"),
            "--source",
            str(fixture_git_repo),
            "--ref",
            "master",
        ]
        assert runner.invoke(app, args).exit_code == 0

        second = runner.invoke(app, args)
        assert second.exit_code != 0
        assert "--force" in second.output

    def test_cli_force_replaces_artifact(self, tmp_path: Path, fixture_git_repo: Path) -> None:
        args = [
            "dclp",
            "--output",
            str(tmp_path / "corpus"),
            "--source",
            str(fixture_git_repo),
            "--ref",
            "master",
        ]
        assert runner.invoke(app, args).exit_code == 0

        forced = runner.invoke(app, [*args, "--force"])
        assert forced.exit_code == 0, forced.output
        validate_artifact(tmp_path / "corpus")
