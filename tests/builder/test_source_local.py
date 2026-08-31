"""Local Git source resolution: commit-accurate reads (SPEC 6.2)."""

import subprocess
from pathlib import Path

import pytest

from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import LocalGitSource


def git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


class TestRefResolution:
    def test_resolves_branch_tag_and_sha(self, tmp_path: Path, fixture_git_repo: Path) -> None:
        source = LocalGitSource(fixture_git_repo)

        branch_sha = source.resolve_commit("master")
        assert len(branch_sha) == 40

        git("tag", "v1", cwd=fixture_git_repo)
        assert source.resolve_commit("v1") == branch_sha
        assert source.resolve_commit(branch_sha) == branch_sha

    def test_unresolvable_ref_fails(self, fixture_git_repo: Path) -> None:
        source = LocalGitSource(fixture_git_repo)

        with pytest.raises(Exception, match="no-such-ref"):
            source.resolve_commit("no-such-ref")

    def test_non_git_directory_is_rejected(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()

        with pytest.raises(Exception, match="[Gg]it"):
            LocalGitSource(plain)


class TestCommitAccurateReads:
    def test_dirty_working_tree_does_not_change_build(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        clean = build_artifact(
            ["dclp"],
            output=tmp_path / "clean",
            source=LocalGitSource(fixture_git_repo),
            source_url="https://github.com/papyri/idp.data.git",
            requested_ref="master",
        )

        # Uncommitted changes must not affect the build: reads come from the
        # resolved commit's tree, never the working tree.
        record = fixture_git_repo / "DCLP" / "23" / "23944.xml"
        record.write_text("<TEI>completely different uncommitted content</TEI>", encoding="utf-8")
        (fixture_git_repo / "DCLP" / "23" / "uncommitted.xml").write_text(
            "<TEI></TEI>", encoding="utf-8"
        )

        dirty = build_artifact(
            ["dclp"],
            output=tmp_path / "dirty",
            source=LocalGitSource(fixture_git_repo),
            source_url="https://github.com/papyri/idp.data.git",
            requested_ref="master",
        )

        assert clean.logical_content_hash == dirty.logical_content_hash


class TestSourceBackedBuild:
    def test_build_through_git_source_records_resolved_commit(
        self, tmp_path: Path, fixture_git_repo: Path
    ) -> None:
        result = build_artifact(
            ["dclp"],
            output=tmp_path / "corpus",
            source=LocalGitSource(fixture_git_repo),
            source_url="https://github.com/papyri/idp.data.git",
            requested_ref="master",
        )

        committed_sha = git("rev-parse", "master", cwd=fixture_git_repo)
        assert result.resolved_commit == committed_sha
        assert result.documents == 2
