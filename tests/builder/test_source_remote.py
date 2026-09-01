"""Remote source acquisition: partial clone, sparse checkout, user cache."""

import subprocess
from pathlib import Path

import pytest

import papyrus_chat.builder.source as source_module
from papyrus_chat.builder.errors import BuildError
from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import RemoteGitSource


def git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


@pytest.fixture()
def remote_repo(tmp_path: Path, fixture_git_repo: Path) -> Path:
    """A bare repo acting as the 'remote' for offline acquisition tests."""
    remote = tmp_path / "remote.git"
    git("clone", "--bare", str(fixture_git_repo), str(remote))
    return remote


class TestRemoteAcquisition:
    def test_resolves_ref_and_reads_from_commit(self, tmp_path: Path, remote_repo: Path) -> None:
        cache = tmp_path / "cache"
        source = RemoteGitSource("file://" + str(remote_repo), cache_dir=cache)

        sha = source.resolve_commit("master")
        assert len(sha) == 40

        files = source.xml_files("DCLP")
        assert "DCLP/23/23944.xml" in files
        assert "DCLP/23/23702.xml" in files

        content = source.read_bytes("DCLP/23/23702.xml")
        assert b"Sb. 20 14258" in content

    def test_only_selected_collection_blobs_are_fetched(
        self, tmp_path: Path, remote_repo: Path
    ) -> None:
        cache = tmp_path / "cache"
        source = RemoteGitSource("file://" + str(remote_repo), cache_dir=cache)
        source.resolve_commit("master")
        source.ensure_sparse_checkout(["dclp"])
        source.read_bytes("DCLP/23/23702.xml")

        # Translations blobs must not be present in the cache working tree
        translations_dir = source.worktree / "Translations"
        assert not translations_dir.exists() or not any(translations_dir.iterdir())

    def test_sparse_checkout_reads_do_not_spawn_per_file_git_processes(
        self,
        tmp_path: Path,
        remote_repo: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = RemoteGitSource("file://" + str(remote_repo), cache_dir=tmp_path / "cache")
        source.resolve_commit("master")
        source.ensure_sparse_checkout(["dclp"])

        def unexpected_subprocess(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"unexpected subprocess: {args!r}, {kwargs!r}")

        monkeypatch.setattr(source_module.subprocess, "run", unexpected_subprocess)

        content = source.read_bytes("DCLP/23/23702.xml")

        assert b"Sb. 20 14258" in content

    def test_sparse_checkout_reads_commit_objects_despite_worktree_changes(
        self, tmp_path: Path, remote_repo: Path
    ) -> None:
        source = RemoteGitSource("file://" + str(remote_repo), cache_dir=tmp_path / "cache")
        source.resolve_commit("master")
        source.ensure_sparse_checkout(["dclp"])
        source_dir = source.worktree / "DCLP" / "23"
        moved_dir = source.worktree / "DCLP" / "records"
        source_dir.rename(moved_dir)
        source_dir.symlink_to(moved_dir.name)

        content = source.read_bytes("DCLP/23/23702.xml")

        assert b"Sb. 20 14258" in content

    def test_cache_is_reused_across_instances(self, tmp_path: Path, remote_repo: Path) -> None:
        cache = tmp_path / "cache"

        first = RemoteGitSource("file://" + str(remote_repo), cache_dir=cache)
        first.resolve_commit("master")
        first.ensure_sparse_checkout(["dclp"])

        # Note the clone directory mtime; a second source reuses the cache
        second = RemoteGitSource("file://" + str(remote_repo), cache_dir=cache)
        second.resolve_commit("master")
        second.ensure_sparse_checkout(["dclp"])

        assert first.cache_key == second.cache_key
        assert second.read_bytes("DCLP/23/23702.xml") == first.read_bytes("DCLP/23/23702.xml")

    def test_unresolvable_ref_fails(self, tmp_path: Path, remote_repo: Path) -> None:
        source = RemoteGitSource("file://" + str(remote_repo), cache_dir=tmp_path / "c")

        with pytest.raises(BuildError, match="no-such-ref"):
            source.resolve_commit("no-such-ref")

    def test_unknown_remote_fails_with_clear_error(self, tmp_path: Path) -> None:
        source = RemoteGitSource("file:///nonexistent/repo.git", cache_dir=tmp_path / "c")

        with pytest.raises(BuildError, match="[Cc]lone"):
            source.resolve_commit("master")

    def test_built_artifact_survives_cache_deletion(
        self, tmp_path: Path, remote_repo: Path
    ) -> None:
        from papyrus_chat.artifact.validation import validate_artifact

        cache = tmp_path / "cache"
        source = RemoteGitSource("file://" + str(remote_repo), cache_dir=cache)
        result = build_artifact(
            ["dclp"],
            output=tmp_path / "corpus",
            source=source,
            source_url="https://github.com/papyri/idp.data.git",
            requested_ref="master",
        )

        import shutil

        shutil.rmtree(cache)

        validate_artifact(result.output_dir)
        assert (result.output_dir / "corpus.sqlite").is_file()
