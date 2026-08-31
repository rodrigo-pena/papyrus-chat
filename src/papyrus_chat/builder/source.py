"""Source acquisition for corpus builds.

A `CorpusSource` resolves a Git ref to an exact commit and reads files from
that commit's tree — never from a working tree, so uncommitted changes and
dirty checkouts cannot alter a build. The artifact remains usable after the
source disappears: files are copied into the artifact at build time.
"""

import subprocess
from pathlib import Path
from typing import Protocol

from papyrus_chat.builder.errors import BuildError


def _run_git(args: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BuildError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


class CorpusSource(Protocol):
    """A build source: ref resolution plus commit-accurate file reads."""

    def resolve_commit(self, ref: str) -> str: ...

    def xml_files(self, upstream_dir: str) -> list[str]: ...

    def read_bytes(self, path: str) -> bytes: ...


def _full_sha(ref: str) -> str | None:
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
        return ref.lower()
    return None


class LocalGitSource:
    """A local Git checkout; files are read from the resolved commit's tree."""

    def __init__(self, checkout: Path) -> None:
        if not (checkout / ".git").exists():
            raise BuildError(
                f"Source is not a Git checkout: {checkout}. "
                "Pass a Git checkout (or remote URL) as --source; "
                "unversioned directories cannot provide reproducible builds."
            )
        self.checkout = checkout
        self._commit: str | None = None

    def resolve_commit(self, ref: str) -> str:
        sha = _full_sha(ref)
        if sha is None:
            resolved = _run_git(
                ["rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=self.checkout, check=False
            )
            if not _full_sha(resolved):
                raise BuildError(
                    f"Cannot resolve ref {ref!r} in {self.checkout}: "
                    "unknown revision. Use a branch, tag, or commit SHA."
                )
            sha = resolved
        self._commit = sha
        return sha

    def _require_commit(self) -> str:
        if self._commit is None:
            raise BuildError("resolve_commit() must be called before reading files")
        return self._commit

    def xml_files(self, upstream_dir: str) -> list[str]:
        commit = self._require_commit()
        output = _run_git(
            ["ls-tree", "-r", "--name-only", commit, "--", upstream_dir],
            cwd=self.checkout,
            check=False,
        )
        if not output:
            return []
        return sorted(
            line for line in output.splitlines() if line.endswith(".xml") and _is_safe_path(line)
        )

    def read_bytes(self, path: str) -> bytes:
        commit = self._require_commit()
        completed = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=self.checkout,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise BuildError(
                f"Cannot read {path!r} from commit {commit[:12]}: "
                + completed.stderr.decode(errors="replace").strip()
            )
        return completed.stdout


def _is_safe_path(path: str) -> bool:
    return not path.startswith("/") and ".." not in Path(path).parts


class RemoteGitSource:
    """A remote Git URL acquired into a user cache with a partial clone.

    Only the selected collections' blobs are downloaded (blob:none filter +
    sparse checkout, SPEC 6.2). The cache lives outside the artifact and the
    built artifact stays usable after the cache is removed.
    """

    COLLECTION_DIRS = {
        "dclp": "DCLP",
        "ddbdp": "DDbDP",
        "hgv": "HGV_meta_EpiDoc",
        "translations": "Translations",
    }

    def __init__(self, url: str, cache_dir: Path | None = None) -> None:
        self.url = url
        if cache_dir is None:
            from platformdirs import user_cache_dir

            cache_dir = Path(user_cache_dir("papyrus-chat", "papyrus-chat")) / "sources"
        self.cache_dir = cache_dir
        self.cache_key = _cache_key(url)
        self.worktree = self.cache_dir / self.cache_key
        self._commit: str | None = None

    def resolve_commit(self, ref: str) -> str:
        self._ensure_clone()
        sha = _full_sha(ref)
        if sha is None:
            resolved = _run_git(
                ["rev-parse", "--verify", f"origin/{ref}^{{commit}}"],
                cwd=self.worktree,
                check=False,
            )
            if not _full_sha(resolved):
                raise BuildError(
                    f"Cannot resolve ref {ref!r} in {self.url}: "
                    "unknown revision. Use a branch, tag, or commit SHA."
                )
            sha = resolved
        self._commit = sha
        return sha

    def _ensure_clone(self) -> None:
        if (self.worktree / ".git").exists():
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                self.url,
                str(self.worktree),
            ],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode(errors="replace").strip()
            raise BuildError(
                f"Cannot clone {self.url} into the local cache: {detail}. "
                "Check the URL and your network connection."
            )

    def ensure_sparse_checkout(self, collections: list[str]) -> None:
        """Limit the working tree to the selected collections' directories.

        Uses cone-mode sparse checkout plus a full checkout of the resolved
        commit: only the selected collections' blobs are fetched from the
        promisor remote.
        """
        dirs = sorted(self.COLLECTION_DIRS.get(c, c) for c in {col.lower() for col in collections})
        _run_git(["sparse-checkout", "set", "--cone", *dirs], cwd=self.worktree)
        _run_git(["checkout", self._require_commit()], cwd=self.worktree)

    def xml_files(self, upstream_dir: str) -> list[str]:
        commit = self._require_commit()
        output = _run_git(
            ["ls-tree", "-r", "--name-only", commit, "--", upstream_dir],
            cwd=self.worktree,
            check=False,
        )
        if not output:
            return []
        return sorted(
            line for line in output.splitlines() if line.endswith(".xml") and _is_safe_path(line)
        )

    def read_bytes(self, path: str) -> bytes:
        commit = self._require_commit()
        completed = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=self.worktree,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode(errors="replace").strip()
            raise BuildError(
                f"Cannot read {path!r} from commit {commit[:12]}: {detail}. "
                "The blob may not be fetched yet; retry the build."
            )
        return completed.stdout

    def _require_commit(self) -> str:
        if self._commit is None:
            raise BuildError("resolve_commit() must be called before reading files")
        return self._commit


def _cache_key(url: str) -> str:
    import hashlib

    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
