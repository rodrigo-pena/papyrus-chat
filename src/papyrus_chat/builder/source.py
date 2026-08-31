"""Source acquisition for corpus builds (SPEC 6.2).

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
