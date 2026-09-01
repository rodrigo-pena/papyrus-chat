"""Source acquisition for corpus builds.

A `CorpusSource` resolves a Git ref to an exact commit and reads files from
that commit's object tree. Remote sources keep one batch object reader open so
record reads do not need one Git process per file. The artifact remains usable
after the source disappears.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Protocol

from papyrus_chat.builder.errors import BuildError

LOGGER = logging.getLogger(__name__)


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
    return (
        not path.startswith("/")
        and ".." not in Path(path).parts
        and not any(character in path for character in ("\0", "\n", "\r"))
    )


class RemoteGitSource:
    """A remote Git URL acquired into a user cache with a partial clone.

    Only the selected collections' blobs are downloaded (blob:none filter +
    sparse checkout, SPEC 6.2). The cache lives outside the artifact and the
    selected commit is force-checked-out to hydrate its blobs, then records are
    read from that commit through one persistent ``git cat-file`` process.
    Dirty or concurrent working-tree changes therefore cannot alter a build.
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
        self._batch_process: subprocess.Popen[bytes] | None = None

    def resolve_commit(self, ref: str) -> str:
        self.close()
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
            LOGGER.debug(
                "Reusing cached source repository at %s",
                self.worktree,
                extra={"event": "source_cache_reused", "path": str(self.worktree)},
            )
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        LOGGER.info(
            "Cloning source repository into the local cache (first build only)",
            extra={"event": "source_clone_started"},
        )
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
        LOGGER.info(
            "Source repository cached in %.1fs",
            time.monotonic() - started,
            extra={
                "event": "source_clone_completed",
                "elapsed_seconds": time.monotonic() - started,
            },
        )

    def ensure_sparse_checkout(self, collections: list[str]) -> None:
        """Limit the working tree to the selected collections' directories.

        Uses cone-mode sparse checkout plus a full checkout of the resolved
        commit: only the selected collections' blobs are fetched from the
        promisor remote.
        """
        dirs = sorted(self.COLLECTION_DIRS.get(c, c) for c in {col.lower() for col in collections})
        _run_git(["sparse-checkout", "set", "--cone", *dirs], cwd=self.worktree)
        commit = self._require_commit()
        _run_git(["checkout", "--force", commit], cwd=self.worktree)
        self._start_batch_reader()

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
        if not _is_safe_path(path):
            raise BuildError(f"Cannot read unsafe source path: {path!r}")
        if self._batch_process is not None:
            return self._read_batch_bytes(path, commit)
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

    def _start_batch_reader(self) -> None:
        self.close()
        try:
            self._batch_process = subprocess.Popen(
                ["git", "cat-file", "--batch"],
                cwd=self.worktree,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise BuildError(f"Cannot start Git batch reader: {error}") from error

    def _read_batch_bytes(self, path: str, commit: str) -> bytes:
        process = self._batch_process
        if process is None or process.stdin is None or process.stdout is None:
            raise BuildError("Git batch reader is not available")
        object_name = f"{commit}:{path}"
        try:
            process.stdin.write(object_name.encode("utf-8") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline()
            if header.endswith(b" missing\n"):
                raise BuildError(
                    f"Cannot read {path!r} from commit {commit[:12]}: object is missing"
                )
            fields = header.rstrip(b"\n").rsplit(b" ", 2)
            if len(fields) != 3 or fields[1] != b"blob":
                raise ValueError(f"unexpected Git response {header!r}")
            size = int(fields[2])
            content = process.stdout.read(size)
            terminator = process.stdout.read(1)
            if len(content) != size or terminator != b"\n":
                raise ValueError("truncated Git object response")
            return content
        except (BrokenPipeError, OSError, ValueError) as error:
            raise BuildError(
                f"Cannot read {path!r} from commit {commit[:12]} via Git batch reader: {error}"
            ) from error

    def close(self) -> None:
        process = getattr(self, "_batch_process", None)
        self._batch_process = None
        if process is None:
            return
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_commit(self) -> str:
        if self._commit is None:
            raise BuildError("resolve_commit() must be called before reading files")
        return self._commit


def _cache_key(url: str) -> str:
    import hashlib

    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
