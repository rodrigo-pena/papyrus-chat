"""Corpus build pipeline: source records → validated artifact (SPEC 6, 7)."""

import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from papyrus_chat.artifact.hashing import logical_content_hash
from papyrus_chat.artifact.manifest import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactManifest,
    BuilderInfo,
    ManifestSource,
    Statistics,
    save_manifest,
)
from papyrus_chat.artifact.records import DocumentRecord, IdentifierRecord, PassageRecord
from papyrus_chat.artifact.schema import ArtifactWriter
from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.builder.collections.dclp import parse_record as parse_dclp
from papyrus_chat.builder.collections.epidoc import ParsedRecord
from papyrus_chat.builder.collections.translations import parse_record as parse_translations

BUILDER_NAME = "papyrus-corpus-build"
BUILDER_VERSION = "0.1.0"

CollectionParser = Callable[..., ParsedRecord]

SUPPORTED_COLLECTIONS: dict[str, tuple[str, CollectionParser]] = {
    "dclp": ("DCLP", parse_dclp),
    "translations": ("Translations", parse_translations),
}


class BuildError(Exception):
    """A user-facing build failure (SPEC 6.4, 11)."""


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    collections: list[str]
    resolved_commit: str
    logical_content_hash: str
    documents: int
    passages: int
    parse_errors: int
    warnings: tuple[str, ...]
    size_bytes: int
    elapsed_seconds: float


def resolve_commit(source_dir: Path, ref: str) -> str:
    """Resolve a full commit SHA, or a ref inside a local Git checkout."""
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
        return ref.lower()
    if (source_dir / ".git").exists():
        completed = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", f"{ref}^{{commit}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
        raise BuildError(f"Cannot resolve ref {ref!r} in {source_dir}: {completed.stderr.strip()}")
    raise BuildError(
        f"Cannot resolve ref {ref!r}: {source_dir} is not a Git checkout. "
        "Pass a full commit SHA or use a Git checkout as --source."
    )


def build_artifact(
    collections: list[str],
    *,
    output: Path,
    source_dir: Path,
    source_url: str,
    requested_ref: str,
    resolved_commit: str,
) -> BuildResult:
    started = time.monotonic()
    canonical = sorted({c.lower() for c in collections})
    unknown = [c for c in canonical if c not in SUPPORTED_COLLECTIONS]
    if unknown:
        raise BuildError(
            f"Unknown collection: {', '.join(unknown)}. Supported: "
            + ", ".join(sorted(SUPPORTED_COLLECTIONS))
        )
    if output.exists():
        raise BuildError(
            f"Output artifact already exists: {output}. "
            "Remove it or pass --force to replace it (replacement arrives in a later build)."
        )
    if not source_dir.is_dir():
        raise BuildError(f"Source directory not found: {source_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        documents, passages, identifiers, warnings = _parse_collections(
            canonical, source_dir=source_dir, source_url=source_url, commit=resolved_commit
        )
        for record in documents:
            if not record.languages:
                warnings.append(
                    f"{record.source.path}: no declared language; stored without languages"
                )

        database = staging / "corpus.sqlite"
        writer = ArtifactWriter(database)
        writer.create_schema()
        for record in sorted(documents, key=lambda d: d.document_id):
            writer.insert_document(record)
        writer.insert_passages(sorted(passages, key=lambda p: (p.document_id, p.sequence)))
        writer.insert_identifiers(
            sorted(identifiers, key=lambda i: (i.document_id, i.namespace, i.value))
        )
        writer.commit()
        writer.close()

        logical_hash = logical_content_hash(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            builder_version=BUILDER_VERSION,
            source_url=source_url,
            resolved_commit=resolved_commit,
            collections=canonical,
            documents=documents,
            passages=passages,
            identifiers=identifiers,
        )

        manifest = ArtifactManifest(
            builder=BuilderInfo(name=BUILDER_NAME, version=BUILDER_VERSION),
            source=ManifestSource(
                url=source_url,
                requested_ref=requested_ref,
                resolved_commit=resolved_commit,
            ),
            collections=canonical,
            statistics=Statistics(
                documents=len(documents),
                passages=len(passages),
                parse_errors=0,
            ),
            logical_content_hash=logical_hash,
            created_at=datetime.now(UTC).isoformat(),
        )
        save_manifest(staging / "manifest.json", manifest)
        (staging / "ATTRIBUTION.md").write_text(
            _attribution_text(source_url, resolved_commit), encoding="utf-8"
        )

        validate_artifact(staging)

        os.rename(staging, output)
    except Exception:
        subprocess.run(["rm", "-rf", str(staging)], check=False)
        raise

    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    return BuildResult(
        output_dir=output,
        collections=canonical,
        resolved_commit=resolved_commit,
        logical_content_hash=logical_hash,
        documents=len(documents),
        passages=len(passages),
        parse_errors=0,
        warnings=tuple(warnings),
        size_bytes=size,
        elapsed_seconds=time.monotonic() - started,
    )


def _parse_collections(
    canonical: list[str], *, source_dir: Path, source_url: str, commit: str
) -> tuple[
    list[DocumentRecord],
    list[PassageRecord],
    list[IdentifierRecord],
    list[str],
]:
    documents = []
    passages: list[PassageRecord] = []
    identifiers: list[IdentifierRecord] = []
    warnings: list[str] = []

    for collection in canonical:
        upstream_dir_name, parser = SUPPORTED_COLLECTIONS[collection]
        collection_dir = source_dir / upstream_dir_name
        if not collection_dir.is_dir():
            raise BuildError(f"Collection directory not found in source: {collection_dir}")

        files = sorted(
            (path for path in collection_dir.rglob("*.xml") if path.is_file()),
            key=lambda path: path.relative_to(source_dir).as_posix(),
        )
        for path in files:
            source_path = path.relative_to(source_dir).as_posix()
            try:
                parsed = parser(
                    path.read_bytes(),
                    collection=collection,
                    source_path=source_path,
                    repository_url=source_url,
                    commit=commit,
                )
            except Exception as error:
                raise BuildError(
                    f"Failed to parse {collection} record {source_path}: {error}"
                ) from error
            documents.append(parsed.document)
            passages.extend(parsed.passages)
            identifiers.extend(parsed.identifiers)
            warnings.extend(parsed.warnings)

    return documents, passages, identifiers, warnings


def _attribution_text(source_url: str, commit: str) -> str:
    return f"""# Attribution

The corpus data in this artifact was extracted verbatim from the
[papyri/idp.data](https://github.com/papyri/idp.data) repository, which is
distributed under CC BY 3.0
([Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/)).

- Upstream repository: {source_url}
- Resolved upstream commit: `{commit}`
- Contributing projects: the Digital Corpus of Literary Papyri (DCLP/LDAB,
  Trismegistos) and APIS-derived translation records, as identified in each
  record's `<authority>` element. See the upstream
  [README](https://github.com/papyri/idp.data#readme) for details.

Model-generated prose produced by the papyrus-chat application is **not**
part of the source corpus and is always labelled as model-generated.
"""
