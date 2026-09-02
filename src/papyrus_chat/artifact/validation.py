"""Validation of corpus artifact directories."""

import sqlite3
from pathlib import Path

from papyrus_chat.artifact.manifest import (
    ARTIFACT_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    ArtifactInvalid,
    load_manifest,
)

CORPUS_FILENAME = "corpus.sqlite"
ATTRIBUTION_FILENAME = "ATTRIBUTION.md"
REQUIRED_FILES = (MANIFEST_FILENAME, CORPUS_FILENAME, ATTRIBUTION_FILENAME)
REQUIRED_TABLES = {
    "documents",
    "identifiers",
    "passages",
    "passages_fts",
    "passage_languages",
    "components",
    "component_identifiers",
    "metadata",
    "semantic_subjects",
    "semantic_subjects_fts",
    "dates",
    "languages",
    "component_links",
    "documents_fts",
}


def validate_artifact(root: Path) -> None:
    """Validate manifest, required files, and SQLite integrity of an artifact."""
    if not root.is_dir():
        raise ArtifactInvalid(f"Artifact directory not found: {root}")

    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            raise ArtifactInvalid(f"Artifact is missing required file: {name}")

    load_manifest(root / MANIFEST_FILENAME)

    database = root / CORPUS_FILENAME
    try:
        connection = sqlite3.connect(database)
    except sqlite3.Error as error:
        raise ArtifactInvalid(f"Cannot open {CORPUS_FILENAME}: {error}") from error
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ArtifactInvalid(f"SQLite integrity check failed for {database}")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        missing_tables = sorted(REQUIRED_TABLES - tables)
        if missing_tables:
            raise ArtifactInvalid(
                f"SQLite artifact is missing schema v{ARTIFACT_SCHEMA_VERSION} table(s): "
                + ", ".join(missing_tables)
                + ". Rebuild this artifact with papyrus-corpus-build 0.3.0 or newer."
            )
        manifest = load_manifest(root / MANIFEST_FILENAME)
        if manifest.semantic_index is not None:
            indexed_files = {
                manifest.semantic_index.subjects_file,
                manifest.semantic_index.embeddings_file,
                *manifest.semantic_index.model_files,
            }
            for relative in sorted(indexed_files):
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ArtifactInvalid(
                        f"Semantic index file path must stay inside the artifact: {relative}"
                    )
                candidate = root / relative
                if not candidate.is_file():
                    raise ArtifactInvalid(f"Semantic index file is missing: {relative}")
                expected = manifest.semantic_index.file_hashes.get(relative)
                if expected is not None:
                    from papyrus_chat.artifact.hashing import file_sha256

                    actual = file_sha256(candidate)
                    if actual != expected:
                        raise ArtifactInvalid(f"Semantic index file hash mismatch: {relative}")
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ArtifactInvalid(f"Foreign-key violations in {database}: {len(violations)} row(s)")
    except sqlite3.DatabaseError as error:
        raise ArtifactInvalid(
            f"SQLite integrity check failed for {database}: file may be corrupt ({error})"
        ) from error
    finally:
        connection.close()
