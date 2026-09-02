"""Validation of corpus artifact directories."""

import json
import math
import sqlite3
import struct
from pathlib import Path

from papyrus_chat.artifact.hashing import file_sha256
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
        connection.row_factory = sqlite3.Row
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
            _validate_semantic_index(root, connection, manifest.semantic_index)
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ArtifactInvalid(f"Foreign-key violations in {database}: {len(violations)} row(s)")
    except sqlite3.DatabaseError as error:
        raise ArtifactInvalid(
            f"SQLite integrity check failed for {database}: file may be corrupt ({error})"
        ) from error
    finally:
        connection.close()


def _validate_semantic_index(
    root: Path, connection: sqlite3.Connection, semantic
) -> None:
    indexed_files = [
        semantic.subjects_file,
        semantic.embeddings_file,
        *semantic.model_files,
    ]
    if len(set(indexed_files)) != len(indexed_files):
        raise ArtifactInvalid("Semantic index files must be unique")
    for relative in indexed_files:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ArtifactInvalid(
                f"Semantic index file path must stay inside the artifact: {relative}"
            )
        candidate = root / relative
        if not candidate.is_file():
            raise ArtifactInvalid(f"Semantic index file is missing: {relative}")
    expected_files = set(indexed_files)
    actual_files = set(semantic.file_hashes)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ArtifactInvalid("Semantic index file hashes are incomplete: " + "; ".join(details))
    for relative in sorted(expected_files):
        candidate = root / relative
        if file_sha256(candidate) != semantic.file_hashes[relative]:
            raise ArtifactInvalid(f"Semantic index file hash mismatch: {relative}")

    model_path = Path("semantic/model") / semantic.model_file
    if model_path.as_posix() not in semantic.model_files:
        raise ArtifactInvalid(
            f"Semantic model file is not included in model_files: {semantic.model_file}"
        )
    subjects_path = root / semantic.subjects_file
    try:
        subject_rows = [
            json.loads(line)
            for line in subjects_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactInvalid(
            f"Semantic subjects file is invalid: {semantic.subjects_file}"
        ) from error
    if len(subject_rows) != semantic.subject_count:
        raise ArtifactInvalid(
            "Semantic subjects row count does not match manifest: "
            f"{len(subject_rows)} != {semantic.subject_count}"
        )
    database_rows = [
        dict(row)
        for row in connection.execute(
            "SELECT subject_id, value, value_norm, document_count "
            "FROM semantic_subjects ORDER BY subject_id"
        )
    ]
    if subject_rows != database_rows:
        raise ArtifactInvalid("Semantic subjects file does not match SQLite rows")

    embeddings_path = root / semantic.embeddings_file
    raw = embeddings_path.read_bytes()
    expected_length = semantic.subject_count * semantic.dimensions * 4
    if len(raw) != expected_length:
        raise ArtifactInvalid(
            "Semantic embedding file length does not match manifest: "
            f"{len(raw)} != {expected_length}"
        )
    for offset in range(0, len(raw), semantic.dimensions * 4):
        vector = struct.unpack_from(f"<{semantic.dimensions}f", raw, offset)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0 or not all(math.isfinite(value) for value in vector):
            raise ArtifactInvalid("Semantic embeddings contain a zero or non-finite vector")
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ArtifactInvalid("Semantic embeddings must be unit-normalized for cosine search")
