"""Ordered content-index mappings and validation, without optional dependencies."""

import hashlib
import math
import sqlite3
import struct
from pathlib import Path
from typing import Literal

from papyrus_chat.artifact.hashing import canonical_json
from papyrus_chat.artifact.manifest import ArtifactInvalid, ContentIndexInfo

ContentKind = Literal["chunks", "profiles"]


def content_rows_hash(connection: sqlite3.Connection, kind: ContentKind) -> str:
    """Hash row-to-vector alignment and every source reference in stable order."""
    digest = hashlib.sha256()
    tables = [(f"semantic_{kind}", "vector_row")]
    if kind == "profiles":
        tables.extend(
            [
                ("semantic_profile_passages", "profile_id, passage_id, char_start"),
                ("semantic_profile_components", "profile_id, component_id"),
            ]
        )
    for table, order in tables:
        digest.update(table.encode())
        for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            digest.update((canonical_json(tuple(row)) + "\n").encode())
    return f"sha256:{digest.hexdigest()}"


def validate_content_index(
    root: Path,
    connection: sqlite3.Connection,
    kind: ContentKind,
    index: ContentIndexInfo,
    dimensions: int,
) -> None:
    table = f"semantic_{kind}"
    count, minimum, maximum = connection.execute(
        f"SELECT count(*), min(vector_row), max(vector_row) FROM {table}"
    ).fetchone()
    if count != index.count or (count and (minimum != 0 or maximum != count - 1)):
        raise ArtifactInvalid(f"Content {kind} row count/alignment does not match manifest")
    if index.preprocessing_version != f"{kind}-v1":
        raise ArtifactInvalid(f"Content {kind} preprocessing version does not match its index")
    if content_rows_hash(connection, kind) != index.rows_hash:
        raise ArtifactInvalid(f"Content {kind} row mapping hash mismatch")
    if kind == "chunks":
        invalid = connection.execute(
            "SELECT 1 FROM semantic_chunks c LEFT JOIN passages p USING (passage_id) "
            "WHERE p.passage_id IS NULL OR c.document_id != p.document_id "
            "OR c.char_end > length(p.display_text) LIMIT 1"
        ).fetchone()
    else:
        invalid = connection.execute(
            "SELECT 1 FROM semantic_profile_passages s "
            "JOIN semantic_profiles d USING (profile_id) LEFT JOIN passages p USING (passage_id) "
            "WHERE p.passage_id IS NULL OR d.document_id != p.document_id "
            "OR s.char_end > length(p.display_text) UNION ALL "
            "SELECT 1 FROM semantic_profiles d WHERE d.metadata_only != NOT EXISTS "
            "(SELECT 1 FROM passages p WHERE p.document_id = d.document_id "
            "AND length(trim(p.display_text)) > 0) UNION ALL "
            "SELECT 1 FROM semantic_profile_components s "
            "JOIN semantic_profiles d USING (profile_id) JOIN components c USING (component_id) "
            "WHERE (c.document_id IS NULL OR c.document_id != d.document_id) AND NOT EXISTS "
            "(SELECT 1 FROM component_links l JOIN components owner "
            "ON owner.component_id = l.ddbdp_component_id "
            "WHERE l.hgv_component_id = c.component_id AND owner.document_id = d.document_id) "
            "LIMIT 1"
        ).fetchone()
    if invalid:
        raise ArtifactInvalid(f"Content {kind} source ownership or offsets are invalid")
    validate_vectors(root / index.embeddings_file, count=count, dimensions=dimensions)


def validate_vectors(path: Path, *, count: int, dimensions: int) -> None:
    """Check a matrix in bounded blocks without loading it all into memory."""
    row_bytes = dimensions * 4
    if path.stat().st_size != count * row_bytes:
        raise ArtifactInvalid("Semantic embedding file length does not match manifest")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(row_bytes * 256), b""):
            for vector in struct.iter_unpack(f"<{dimensions}f", block):
                norm = sum(value * value for value in vector)
                if not math.isfinite(norm) or norm == 0:
                    raise ArtifactInvalid("Semantic embeddings contain a zero or non-finite vector")
                if not math.isclose(norm, 1.0, rel_tol=2e-4, abs_tol=2e-4):
                    raise ArtifactInvalid("Semantic embeddings must be unit-normalized")
