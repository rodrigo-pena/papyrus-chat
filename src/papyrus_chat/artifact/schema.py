"""SQLite logical schema for corpus artifacts (SPEC 7.3).

Schema version 1 exposes: documents, identifiers, passages, and an FTS5
index over passage search text and document titles. Stable IDs derive from
collection, source identity (path), and structural location — never from
insertion order.
"""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from papyrus_chat.artifact.records import (
    DocumentRecord,
    IdentifierRecord,
    PassageRecord,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE documents (
    document_id   TEXT PRIMARY KEY,
    collection    TEXT NOT NULL,
    title         TEXT NOT NULL,
    languages     TEXT NOT NULL,
    metadata      TEXT NOT NULL,
    source_url    TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    locator       TEXT,
    canonical_url TEXT
);

CREATE TABLE identifiers (
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    namespace   TEXT NOT NULL,
    value       TEXT NOT NULL,
    PRIMARY KEY (document_id, namespace, value)
);
CREATE INDEX identifiers_lookup ON identifiers(namespace, value);

CREATE TABLE passages (
    passage_id     TEXT PRIMARY KEY,
    document_id    TEXT NOT NULL REFERENCES documents(document_id),
    kind           TEXT NOT NULL CHECK (kind IN ('edition', 'translation')),
    sequence       INTEGER NOT NULL,
    textpart       TEXT,
    line_reference TEXT,
    display_text   TEXT NOT NULL,
    search_text    TEXT NOT NULL,
    uncertainty    TEXT NOT NULL,
    source_url     TEXT NOT NULL,
    source_commit  TEXT NOT NULL,
    source_path    TEXT NOT NULL,
    locator        TEXT
);
CREATE INDEX passages_by_document ON passages(document_id, sequence);

CREATE VIRTUAL TABLE passages_fts USING fts5(
    search_text,
    title,
    passage_id UNINDEXED
);
"""


class FTS5Unavailable(Exception):
    """The runtime SQLite build does not provide the FTS5 module."""


def derive_document_id(collection: str, source_path: str) -> str:
    """Stable document ID from collection and source-relative path."""
    return f"{collection}:{source_path}"


def derive_passage_id(document_id: str, kind: str, sequence: int, locator: str | None) -> str:
    """Stable passage ID from document, kind, sequence, and structural locator."""
    return f"{document_id}#{kind}:{sequence}:{locator or ''}"


def _ensure_fts5(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(x)")
    connection.execute("DROP TABLE temp.fts5_probe")


class ArtifactWriter:
    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute("PRAGMA foreign_keys = ON")

    def create_schema(self) -> None:
        try:
            _ensure_fts5(self._connection)
        except sqlite3.OperationalError as error:
            raise FTS5Unavailable(
                "The SQLite build in use does not provide FTS5 "
                f"({error}). Use a CPython build with FTS5 enabled."
            ) from error
        self._connection.executescript(_SCHEMA)

    def insert_document(self, record: DocumentRecord) -> None:
        self._connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.document_id,
                record.collection,
                record.title,
                json.dumps(record.languages, ensure_ascii=False, sort_keys=True),
                json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                record.source.repository_url,
                record.source.commit,
                record.source.path,
                record.source.locator,
                record.canonical_url,
            ),
        )

    def insert_passages(self, records: Sequence[PassageRecord]) -> None:
        self._connection.executemany(
            "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    record.passage_id,
                    record.document_id,
                    record.kind,
                    record.sequence,
                    record.textpart,
                    record.line_reference,
                    record.display_text,
                    record.search_text,
                    json.dumps(record.uncertainty, ensure_ascii=False, sort_keys=True),
                    record.source.repository_url,
                    record.source.commit,
                    record.source.path,
                    record.source.locator,
                )
                for record in records
            ],
        )
        for record in records:
            title = self._connection.execute(
                "SELECT title FROM documents WHERE document_id = ?",
                (record.document_id,),
            ).fetchone()
            if title is None:
                raise sqlite3.IntegrityError(
                    f"passage references unknown document {record.document_id}"
                )
            self._connection.execute(
                "INSERT INTO passages_fts (search_text, title, passage_id) VALUES (?, ?, ?)",
                (record.search_text, title[0], record.passage_id),
            )

    def insert_identifiers(self, records: Sequence[IdentifierRecord]) -> None:
        self._connection.executemany(
            "INSERT INTO identifiers VALUES (?, ?, ?)",
            [(record.document_id, record.namespace, record.value) for record in records],
        )

    def commit(self) -> None:
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


class ArtifactReader:
    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.row_factory = sqlite3.Row

    def get_document(self, document_id: str) -> DocumentRecord | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            document_id=row["document_id"],
            collection=row["collection"],
            title=row["title"],
            languages=json.loads(row["languages"]),
            metadata=json.loads(row["metadata"]),
            source=SourceRefFor(row),
            canonical_url=row["canonical_url"],
        )

    def get_passages(self, document_id: str) -> list[PassageRecord]:
        rows = self._connection.execute(
            "SELECT * FROM passages WHERE document_id = ? ORDER BY sequence",
            (document_id,),
        ).fetchall()
        return [
            PassageRecord(
                passage_id=row["passage_id"],
                document_id=row["document_id"],
                kind=row["kind"],
                sequence=row["sequence"],
                textpart=row["textpart"],
                line_reference=row["line_reference"],
                display_text=row["display_text"],
                search_text=row["search_text"],
                uncertainty=json.loads(row["uncertainty"]),
                source=SourceRefFor(row),
            )
            for row in rows
        ]

    def get_identifiers(self, document_id: str) -> list[IdentifierRecord]:
        rows = self._connection.execute(
            "SELECT document_id, namespace, value FROM identifiers"
            " WHERE document_id = ? ORDER BY namespace, value",
            (document_id,),
        ).fetchall()
        return [
            IdentifierRecord(
                document_id=row["document_id"],
                namespace=row["namespace"],
                value=row["value"],
            )
            for row in rows
        ]

    def fts_count(self) -> int:
        row = self._connection.execute("SELECT count(*) FROM passages_fts").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._connection.close()


def SourceRefFor(row: sqlite3.Row):  # noqa: N802 - internal row mapper
    from papyrus_chat.artifact.records import SourceReference

    return SourceReference(
        repository_url=row["source_url"],
        commit=row["source_commit"],
        path=row["source_path"],
    )
