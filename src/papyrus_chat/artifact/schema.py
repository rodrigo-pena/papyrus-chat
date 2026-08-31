"""SQLite logical schema for corpus artifacts.

Schema version 2 adds source components and their linked metadata while
retaining the document, identifier, and passage tables used by retrieval.
Stable IDs derive from collection, source identity (path), and structural
location — never from insertion order.
"""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from papyrus_chat.artifact.records import (
    ComponentDateRecord,
    ComponentIdentifierRecord,
    ComponentLinkRecord,
    ComponentRecord,
    DocumentRecord,
    IdentifierRecord,
    PassageRecord,
)
from papyrus_chat.textnorm import normalize_identifier_value

SCHEMA_VERSION = 2

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
    document_id    TEXT NOT NULL REFERENCES documents(document_id),
    namespace      TEXT NOT NULL,
    value          TEXT NOT NULL,
    namespace_norm TEXT NOT NULL,
    value_norm     TEXT NOT NULL,
    PRIMARY KEY (document_id, namespace, value)
);
CREATE INDEX identifiers_lookup ON identifiers(namespace, value);
CREATE INDEX identifiers_norm_lookup ON identifiers(namespace_norm, value_norm);

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

CREATE TABLE passage_languages (
    passage_id TEXT PRIMARY KEY REFERENCES passages(passage_id),
    language   TEXT NOT NULL
);
CREATE INDEX passage_languages_lookup ON passage_languages(language, passage_id);

CREATE VIRTUAL TABLE passages_fts USING fts5(
    search_text,
    title,
    passage_id UNINDEXED
);

CREATE TABLE components (
    component_id  TEXT PRIMARY KEY,
    document_id   TEXT REFERENCES documents(document_id),
    kind          TEXT NOT NULL,
    title         TEXT NOT NULL,
    canonical_url TEXT,
    source_url    TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    locator       TEXT
);
CREATE INDEX components_by_document ON components(document_id, kind);

CREATE TABLE component_identifiers (
    component_id   TEXT NOT NULL REFERENCES components(component_id),
    namespace      TEXT NOT NULL,
    value          TEXT NOT NULL,
    namespace_norm TEXT NOT NULL,
    value_norm     TEXT NOT NULL,
    source_url     TEXT NOT NULL,
    source_commit  TEXT NOT NULL,
    source_path    TEXT NOT NULL,
    locator        TEXT,
    PRIMARY KEY (component_id, namespace, value)
);
CREATE INDEX component_identifiers_lookup
    ON component_identifiers(namespace, value);
CREATE INDEX component_identifiers_norm_lookup
    ON component_identifiers(namespace_norm, value_norm);

CREATE TABLE metadata (
    component_id  TEXT NOT NULL REFERENCES components(component_id),
    key           TEXT NOT NULL,
    value         TEXT NOT NULL,
    source_url    TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    locator       TEXT,
    PRIMARY KEY (component_id, key, value)
);
CREATE INDEX metadata_lookup ON metadata(key, value);

CREATE TABLE dates (
    component_id  TEXT NOT NULL REFERENCES components(component_id),
    sequence      INTEGER NOT NULL,
    not_before    TEXT,
    not_after     TEXT,
    when_value    TEXT,
    text          TEXT,
    source_url    TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    locator       TEXT,
    PRIMARY KEY (component_id, sequence)
);
CREATE INDEX dates_by_bounds ON dates(not_before, not_after);

CREATE TABLE languages (
    component_id  TEXT NOT NULL REFERENCES components(component_id),
    language      TEXT NOT NULL,
    role          TEXT NOT NULL,
    source_url    TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    locator       TEXT,
    PRIMARY KEY (component_id, language, role)
);
CREATE INDEX languages_lookup ON languages(language, role);

CREATE TABLE component_links (
    ddbdp_component_id TEXT NOT NULL REFERENCES components(component_id),
    hgv_component_id   TEXT NOT NULL REFERENCES components(component_id),
    PRIMARY KEY (ddbdp_component_id, hgv_component_id)
);
CREATE INDEX component_links_hgv ON component_links(hgv_component_id);

CREATE VIRTUAL TABLE documents_fts USING fts5(
    document_id UNINDEXED,
    title,
    metadata
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
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
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
        self._refresh_document_fts(record.document_id)

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
        self._connection.executemany(
            "INSERT INTO passage_languages VALUES (?, ?)",
            [(record.passage_id, record.language) for record in records if record.language],
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
            "INSERT INTO identifiers VALUES (?, ?, ?, ?, ?)",
            [
                (
                    record.document_id,
                    record.namespace,
                    record.value,
                    normalize_identifier_value(record.namespace),
                    normalize_identifier_value(record.value),
                )
                for record in records
            ],
        )

    def insert_components(
        self,
        records: Sequence[ComponentRecord],
        links: Sequence[ComponentLinkRecord] = (),
    ) -> None:
        """Persist components and all source-attributed child records."""
        self._connection.executemany(
            "INSERT INTO components VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    record.component_id,
                    record.document_id,
                    record.kind,
                    record.title,
                    record.canonical_url,
                    record.source.repository_url,
                    record.source.commit,
                    record.source.path,
                    record.source.locator,
                )
                for record in records
            ],
        )
        self._connection.executemany(
            "INSERT INTO component_identifiers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    identifier.component_id,
                    identifier.namespace,
                    identifier.value,
                    normalize_identifier_value(identifier.namespace),
                    normalize_identifier_value(identifier.value),
                    record.source.repository_url,
                    record.source.commit,
                    record.source.path,
                    record.source.locator,
                )
                for record in records
                for identifier in record.identifiers
            ],
        )
        self._connection.executemany(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    record.component_id,
                    key,
                    value,
                    record.source.repository_url,
                    record.source.commit,
                    record.source.path,
                    record.source.locator,
                )
                for record in records
                for key, values in sorted(record.metadata.items())
                for value in values
            ],
        )
        self._connection.executemany(
            "INSERT INTO dates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    date.component_id,
                    date.sequence,
                    date.not_before,
                    date.not_after,
                    date.when,
                    date.text,
                    record.source.repository_url,
                    record.source.commit,
                    record.source.path,
                    record.source.locator,
                )
                for record in records
                for date in record.dates
            ],
        )
        self._connection.executemany(
            "INSERT INTO languages VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    record.component_id,
                    language,
                    "edition",
                    record.source.repository_url,
                    record.source.commit,
                    record.source.path,
                    record.source.locator,
                )
                for record in records
                for language in record.languages
            ],
        )
        self._connection.executemany(
            "INSERT INTO component_links VALUES (?, ?)",
            [(link.ddbdp_component_id, link.hgv_component_id) for link in links],
        )
        for document_id in self._document_ids():
            self._refresh_document_fts(document_id)

    def _document_ids(self) -> list[str]:
        return [row[0] for row in self._connection.execute("SELECT document_id FROM documents")]

    def _refresh_document_fts(self, document_id: str) -> None:
        document = self._connection.execute(
            "SELECT title, metadata FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if document is None:
            return
        component_rows = self._connection.execute(
            "SELECT title FROM components WHERE document_id = ? "
            "UNION SELECT h.title FROM component_links l "
            "JOIN components d ON d.component_id = l.ddbdp_component_id "
            "JOIN components h ON h.component_id = l.hgv_component_id "
            "WHERE d.document_id = ? ORDER BY title",
            (document_id, document_id),
        ).fetchall()
        title = " ".join([document[0], *(row[0] for row in component_rows)])
        metadata = document[1]
        component_metadata = self._connection.execute(
            "SELECT value FROM metadata WHERE component_id IN ("
            "SELECT component_id FROM components WHERE document_id = ? "
            "UNION SELECT l.hgv_component_id FROM component_links l "
            "JOIN components d ON d.component_id = l.ddbdp_component_id "
            "WHERE d.document_id = ?) "
            "ORDER BY component_id, key, value",
            (document_id, document_id),
        ).fetchall()
        if component_metadata:
            metadata += " " + " ".join(row[0] for row in component_metadata)
        self._connection.execute("DELETE FROM documents_fts WHERE document_id = ?", (document_id,))
        self._connection.execute(
            "INSERT INTO documents_fts (document_id, title, metadata) VALUES (?, ?, ?)",
            (document_id, title, metadata),
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
            "SELECT p.*, pl.language FROM passages p "
            "LEFT JOIN passage_languages pl ON pl.passage_id = p.passage_id "
            "WHERE p.document_id = ? ORDER BY p.sequence",
            (document_id,),
        ).fetchall()
        return [
            PassageRecord(
                passage_id=row["passage_id"],
                document_id=row["document_id"],
                kind=row["kind"],
                language=row["language"],
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

    def get_components(self, document_id: str) -> list[ComponentRecord]:
        rows = self._connection.execute(
            "SELECT * FROM components WHERE document_id = ? "
            "OR component_id IN ("
            "SELECT l.hgv_component_id FROM component_links l "
            "JOIN components d ON d.component_id = l.ddbdp_component_id "
            "WHERE d.document_id = ?) ORDER BY component_id",
            (document_id, document_id),
        ).fetchall()
        components: list[ComponentRecord] = []
        for row in rows:
            identifiers = self._connection.execute(
                "SELECT component_id, namespace, value FROM component_identifiers "
                "WHERE component_id = ? ORDER BY namespace, value",
                (row["component_id"],),
            ).fetchall()
            metadata_rows = self._connection.execute(
                "SELECT key, value FROM metadata WHERE component_id = ? ORDER BY key, value",
                (row["component_id"],),
            ).fetchall()
            metadata: dict[str, list[str]] = {}
            for metadata_row in metadata_rows:
                metadata.setdefault(metadata_row["key"], []).append(metadata_row["value"])
            date_rows = self._connection.execute(
                "SELECT * FROM dates WHERE component_id = ? ORDER BY sequence",
                (row["component_id"],),
            ).fetchall()
            language_rows = self._connection.execute(
                "SELECT language FROM languages WHERE component_id = ? ORDER BY language",
                (row["component_id"],),
            ).fetchall()
            components.append(
                ComponentRecord(
                    component_id=row["component_id"],
                    document_id=row["document_id"],
                    kind=row["kind"],
                    title=row["title"],
                    languages=tuple(language_row["language"] for language_row in language_rows),
                    metadata={key: tuple(values) for key, values in metadata.items()},
                    dates=tuple(
                        ComponentDateRecord(
                            component_id=date_row["component_id"],
                            sequence=date_row["sequence"],
                            not_before=date_row["not_before"],
                            not_after=date_row["not_after"],
                            when=date_row["when_value"],
                            text=date_row["text"],
                        )
                        for date_row in date_rows
                    ),
                    identifiers=tuple(
                        ComponentIdentifierRecord(
                            component_id=identifier_row["component_id"],
                            namespace=identifier_row["namespace"],
                            value=identifier_row["value"],
                        )
                        for identifier_row in identifiers
                    ),
                    source=SourceRefFor(row),
                    canonical_url=row["canonical_url"],
                )
            )
        return components

    def get_component_links(self, document_id: str) -> list[ComponentLinkRecord]:
        rows = self._connection.execute(
            "SELECT l.ddbdp_component_id, l.hgv_component_id FROM component_links l "
            "JOIN components c ON c.component_id = l.ddbdp_component_id "
            "WHERE c.document_id = ? ORDER BY l.ddbdp_component_id, l.hgv_component_id",
            (document_id,),
        ).fetchall()
        return [
            ComponentLinkRecord(
                ddbdp_component_id=row["ddbdp_component_id"],
                hgv_component_id=row["hgv_component_id"],
            )
            for row in rows
        ]

    def fts_count(self) -> int:
        row = self._connection.execute("SELECT count(*) FROM passages_fts").fetchone()
        return int(row[0])

    def document_fts_count(self) -> int:
        row = self._connection.execute("SELECT count(*) FROM documents_fts").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._connection.close()


def SourceRefFor(row: sqlite3.Row):  # noqa: N802 - internal row mapper
    from papyrus_chat.artifact.records import SourceReference

    return SourceReference(
        repository_url=row["source_url"],
        commit=row["source_commit"],
        path=row["source_path"],
        locator=row["locator"],
    )
