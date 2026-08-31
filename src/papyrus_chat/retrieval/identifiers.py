"""Normalized exact identifier lookup (SPEC 8, step 1)."""

import sqlite3
from pathlib import Path

from papyrus_chat.artifact.records import DocumentRecord
from papyrus_chat.textnorm import normalize_identifier_query


class IdentifierLookup:
    """Exact identifier lookup over the artifact's normalized identifier index."""

    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row

    def lookup(self, query: str) -> list[DocumentRecord]:
        namespace, value = normalize_identifier_query(query)

        if namespace:
            rows = self._connection.execute(
                "SELECT DISTINCT d.* FROM identifiers i"
                " JOIN documents d ON d.document_id = i.document_id"
                " WHERE i.namespace_norm = ? AND i.value_norm = ?"
                " ORDER BY d.collection, d.document_id",
                (namespace, value),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT DISTINCT d.* FROM identifiers i"
                " JOIN documents d ON d.document_id = i.document_id"
                " WHERE i.value_norm = ?"
                " ORDER BY d.collection, d.document_id",
                (value,),
            ).fetchall()

        return [_document_from_row(row) for row in rows]

    def close(self) -> None:
        self._connection.close()


def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
    import json

    from papyrus_chat.artifact.records import SourceReference

    return DocumentRecord(
        document_id=row["document_id"],
        collection=row["collection"],
        title=row["title"],
        languages=json.loads(row["languages"]),
        metadata=json.loads(row["metadata"]),
        source=SourceReference(
            repository_url=row["source_url"],
            commit=row["source_commit"],
            path=row["source_path"],
            locator=row["locator"],
        ),
        canonical_url=row["canonical_url"],
    )
