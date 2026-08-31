"""Corpus search: identifier lookup first, then FTS5 (SPEC 8).

Ranking uses a fixed, explicit BM25 configuration over the FTS5 table's two
indexed columns (search text, title): passage text is weighted 10x over
titles. Unicode61 tokenization plus shared normalization (see textnorm)
makes Greek matches case- and diacritic-insensitive. Results are ordered
deterministically: score, then collection, document ID, and sequence.
"""

import re
import sqlite3
from pathlib import Path

from papyrus_chat.artifact.records import DocumentRecord
from papyrus_chat.retrieval.evidence import (
    EvidenceItem,
    EvidenceKind,
    EvidencePacket,
    snippet_for,
)
from papyrus_chat.retrieval.identifiers import IdentifierLookup
from papyrus_chat.textnorm import normalize_identifier_value, normalize_search_text

# Explicit FTS5/BM25 configuration (SPEC 8). Column order matches the
# passages_fts definition: (search_text, title).
BM25_WEIGHTS = (10.0, 1.0)

_FTS_SPECIALS = re.compile(r'[()":*^{}[\]\-]')


def build_fts_query(user_query: str) -> str:
    """Convert user text into a safe FTS5 query of quoted prefix tokens."""
    tokens = _FTS_SPECIALS.sub(" ", user_query).split()
    normalized = [normalize_search_text(token) for token in tokens]
    return " ".join(f'"{token}"*' for token in normalized if token)


class SearchFilters:
    def __init__(
        self,
        collection: str | None = None,
        kind: str | None = None,
        document_id: str | None = None,
    ) -> None:
        self.collection = collection
        self.kind = kind
        self.document_id = document_id


class CorpusSearch:
    BM25_WEIGHTS = BM25_WEIGHTS

    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row
        self._identifiers = IdentifierLookup(database_path)
        self._weights = BM25_WEIGHTS

    def search(
        self, query: str, filters: SearchFilters | None = None, limit: int = 20
    ) -> EvidencePacket:
        filters = filters or SearchFilters()

        if filters.collection is None and filters.kind is None and filters.document_id is None:
            identifier_hits = self._identifiers.lookup(query)
            if identifier_hits:
                return EvidencePacket(
                    query=query,
                    strategy="identifier",
                    items=tuple(self._metadata_items(identifier_hits)),
                )

        items = self._full_text(query, filters, limit)
        return EvidencePacket(query=query, strategy="full-text", items=tuple(items))

    def _full_text(self, query: str, filters: SearchFilters, limit: int) -> list[EvidenceItem]:
        fts_query = build_fts_query(query)
        if not fts_query:
            return []

        sql = (
            "SELECT p.passage_id, p.kind, p.sequence, p.display_text,"
            " p.source_commit, p.source_path, p.locator,"
            " d.document_id, d.title, d.collection,"
            " bm25(passages_fts, ?, ?) AS score"
            " FROM passages_fts f"
            " JOIN passages p ON p.passage_id = f.passage_id"
            " JOIN documents d ON d.document_id = p.document_id"
            " WHERE passages_fts MATCH ?"
        )
        params: list[object] = [*self._weights, fts_query]

        if filters.collection:
            sql += " AND d.collection = ?"
            params.append(filters.collection)
        if filters.kind:
            sql += " AND p.kind = ?"
            params.append(filters.kind)
        if filters.document_id:
            sql += " AND p.document_id = ?"
            params.append(filters.document_id)

        sql += " ORDER BY score, d.collection, d.document_id, p.sequence LIMIT ?"
        params.append(limit)

        rows = self._connection.execute(sql, params).fetchall()
        items = [self._passage_item(row) for row in rows]

        if filters.collection is None and filters.kind is None:
            items.extend(self._metadata_only_matches(query, limit))

        return items

    def _metadata_only_matches(self, query: str, limit: int) -> list[EvidenceItem]:
        """Find textless documents by title/metadata substring (SPEC 8)."""
        like = f"%{normalize_identifier_value(query)}%"
        rows = self._connection.execute(
            "SELECT d.* FROM documents d"
            " WHERE (d.title LIKE ? OR d.metadata LIKE ?)"
            "   AND NOT EXISTS (SELECT 1 FROM passages p WHERE p.document_id = d.document_id)"
            " ORDER BY d.collection, d.document_id LIMIT ?",
            (like, like, limit),
        ).fetchall()

        seen = {item.document_id for item in []}
        return [item for item in self._metadata_items_map(rows) if item.document_id not in seen]

    def _passage_item(self, row: sqlite3.Row) -> EvidenceItem:
        kind: EvidenceKind = row["kind"]
        locator = row["locator"] if row["locator"] != row["kind"] else None
        citation = (
            f"{row['collection']}:{self._first_identifier(row['document_id'])} "
            f"({row['title']}), {row['kind']}, {row['source_path']}"
            + (f", {locator}" if locator else "")
        )
        return EvidenceItem(
            document_id=row["document_id"],
            title=row["title"],
            collection=row["collection"],
            passage_id=row["passage_id"],
            kind=kind,
            display_text=row["display_text"],
            snippet=snippet_for(row["display_text"]),
            commit=row["source_commit"],
            source_path=row["source_path"],
            locator=row["locator"],
            citation_label=citation,
        )

    def _metadata_items(self, documents: list[DocumentRecord]) -> list[EvidenceItem]:
        return [
            EvidenceItem(
                document_id=doc.document_id,
                title=doc.title,
                collection=doc.collection,
                commit=doc.source.commit,
                source_path=doc.source.path,
                locator=doc.source.locator,
                citation_label=(
                    f"{doc.collection}:{self._first_identifier(doc.document_id)} ({doc.title})"
                ),
            )
            for doc in documents
        ]

    def _metadata_items_map(self, rows: list[sqlite3.Row]) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for row in rows:
            citation = (
                f"{row['collection']}:{self._first_identifier(row['document_id'])} "
                f"({row['title']}), metadata only"
            )
            items.append(
                EvidenceItem(
                    document_id=row["document_id"],
                    title=row["title"],
                    collection=row["collection"],
                    kind=None,
                    display_text=None,
                    snippet=snippet_for(row["metadata"] or "", 200),
                    commit=row["source_commit"],
                    source_path=row["source_path"],
                    locator=row["locator"],
                    citation_label=citation,
                )
            )
        return items

    def _first_identifier(self, document_id: str) -> str:
        row = self._connection.execute(
            "SELECT namespace, value FROM identifiers WHERE document_id = ?"
            " ORDER BY CASE WHEN LOWER(namespace) = 'tm' THEN 0 ELSE 1 END,"
            " namespace, value LIMIT 1",
            (document_id,),
        ).fetchone()
        if row is None:
            return document_id
        preferred = normalize_identifier_value(row["namespace"])
        return f"{row['namespace']} {row['value']}" if preferred else row["value"]
