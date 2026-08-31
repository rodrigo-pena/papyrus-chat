"""Structured, distinct-document retrieval over a corpus artifact."""

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from papyrus_chat.artifact.records import (
    ComponentDateRecord,
    ComponentIdentifierRecord,
    ComponentRecord,
    SourceReference,
)
from papyrus_chat.retrieval.evidence import snippet_for
from papyrus_chat.retrieval.search import build_fts_query

CorpusField = Literal["title", "metadata", "transcription", "translation"]
FacetField = Literal["collection", "language", "subject", "material", "origin", "kind"]


class CorpusDateInterval(BaseModel):
    """Inclusive numeric bounds used to overlap linked HGV intervals."""

    model_config = ConfigDict(frozen=True)

    not_before: int
    not_after: int

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> "CorpusDateInterval":
        if self.not_before > self.not_after:
            raise ValueError("not_before must be less than or equal to not_after")
        return self


class CorpusQuery(BaseModel):
    """A bounded query with OR semantics inside groups and AND between groups."""

    model_config = ConfigDict(frozen=True)

    collections: tuple[str, ...] = ()
    term_groups: tuple[tuple[str, ...], ...] = ()
    fields: tuple[CorpusField, ...] = (
        "transcription",
        "translation",
        "title",
        "metadata",
    )
    transcription_languages: tuple[str, ...] = ()
    date_interval: CorpusDateInterval | None = None
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("collections", "transcription_languages", mode="before")
    @classmethod
    def normalize_values(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("expected a list of strings")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("values must be non-empty strings")
            cleaned = " ".join(item.split())
            if cleaned.casefold() not in {entry.casefold() for entry in normalized}:
                normalized.append(cleaned.casefold())
        return tuple(sorted(normalized))

    @field_validator("fields", mode="before")
    @classmethod
    def normalize_fields(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("fields must contain at least one field")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("fields must contain non-empty strings")
            cleaned = item.strip().casefold()
            if cleaned not in normalized:
                normalized.append(cleaned)
        return tuple(normalized)

    @field_validator("term_groups", mode="before")
    @classmethod
    def normalize_term_groups(cls, value: object) -> tuple[tuple[str, ...], ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("term_groups must be a list of term lists")
        if len(value) > 8:
            raise ValueError("at most 8 term groups are allowed")
        groups: list[tuple[str, ...]] = []
        for raw_group in value:
            if not isinstance(raw_group, (list, tuple)) or not raw_group:
                raise ValueError("each term group must contain at least one term")
            if len(raw_group) > 16:
                raise ValueError("each term group may contain at most 16 terms")
            terms: list[str] = []
            for raw_term in raw_group:
                if not isinstance(raw_term, str):
                    raise ValueError("terms must be strings")
                term = " ".join(raw_term.split())
                if not term:
                    raise ValueError("terms must be non-empty")
                if len(term) > 200:
                    raise ValueError("terms may contain at most 200 characters")
                if term.casefold() not in {entry.casefold() for entry in terms}:
                    terms.append(term)
            groups.append(tuple(terms))
        return tuple(groups)


class CorpusHit(BaseModel):
    """One distinct candidate document with an optional located passage."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    collection: str
    languages: tuple[str, ...]
    metadata: dict[str, str]
    passage_id: str | None = None
    passage_kind: Literal["edition", "translation"] | None = None
    passage_language: str | None = None
    passage_text: str | None = None
    snippet: str | None = None
    line_reference: str | None = None
    components: tuple[ComponentRecord, ...] = ()
    source: SourceReference
    canonical_url: str | None = None


class CorpusSearchResult(BaseModel):
    """Complete normalized query, exact candidate count, and bounded hits."""

    model_config = ConfigDict(frozen=True)

    query: CorpusQuery
    assumptions: tuple[str, ...] = ()
    candidate_count: int
    truncated: bool
    hits: tuple[CorpusHit, ...]

    @property
    def normalized_query(self) -> CorpusQuery:
        """Alias useful to callers that name the returned query explicitly."""
        return self.query


class CorpusDescription(BaseModel):
    """Small inventory summary safe to expose to an agent."""

    model_config = ConfigDict(frozen=True)

    collections: tuple[str, ...]
    documents: int
    passages: int
    components: int
    languages: tuple[str, ...]


class CorpusInspection(BaseModel):
    """A selected document and a bounded set of located passages."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    collection: str
    languages: tuple[str, ...]
    metadata: dict[str, str]
    source: SourceReference
    canonical_url: str | None
    components: tuple[ComponentRecord, ...] = ()
    passages: tuple[CorpusHit, ...]


class CorpusFacetValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str
    count: int


class CorpusFacetResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: CorpusQuery
    field: FacetField
    values: tuple[CorpusFacetValue, ...]


class StructuredCorpusSearch:
    """Read-only structured query service for an artifact SQLite database."""

    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row

    def query(
        self,
        query: CorpusQuery | dict[str, object],
        *,
        assumptions: Iterable[str] = (),
    ) -> CorpusSearchResult:
        normalized = CorpusQuery.model_validate(query)
        where, params = self._where_clause(normalized)
        where_sql = " AND ".join(where)
        count = int(
            self._connection.execute(
                f"SELECT count(*) FROM documents d WHERE {where_sql}", params
            ).fetchone()[0]
        )
        rows = self._connection.execute(
            f"SELECT d.* FROM documents d WHERE {where_sql} "
            "ORDER BY d.collection, d.document_id LIMIT ?",
            [*params, normalized.limit],
        ).fetchall()
        components = self._components_by_document(row["document_id"] for row in rows)
        hits = tuple(
            self._hit(
                row,
                normalized,
                components=components.get(row["document_id"], ()),
            )
            for row in rows
        )
        return CorpusSearchResult(
            query=normalized,
            assumptions=tuple(assumptions),
            candidate_count=count,
            truncated=count > normalized.limit,
            hits=hits,
        )

    def facet_documents(
        self,
        query: CorpusQuery | dict[str, object],
        field: FacetField,
    ) -> CorpusFacetResult:
        if field not in {"collection", "language", "subject", "material", "origin", "kind"}:
            raise ValueError(f"Unsupported facet field: {field}")
        normalized = CorpusQuery.model_validate(query)
        where, params = self._where_clause(normalized)
        where_sql = " AND ".join(where)
        rows = self._connection.execute(
            f"SELECT d.document_id FROM documents d WHERE {where_sql}", params
        ).fetchall()
        document_ids = [row["document_id"] for row in rows]
        counts = self._facet_counts(field, document_ids)
        values = tuple(
            CorpusFacetValue(value=value, count=count)
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )
        return CorpusFacetResult(query=normalized, field=field, values=values)

    def describe(self) -> CorpusDescription:
        collections = tuple(
            row["collection"]
            for row in self._connection.execute(
                "SELECT DISTINCT collection FROM documents ORDER BY collection"
            )
        )
        languages = {
            row["language"]
            for row in self._connection.execute(
                "SELECT DISTINCT pl.language FROM passage_languages pl "
                "JOIN passages p ON p.passage_id = pl.passage_id "
                "WHERE p.kind = 'edition'"
            )
        }
        return CorpusDescription(
            collections=collections,
            documents=int(self._connection.execute("SELECT count(*) FROM documents").fetchone()[0]),
            passages=int(self._connection.execute("SELECT count(*) FROM passages").fetchone()[0]),
            components=int(
                self._connection.execute("SELECT count(*) FROM components").fetchone()[0]
            ),
            languages=tuple(sorted(languages)),
        )

    def inspect_documents(
        self,
        document_ids: Iterable[str],
        *,
        excerpt_limit: int = 3,
    ) -> tuple[CorpusInspection, ...]:
        ids = tuple(dict.fromkeys(document_ids))
        if len(ids) > 20:
            raise ValueError("at most 20 documents may be inspected")
        if not 1 <= excerpt_limit <= 10:
            raise ValueError("excerpt_limit must be between 1 and 10")
        if not ids:
            return ()
        placeholders = ", ".join("?" for _ in ids)
        rows = self._connection.execute(
            f"SELECT * FROM documents WHERE document_id IN ({placeholders})", ids
        ).fetchall()
        by_id = {row["document_id"]: row for row in rows}
        components = self._components_by_document(by_id)
        query = CorpusQuery(fields=("transcription", "translation"))
        inspections: list[CorpusInspection] = []
        for document_id in ids:
            row = by_id.get(document_id)
            if row is None:
                continue
            passages = self._connection.execute(
                "SELECT p.*, pl.language FROM passages p "
                "LEFT JOIN passage_languages pl ON pl.passage_id = p.passage_id "
                "WHERE p.document_id = ? ORDER BY p.sequence LIMIT ?",
                (document_id, excerpt_limit),
            ).fetchall()
            source = SourceReference(
                repository_url=row["source_url"],
                commit=row["source_commit"],
                path=row["source_path"],
                locator=row["locator"],
            )
            inspections.append(
                CorpusInspection(
                    document_id=document_id,
                    title=row["title"],
                    collection=row["collection"],
                    languages=tuple(json.loads(row["languages"])),
                    metadata=json.loads(row["metadata"]),
                    source=source,
                    canonical_url=row["canonical_url"],
                    components=components.get(document_id, ()),
                    passages=tuple(
                        self._hit(
                            row,
                            query,
                            passage=passage,
                            components=components.get(document_id, ()),
                        )
                        for passage in passages
                    ),
                )
            )
        return tuple(inspections)

    def close(self) -> None:
        self._connection.close()

    def _where_clause(self, query: CorpusQuery) -> tuple[list[str], list[object]]:
        where = ["1 = 1"]
        params: list[object] = []
        if query.collections:
            placeholders = ", ".join("?" for _ in query.collections)
            where.append(f"d.collection IN ({placeholders})")
            params.extend(query.collections)
        for group in query.term_groups:
            alternatives: list[str] = []
            for term in group:
                alternatives.extend(self._term_conditions(term, query.fields, params))
            where.append("(" + " OR ".join(alternatives) + ")")
        if query.transcription_languages:
            placeholders = ", ".join("?" for _ in query.transcription_languages)
            where.append(
                "EXISTS (SELECT 1 FROM passages p "
                "JOIN passage_languages pl ON pl.passage_id = p.passage_id "
                "WHERE p.document_id = d.document_id AND p.kind = 'edition' "
                f"AND pl.language IN ({placeholders}))"
            )
            params.extend(query.transcription_languages)
        if query.date_interval is not None:
            where.append(
                "EXISTS ("
                "SELECT 1 FROM components ddc "
                "JOIN component_links link "
                "ON link.ddbdp_component_id = ddc.component_id "
                "JOIN dates date_row ON date_row.component_id = link.hgv_component_id "
                "WHERE ddc.document_id = d.document_id AND ddc.kind = 'ddbdp' "
                "AND (NULLIF(date_row.not_before, '') IS NOT NULL "
                "OR NULLIF(date_row.not_after, '') IS NOT NULL "
                "OR NULLIF(date_row.when_value, '') IS NOT NULL) "
                "AND (COALESCE(NULLIF(date_row.not_before, ''), "
                "NULLIF(date_row.when_value, '')) IS NULL "
                "OR CAST(COALESCE(NULLIF(date_row.not_before, ''), "
                "NULLIF(date_row.when_value, '')) AS INTEGER) <= ?) "
                "AND (COALESCE(NULLIF(date_row.not_after, ''), "
                "NULLIF(date_row.when_value, '')) IS NULL "
                "OR CAST(COALESCE(NULLIF(date_row.not_after, ''), "
                "NULLIF(date_row.when_value, '')) AS INTEGER) >= ?)"
                ")"
            )
            params.extend([query.date_interval.not_after, query.date_interval.not_before])
        return where, params

    def _term_conditions(
        self,
        term: str,
        fields: tuple[CorpusField, ...],
        params: list[object],
    ) -> list[str]:
        fts_query = build_fts_query(term)
        if not fts_query:
            return ["0 = 1"]
        conditions: list[str] = []
        if "title" in fields:
            conditions.append(
                "EXISTS (SELECT 1 FROM documents_fts df "
                "WHERE df.document_id = d.document_id AND documents_fts MATCH ?)"
            )
            params.append(_column_fts_query("title", fts_query))
        if "metadata" in fields:
            conditions.append(
                "EXISTS (SELECT 1 FROM documents_fts df "
                "WHERE df.document_id = d.document_id AND documents_fts MATCH ?)"
            )
            params.append(_column_fts_query("metadata", fts_query))
        for kind, field in (("edition", "transcription"), ("translation", "translation")):
            if field not in fields:
                continue
            conditions.append(
                "EXISTS (SELECT 1 FROM passages_fts pf JOIN passages p "
                "ON p.passage_id = pf.passage_id "
                "WHERE p.document_id = d.document_id AND p.kind = ? AND passages_fts MATCH ?)"
            )
            params.extend([kind, _column_fts_query("search_text", fts_query)])
        return conditions or ["0 = 1"]

    def _hit(
        self,
        row: sqlite3.Row,
        query: CorpusQuery,
        *,
        passage: sqlite3.Row | None = None,
        components: tuple[ComponentRecord, ...] = (),
    ) -> CorpusHit:
        if passage is None:
            passage = self._matched_passage(row["document_id"], query)
        if passage is None:
            source = SourceReference(
                repository_url=row["source_url"],
                commit=row["source_commit"],
                path=row["source_path"],
                locator=row["locator"],
            )
            return CorpusHit(
                document_id=row["document_id"],
                title=row["title"],
                collection=row["collection"],
                languages=tuple(json.loads(row["languages"])),
                metadata=json.loads(row["metadata"]),
                components=components,
                source=source,
                canonical_url=row["canonical_url"],
            )
        source = SourceReference(
            repository_url=passage["source_url"],
            commit=passage["source_commit"],
            path=passage["source_path"],
            locator=passage["locator"],
        )
        return CorpusHit(
            document_id=row["document_id"],
            title=row["title"],
            collection=row["collection"],
            languages=tuple(json.loads(row["languages"])),
            metadata=json.loads(row["metadata"]),
            passage_id=passage["passage_id"],
            passage_kind=passage["kind"],
            passage_language=passage["language"],
            passage_text=passage["display_text"],
            snippet=snippet_for(passage["display_text"]),
            line_reference=passage["line_reference"],
            components=components,
            source=source,
            canonical_url=row["canonical_url"],
        )

    def _matched_passage(self, document_id: str, query: CorpusQuery) -> sqlite3.Row | None:
        kinds = [
            kind
            for kind, field in (("edition", "transcription"), ("translation", "translation"))
            if field in query.fields
        ]
        if not kinds:
            return None
        for group in query.term_groups:
            for term in group:
                fts_query = build_fts_query(term)
                if not fts_query:
                    continue
                placeholders = ", ".join("?" for _ in kinds)
                row = self._connection.execute(
                    "SELECT p.*, pl.language FROM passages_fts pf JOIN passages p "
                    "ON p.passage_id = pf.passage_id "
                    "LEFT JOIN passage_languages pl ON pl.passage_id = p.passage_id "
                    f"WHERE p.document_id = ? AND p.kind IN ({placeholders}) "
                    "AND passages_fts MATCH ? "
                    "ORDER BY p.sequence LIMIT 1",
                    [document_id, *kinds, _column_fts_query("search_text", fts_query)],
                ).fetchone()
                if row is not None:
                    return row
        return None

    def _components_by_document(
        self, document_ids: Iterable[str]
    ) -> dict[str, tuple[ComponentRecord, ...]]:
        ids = tuple(dict.fromkeys(document_ids))
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        rows = self._connection.execute(
            "SELECT c.*, c.document_id AS owner_document_id FROM components c "
            f"WHERE c.document_id IN ({placeholders}) "
            "UNION ALL "
            "SELECT h.*, d.document_id AS owner_document_id FROM components h "
            "JOIN component_links l ON l.hgv_component_id = h.component_id "
            "JOIN components d ON d.component_id = l.ddbdp_component_id "
            f"WHERE d.document_id IN ({placeholders}) "
            "ORDER BY owner_document_id, component_id",
            [*ids, *ids],
        ).fetchall()
        component_ids = tuple(dict.fromkeys(row["component_id"] for row in rows))
        if not component_ids:
            return {document_id: () for document_id in ids}
        component_placeholders = ", ".join("?" for _ in component_ids)
        identifier_rows = self._connection.execute(
            "SELECT component_id, namespace, value FROM component_identifiers "
            f"WHERE component_id IN ({component_placeholders}) ORDER BY namespace, value",
            component_ids,
        ).fetchall()
        metadata_rows = self._connection.execute(
            "SELECT component_id, key, value FROM metadata "
            f"WHERE component_id IN ({component_placeholders}) ORDER BY key, value",
            component_ids,
        ).fetchall()
        date_rows = self._connection.execute(
            "SELECT * FROM dates "
            f"WHERE component_id IN ({component_placeholders}) ORDER BY sequence",
            component_ids,
        ).fetchall()
        language_rows = self._connection.execute(
            "SELECT component_id, language FROM languages "
            f"WHERE component_id IN ({component_placeholders}) ORDER BY language",
            component_ids,
        ).fetchall()

        identifiers: dict[str, list[ComponentIdentifierRecord]] = {}
        for child in identifier_rows:
            identifiers.setdefault(child["component_id"], []).append(
                ComponentIdentifierRecord(
                    component_id=child["component_id"],
                    namespace=child["namespace"],
                    value=child["value"],
                )
            )
        metadata: dict[str, dict[str, list[str]]] = {}
        for child in metadata_rows:
            metadata.setdefault(child["component_id"], {}).setdefault(child["key"], []).append(
                child["value"]
            )
        dates: dict[str, list[ComponentDateRecord]] = {}
        for child in date_rows:
            dates.setdefault(child["component_id"], []).append(
                ComponentDateRecord(
                    component_id=child["component_id"],
                    sequence=child["sequence"],
                    not_before=child["not_before"],
                    not_after=child["not_after"],
                    when=child["when_value"],
                    text=child["text"],
                )
            )
        languages: dict[str, list[str]] = {}
        for child in language_rows:
            languages.setdefault(child["component_id"], []).append(child["language"])

        by_document: dict[str, list[ComponentRecord]] = {document_id: [] for document_id in ids}
        for row in rows:
            component_id = row["component_id"]
            by_document[row["owner_document_id"]].append(
                ComponentRecord(
                    component_id=component_id,
                    document_id=row["document_id"],
                    kind=row["kind"],
                    title=row["title"],
                    languages=tuple(languages.get(component_id, ())),
                    metadata={
                        key: tuple(values) for key, values in metadata.get(component_id, {}).items()
                    },
                    dates=tuple(dates.get(component_id, ())),
                    identifiers=tuple(identifiers.get(component_id, ())),
                    source=SourceReference(
                        repository_url=row["source_url"],
                        commit=row["source_commit"],
                        path=row["source_path"],
                        locator=row["locator"],
                    ),
                    canonical_url=row["canonical_url"],
                )
            )
        return {document_id: tuple(values) for document_id, values in by_document.items()}

    def _facet_counts(self, field: FacetField, document_ids: list[str]) -> dict[str, int]:
        if not document_ids:
            return {}
        placeholders = ", ".join("?" for _ in document_ids)
        if field == "collection":
            rows = self._connection.execute(
                "SELECT collection AS value, count(*) AS count FROM documents "
                f"WHERE document_id IN ({placeholders}) GROUP BY collection",
                document_ids,
            ).fetchall()
        elif field == "language":
            rows = self._connection.execute(
                "SELECT lang.language AS value, count(DISTINCT c.document_id) AS count "
                "FROM languages lang JOIN components c ON c.component_id = lang.component_id "
                f"WHERE c.document_id IN ({placeholders}) GROUP BY lang.language",
                document_ids,
            ).fetchall()
        elif field == "kind":
            rows = self._connection.execute(
                "SELECT p.kind AS value, count(DISTINCT p.document_id) AS count "
                "FROM passages p "
                f"WHERE p.document_id IN ({placeholders}) GROUP BY p.kind",
                document_ids,
            ).fetchall()
        else:
            key = field
            rows = self._connection.execute(
                "SELECT value, count(DISTINCT document_id) AS count FROM ("
                "SELECT m.value, c.document_id FROM metadata m "
                "JOIN components c ON c.component_id = m.component_id "
                f"WHERE c.document_id IN ({placeholders}) AND m.key = ? "
                "UNION ALL SELECT m.value, d.document_id FROM metadata m "
                "JOIN component_links l ON l.hgv_component_id = m.component_id "
                "JOIN components d ON d.component_id = l.ddbdp_component_id "
                f"WHERE d.document_id IN ({placeholders}) AND m.key = ?"
                ") GROUP BY value",
                [*document_ids, key, *document_ids, key],
            ).fetchall()
        return {row["value"]: int(row["count"]) for row in rows}


def _column_fts_query(column: str, query: str) -> str:
    return f"{column} : ({query})"
