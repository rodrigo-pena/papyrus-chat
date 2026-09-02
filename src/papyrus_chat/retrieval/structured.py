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


def _loads_json_text(text: str) -> object:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _recover_swallowed_members(text: str, *, field: str) -> dict[str, object] | None:
    """Recover argument members a model swallowed into one field's string value.

    The malformed text leads with this field's JSON value and ends with the
    arguments object's closing brace; wrapping the text minus that brace parses
    the field's value together with the members that followed it.
    """
    if not text.rstrip().endswith("}"):
        return None
    try:
        recovered = json.loads('{"' + field + '": ' + text.rstrip()[:-1] + "}")
    except ValueError:
        return None
    return recovered if isinstance(recovered, dict) else None


class CorpusDateInterval(BaseModel):
    """Inclusive numeric bounds used to overlap linked HGV intervals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    not_before: int = Field(
        description="Inclusive lower bound as a proleptic year; BCE years are negative (e.g. -300)."
    )
    not_after: int = Field(
        description="Inclusive upper bound as a proleptic year; BCE years are negative; "
        "must be greater than or equal to not_before."
    )

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_text(cls, value: object) -> object:
        if isinstance(value, str):
            return _loads_json_text(value)
        return value

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> "CorpusDateInterval":
        if self.not_before > self.not_after:
            raise ValueError(
                "not_before must be less than or equal to not_after (BCE years are negative)"
            )
        return self


class CorpusQuery(BaseModel):
    """A bounded query with OR semantics inside groups and AND between groups."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collections: tuple[str, ...] = Field(
        default=(),
        description="Collection identifiers to restrict the search, as reported by "
        "describe_corpus.",
    )
    term_groups: tuple[tuple[str, ...], ...] = Field(
        default=(),
        description="Up to 8 groups (AND between groups) of at most 16 alternative terms each "
        "(OR within a group); each term is at most 200 characters. Terms are prefix-matched "
        "against diacritic-folded word tokens: include inflected variants, since different "
        "stems (e.g. Greek augmented διεσ- vs unaugmented δια-) do not match. A flat list of "
        "strings is not accepted: every group must itself be a list of terms.",
    )
    subject_groups: tuple[tuple[str, ...], ...] = Field(
        default=(),
        description="Up to 8 groups of exact HGV subject labels (OR within a group, AND "
        "between groups). Use semantic suggestions to discover labels before filtering.",
    )
    fields: tuple[CorpusField, ...] = Field(
        default=("transcription", "translation", "title", "metadata"),
        description="Fields to search; a non-empty subset of title, metadata, transcription, "
        "translation.",
    )
    transcription_languages: tuple[str, ...] = Field(
        default=(),
        description="Edition language codes (e.g. grc) that a document's transcription must use.",
    )
    date_interval: CorpusDateInterval | None = Field(
        default=None,
        description="Optional inclusive interval overlapped with linked HGV date ranges; "
        "BCE years are negative and not_before must not exceed not_after. An HGV range "
        "missing one bound (e.g. 'nach 48') counts as its known bound.",
    )
    limit: int = Field(
        default=20, ge=1, le=100, description="Maximum documents returned, 1 to 100."
    )

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_text(cls, value: object) -> object:
        if isinstance(value, str):
            return _loads_json_text(value)
        if isinstance(value, dict):
            repaired: dict[str, object] = dict(value)
            known_fields = set(cls.model_fields)
            for key, item in value.items():
                if not isinstance(item, str):
                    continue
                if recovered := _recover_swallowed_members(item, field=key):
                    for name, member in recovered.items():
                        if name in known_fields:
                            repaired[name] = member
            return repaired
        return value

    @field_validator("collections", "transcription_languages", mode="before")
    @classmethod
    def normalize_values(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            decoded = _loads_json_text(value)
            value = decoded if isinstance(decoded, (list, tuple)) else [decoded]
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
            decoded = _loads_json_text(value)
            value = decoded if isinstance(decoded, (list, tuple)) else [decoded]
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
        if isinstance(value, str):
            decoded = _loads_json_text(value)
            if isinstance(decoded, (list, tuple)):
                value = decoded
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

    @field_validator("subject_groups", mode="before")
    @classmethod
    def normalize_subject_groups(cls, value: object) -> tuple[tuple[str, ...], ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            decoded = _loads_json_text(value)
            if isinstance(decoded, (list, tuple)):
                value = decoded
        if not isinstance(value, (list, tuple)):
            raise ValueError("subject_groups must be a list of subject-label lists")
        if len(value) > 8:
            raise ValueError("at most 8 subject groups are allowed")
        groups: list[tuple[str, ...]] = []
        for raw_group in value:
            if not isinstance(raw_group, (list, tuple)) or not raw_group:
                raise ValueError("each subject group must contain at least one label")
            if len(raw_group) > 32:
                raise ValueError("each subject group may contain at most 32 labels")
            labels: list[str] = []
            for raw_label in raw_group:
                if not isinstance(raw_label, str):
                    raise ValueError("subject labels must be strings")
                label = " ".join(raw_label.split())
                if not label:
                    raise ValueError("subject labels must be non-empty")
                if len(label) > 300:
                    raise ValueError("subject labels may contain at most 300 characters")
                if label.casefold() not in {entry.casefold() for entry in labels}:
                    labels.append(label)
            groups.append(tuple(labels))
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
    group_candidate_counts: tuple[int, ...] | None = None
    """Per-term-group candidate counts when the full conjunction matched nothing.

    Counts each group alone under the same non-term filters, so an empty result
    shows which group eliminated every document.
    """

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


class CorpusDocumentMatch(BaseModel):
    """A corpus document identified by its canonical citation URL."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    collection: str


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
        group_counts = self._group_candidate_counts(normalized) if count == 0 else None
        if normalized.term_groups:
            rows = self._ranked_rows(normalized, where_sql, params)
        else:
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
            group_candidate_counts=group_counts,
        )

    def _group_candidate_counts(self, query: CorpusQuery) -> tuple[int, ...] | None:
        if len(query.term_groups) < 2:
            return None
        counts: list[int] = []
        for group in query.term_groups:
            solo = query.model_copy(update={"term_groups": (group,)})
            where, params = self._where_clause(solo)
            counts.append(
                int(
                    self._connection.execute(
                        f"SELECT count(*) FROM documents d WHERE {' AND '.join(where)}",
                        params,
                    ).fetchone()[0]
                )
            )
        return tuple(counts)

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
        rows = self._facet_rows(field, where_sql, params)
        values = tuple(
            CorpusFacetValue(value=row["value"], count=int(row["count"])) for row in rows
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

    def document_for_citation(self, canonical_url: str) -> CorpusDocumentMatch | None:
        """Resolve a citation URL to its corpus document, if one exists.

        Used by the output validator to distinguish a citation the model
        invented from one that names a real document no tool returned in
        this conversation. The documents table has no index on
        canonical_url, so this scans; it only runs on validation failure.
        """
        row = self._connection.execute(
            "SELECT document_id, title, collection FROM documents WHERE canonical_url = ?",
            (canonical_url,),
        ).fetchone()
        if row is None:
            return None
        return CorpusDocumentMatch(
            document_id=row["document_id"],
            title=row["title"],
            collection=row["collection"],
        )

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
            alternatives = self._term_group_conditions(group, query.fields, params)
            where.append("(" + " OR ".join(alternatives) + ")" if alternatives else "0 = 1")
        for group in query.subject_groups:
            where.append(self._subject_group_condition(group, params))
        if query.transcription_languages:
            placeholders = ", ".join("?" for _ in query.transcription_languages)
            where.append(
                "EXISTS (SELECT 1 FROM passages p "
                "WHERE p.document_id = d.document_id AND p.kind = 'edition' "
                "AND EXISTS (SELECT 1 FROM passage_languages pl "
                "WHERE pl.passage_id = p.passage_id "
                f"AND pl.language IN ({placeholders})))"
            )
            params.extend(query.transcription_languages)
        if query.date_interval is not None:
            # Clamp open-ended ranges ("nach 244 v.Chr.") to their known bound so a
            # terminus post quem alone cannot overlap an unrelated interval.
            start = (
                "CAST(COALESCE(NULLIF(date_row.not_before, ''), "
                "NULLIF(date_row.not_after, ''), NULLIF(date_row.when_value, '')) AS INTEGER)"
            )
            end = (
                "CAST(COALESCE(NULLIF(date_row.not_after, ''), "
                "NULLIF(date_row.not_before, ''), NULLIF(date_row.when_value, '')) AS INTEGER)"
            )
            where.append(
                "EXISTS ("
                "SELECT 1 FROM components ddc "
                "JOIN component_links link "
                "ON link.ddbdp_component_id = ddc.component_id "
                "JOIN dates date_row ON date_row.component_id = link.hgv_component_id "
                "WHERE ddc.document_id = d.document_id AND ddc.kind = 'ddbdp' "
                f"AND {start} <= ? AND {end} >= ?"
                ")"
            )
            params.extend([query.date_interval.not_after, query.date_interval.not_before])
        return where, params

    @staticmethod
    def _subject_group_condition(group: tuple[str, ...], params: list[object]) -> str:
        placeholders = ", ".join("?" for _ in group)
        params.extend([*group, *group])
        return (
            "d.document_id IN ("
            "SELECT owner.document_id FROM components owner "
            "JOIN metadata subject ON subject.component_id = owner.component_id "
            "WHERE owner.document_id IS NOT NULL AND subject.key = 'subject' "
            f"AND subject.value IN ({placeholders}) "
            "UNION "
            "SELECT ddbdp.document_id FROM components ddbdp "
            "JOIN component_links link ON link.ddbdp_component_id = ddbdp.component_id "
            "JOIN metadata subject ON subject.component_id = link.hgv_component_id "
            "WHERE ddbdp.document_id IS NOT NULL AND subject.key = 'subject' "
            f"AND subject.value IN ({placeholders})"
            ")"
        )

    def _term_group_conditions(
        self,
        group: tuple[str, ...],
        fields: tuple[CorpusField, ...],
        params: list[object],
    ) -> list[str]:
        """Match one term group with uncorrelated full-text subqueries.

        Each FTS table is probed once per group with the OR of its terms instead
        of once per document row: FTS5 cannot constrain a MATCH by the outer
        document_id, so a correlated EXISTS re-scans the posting list for every
        row of the documents table.
        """
        fts_queries = tuple(
            dict.fromkeys(fts_query for term in group if (fts_query := build_fts_query(term)))
        )
        if not fts_queries:
            return []
        conditions: list[str] = []
        document_columns = [
            column
            for field, column in (("title", "title"), ("metadata", "metadata"))
            if field in fields
        ]
        if document_columns:
            match_query = " OR ".join(
                f"({_column_fts_query(column, fts_query)})"
                for fts_query in fts_queries
                for column in document_columns
            )
            conditions.append(
                "d.document_id IN (SELECT df.document_id FROM documents_fts df "
                "WHERE documents_fts MATCH ?)"
            )
            params.append(match_query)
        for kind, field in (("edition", "transcription"), ("translation", "translation")):
            if field not in fields:
                continue
            match_query = " OR ".join(
                f"({_column_fts_query('search_text', fts_query)})" for fts_query in fts_queries
            )
            conditions.append(
                "d.document_id IN (SELECT p.document_id FROM passages_fts pf "
                "JOIN passages p ON p.passage_id = pf.passage_id "
                "WHERE p.kind = ? AND passages_fts MATCH ?)"
            )
            params.extend([kind, match_query])
        return conditions

    def _ranked_rows(
        self, query: CorpusQuery, where_sql: str, params: list[object]
    ) -> list[sqlite3.Row]:
        fts_queries = tuple(
            dict.fromkeys(
                fts_query
                for group in query.term_groups
                for term in group
                if (fts_query := build_fts_query(term))
            )
        )
        document_alternatives = [
            _column_fts_query(column, fts_query)
            for fts_query in fts_queries
            for field, column in (("title", "title"), ("metadata", "metadata"))
            if field in query.fields
        ]
        passage_query = " OR ".join(
            f"({_column_fts_query('search_text', fts_query)})" for fts_query in fts_queries
        )
        passage_kinds = [
            kind
            for kind, field in (("edition", "transcription"), ("translation", "translation"))
            if field in query.fields
        ]

        ranking_params: list[object] = [*params]
        if document_alternatives:
            document_scores = (
                "SELECT document_id, bm25(documents_fts, 0.0, 5.0, 2.0) AS score "
                "FROM documents_fts WHERE documents_fts MATCH ?"
            )
            ranking_params.append(" OR ".join(f"({item})" for item in document_alternatives))
        else:
            document_scores = "SELECT NULL AS document_id, 0.0 AS score WHERE 0"

        if passage_kinds and passage_query:
            placeholders = ", ".join("?" for _ in passage_kinds)
            passage_matches = (
                "SELECT p.document_id, "
                "(bm25(passages_fts, 1.0, 0.0, 0.0) * 10.0) AS score "
                "FROM passages_fts JOIN passages p ON p.passage_id = passages_fts.passage_id "
                f"WHERE p.kind IN ({placeholders}) AND passages_fts MATCH ?"
            )
            ranking_params.extend([*passage_kinds, passage_query])
        else:
            passage_matches = "SELECT NULL AS document_id, 0.0 AS score WHERE 0"

        ranking_params.append(query.limit)
        return self._connection.execute(
            f"WITH candidates AS (SELECT d.* FROM documents d WHERE {where_sql}), "
            f"document_scores AS MATERIALIZED ({document_scores}), "
            f"passage_matches AS MATERIALIZED ({passage_matches}), "
            "passage_scores AS ("
            "SELECT document_id, sum(score) AS score FROM passage_matches GROUP BY document_id"
            ") SELECT candidates.* FROM candidates "
            "LEFT JOIN document_scores ds ON ds.document_id = candidates.document_id "
            "LEFT JOIN passage_scores ps ON ps.document_id = candidates.document_id "
            "ORDER BY (coalesce(ds.score, 0.0) + coalesce(ps.score, 0.0)) ASC, "
            "candidates.collection, candidates.document_id LIMIT ?",
            ranking_params,
        ).fetchall()

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
        fts_queries = tuple(
            dict.fromkeys(
                fts_query
                for group in query.term_groups
                for term in group
                if (fts_query := build_fts_query(term))
            )
        )
        if not fts_queries:
            return None
        match_query = " OR ".join(
            f"({_column_fts_query('search_text', fts_query)})" for fts_query in fts_queries
        )
        placeholders = ", ".join("?" for _ in kinds)
        return self._connection.execute(
            "SELECT p.*, pl.language FROM passages p "
            "LEFT JOIN passage_languages pl ON pl.passage_id = p.passage_id "
            f"WHERE p.document_id = ? AND p.kind IN ({placeholders}) "
            "AND p.passage_id IN (SELECT pf.passage_id FROM passages_fts pf "
            "WHERE passages_fts MATCH ?) "
            "ORDER BY p.sequence LIMIT 1",
            [document_id, *kinds, match_query],
        ).fetchone()

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

    def _facet_rows(
        self, field: FacetField, where_sql: str, params: list[object]
    ) -> list[sqlite3.Row]:
        # MATERIALIZED: component_owners references filtered_documents in both UNION
        # legs, and without it each leg re-evaluates the whole filter.
        filtered = (
            f"WITH filtered_documents AS MATERIALIZED "
            f"(SELECT d.* FROM documents d WHERE {where_sql}) "
        )
        if field == "collection":
            rows = self._connection.execute(
                filtered + "SELECT collection AS value, count(*) AS count FROM filtered_documents "
                "GROUP BY collection ORDER BY count DESC, value ASC",
                params,
            ).fetchall()
        elif field == "language":
            rows = self._connection.execute(
                filtered + "SELECT pl.language AS value, count(DISTINCT fd.document_id) AS count "
                "FROM filtered_documents fd "
                "JOIN passages p ON p.document_id = fd.document_id "
                "JOIN passage_languages pl ON pl.passage_id = p.passage_id "
                "WHERE p.kind = 'edition' GROUP BY pl.language "
                "ORDER BY count DESC, value ASC",
                params,
            ).fetchall()
        elif field == "kind":
            rows = self._connection.execute(
                filtered + "SELECT p.kind AS value, count(DISTINCT fd.document_id) AS count "
                "FROM filtered_documents fd "
                "JOIN passages p ON p.document_id = fd.document_id GROUP BY p.kind "
                "ORDER BY count DESC, value ASC",
                params,
            ).fetchall()
        else:
            rows = self._connection.execute(
                filtered + ", component_owners AS ("
                "SELECT fd.document_id, c.component_id FROM filtered_documents fd "
                "JOIN components c ON c.document_id = fd.document_id "
                "UNION SELECT fd.document_id, l.hgv_component_id FROM filtered_documents fd "
                "JOIN components d ON d.document_id = fd.document_id "
                "JOIN component_links l ON l.ddbdp_component_id = d.component_id"
                ") SELECT m.value AS value, count(DISTINCT owners.document_id) AS count "
                "FROM component_owners owners "
                "JOIN metadata m ON m.component_id = owners.component_id "
                "WHERE m.key = ? GROUP BY m.value ORDER BY count DESC, value ASC",
                [*params, field],
            ).fetchall()
        return rows


def _column_fts_query(column: str, query: str) -> str:
    return f"{column} : ({query})"
