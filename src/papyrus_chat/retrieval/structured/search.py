"""Read-only structured query service for an artifact SQLite database."""

import json
import sqlite3
from collections.abc import Callable, Iterable
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from papyrus_chat.artifact.records import (
    ComponentDateRecord,
    ComponentIdentifierRecord,
    ComponentRecord,
    SourceReference,
)
from papyrus_chat.catalog import COLLECTION_NAMES, METADATA_SOURCE_NAMES
from papyrus_chat.retrieval.evidence import snippet_for
from papyrus_chat.retrieval.scope import document_scope_where
from papyrus_chat.retrieval.search import build_fts_query
from papyrus_chat.retrieval.structured.models import (
    CorpusDescription,
    CorpusDocumentMatch,
    CorpusFacetResult,
    CorpusFacetValue,
    CorpusField,
    CorpusHit,
    CorpusInspection,
    CorpusQuery,
    CorpusSearchResult,
    FacetField,
)

if TYPE_CHECKING:
    from papyrus_chat.retrieval.semantic import QueryEncoder


def _serialized(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapper(self: "StructuredCorpusSearch", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class StructuredCorpusSearch:
    """Read-only structured query service for an artifact SQLite database."""

    def __init__(
        self,
        database_path: Path,
        *,
        read_only: bool = False,
        lock: Any = None,
        semantic_encoder: "QueryEncoder | None" = None,
    ) -> None:
        if read_only:
            self._connection = sqlite3.connect(
                f"{database_path.resolve().as_uri()}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
        else:
            self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._database_path = database_path
        self._read_only = read_only
        self._lock = lock or RLock()
        self._semantic = None
        self._discovery = None
        self._semantic_encoder = semantic_encoder
        self._closed = False

    @property
    def semantic(self):
        """Lazily open the optional semantic subject index beside this database."""
        with self._lock:
            if self._semantic is None:
                from papyrus_chat.retrieval.semantic import SemanticSubjectSearch

                self._semantic = SemanticSubjectSearch(
                    self._database_path,
                    read_only=self._read_only,
                    lock=self._lock,
                    encoder=self.embedding_encoder,
                )
            return self._semantic

    @property
    def embedding_encoder(self) -> "QueryEncoder":
        with self._lock:
            if self._semantic_encoder is None:
                from papyrus_chat.artifact.manifest import load_manifest
                from papyrus_chat.semantic.embeddings import EmbeddingModelSpec, LazyLocalEncoder

                semantic = load_manifest(
                    self._database_path.parent / "manifest.json"
                ).semantic_index
                if semantic is None:
                    raise RuntimeError("artifact has no semantic model")
                spec = EmbeddingModelSpec(
                    model_id=semantic.model_id,
                    revision=semantic.revision,
                    dimensions=semantic.dimensions,
                    model_file=semantic.model_file,
                    query_prefix=semantic.query_prefix,
                    passage_prefix=semantic.passage_prefix,
                    pooling=semantic.pooling,
                )
                self._semantic_encoder = LazyLocalEncoder(
                    self._database_path.parent / "semantic/model",
                    model_spec=spec,
                )
            return self._semantic_encoder

    @property
    def discovery(self):
        with self._lock:
            if self._discovery is None:
                from papyrus_chat.artifact.manifest import load_manifest
                from papyrus_chat.retrieval.discovery.search import SemanticDocumentSearch

                semantic = load_manifest(
                    self._database_path.parent / "manifest.json"
                ).semantic_index
                if semantic is None:
                    raise RuntimeError("artifact has no semantic content index")
                self._discovery = SemanticDocumentSearch(
                    self._database_path.parent,
                    self._connection,
                    semantic,
                    self.embedding_encoder,
                )
            return self._discovery

    @_serialized
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
                "ORDER BY d.collection, d.document_id LIMIT ? OFFSET ?",
                [*params, normalized.limit, normalized.offset],
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
            truncated=count > normalized.offset + len(hits),
            offset=normalized.offset,
            next_offset=(normalized.offset + len(hits))
            if normalized.offset + len(hits) < count
            else None,
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

    @_serialized
    def facet_documents(
        self,
        query: CorpusQuery | dict[str, object],
        field: FacetField,
        *,
        limit: int | None = None,
    ) -> CorpusFacetResult:
        if field not in {"collection", "language", "subject", "material", "origin", "kind"}:
            raise ValueError(f"Unsupported facet field: {field}")
        if limit is not None and not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        normalized = CorpusQuery.model_validate(query)
        where, params = self._where_clause(normalized)
        where_sql = " AND ".join(where)
        total_values = self._facet_value_count(field, where_sql, params)
        rows = self._facet_rows(field, where_sql, params, limit=limit)
        values = tuple(
            CorpusFacetValue(value=row["value"], count=int(row["count"])) for row in rows
        )
        return CorpusFacetResult(
            query=normalized,
            field=field,
            values=values,
            total_values=total_values,
            truncated=limit is not None and total_values > limit,
            limit=limit,
        )

    @_serialized
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
            collection_names={
                key: COLLECTION_NAMES[key] for key in collections if key in COLLECTION_NAMES
            },
            metadata_source_names={
                row["kind"]: METADATA_SOURCE_NAMES[row["kind"]]
                for row in self._connection.execute(
                    "SELECT DISTINCT kind FROM components ORDER BY kind"
                )
                if row["kind"] in METADATA_SOURCE_NAMES
            },
            documents=int(self._connection.execute("SELECT count(*) FROM documents").fetchone()[0]),
            passages=int(self._connection.execute("SELECT count(*) FROM passages").fetchone()[0]),
            components=int(
                self._connection.execute("SELECT count(*) FROM components").fetchone()[0]
            ),
            languages=tuple(sorted(languages)),
        )

    @_serialized
    def inspect_documents(
        self,
        document_ids: Iterable[str],
        *,
        excerpt_limit: int = 3,
        chunk_ids: Iterable[str] = (),
    ) -> tuple[CorpusInspection, ...]:
        ids = tuple(dict.fromkeys(document_ids))
        if not ids:
            raise ValueError("at least 1 document must be inspected")
        if len(ids) > 20:
            raise ValueError("at most 20 documents may be inspected")
        if not 1 <= excerpt_limit <= 10:
            raise ValueError("excerpt_limit must be between 1 and 10")
        selected_chunks = tuple(dict.fromkeys(chunk_ids))
        if len(selected_chunks) > 40:
            raise ValueError("at most 40 chunk ids may be inspected")
        focused: dict[str, list[sqlite3.Row]] = {}
        if selected_chunks:
            chunk_placeholders = ", ".join("?" for _ in selected_chunks)
            chunks = {
                row["chunk_id"]: row
                for row in self._connection.execute(
                    "SELECT p.*, pl.language, c.chunk_id, c.char_start, c.char_end "
                    "FROM semantic_chunks c JOIN passages p USING (passage_id) "
                    "LEFT JOIN passage_languages pl USING (passage_id) "
                    f"WHERE c.chunk_id IN ({chunk_placeholders})",
                    selected_chunks,
                )
            }
            for chunk_id in selected_chunks:
                chunk = chunks.get(chunk_id)
                if chunk is None or chunk["document_id"] not in ids:
                    raise ValueError("chunk ids must exist and belong to the requested documents")
                focused.setdefault(chunk["document_id"], []).append(chunk)
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
            ordinary_passages = self._connection.execute(
                "SELECT p.*, pl.language FROM passages p "
                "LEFT JOIN passage_languages pl ON pl.passage_id = p.passage_id "
                "WHERE p.document_id = ? ORDER BY p.sequence LIMIT ?",
                (document_id, excerpt_limit + len(focused.get(document_id, ()))),
            ).fetchall()
            selected = focused.get(document_id, [])[:excerpt_limit]
            selected_passage_ids = {p["passage_id"] for p in selected}
            passages = [
                *selected,
                *(p for p in ordinary_passages if p["passage_id"] not in selected_passage_ids),
            ][:excerpt_limit]
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
                        ).model_copy(
                            update={
                                "chunk_id": passage["chunk_id"]
                                if "chunk_id" in passage.keys()
                                else None,
                                "chunk_char_start": passage["char_start"]
                                if "char_start" in passage.keys()
                                else None,
                                "chunk_char_end": passage["char_end"]
                                if "char_end" in passage.keys()
                                else None,
                            }
                        )
                        for passage in passages
                    ),
                )
            )
        return tuple(inspections)

    @_serialized
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

    @_serialized
    def close(self) -> None:
        if self._closed:
            return
        if self._semantic is not None:
            self._semantic.close()
        if self._discovery is not None:
            self._discovery.close()
        self._semantic_encoder = None
        self._connection.close()
        self._closed = True

    def _where_clause(self, query: CorpusQuery) -> tuple[list[str], list[object]]:
        where, params = document_scope_where(query)
        for group in query.term_groups:
            alternatives = self._term_group_conditions(group, query.fields, params)
            where.append("(" + " OR ".join(alternatives) + ")" if alternatives else "0 = 1")
        for group in query.subject_groups:
            where.append(self._subject_group_condition(group, params))
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

        ranking_params.extend([query.limit, query.offset])
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
            "candidates.collection, candidates.document_id LIMIT ? OFFSET ?",
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

    def _facet_value_count(self, field: FacetField, where_sql: str, params: list[object]) -> int:
        filtered = (
            "WITH filtered_documents AS MATERIALIZED "
            f"(SELECT d.* FROM documents d WHERE {where_sql}) "
        )
        if field == "collection":
            sql = filtered + "SELECT count(*) FROM ("
            sql += "SELECT collection FROM filtered_documents GROUP BY collection)"
            count_params = params
        elif field == "language":
            sql = filtered + "SELECT count(*) FROM ("
            sql += (
                "SELECT pl.language FROM filtered_documents fd "
                "JOIN passages p ON p.document_id = fd.document_id "
                "JOIN passage_languages pl ON pl.passage_id = p.passage_id "
                "WHERE p.kind = 'edition' GROUP BY pl.language)"
            )
            count_params = params
        elif field == "kind":
            sql = filtered + "SELECT count(*) FROM ("
            sql += (
                "SELECT p.kind FROM filtered_documents fd "
                "JOIN passages p ON p.document_id = fd.document_id GROUP BY p.kind)"
            )
            count_params = params
        else:
            sql = filtered + ", component_owners AS ("
            sql += (
                "SELECT fd.document_id, c.component_id FROM filtered_documents fd "
                "JOIN components c ON c.document_id = fd.document_id "
                "UNION SELECT fd.document_id, l.hgv_component_id FROM filtered_documents fd "
                "JOIN components d ON d.document_id = fd.document_id "
                "JOIN component_links l ON l.ddbdp_component_id = d.component_id"
                ") SELECT count(*) FROM ("
                "SELECT m.value FROM component_owners owners "
                "JOIN metadata m ON m.component_id = owners.component_id "
                "WHERE m.key = ? GROUP BY m.value)"
            )
            count_params = [*params, field]
        return int(self._connection.execute(sql, count_params).fetchone()[0])

    def _facet_rows(
        self,
        field: FacetField,
        where_sql: str,
        params: list[object],
        *,
        limit: int | None,
    ) -> list[sqlite3.Row]:
        # MATERIALIZED: component_owners references filtered_documents in both UNION
        # legs, and without it each leg re-evaluates the whole filter.
        filtered = (
            f"WITH filtered_documents AS MATERIALIZED "
            f"(SELECT d.* FROM documents d WHERE {where_sql}) "
        )
        if field == "collection":
            sql = self._connection.execute(
                filtered + "SELECT collection AS value, count(*) AS count FROM filtered_documents "
                "GROUP BY collection ORDER BY count DESC, value ASC"
                + (" LIMIT ?" if limit is not None else ""),
                [*params, limit] if limit is not None else params,
            )
            rows = sql.fetchall()
        elif field == "language":
            sql = self._connection.execute(
                filtered + "SELECT pl.language AS value, count(DISTINCT fd.document_id) AS count "
                "FROM filtered_documents fd "
                "JOIN passages p ON p.document_id = fd.document_id "
                "JOIN passage_languages pl ON pl.passage_id = p.passage_id "
                "WHERE p.kind = 'edition' GROUP BY pl.language "
                "ORDER BY count DESC, value ASC" + (" LIMIT ?" if limit is not None else ""),
                [*params, limit] if limit is not None else params,
            )
            rows = sql.fetchall()
        elif field == "kind":
            sql = self._connection.execute(
                filtered + "SELECT p.kind AS value, count(DISTINCT fd.document_id) AS count "
                "FROM filtered_documents fd "
                "JOIN passages p ON p.document_id = fd.document_id GROUP BY p.kind "
                "ORDER BY count DESC, value ASC" + (" LIMIT ?" if limit is not None else ""),
                [*params, limit] if limit is not None else params,
            )
            rows = sql.fetchall()
        else:
            sql = self._connection.execute(
                filtered + ", component_owners AS ("
                "SELECT fd.document_id, c.component_id FROM filtered_documents fd "
                "JOIN components c ON c.document_id = fd.document_id "
                "UNION SELECT fd.document_id, l.hgv_component_id FROM filtered_documents fd "
                "JOIN components d ON d.document_id = fd.document_id "
                "JOIN component_links l ON l.ddbdp_component_id = d.component_id"
                ") SELECT m.value AS value, count(DISTINCT owners.document_id) AS count "
                "FROM component_owners owners "
                "JOIN metadata m ON m.component_id = owners.component_id "
                "WHERE m.key = ? GROUP BY m.value ORDER BY count DESC, value ASC"
                + (" LIMIT ?" if limit is not None else ""),
                [*params, field, limit] if limit is not None else [*params, field],
            )
            rows = sql.fetchall()
        return rows


def _column_fts_query(column: str, query: str) -> str:
    return f"{column} : ({query})"
