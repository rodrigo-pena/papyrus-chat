"""Scoped semantic suggestions over the artifact's HGV subject vocabulary."""

import sqlite3
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, computed_field

from papyrus_chat.artifact.manifest import SemanticIndexInfo, load_manifest
from papyrus_chat.retrieval.scope import document_scope_where
from papyrus_chat.retrieval.structured import CorpusQuery
from papyrus_chat.semantic.embeddings import (
    EmbeddingKind,
    EmbeddingModelSpec,
    cosine_similarity,
    normalize_embedding,
)


class QueryEncoder(Protocol):
    def encode(
        self, texts: Sequence[str], *, kind: EmbeddingKind
    ) -> tuple[tuple[float, ...], ...]: ...


class SubjectSuggestion(BaseModel):
    """One exact label suggested for a natural-language concept."""

    model_config = ConfigDict(frozen=True)

    value: str
    score: float
    strategy: str
    document_count: int
    scoped_document_count: int
    scope_document_count: int
    subject_annotated_document_count: int

    @computed_field
    @property
    def label_prevalence(self) -> float:
        """Fraction of scoped documents carrying this exact subject label."""
        if self.scope_document_count == 0:
            return 0.0
        return self.scoped_document_count / self.scope_document_count

    @computed_field
    @property
    def subject_annotation_coverage(self) -> float:
        """Fraction of scoped documents carrying at least one subject annotation."""
        if self.scope_document_count == 0:
            return 0.0
        return self.subject_annotated_document_count / self.scope_document_count


@dataclass(frozen=True)
class SubjectScopeStats:
    scope_document_count: int
    annotated_document_count: int
    scoped_label_counts: dict[str, int]


class SemanticSubjectSearch:
    """Fuse lexical vocabulary matching with local dense vectors and scope counts."""

    def __init__(self, database_path: Path, *, encoder: QueryEncoder | None = None) -> None:
        self.database_path = database_path
        self.artifact_root = database_path.parent
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._manifest = load_manifest(self.artifact_root / "manifest.json")
        self._encoder = encoder

    def suggest_subject_values(
        self,
        concept: str,
        *,
        scope: CorpusQuery | dict[str, object] | None = None,
        limit: int = 20,
    ) -> tuple[SubjectSuggestion, ...]:
        if not concept.strip():
            raise ValueError("concept must be non-empty")
        if not 1 <= limit <= 30:
            raise ValueError("limit must be between 1 and 30")
        semantic = self._manifest.semantic_index
        if semantic is None or semantic.subject_count == 0:
            return ()
        normalized_scope = CorpusQuery.model_validate(scope or {}).model_copy(
            update={"term_groups": (), "subject_groups": (), "limit": 1}
        )
        scope_where, scope_params = self._scope_where(normalized_scope)
        scope_stats = self._subject_scope_stats(scope_where, scope_params)
        scope_count = scope_stats.scope_document_count
        rows = self._connection.execute(
            "SELECT subject_id, value, value_norm, document_count FROM semantic_subjects "
            "ORDER BY value_norm, value"
        ).fetchall()
        lexical_ranks = self._lexical_ranks(concept, rows)
        dense_ranks, dense_scores = self._dense_ranks(concept, rows, semantic)
        ranked: list[tuple[float, sqlite3.Row, str]] = []
        for index, row in enumerate(rows):
            lexical_rank = lexical_ranks.get(index)
            dense_rank = dense_ranks.get(index)
            if lexical_rank is None and dense_rank is None:
                continue
            score = sum(
                1.0 / (60 + rank) for rank in (lexical_rank, dense_rank) if rank is not None
            )
            strategy = (
                "hybrid"
                if lexical_rank is not None and dense_rank is not None
                else ("lexical" if lexical_rank is not None else "semantic")
            )
            ranked.append((score, row, strategy))
        ranked.sort(key=lambda item: (-item[0], item[1]["value_norm"], item[1]["value"]))
        suggestions: list[SubjectSuggestion] = []
        for score, row, strategy in ranked:
            scoped_count = scope_stats.scoped_label_counts.get(row["value"], 0)
            if scoped_count == 0:
                continue
            suggestions.append(
                SubjectSuggestion(
                    value=row["value"],
                    score=score + dense_scores.get(row["subject_id"], 0.0) * 1e-6,
                    strategy=strategy,
                    document_count=int(row["document_count"]),
                    scoped_document_count=scoped_count,
                    scope_document_count=scope_count,
                    subject_annotated_document_count=scope_stats.annotated_document_count,
                )
            )
            if len(suggestions) >= limit:
                break
        return tuple(suggestions)

    def close(self) -> None:
        self._connection.close()

    def _scope_where(self, scope: CorpusQuery) -> tuple[str, list[object]]:
        where, params = document_scope_where(scope)
        return " AND ".join(where), params

    def _subject_scope_stats(
        self, scope_where: str, scope_params: list[object]
    ) -> SubjectScopeStats:
        rows = self._connection.execute(
            f"""
            WITH scoped_documents AS (
                SELECT d.document_id FROM documents d WHERE {scope_where}
            ), subject_documents AS (
                SELECT owner.document_id, subject.value
                FROM components owner
                JOIN metadata subject ON subject.component_id = owner.component_id
                WHERE owner.document_id IS NOT NULL AND subject.key = 'subject'
                UNION
                SELECT ddbdp.document_id, subject.value
                FROM components ddbdp
                JOIN component_links link ON link.ddbdp_component_id = ddbdp.component_id
                JOIN metadata subject ON subject.component_id = link.hgv_component_id
                WHERE ddbdp.document_id IS NOT NULL AND subject.key = 'subject'
            ), scoped_subject_documents AS (
                SELECT DISTINCT scoped.document_id, subjects.value
                FROM scoped_documents scoped
                JOIN subject_documents subjects ON subjects.document_id = scoped.document_id
            ), label_counts AS (
                SELECT value, count(*) AS document_count
                FROM scoped_subject_documents
                GROUP BY value
            )
            SELECT
                (SELECT count(*) FROM scoped_documents) AS scope_document_count,
                (SELECT count(DISTINCT document_id) FROM scoped_subject_documents)
                    AS annotated_document_count,
                label_counts.value,
                label_counts.document_count
            FROM label_counts
            UNION ALL
            SELECT
                (SELECT count(*) FROM scoped_documents),
                (SELECT count(DISTINCT document_id) FROM scoped_subject_documents),
                NULL,
                NULL
            WHERE NOT EXISTS (SELECT 1 FROM label_counts)
            ORDER BY value
            """,
            scope_params,
        ).fetchall()
        if not rows:
            return SubjectScopeStats(0, 0, {})
        first = rows[0]
        return SubjectScopeStats(
            scope_document_count=int(first["scope_document_count"]),
            annotated_document_count=int(first["annotated_document_count"]),
            scoped_label_counts={
                str(row["value"]): int(row["document_count"])
                for row in rows
                if row["value"] is not None
            },
        )

    @staticmethod
    def _lexical_ranks(concept: str, rows: Sequence[sqlite3.Row]) -> dict[int, int]:
        tokens = {token.casefold() for token in concept.split() if token.strip()}
        scored = []
        for index, row in enumerate(rows):
            value_tokens = {token.casefold() for token in row["value"].split()}
            overlap = len(tokens & value_tokens)
            if overlap:
                scored.append((overlap, row["value_norm"], row["value"], index))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return {item[3]: rank for rank, item in enumerate(scored, start=1)}

    def _dense_ranks(
        self, concept: str, rows: Sequence[sqlite3.Row], manifest: SemanticIndexInfo
    ) -> tuple[dict[int, int], dict[str, float]]:
        semantic = manifest
        encoder = self._encoder
        if encoder is None:
            from papyrus_chat.semantic.embeddings import LocalEmbeddingEncoder

            model_spec = EmbeddingModelSpec(
                model_id=semantic.model_id,
                revision=semantic.revision,
                dimensions=semantic.dimensions,
                model_file=semantic.model_file,
                query_prefix=semantic.query_prefix,
                passage_prefix=semantic.passage_prefix,
                pooling=semantic.pooling,
            )
            encoder = LocalEmbeddingEncoder(
                self.artifact_root / "semantic/model", model_spec=model_spec
            )
            self._encoder = encoder
        query_vector = normalize_embedding(
            encoder.encode([concept], kind="query")[0], dimensions=semantic.dimensions
        )
        raw = (self.artifact_root / semantic.embeddings_file).read_bytes()
        expected = len(rows) * semantic.dimensions * 4
        if len(raw) != expected:
            raise ValueError("semantic embedding file length does not match manifest")
        scores: list[tuple[float, str, int]] = []
        for index, row in enumerate(rows):
            raw_vector = struct.unpack_from(
                f"<{semantic.dimensions}f", raw, index * semantic.dimensions * 4
            )
            vector = normalize_embedding(raw_vector, dimensions=semantic.dimensions)
            score = cosine_similarity(query_vector, vector)
            scores.append((score, row["value_norm"], index))
        scores.sort(key=lambda item: (-item[0], item[1], rows[item[2]]["value"]))
        return (
            {item[2]: rank for rank, item in enumerate(scores, start=1)},
            {rows[item[2]]["subject_id"]: item[0] for item in scores},
        )


__all__ = ["SemanticSubjectSearch", "SubjectSuggestion"]
