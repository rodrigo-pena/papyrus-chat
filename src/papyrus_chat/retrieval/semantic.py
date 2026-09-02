"""Scoped semantic suggestions over the artifact's HGV subject vocabulary."""

import sqlite3
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from papyrus_chat.artifact.manifest import SemanticIndexInfo, load_manifest
from papyrus_chat.retrieval.structured import CorpusQuery
from papyrus_chat.semantic.embeddings import EmbeddingKind


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

    @property
    def coverage(self) -> float:
        """Fraction of scoped documents annotated with this subject label."""
        if self.scope_document_count == 0:
            return 0.0
        return self.scoped_document_count / self.scope_document_count


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
        scope_count = int(
            self._connection.execute(
                f"SELECT count(*) FROM documents d WHERE {scope_where}", scope_params
            ).fetchone()[0]
        )
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
            scoped_count = self._subject_count(row["value"], normalized_scope)
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
                )
            )
            if len(suggestions) >= limit:
                break
        return tuple(suggestions)

    def close(self) -> None:
        self._connection.close()

    def _scope_where(self, scope: CorpusQuery) -> tuple[str, list[object]]:
        # Reuse StructuredCorpusSearch's proven scope SQL without importing its
        # connection-bound service (which would create a second database handle).
        where = ["1 = 1"]
        params: list[object] = []
        if scope.collections:
            placeholders = ", ".join("?" for _ in scope.collections)
            where.append(f"d.collection IN ({placeholders})")
            params.extend(scope.collections)
        if scope.transcription_languages:
            placeholders = ", ".join("?" for _ in scope.transcription_languages)
            where.append(
                "EXISTS (SELECT 1 FROM passages p WHERE p.document_id = d.document_id "
                "AND p.kind = 'edition' AND EXISTS (SELECT 1 FROM passage_languages pl "
                f"WHERE pl.passage_id = p.passage_id AND pl.language IN ({placeholders})))"
            )
            params.extend(scope.transcription_languages)
        if scope.date_interval is not None:
            start = (
                "CAST(COALESCE(NULLIF(date_row.not_before, ''), "
                "NULLIF(date_row.not_after, ''), NULLIF(date_row.when_value, '')) AS INTEGER)"
            )
            end = (
                "CAST(COALESCE(NULLIF(date_row.not_after, ''), "
                "NULLIF(date_row.not_before, ''), NULLIF(date_row.when_value, '')) AS INTEGER)"
            )
            where.append(
                "EXISTS (SELECT 1 FROM components ddc JOIN component_links link "
                "ON link.ddbdp_component_id = ddc.component_id JOIN dates date_row "
                "ON date_row.component_id = link.hgv_component_id "
                "WHERE ddc.document_id = d.document_id AND ddc.kind = 'ddbdp' "
                f"AND {start} <= ? AND {end} >= ?)"
            )
            params.extend([scope.date_interval.not_after, scope.date_interval.not_before])
        return " AND ".join(where), params

    def _subject_count(self, value: str, scope: CorpusQuery) -> int:
        where, params = self._scope_where(scope)
        placeholders = "?"
        params.extend([value, value])
        return int(
            self._connection.execute(
                f"SELECT count(*) FROM documents d WHERE {where} AND d.document_id IN ("
                "SELECT owner.document_id FROM components owner JOIN metadata subject "
                "ON subject.component_id = owner.component_id "
                "WHERE owner.document_id IS NOT NULL AND subject.key = 'subject' "
                f"AND subject.value IN ({placeholders}) "
                "UNION SELECT ddbdp.document_id FROM components ddbdp JOIN component_links link "
                "ON link.ddbdp_component_id = ddbdp.component_id JOIN metadata subject "
                "ON subject.component_id = link.hgv_component_id "
                "WHERE ddbdp.document_id IS NOT NULL AND subject.key = 'subject' "
                f"AND subject.value IN ({placeholders}))",
                params,
            ).fetchone()[0]
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

            encoder = LocalEmbeddingEncoder(self.artifact_root / "semantic/model")
            self._encoder = encoder
        query_vector = encoder.encode([concept], kind="query")[0]
        raw = (self.artifact_root / semantic.embeddings_file).read_bytes()
        expected = len(rows) * semantic.dimensions * 4
        if len(raw) != expected:
            raise ValueError("semantic embedding file length does not match manifest")
        scores: list[tuple[float, str, int]] = []
        for index, row in enumerate(rows):
            vector = struct.unpack_from(
                f"<{semantic.dimensions}f", raw, index * semantic.dimensions * 4
            )
            score = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            scores.append((score, row["value_norm"], index))
        scores.sort(key=lambda item: (-item[0], item[1], rows[item[2]]["value"]))
        return (
            {item[2]: rank for rank, item in enumerate(scores, start=1)},
            {rows[item[2]]["subject_id"]: item[0] for item in scores},
        )


__all__ = ["SemanticSubjectSearch", "SubjectSuggestion"]
