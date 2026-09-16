"""Local hybrid discovery over scoped profile, chunk, and lexical candidates."""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from papyrus_chat.artifact.manifest import SemanticIndexInfo
from papyrus_chat.artifact.records import SourceReference
from papyrus_chat.retrieval.discovery.models import (
    DiscoveryChannel,
    DiscoveryChunk,
    DiscoveryHit,
    DiscoveryQuery,
    DiscoveryResult,
)
from papyrus_chat.retrieval.discovery.ranking import fuse_rankings
from papyrus_chat.retrieval.scope import document_scope_where
from papyrus_chat.retrieval.search import build_fts_query
from papyrus_chat.retrieval.semantic import QueryEncoder
from papyrus_chat.semantic.embeddings import normalize_embedding
from papyrus_chat.semantic.vectors import LocalVectorStore


def passage_where(query: DiscoveryQuery) -> tuple[list[str], list[object]]:
    """Passage predicates consistently use aliases p and pl in every channel."""
    where: list[str] = []
    params: list[object] = []
    for column, values in (
        ("p.kind", query.passage_kinds),
        ("lower(pl.language)", query.passage_languages),
    ):
        if values:
            where.append(f"{column} IN ({', '.join('?' for _ in values)})")
            params.extend(values)
    return where, params


class SemanticDocumentSearch:
    """Called under the owning corpus service's shared connection lock."""

    def __init__(
        self,
        root: Path,
        connection: sqlite3.Connection,
        manifest: SemanticIndexInfo,
        encoder: QueryEncoder,
    ) -> None:
        self._connection = connection
        self._manifest = manifest
        self._encoder = encoder
        self._vectors = LocalVectorStore(root)

    def search(
        self,
        query: DiscoveryQuery,
        *,
        channels: Sequence[DiscoveryChannel] = ("profiles", "chunks", "lexical"),
    ) -> DiscoveryResult:
        where, params = document_scope_where(query)
        passage_clauses, passage_params = passage_where(query)
        if passage_clauses:
            where.append(
                "EXISTS (SELECT 1 FROM passages p "
                "LEFT JOIN passage_languages pl USING (passage_id) "
                "WHERE p.document_id = d.document_id AND " + " AND ".join(passage_clauses) + ")"
            )
            params.extend(passage_params)
        scope_sql = " AND ".join(where)
        documents = {
            row["document_id"]: row
            for row in self._connection.execute(
                "SELECT d.document_id, d.collection, d.title, d.languages, d.canonical_url "
                f"FROM documents d WHERE {scope_sql} ORDER BY d.collection, d.document_id",
                params,
            )
        }
        if not documents:
            return DiscoveryResult(
                query=query,
                scope_document_count=0,
                indexed_document_count=0,
                offset=query.offset,
                ranked_candidate_count=0,
            )
        identities = {key: (row["collection"], key) for key, row in documents.items()}
        eligible_passage_sql = " AND " + " AND ".join(passage_clauses) if passage_clauses else ""
        chunk_rows = (
            self._connection.execute(
                "SELECT c.chunk_id, c.document_id, c.vector_row FROM semantic_chunks c "
                "JOIN documents d ON d.document_id = c.document_id "
                "JOIN passages p ON p.passage_id = c.passage_id "
                "LEFT JOIN passage_languages pl ON pl.passage_id = p.passage_id "
                f"WHERE {scope_sql}{eligible_passage_sql} ORDER BY c.vector_row",
                [*params, *passage_params],
            ).fetchall()
            if self._manifest.chunks is not None
            else []
        )
        profile_rows = (
            self._connection.execute(
                "SELECT s.document_id, s.vector_row FROM semantic_profiles s "
                "JOIN documents d ON d.document_id = s.document_id "
                f"WHERE {scope_sql} ORDER BY s.vector_row",
                params,
            ).fetchall()
            if self._manifest.profiles is not None and not passage_clauses
            else []
        )
        indexed_ids = {row["document_id"] for row in [*chunk_rows, *profile_rows]}
        if not indexed_ids:
            return DiscoveryResult(
                query=query,
                scope_document_count=len(documents),
                indexed_document_count=0,
                offset=query.offset,
                ranked_candidate_count=0,
            )
        query_vector = normalize_embedding(
            self._encoder.encode([query.text], kind="query")[0],
            dimensions=self._manifest.dimensions,
        )
        channel_scores: dict[DiscoveryChannel, dict[str, float]] = {}
        best_chunks: dict[str, list[tuple[float, str]]] = {}
        if profile_rows and self._manifest.profiles is not None and "profiles" in channels:
            scores = self._vectors.scores(
                self._manifest.profiles.embeddings_file,
                count=self._manifest.profiles.count,
                dimensions=self._manifest.dimensions,
                rows=[row["vector_row"] for row in profile_rows],
                query=query_vector,
            )
            channel_scores["profiles"] = {
                row["document_id"]: score for row, score in zip(profile_rows, scores, strict=True)
            }
        if chunk_rows and self._manifest.chunks is not None:
            # Even profile-only evaluation returns real inspectable source locations.
            scores = self._vectors.scores(
                self._manifest.chunks.embeddings_file,
                count=self._manifest.chunks.count,
                dimensions=self._manifest.dimensions,
                rows=[row["vector_row"] for row in chunk_rows],
                query=query_vector,
            )
            for row, score in zip(chunk_rows, scores, strict=True):
                best = best_chunks.setdefault(row["document_id"], [])
                best.append((score, row["chunk_id"]))
                best.sort(key=lambda item: (-item[0], item[1]))
                del best[2:]
            if "chunks" in channels:
                channel_scores["chunks"] = {doc: best[0][0] for doc, best in best_chunks.items()}
        if "lexical" in channels:
            lexical: dict[str, float] = {}
            if fts_query := build_fts_query(query.text, operator="OR"):
                rows = self._connection.execute(
                    "SELECT p.document_id, bm25(passages_fts, 1.0, 0.0) AS score "
                    "FROM passages_fts JOIN passages p ON p.passage_id = passages_fts.passage_id "
                    "JOIN documents d ON d.document_id = p.document_id "
                    "LEFT JOIN passage_languages pl ON pl.passage_id = p.passage_id "
                    f"WHERE passages_fts MATCH ? AND {scope_sql}{eligible_passage_sql}",
                    [fts_query, *params, *passage_params],
                )
                for row in rows:
                    document_id = row["document_id"]
                    if document_id in indexed_ids:
                        lexical[document_id] = min(lexical.get(document_id, 0.0), row["score"])
            channel_scores["lexical"] = lexical
        rankings = {
            channel: sorted(
                scores,
                key=lambda doc: (
                    scores[doc] if channel == "lexical" else -scores[doc],
                    identities[doc],
                ),
            )
            for channel, scores in channel_scores.items()
        }
        fused = fuse_rankings(rankings, identities)
        selected = fused[query.offset : query.offset + query.limit]
        chunk_ids = [chunk_id for doc, _, _ in selected for _, chunk_id in best_chunks.get(doc, [])]
        chunks = self._chunks_by_id(chunk_ids)
        profiles = (
            self._profiles_by_document([doc for doc, _, _ in selected]) if profile_rows else {}
        )
        hits = []
        for document_id, score, contributing in selected:
            row = documents[document_id]
            profile = profiles.get(document_id) if "profiles" in contributing else None
            hits.append(
                DiscoveryHit(
                    document_id=document_id,
                    title=row["title"],
                    collection=row["collection"],
                    languages=tuple(json.loads(row["languages"])),
                    canonical_url=row["canonical_url"],
                    score=score,
                    channels=contributing,
                    channel_scores={
                        channel: channel_scores[channel][document_id] for channel in contributing
                    },
                    chunks=tuple(
                        chunks[chunk_id] for _, chunk_id in best_chunks.get(document_id, [])
                    ),
                    profile_snippet=profile["profile_text"][:500] if profile else None,
                    profile_metadata_only=bool(profile["metadata_only"]) if profile else None,
                )
            )
        return DiscoveryResult(
            query=query,
            scope_document_count=len(documents),
            indexed_document_count=len(indexed_ids),
            hits=tuple(hits),
            offset=query.offset,
            ranked_candidate_count=len(fused),
            next_offset=query.offset + len(hits) if query.offset + len(hits) < len(fused) else None,
            ranked_candidates_truncated=query.offset + len(hits) < len(fused),
            channels_used=tuple(channel_scores),
        )

    def _profiles_by_document(self, ids: list[str]) -> dict[str, sqlite3.Row]:
        if not ids:
            return {}
        return {
            row["document_id"]: row
            for row in self._connection.execute(
                "SELECT document_id, profile_text, metadata_only FROM semantic_profiles "
                f"WHERE document_id IN ({', '.join('?' for _ in ids)})",
                ids,
            )
        }

    def _chunks_by_id(self, ids: list[str]) -> dict[str, DiscoveryChunk]:
        if not ids:
            return {}
        rows = self._connection.execute(
            "SELECT c.*, p.kind, p.display_text, p.line_reference, pl.language, p.source_url, "
            "p.source_commit, p.source_path, p.locator FROM semantic_chunks c "
            "JOIN passages p USING (passage_id) LEFT JOIN passage_languages pl USING (passage_id) "
            f"WHERE c.chunk_id IN ({', '.join('?' for _ in ids)})",
            ids,
        )
        return {
            row["chunk_id"]: DiscoveryChunk(
                chunk_id=row["chunk_id"],
                passage_id=row["passage_id"],
                passage_kind=row["kind"],
                passage_language=row["language"],
                char_start=row["char_start"],
                char_end=row["char_end"],
                snippet=row["display_text"][row["char_start"] : row["char_end"]][:500],
                line_reference=row["line_reference"],
                source=SourceReference(
                    repository_url=row["source_url"],
                    commit=row["source_commit"],
                    path=row["source_path"],
                    locator=row["locator"],
                ),
            )
            for row in rows
        }

    def close(self) -> None:
        self._vectors.close()
