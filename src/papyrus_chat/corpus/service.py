"""Lifecycle-managed, transport-neutral access to one corpus artifact."""

import importlib.util
import json
from collections.abc import Iterable
from pathlib import Path

from papyrus_chat.artifact.manifest import ArtifactManifest, load_manifest
from papyrus_chat.corpus.models import (
    CorpusDescription,
    CorpusDocumentMatch,
    CorpusDocumentSummary,
    CorpusFacetResult,
    CorpusIdentifierLookupResult,
    CorpusInfo,
    CorpusInspectionResult,
    CorpusQuery,
    CorpusSearchResult,
    CorpusSemanticCapability,
    CorpusSubjectSuggestionSummary,
)
from papyrus_chat.retrieval.identifiers import normalize_identifier_query
from papyrus_chat.retrieval.structured import FacetField, StructuredCorpusSearch

_SEMANTIC_RUNTIME_REASON = "semantic subject suggestions require the [mcp,semantic] extras"


class CorpusService:
    """Read-only corpus operations bound to a single artifact root."""

    def __init__(
        self,
        search: StructuredCorpusSearch,
        *,
        artifact_root: Path | None = None,
        manifest: ArtifactManifest | None = None,
    ) -> None:
        self._search = search
        self._connection = search._connection  # noqa: SLF001 - lifecycle exposure
        self._lock = search._lock  # noqa: SLF001 - shared connection serialization
        self.artifact_root = (artifact_root or search._database_path.parent).resolve()  # noqa: SLF001
        self.manifest = manifest or load_manifest(self.artifact_root / "manifest.json")

    @classmethod
    def open(cls, artifact_root: Path) -> "CorpusService":
        """Open a validated artifact's manifest and its database read-only."""
        root = artifact_root.expanduser().resolve()
        manifest = load_manifest(root / "manifest.json")
        search = StructuredCorpusSearch(root / "corpus.sqlite", read_only=True)
        return cls(search, artifact_root=root, manifest=manifest)

    def describe_corpus(self) -> CorpusDescription:
        return self._search.describe()

    def get_corpus_info(self) -> CorpusInfo:
        """Return artifact provenance and runtime semantic capability."""
        with self._lock:
            semantic = self.manifest.semantic_index
            if semantic is None or semantic.subject_count == 0:
                capability = CorpusSemanticCapability(
                    available=False,
                    unavailable_reason="artifact has no bundled semantic subject index",
                )
            elif not _semantic_runtime_available():
                capability = CorpusSemanticCapability(
                    available=False,
                    model_id=semantic.model_id,
                    revision=semantic.revision,
                    subject_count=semantic.subject_count,
                    unavailable_reason=_SEMANTIC_RUNTIME_REASON,
                )
            else:
                capability = CorpusSemanticCapability(
                    available=True,
                    model_id=semantic.model_id,
                    revision=semantic.revision,
                    subject_count=semantic.subject_count,
                )
            return CorpusInfo(
                artifact_schema_version=self.manifest.artifact_schema_version,
                builder=self.manifest.builder,
                source=self.manifest.source,
                collections=tuple(self.manifest.collections),
                statistics=self.manifest.statistics,
                languages=self._search.describe().languages,
                logical_content_hash=self.manifest.logical_content_hash,
                created_at=self.manifest.created_at,
                semantic_capability=capability,
            )

    def lookup_document(self, identifier: str, *, limit: int = 20) -> CorpusIdentifierLookupResult:
        """Look up exact normalized identifier values with bounded lean matches."""
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("identifier must be non-empty")
        if len(identifier) > 200:
            raise ValueError("identifier must be at most 200 characters")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        namespace, value = normalize_identifier_query(identifier)
        if not value:
            raise ValueError("identifier must contain a value")
        normalized_identifier = f"{namespace}:{value}" if namespace else value
        with self._lock:
            if namespace:
                where = "i.namespace_norm = ? AND i.value_norm = ?"
                params: tuple[object, ...] = (namespace, value)
            else:
                where = "i.value_norm = ?"
                params = (value,)
            count = int(
                self._connection.execute(
                    "SELECT count(DISTINCT d.document_id) FROM identifiers i "
                    "JOIN documents d ON d.document_id = i.document_id WHERE " + where,
                    params,
                ).fetchone()[0]
            )
            rows = self._connection.execute(
                "SELECT DISTINCT d.document_id, d.title, d.collection, d.languages, "
                "d.canonical_url FROM identifiers i JOIN documents d "
                "ON d.document_id = i.document_id WHERE "
                + where
                + " ORDER BY d.collection, d.document_id LIMIT ?",
                [*params, limit],
            ).fetchall()
            matches = tuple(
                CorpusDocumentSummary(
                    document_id=row["document_id"],
                    title=row["title"],
                    collection=row["collection"],
                    languages=tuple(json.loads(row["languages"])),
                    canonical_url=row["canonical_url"],
                )
                for row in rows
            )
            return CorpusIdentifierLookupResult(
                normalized_identifier=normalized_identifier,
                exact_match_count=count,
                truncated=count > limit,
                matches=matches,
                limit=limit,
            )

    def search_documents(
        self,
        query: CorpusQuery,
        *,
        assumptions: tuple[str, ...] = (),
    ) -> CorpusSearchResult:
        return self._search.query(query, assumptions=assumptions)

    def inspect_documents(
        self,
        document_ids: Iterable[str],
        *,
        excerpt_limit: int = 3,
    ) -> CorpusInspectionResult:
        return CorpusInspectionResult(
            inspections=self._search.inspect_documents(
                document_ids,
                excerpt_limit=excerpt_limit,
            )
        )

    def facet_documents(
        self, query: CorpusQuery, field: FacetField, *, limit: int = 50
    ) -> CorpusFacetResult:
        return self._search.facet_documents(query, field, limit=limit)

    def suggest_subjects(
        self, concept: str, *, scope: CorpusQuery, limit: int = 20
    ) -> CorpusSubjectSuggestionSummary:
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError("concept must be non-empty")
        if len(concept) > 500:
            raise ValueError("concept must be at most 500 characters")
        if not 1 <= limit <= 30:
            raise ValueError("limit must be between 1 and 30")
        normalized_scope = scope.model_copy(update={"term_groups": (), "subject_groups": ()})
        capability = self.get_corpus_info().semantic_capability
        if not capability.available:
            return CorpusSubjectSuggestionSummary(
                concept=concept,
                scope=normalized_scope,
                suggestions=(),
                available=False,
                unavailable_reason=capability.unavailable_reason,
            )
        try:
            suggestions = self._search.semantic.suggest_subject_values(
                concept, scope=scope, limit=limit
            )
        except (ImportError, RuntimeError) as error:
            return CorpusSubjectSuggestionSummary(
                concept=concept,
                scope=normalized_scope,
                suggestions=(),
                available=False,
                unavailable_reason=_SEMANTIC_RUNTIME_REASON + f": {error}",
            )
        return CorpusSubjectSuggestionSummary(
            concept=concept,
            scope=normalized_scope,
            suggestions=suggestions,
            available=True,
        )

    def suggest_subject_values(
        self, concept: str, *, scope: CorpusQuery, limit: int = 20
    ) -> CorpusSubjectSuggestionSummary:
        """Compatibility alias for the original Pydantic-AI tool service method."""
        return self.suggest_subjects(concept, scope=scope, limit=limit)

    def document_for_citation(self, canonical_url: str) -> CorpusDocumentMatch | None:
        return self._search.document_for_citation(canonical_url)

    def close(self) -> None:
        self._search.close()


__all__ = ["CorpusService"]


def _semantic_runtime_available() -> bool:
    try:
        return importlib.util.find_spec("fastembed") is not None
    except (ImportError, ModuleNotFoundError):
        return False
