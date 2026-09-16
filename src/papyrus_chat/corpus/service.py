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
    SemanticIndexCapability,
)
from papyrus_chat.corpus.passages import DocumentPassagePage
from papyrus_chat.retrieval.discovery.models import DiscoveryQuery, DiscoveryResult
from papyrus_chat.retrieval.identifiers import normalize_identifier_query
from papyrus_chat.retrieval.semantic import QueryEncoder
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
        self._content_failure: str | None = None

    @classmethod
    def open(
        cls,
        artifact_root: Path,
        *,
        semantic_encoder: QueryEncoder | None = None,
    ) -> "CorpusService":
        """Open a validated artifact's manifest and its database read-only."""
        root = artifact_root.expanduser().resolve()
        manifest = load_manifest(root / "manifest.json")
        search = StructuredCorpusSearch(
            root / "corpus.sqlite",
            read_only=True,
            semantic_encoder=semantic_encoder,
        )
        return cls(search, artifact_root=root, manifest=manifest)

    def read_document_passages(
        self, document_id: str, *, cursor: str | None = None
    ) -> DocumentPassagePage:
        """Read consecutive exact text windows; continue with the returned cursor."""
        from papyrus_chat.corpus.passages import read_passages

        with self._lock:
            return read_passages(
                self._connection, self.manifest.logical_content_hash, document_id, cursor
            )

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
            capability = capability.model_copy(
                update={
                    "subjects": SemanticIndexCapability(
                        available=capability.available,
                        count=capability.subject_count,
                        unavailable_reason=capability.unavailable_reason,
                    ),
                    "profiles": self._content_capability("profiles"),
                    "chunks": self._content_capability("chunks"),
                }
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

    def _content_capability(self, kind: str) -> SemanticIndexCapability:
        semantic = self.manifest.semantic_index
        index = getattr(semantic, kind) if semantic is not None else None
        reason = self._content_failure
        if index is None or index.count == 0:
            reason = f"artifact has no bundled semantic {kind} index"
        elif not _semantic_runtime_available():
            reason = "semantic discovery requires the [semantic] extra"
        return SemanticIndexCapability(
            available=reason is None,
            count=index.count if index is not None else 0,
            unavailable_reason=reason,
        )

    def discover_documents(self, query: DiscoveryQuery) -> DiscoveryResult:
        query = DiscoveryQuery.model_validate(query)
        with self._lock:
            # Re-evaluate transient runtime failures on every attempt instead of
            # letting one error disable discovery or capability reporting for
            # the rest of the process.
            self._content_failure = None
            capabilities = [self._content_capability("chunks")]
            if not query.passage_languages and not query.passage_kinds:
                capabilities.append(self._content_capability("profiles"))
            if not any(capability.available for capability in capabilities):
                return DiscoveryResult(
                    query=query,
                    available=False,
                    unavailable_reason="; ".join(
                        dict.fromkeys(
                            capability.unavailable_reason or "semantic discovery unavailable"
                            for capability in capabilities
                        )
                    ),
                )
            try:
                result = self._search.discovery.search(query)
            except (ImportError, RuntimeError) as error:
                self._content_failure = f"Local semantic discovery unavailable: {error}"
                return DiscoveryResult(
                    query=query,
                    available=False,
                    unavailable_reason=self._content_failure,
                )
            self._content_failure = None
            return result

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
        chunk_ids: Iterable[str] = (),
    ) -> CorpusInspectionResult:
        return CorpusInspectionResult(
            inspections=self._search.inspect_documents(
                document_ids,
                excerpt_limit=excerpt_limit,
                chunk_ids=chunk_ids,
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

    def document_for_citation(self, canonical_url: str) -> CorpusDocumentMatch | None:
        return self._search.document_for_citation(canonical_url)

    def close(self) -> None:
        self._search.close()


__all__ = ["CorpusService"]


def _semantic_runtime_available() -> bool:
    try:
        return all(
            importlib.util.find_spec(name) is not None
            for name in ("fastembed", "numpy", "tokenizers")
        )
    except (ImportError, ModuleNotFoundError):
        return False
