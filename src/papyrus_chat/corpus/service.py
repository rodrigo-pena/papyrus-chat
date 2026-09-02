"""Lifecycle-managed, transport-neutral access to one corpus artifact."""

from collections.abc import Iterable
from pathlib import Path

from papyrus_chat.artifact.manifest import ArtifactManifest, load_manifest
from papyrus_chat.corpus.models import (
    CorpusDescription,
    CorpusDocumentMatch,
    CorpusFacetResult,
    CorpusInspectionResult,
    CorpusQuery,
    CorpusSearchResult,
    CorpusSubjectSuggestionSummary,
)
from papyrus_chat.retrieval.structured import FacetField, StructuredCorpusSearch


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

    def facet_documents(self, query: CorpusQuery, field: FacetField) -> CorpusFacetResult:
        return self._search.facet_documents(query, field)

    def suggest_subject_values(
        self, concept: str, *, scope: CorpusQuery, limit: int = 20
    ) -> CorpusSubjectSuggestionSummary:
        return CorpusSubjectSuggestionSummary(
            concept=concept,
            scope=scope.model_copy(update={"term_groups": (), "subject_groups": ()}),
            suggestions=self._search.semantic.suggest_subject_values(
                concept, scope=scope, limit=limit
            ),
        )

    def document_for_citation(self, canonical_url: str) -> CorpusDocumentMatch | None:
        return self._search.document_for_citation(canonical_url)

    def close(self) -> None:
        self._search.close()


__all__ = ["CorpusService"]
