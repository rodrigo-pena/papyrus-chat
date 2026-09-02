"""Shared SQL for document-level corpus scopes."""

from collections.abc import Sequence
from typing import Protocol


class DateInterval(Protocol):
    @property
    def not_before(self) -> int: ...

    @property
    def not_after(self) -> int: ...


class DocumentScope(Protocol):
    @property
    def collections(self) -> Sequence[str]: ...

    @property
    def transcription_languages(self) -> Sequence[str]: ...

    @property
    def date_interval(self) -> DateInterval | None: ...


def document_scope_where(scope: DocumentScope) -> tuple[list[str], list[object]]:
    """Compile collection, language, and date filters for documents aliased as ``d``."""
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
    return where, params


__all__ = ["DocumentScope", "document_scope_where"]
