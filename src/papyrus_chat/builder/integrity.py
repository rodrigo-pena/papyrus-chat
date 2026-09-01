"""Validate normalized corpus relationships before SQLite persistence."""

from collections.abc import Hashable, Iterable, Sequence

from papyrus_chat.artifact.records import (
    ComponentLinkRecord,
    ComponentRecord,
    DocumentRecord,
    IdentifierRecord,
    PassageRecord,
)
from papyrus_chat.builder.errors import BuildError

_MAX_REPORTED_ISSUES = 20


class _Issues:
    def __init__(self) -> None:
        self.total = 0
        self.reported: list[str] = []

    def add(self, message: str) -> None:
        self.total += 1
        if len(self.reported) < _MAX_REPORTED_ISSUES:
            self.reported.append(message)

    def raise_if_any(self) -> None:
        if not self.total:
            return
        omitted = self.total - len(self.reported)
        detail = "\n".join(f"- {message}" for message in self.reported)
        if omitted:
            detail += f"\n- ... and {omitted} more issue(s)"
        raise BuildError(
            f"Parsed corpus integrity check failed with {self.total} issue(s); "
            f"no database was written:\n{detail}"
        )


def _check_unique(rows: Iterable[tuple[Hashable, str]], *, label: str, issues: _Issues) -> None:
    seen: dict[Hashable, str] = {}
    for key, source_path in rows:
        previous = seen.get(key)
        if previous is not None:
            issues.add(f"duplicate {label} {key!r} in {previous!r} and {source_path!r}")
        else:
            seen[key] = source_path


def validate_record_graph(
    *,
    documents: Sequence[DocumentRecord],
    passages: Sequence[PassageRecord],
    identifiers: Sequence[IdentifierRecord],
    components: Sequence[ComponentRecord],
    links: Sequence[ComponentLinkRecord],
) -> None:
    """Collect uniqueness and relationship errors before database insertion."""

    issues = _Issues()
    _check_unique(
        ((record.document_id, record.source.path) for record in documents),
        label="document ID",
        issues=issues,
    )
    document_ids = {record.document_id for record in documents}

    _check_unique(
        ((record.passage_id, record.source.path) for record in passages),
        label="passage ID",
        issues=issues,
    )
    for record in passages:
        if record.document_id not in document_ids:
            issues.add(
                f"passage references unknown document {record.document_id!r} "
                f"in {record.source.path!r}"
            )

    _check_unique(
        (
            ((record.document_id, record.namespace, record.value), record.document_id)
            for record in identifiers
        ),
        label="identifier row",
        issues=issues,
    )
    for record in identifiers:
        if record.document_id not in document_ids:
            issues.add(f"identifier references unknown document {record.document_id!r}")

    _check_unique(
        ((record.component_id, record.source.path) for record in components),
        label="component ID",
        issues=issues,
    )
    components_by_id = {record.component_id: record for record in components}
    for record in components:
        if record.document_id is not None and record.document_id not in document_ids:
            issues.add(
                f"component references unknown document {record.document_id!r} "
                f"in {record.source.path!r}"
            )
        _check_component_children(record, issues)

    _check_unique(
        (
            ((link.ddbdp_component_id, link.hgv_component_id), link.ddbdp_component_id)
            for link in links
        ),
        label="component link",
        issues=issues,
    )
    for link in links:
        ddbdp = components_by_id.get(link.ddbdp_component_id)
        if ddbdp is None or ddbdp.kind != "ddbdp":
            issues.add(f"link references unknown DDbDP component {link.ddbdp_component_id!r}")
        hgv = components_by_id.get(link.hgv_component_id)
        if hgv is None or hgv.kind != "hgv":
            issues.add(f"link references unknown HGV component {link.hgv_component_id!r}")

    issues.raise_if_any()


def _check_component_children(record: ComponentRecord, issues: _Issues) -> None:
    _check_unique(
        (
            (
                (identifier.component_id, identifier.namespace, identifier.value),
                record.source.path,
            )
            for identifier in record.identifiers
        ),
        label="component identifier row",
        issues=issues,
    )
    for identifier in record.identifiers:
        if identifier.component_id != record.component_id:
            issues.add(
                f"component identifier owner {identifier.component_id!r} does not match "
                f"{record.component_id!r} in {record.source.path!r}"
            )

    _check_unique(
        (
            ((record.component_id, key, value), record.source.path)
            for key, values in record.metadata.items()
            for value in values
        ),
        label="metadata row",
        issues=issues,
    )
    _check_unique(
        (((date.component_id, date.sequence), record.source.path) for date in record.dates),
        label="date row",
        issues=issues,
    )
    for date in record.dates:
        if date.component_id != record.component_id:
            issues.add(
                f"date owner {date.component_id!r} does not match {record.component_id!r} "
                f"in {record.source.path!r}"
            )

    _check_unique(
        (
            ((record.component_id, language, "edition"), record.source.path)
            for language in record.languages
        ),
        label="language row",
        issues=issues,
    )
