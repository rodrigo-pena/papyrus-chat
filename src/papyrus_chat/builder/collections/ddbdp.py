"""DDbDP collection adapter for documentary transcriptions."""

from dataclasses import dataclass

from papyrus_chat.artifact.records import DocumentRecord, IdentifierRecord, PassageRecord
from papyrus_chat.builder.collections.epidoc import parse_epidoc_record
from papyrus_chat.builder.components import ComponentIdentifier, DDbDPComponent


@dataclass(frozen=True)
class ParsedDDbDP:
    """Parsed DDbDP data in both the current artifact and component shapes."""

    document: DocumentRecord
    identifiers: tuple[IdentifierRecord, ...]
    passages: tuple[PassageRecord, ...]
    component: DDbDPComponent
    warnings: tuple[str, ...]


def parse_record(
    data: bytes,
    *,
    collection: str,
    source_path: str,
    repository_url: str,
    commit: str,
) -> ParsedDDbDP:
    """Parse one DDbDP EpiDoc transcription.

    DDbDP's header language list describes available interface languages, so
    the document language is taken from the ``xml:lang`` value on edition
    divs. Only edition divs become transcription passages.
    """

    parsed = parse_epidoc_record(
        data,
        collection=collection,
        source_path=source_path,
        repository_url=repository_url,
        commit=commit,
        languages_from_editions=True,
        include_translations=False,
    )
    component = DDbDPComponent(
        component_id=f"ddbdp:{source_path}",
        source=parsed.document.source,
        identifiers=tuple(
            ComponentIdentifier(namespace=identifier.namespace, value=identifier.value)
            for identifier in parsed.identifiers
        ),
        title=parsed.document.title,
        edition_languages=tuple(parsed.document.languages),
        passages=tuple(parsed.passages),
        canonical_url=parsed.document.canonical_url,
    )
    return ParsedDDbDP(
        document=parsed.document,
        identifiers=parsed.identifiers,
        passages=parsed.passages,
        component=component,
        warnings=parsed.warnings,
    )
