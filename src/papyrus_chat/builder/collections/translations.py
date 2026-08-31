"""Translations collection adapter (published translation records, SPEC 3.1)."""

from papyrus_chat.builder.collections.epidoc import ParsedRecord, parse_epidoc_record


def parse_record(
    data: bytes,
    *,
    collection: str,
    source_path: str,
    repository_url: str,
    commit: str,
) -> ParsedRecord:
    return parse_epidoc_record(
        data,
        collection=collection,
        source_path=source_path,
        repository_url=repository_url,
        commit=commit,
    )
