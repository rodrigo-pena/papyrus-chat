"""Deterministic logical content hash of a corpus artifact.

The hash is computed over a canonical JSON representation of everything that
defines the artifact's logical content:

- artifact schema version and builder version;
- source URL and resolved commit;
- the canonically sorted collection names;
- every document, identifier, and passage record as frozen-model dumps.

Records are sorted before serialization, so input order cannot change the
hash. Volatile values (timestamps, SQLite page layout, file sizes) are not
part of the payload; identical inputs and builder versions therefore always
produce the same logical content hash. Byte-identical SQLite files are not
required.
"""

import hashlib
import json
from collections.abc import Iterable

from papyrus_chat.artifact.records import (
    ComponentLinkRecord,
    ComponentRecord,
    DocumentRecord,
    IdentifierRecord,
    PassageRecord,
)


def canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def logical_content_hash(
    *,
    schema_version: int,
    builder_version: str,
    source_url: str,
    resolved_commit: str,
    collections: Iterable[str],
    documents: Iterable[DocumentRecord],
    passages: Iterable[PassageRecord],
    identifiers: Iterable[IdentifierRecord],
    components: Iterable[ComponentRecord] = (),
    links: Iterable[ComponentLinkRecord] = (),
) -> str:
    documents = sorted(documents, key=lambda d: d.document_id)
    passages = sorted(passages, key=lambda p: p.passage_id)
    identifiers = sorted(identifiers, key=lambda i: (i.document_id, i.namespace, i.value))
    components = sorted(components, key=lambda component: component.component_id)
    links = sorted(
        links,
        key=lambda link: (link.ddbdp_component_id, link.hgv_component_id),
    )

    payload: dict[str, object] = {
        "artifact_schema_version": schema_version,
        "builder_version": builder_version,
        "source_url": source_url,
        "resolved_commit": resolved_commit,
        "collections": sorted(collections),
        "documents": [d.model_dump(mode="json") for d in documents],
        "passages": [p.model_dump(mode="json") for p in passages],
        "identifiers": [i.model_dump(mode="json") for i in identifiers],
    }
    if components:
        payload["components"] = [component.model_dump(mode="json") for component in components]
    if links:
        payload["links"] = [link.model_dump(mode="json") for link in links]
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
