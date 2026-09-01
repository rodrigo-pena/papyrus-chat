"""Corpus build pipeline: source records → validated artifact."""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from papyrus_chat.artifact.hashing import logical_content_hash
from papyrus_chat.artifact.manifest import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactManifest,
    BuilderInfo,
    ManifestSource,
    Statistics,
    save_manifest,
)
from papyrus_chat.artifact.records import ComponentDateRecord as ArtifactDateRecord
from papyrus_chat.artifact.records import ComponentIdentifierRecord as ArtifactIdentifierRecord
from papyrus_chat.artifact.records import (
    ComponentLinkRecord,
    DocumentRecord,
    IdentifierRecord,
    PassageRecord,
)
from papyrus_chat.artifact.records import ComponentRecord as ArtifactComponentRecord
from papyrus_chat.artifact.schema import ArtifactWriter
from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.builder.collections.dclp import parse_record as parse_dclp
from papyrus_chat.builder.collections.ddbdp import ParsedDDbDP
from papyrus_chat.builder.collections.ddbdp import parse_record as parse_ddbdp
from papyrus_chat.builder.collections.epidoc import ParsedRecord
from papyrus_chat.builder.collections.hgv import parse_record as parse_hgv
from papyrus_chat.builder.collections.translations import parse_record as parse_translations
from papyrus_chat.builder.components import (
    DDbDPComponent,
    HGVComponent,
    LinkedDDbDPComponent,
    link_hgv_metadata,
)
from papyrus_chat.builder.errors import BuildError
from papyrus_chat.builder.source import CorpusSource

BUILDER_NAME = "papyrus-corpus-build"
BUILDER_VERSION = "0.2.1"

LOGGER = logging.getLogger(__name__)

CollectionParser = Callable[..., ParsedRecord | ParsedDDbDP]

SUPPORTED_COLLECTIONS: dict[str, tuple[str, CollectionParser]] = {
    "dclp": ("DCLP", parse_dclp),
    "ddbdp": ("DDbDP", parse_ddbdp),
    "translations": ("Translations", parse_translations),
}


@dataclass(frozen=True)
class ParsedCorpus:
    documents: list[DocumentRecord]
    passages: list[PassageRecord]
    identifiers: list[IdentifierRecord]
    components: list[ArtifactComponentRecord]
    links: list[ComponentLinkRecord]
    warnings: list[str]


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    collections: list[str]
    resolved_commit: str
    logical_content_hash: str
    documents: int
    passages: int
    components: int
    links: int
    parse_errors: int
    warnings: tuple[str, ...]
    size_bytes: int
    elapsed_seconds: float


def build_artifact(
    collections: list[str],
    *,
    output: Path,
    source: CorpusSource,
    source_url: str,
    requested_ref: str,
    force: bool = False,
) -> BuildResult:
    started = time.monotonic()
    canonical = sorted({c.lower() for c in collections})
    LOGGER.info(
        "Starting corpus build: collections=%s output=%s source=%s",
        ",".join(canonical),
        output,
        type(source).__name__,
        extra={
            "event": "corpus_build_started",
            "collections": canonical,
            "output": str(output),
            "source_type": type(source).__name__,
        },
    )
    unknown = [c for c in canonical if c not in SUPPORTED_COLLECTIONS]
    if unknown:
        raise BuildError(
            f"Unknown collection: {', '.join(unknown)}. Supported: "
            + ", ".join(sorted(SUPPORTED_COLLECTIONS))
        )
    if output.exists() and not force:
        raise BuildError(
            f"Output artifact already exists: {output}. "
            "Remove it or pass --force to replace exactly this artifact."
        )
    LOGGER.info(
        "Resolving source ref %r",
        requested_ref,
        extra={"event": "source_ref_resolving", "ref": requested_ref},
    )
    resolved_commit = source.resolve_commit(requested_ref)
    LOGGER.info(
        "Resolved source commit %s",
        resolved_commit[:12],
        extra={"event": "source_ref_resolved", "commit": resolved_commit},
    )
    sparse = getattr(source, "ensure_sparse_checkout", None)
    if callable(sparse):
        sparse_collections = [*canonical, "hgv"] if "ddbdp" in canonical else canonical
        LOGGER.info(
            "Preparing source checkout for collections=%s",
            ",".join(sparse_collections),
            extra={
                "event": "source_checkout_started",
                "collections": sparse_collections,
            },
        )
        sparse(sparse_collections)
        LOGGER.info(
            "Source checkout ready",
            extra={"event": "source_checkout_completed"},
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        parsed_corpus = _parse_collections(
            canonical, source=source, source_url=source_url, commit=resolved_commit
        )
        documents = parsed_corpus.documents
        passages = parsed_corpus.passages
        identifiers = parsed_corpus.identifiers
        components = parsed_corpus.components
        links = parsed_corpus.links
        warnings = parsed_corpus.warnings
        # Upstream records can declare the same idno more than once; keep one,
        # so the database and the logical hash see identical content.
        identifiers = list({(i.document_id, i.namespace, i.value): i for i in identifiers}.values())
        for record in documents:
            if not record.languages:
                warnings.append(
                    f"{record.source.path}: no declared language; stored without languages"
                )

        database = staging / "corpus.sqlite"
        LOGGER.info(
            "Writing corpus database: documents=%d passages=%d identifiers=%d components=%d",
            len(documents),
            len(passages),
            len(identifiers),
            len(components),
            extra={
                "event": "artifact_database_write_started",
                "documents": len(documents),
                "passages": len(passages),
                "identifiers": len(identifiers),
                "components": len(components),
            },
        )
        writer = ArtifactWriter(database)
        writer.create_schema()
        for record in sorted(documents, key=lambda d: d.document_id):
            writer.insert_document(record)
        writer.insert_passages(sorted(passages, key=lambda p: (p.document_id, p.sequence)))
        writer.insert_identifiers(
            sorted(identifiers, key=lambda i: (i.document_id, i.namespace, i.value))
        )
        writer.insert_components(
            sorted(components, key=lambda component: component.component_id),
            sorted(
                links,
                key=lambda link: (link.ddbdp_component_id, link.hgv_component_id),
            ),
        )
        writer.commit()
        writer.close()

        LOGGER.info(
            "Computing logical content hash",
            extra={"event": "artifact_hash_started"},
        )
        logical_hash = logical_content_hash(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            builder_version=BUILDER_VERSION,
            source_url=source_url,
            resolved_commit=resolved_commit,
            collections=canonical,
            documents=documents,
            passages=passages,
            identifiers=identifiers,
            components=components,
            links=links,
        )

        LOGGER.info(
            "Writing artifact manifest and attribution",
            extra={"event": "artifact_metadata_write_started"},
        )
        manifest = ArtifactManifest(
            builder=BuilderInfo(name=BUILDER_NAME, version=BUILDER_VERSION),
            source=ManifestSource(
                url=source_url,
                requested_ref=requested_ref,
                resolved_commit=resolved_commit,
            ),
            collections=canonical,
            statistics=Statistics(
                documents=len(documents),
                passages=len(passages),
                components=len(components),
                links=len(links),
                parse_errors=0,
            ),
            logical_content_hash=logical_hash,
            created_at=datetime.now(UTC).isoformat(),
        )
        save_manifest(staging / "manifest.json", manifest)
        (staging / "ATTRIBUTION.md").write_text(
            _attribution_text(source_url, resolved_commit), encoding="utf-8"
        )

        LOGGER.info(
            "Validating staged artifact",
            extra={"event": "artifact_validation_started"},
        )
        validate_artifact(staging)

        LOGGER.info(
            "Publishing artifact to %s",
            output,
            extra={"event": "artifact_publish_started", "output": str(output)},
        )
        if output.exists():
            previous = output.with_name(f".{output.name}.old-{os.getpid()}")
            os.rename(output, previous)
            os.rename(staging, output)
            shutil.rmtree(previous, ignore_errors=True)
        else:
            os.rename(staging, output)
    except Exception:
        subprocess.run(["rm", "-rf", str(staging)], check=False)
        raise

    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    result = BuildResult(
        output_dir=output,
        collections=canonical,
        resolved_commit=resolved_commit,
        logical_content_hash=logical_hash,
        documents=len(documents),
        passages=len(passages),
        components=len(components),
        links=len(links),
        parse_errors=0,
        warnings=tuple(warnings),
        size_bytes=size,
        elapsed_seconds=time.monotonic() - started,
    )
    LOGGER.info(
        "Corpus build completed: documents=%d passages=%d size=%d bytes elapsed=%.2fs",
        result.documents,
        result.passages,
        result.size_bytes,
        result.elapsed_seconds,
        extra={
            "event": "corpus_build_completed",
            "documents": result.documents,
            "passages": result.passages,
            "size_bytes": result.size_bytes,
            "elapsed_seconds": result.elapsed_seconds,
        },
    )
    return result


def _parse_collections(
    canonical: list[str], *, source: CorpusSource, source_url: str, commit: str
) -> ParsedCorpus:
    documents = []
    passages: list[PassageRecord] = []
    identifiers: list[IdentifierRecord] = []
    ddbdp_components: list[DDbDPComponent] = []
    warnings: list[str] = []

    for collection in canonical:
        upstream_dir_name, parser = SUPPORTED_COLLECTIONS[collection]

        files = source.xml_files(upstream_dir_name)
        if not files:
            raise BuildError(
                f"No XML records found for collection '{collection}' at commit "
                f"{commit[:12]} (looked in {upstream_dir_name}/)"
            )
        collection_started = time.monotonic()
        documents_before = len(documents)
        passages_before = len(passages)
        warnings_before = len(warnings)
        LOGGER.info(
            "Parsing %s collection (%d XML records)",
            collection,
            len(files),
            extra={
                "event": "collection_parse_started",
                "collection": collection,
                "records": len(files),
            },
        )
        next_progress_percent = 10
        for processed, source_path in enumerate(files, start=1):
            try:
                parsed = parser(
                    source.read_bytes(source_path),
                    collection=collection,
                    source_path=source_path,
                    repository_url=source_url,
                    commit=commit,
                )
            except BuildError:
                raise
            except Exception as error:
                raise BuildError(
                    f"Failed to parse {collection} record {source_path}: {error}"
                ) from error
            documents.append(parsed.document)
            passages.extend(parsed.passages)
            identifiers.extend(parsed.identifiers)
            warnings.extend(parsed.warnings)
            if isinstance(parsed, ParsedDDbDP):
                ddbdp_components.append(parsed.component)
            percent = processed * 100 // len(files)
            if percent >= next_progress_percent or processed == len(files):
                LOGGER.info(
                    "Parsed %s records: %d/%d (%d%%) elapsed=%.1fs",
                    collection,
                    processed,
                    len(files),
                    percent,
                    time.monotonic() - collection_started,
                    extra={
                        "event": "collection_parse_progress",
                        "collection": collection,
                        "processed": processed,
                        "total": len(files),
                        "percent": percent,
                    },
                )
                next_progress_percent = (percent // 10 + 1) * 10
        LOGGER.info(
            "Finished %s collection: documents=%d passages=%d warnings=%d elapsed=%.1fs",
            collection,
            len(documents) - documents_before,
            len(passages) - passages_before,
            len(warnings) - warnings_before,
            time.monotonic() - collection_started,
            extra={
                "event": "collection_parse_completed",
                "collection": collection,
                "documents": len(documents) - documents_before,
                "passages": len(passages) - passages_before,
                "warnings": len(warnings) - warnings_before,
            },
        )

    components: list[ArtifactComponentRecord] = []
    links: list[ComponentLinkRecord] = []
    if ddbdp_components:
        hgv_paths = source.xml_files("HGV_meta_EpiDoc")
        if not hgv_paths:
            warnings.append(
                f"No HGV metadata files found at commit {commit[:12]}; "
                "DDbDP records were retained without linked descriptive metadata"
            )
        hgv_components: list[HGVComponent] = []
        hgv_started = time.monotonic()
        if hgv_paths:
            LOGGER.info(
                "Parsing linked HGV metadata (%d XML records)",
                len(hgv_paths),
                extra={
                    "event": "hgv_parse_started",
                    "records": len(hgv_paths),
                },
            )
        next_progress_percent = 10
        for processed, source_path in enumerate(hgv_paths, start=1):
            try:
                hgv_components.append(
                    parse_hgv(
                        source.read_bytes(source_path),
                        source_path=source_path,
                        repository_url=source_url,
                        commit=commit,
                    )
                )
            except Exception as error:
                raise BuildError(f"Failed to parse hgv record {source_path}: {error}") from error
            percent = processed * 100 // len(hgv_paths)
            if percent >= next_progress_percent or processed == len(hgv_paths):
                LOGGER.info(
                    "Parsed hgv records: %d/%d (%d%%) elapsed=%.1fs",
                    processed,
                    len(hgv_paths),
                    percent,
                    time.monotonic() - hgv_started,
                    extra={
                        "event": "hgv_parse_progress",
                        "processed": processed,
                        "total": len(hgv_paths),
                        "percent": percent,
                    },
                )
                next_progress_percent = (percent // 10 + 1) * 10
        linked = link_hgv_metadata(ddbdp_components, hgv_components)
        components, links = _artifact_components(linked)
        LOGGER.info(
            "Linked DDbDP and HGV metadata: ddbdp=%d hgv=%d links=%d elapsed=%.1fs",
            len(ddbdp_components),
            len(hgv_components),
            len(links),
            time.monotonic() - hgv_started,
            extra={
                "event": "hgv_link_completed",
                "ddbdp_components": len(ddbdp_components),
                "hgv_components": len(hgv_components),
                "links": len(links),
            },
        )

    return ParsedCorpus(
        documents=documents,
        passages=passages,
        identifiers=identifiers,
        components=components,
        links=links,
        warnings=warnings,
    )


def _artifact_components(
    linked: tuple[LinkedDDbDPComponent, ...],
) -> tuple[list[ArtifactComponentRecord], list[ComponentLinkRecord]]:
    components: list[ArtifactComponentRecord] = []
    links: list[ComponentLinkRecord] = []
    seen_hgv: set[str] = set()
    for item in linked:
        ddbdp = item.component
        components.append(
            ArtifactComponentRecord(
                component_id=ddbdp.component_id,
                document_id=ddbdp.component_id,
                kind=ddbdp.kind,
                title=ddbdp.title,
                languages=ddbdp.edition_languages,
                metadata=_single_value_metadata(ddbdp.metadata),
                identifiers=tuple(
                    ArtifactIdentifierRecord(
                        component_id=ddbdp.component_id,
                        namespace=identifier.namespace,
                        value=identifier.value,
                    )
                    for identifier in ddbdp.identifiers
                ),
                source=ddbdp.source,
                canonical_url=ddbdp.canonical_url,
            )
        )
        for hgv in item.hgv_components:
            links.append(
                ComponentLinkRecord(
                    ddbdp_component_id=ddbdp.component_id,
                    hgv_component_id=hgv.component_id,
                )
            )
            if hgv.component_id in seen_hgv:
                continue
            seen_hgv.add(hgv.component_id)
            components.append(_hgv_artifact_component(hgv))
    return components, links


def _hgv_artifact_component(component: HGVComponent) -> ArtifactComponentRecord:
    metadata: dict[str, tuple[str, ...]] = {}
    if component.subjects:
        metadata["subject"] = component.subjects
    if component.commentary:
        metadata["commentary"] = component.commentary
    if component.material:
        metadata["material"] = (component.material,)
    if component.origins:
        metadata["origin"] = component.origins
    return ArtifactComponentRecord(
        component_id=component.component_id,
        document_id=None,
        kind=component.kind,
        title=component.title,
        metadata=metadata,
        dates=tuple(
            ArtifactDateRecord(
                component_id=component.component_id,
                sequence=sequence,
                not_before=date.not_before,
                not_after=date.not_after,
                when=date.when,
                text=date.text,
            )
            for sequence, date in enumerate(component.dates)
        ),
        identifiers=tuple(
            ArtifactIdentifierRecord(
                component_id=component.component_id,
                namespace=identifier.namespace,
                value=identifier.value,
            )
            for identifier in component.identifiers
        ),
        source=component.source,
    )


def _single_value_metadata(metadata: dict[str, str]) -> dict[str, tuple[str, ...]]:
    return {key: (value,) for key, value in metadata.items()}


def _attribution_text(source_url: str, commit: str) -> str:
    return f"""# Attribution

The corpus data in this artifact was extracted verbatim from the
[papyri/idp.data](https://github.com/papyri/idp.data) repository, which is
distributed under CC BY 3.0
([Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/)).

- Upstream repository: {source_url}
- Resolved upstream commit: `{commit}`
- Contributing projects: the Digital Corpus of Literary Papyri (DCLP/LDAB,
  Trismegistos) and APIS-derived translation records, as identified in each
  record's `<authority>` element. See the upstream
  [README](https://github.com/papyri/idp.data#readme) for details.

Model-generated prose produced by the papyrus-chat application is **not**
part of the source corpus and is always labelled as model-generated.
"""
