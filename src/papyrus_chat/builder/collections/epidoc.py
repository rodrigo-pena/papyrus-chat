"""Shared EpiDoc extraction for collection adapters (SPEC 6.3).

Per-collection adapters (`dclp.py`, `translations.py`) reuse this generic
parser; collection-specific behavior can be layered on top. Records without
text stay discoverable as metadata-only documents. `display_text` keeps
visible uncertainty signals; `search_text` is normalized separately;
unsupported elements are recorded as warnings, never silently dropped.
"""

from dataclasses import dataclass
from typing import Literal

from lxml import etree as _etree

from papyrus_chat.artifact.records import (
    DocumentRecord,
    IdentifierRecord,
    PassageRecord,
    SourceReference,
)
from papyrus_chat.artifact.schema import derive_document_id, derive_passage_id
from papyrus_chat.builder.xml import ParseError, element_text, parse_xml
from papyrus_chat.textnorm import normalize_search_text

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

_PASSAGE_DIV_KINDS = (("edition", "edition"), ("translation", "translation"))
_WARNED_TAGS = ("g", "surplus", "del", "add")


@dataclass(frozen=True)
class ParsedRecord:
    document: DocumentRecord
    identifiers: tuple[IdentifierRecord, ...]
    passages: tuple[PassageRecord, ...]
    warnings: tuple[str, ...]


def parse_epidoc_record(
    data: bytes,
    *,
    collection: str,
    source_path: str,
    repository_url: str,
    commit: str,
) -> ParsedRecord:
    try:
        tree = parse_xml(data)
    except ParseError as error:
        raise ParseError(f"Failed to parse {collection} record {source_path}: {error}") from error

    root = tree.getroot()
    header = root.find(f"{TEI}teiHeader")
    if header is None:
        raise ParseError(f"Failed to parse {collection} record {source_path}: no teiHeader")

    identifiers = tuple(
        IdentifierRecord(
            document_id=derive_document_id(collection, source_path),
            namespace=element.get("type") or "untyped",
            value=element_text(element),
        )
        for element in header.iterfind(f".//{TEI}idno")
        if element_text(element)
    )

    title_element = header.find(f".//{TEI}titleStmt/{TEI}title")
    title = element_text(title_element) if title_element is not None else "(untitled)"

    languages = _languages(root)
    metadata = _metadata(header)
    canonical_url = _canonical_url(identifiers)

    source = SourceReference(repository_url=repository_url, commit=commit, path=source_path)
    document = DocumentRecord(
        document_id=derive_document_id(collection, source_path),
        collection=collection,
        title=title,
        languages=languages,
        metadata=metadata,
        source=source,
        canonical_url=canonical_url,
    )

    passages: list[PassageRecord] = []
    warnings: list[str] = []
    for div_kind, passage_kind in _PASSAGE_DIV_KINDS:
        for div in root.iterfind(f".//{TEI}div[@type='{div_kind}']"):
            passages.extend(
                _passages_from_div(
                    div,
                    document=document,
                    kind=passage_kind,
                    start_sequence=len(passages),
                )
            )
            warnings.extend(_structure_warnings(div, source_path))

    return ParsedRecord(
        document=document,
        identifiers=identifiers,
        passages=tuple(passages),
        warnings=tuple(warnings),
    )


def _languages(root: _etree._Element) -> list[str]:
    languages: list[str] = []
    for element in root.iterfind(f".//{TEI}language"):
        ident = element.get("ident")
        if ident:
            languages.append(ident)
    if not languages:
        declared = root.get(XML_LANG) or root.findtext(f".//{TEI}div[@type='edition']")
        if declared:
            languages = [declared]
    unique: list[str] = []
    for language in languages:
        if language not in unique:
            unique.append(language)
    return unique


def _metadata(header: _etree._Element) -> dict[str, str]:
    metadata: dict[str, str] = {}

    def put(key: str, value: str | None) -> None:
        if value and value.strip():
            metadata.setdefault(key, " ".join(value.split()))

    put("authority", header.findtext(f".//{TEI}authority"))
    put("origPlace", header.findtext(f".//{TEI}origPlace"))
    put("material", header.findtext(f".//{TEI}material"))
    for element in header.iterfind(f".//{TEI}origDate"):
        put("origDate", element_text(element))
        if element.get("notBefore"):
            put("notBefore", element.get("notBefore"))
        if element.get("notAfter"):
            put("notAfter", element.get("notAfter"))
    for element in header.iterfind(f".//{TEI}term"):
        term_type = element.get("type")
        if term_type:
            put(f"term_{term_type}", element_text(element))
    return metadata


def _canonical_url(identifiers: tuple[IdentifierRecord, ...]) -> str | None:
    by_namespace = {identifier.namespace: identifier.value for identifier in identifiers}
    if "dclp" in by_namespace:
        return f"https://papyri.info/dclp/{by_namespace['dclp']}"
    if "ddb-hybrid" in by_namespace:
        return f"https://papyri.info/ddbdp/{by_namespace['ddb-hybrid']}"
    return None


def _innermost_textparts(div: _etree._Element) -> list[_etree._Element]:
    all_parts = div.findall(f".//{TEI}div[@type='textpart']")
    return [part for part in all_parts if not part.findall(f".//{TEI}div[@type='textpart']")]


def _textpart_label(element: _etree._Element) -> str | None:
    ancestor_labels: list[str] = []
    for ancestor in element.iterancestors():
        if ancestor.get("type") != "textpart":
            break
        if ancestor.get("n"):
            ancestor_labels.append(ancestor.get("n") or "")
    labels = list(reversed(ancestor_labels))
    if element.get("n"):
        labels.append(element.get("n") or "")
    return ".".join(labels) if labels else None


def _line_reference(element: _etree._Element) -> str | None:
    numbers = [line.get("n") for line in element.findall(f".//{TEI}lb") if line.get("n")]
    if not numbers:
        numbers = [
            milestone.get("n")
            for milestone in element.findall(f".//{TEI}milestone")
            if milestone.get("unit") == "line" and milestone.get("n")
        ]
    if not numbers:
        return None
    return f"lines {numbers[0]}-{numbers[-1]}" if len(numbers) > 1 else f"line {numbers[0]}"


def _render(element: _etree._Element) -> str:
    if element.tag == f"{TEI}gap":
        return "[...]"

    content = [element.text or ""]
    for child in element:
        content.append(_render(child))
        content.append(child.tail or "")

    if element.tag == f"{TEI}supplied":
        return "[" + "".join(content) + "]"
    return "".join(content)


def _uncertainty(element: _etree._Element) -> dict[str, int]:
    return {
        "supplied": len(element.findall(f".//{TEI}supplied")),
        "unclear": len(element.findall(f".//{TEI}unclear")),
        "gap": len(element.findall(f".//{TEI}gap")),
        "cert": len(element.findall(".//*[@cert]")),
    }


def _structure_warnings(div: _etree._Element, source_path: str) -> list[str]:
    warnings: list[str] = []
    for tag in _WARNED_TAGS:
        count = len(div.findall(f".//{TEI}{tag}"))
        if count:
            warnings.append(
                f"{count} <{tag}> element(s) in {source_path} have no normalized "
                "search-text representation; display text is unaffected"
            )
    return warnings


def _passages_from_div(
    div: _etree._Element,
    *,
    document: DocumentRecord,
    kind: Literal["edition", "translation"],
    start_sequence: int,
) -> list[PassageRecord]:
    units = _innermost_textparts(div) or [div]
    passages: list[PassageRecord] = []

    for offset, unit in enumerate(units):
        display_text = " ".join(_render(unit).split())
        if not display_text:
            continue
        sequence = start_sequence + offset + 1
        label = _textpart_label(unit) if unit is not div else None
        locator = label or kind
        passages.append(
            PassageRecord(
                passage_id=derive_passage_id(document.document_id, kind, sequence, locator),
                document_id=document.document_id,
                kind=kind,
                sequence=sequence,
                textpart=label,
                line_reference=_line_reference(unit),
                display_text=display_text,
                search_text=normalize_search_text(display_text),
                source=document.source.model_copy(update={"locator": locator}),
                uncertainty=_uncertainty(unit),
            )
        )
    return passages
