"""HGV metadata collection adapter."""

from lxml import etree as _etree

from papyrus_chat.artifact.records import SourceReference
from papyrus_chat.builder.components import ComponentIdentifier, DateInterval, HGVComponent
from papyrus_chat.builder.xml import ParseError, element_text, parse_xml

TEI = "{http://www.tei-c.org/ns/1.0}"


def parse_record(
    data: bytes,
    *,
    collection: str = "hgv",
    source_path: str,
    repository_url: str,
    commit: str,
) -> HGVComponent:
    """Parse one HGV metadata component without treating it as a collection document."""

    try:
        tree = parse_xml(data)
    except ParseError as error:
        raise ParseError(f"Failed to parse {collection} record {source_path}: {error}") from error

    root = tree.getroot()
    header = root.find(f"{TEI}teiHeader")
    if header is None:
        raise ParseError(f"Failed to parse {collection} record {source_path}: no teiHeader")

    identifiers = _identifiers(header)
    title_element = header.find(f".//{TEI}titleStmt/{TEI}title")
    title = element_text(title_element) if title_element is not None else "(untitled)"
    source = _source(repository_url, commit, source_path)
    origin = header.findall(f".//{TEI}history/{TEI}origin")

    return HGVComponent(
        component_id=f"hgv:{source_path}",
        source=source,
        identifiers=identifiers,
        title=title,
        subjects=tuple(_subjects(header)),
        commentary=tuple(_commentary(root)),
        material=_material(header),
        origins=tuple(
            element_text(place)
            for item in origin
            for place in item.findall(f"{TEI}origPlace")
            if element_text(place)
        ),
        dates=tuple(
            DateInterval(
                not_before=date.get("notBefore"),
                not_after=date.get("notAfter"),
                when=date.get("when"),
                text=element_text(date) or None,
            )
            for item in origin
            for date in item.findall(f"{TEI}origDate")
        ),
    )


def _source(repository_url: str, commit: str, source_path: str) -> SourceReference:
    return SourceReference(repository_url=repository_url, commit=commit, path=source_path)


def _identifiers(header: _etree._Element) -> tuple[ComponentIdentifier, ...]:
    identifiers: list[ComponentIdentifier] = []
    seen: set[tuple[str, str]] = set()
    for element in header.iterfind(f".//{TEI}idno"):
        value = element_text(element)
        namespace = element.get("type") or "untyped"
        if not value or (namespace, value) in seen:
            continue
        seen.add((namespace, value))
        identifiers.append(ComponentIdentifier(namespace=namespace, value=value))
    if not any(identifier.namespace.lower() == "hgv" for identifier in identifiers):
        filename = next(
            (
                identifier.value
                for identifier in identifiers
                if identifier.namespace.lower() == "filename"
            ),
            None,
        )
        if filename is not None:
            identifiers.append(ComponentIdentifier(namespace="HGV", value=filename))
    return tuple(identifiers)


def _subjects(header: _etree._Element) -> list[str]:
    terms = header.findall(f".//{TEI}keywords[@scheme='hgv']/{TEI}term")
    if not terms:
        terms = header.findall(f".//{TEI}textClass//{TEI}term")
    return [value for term in terms if (value := element_text(term))]


def _commentary(root: _etree._Element) -> list[str]:
    commentary: list[str] = []
    for div in root.findall(f".//{TEI}div[@type='commentary'][@subtype='general']"):
        paragraphs = div.findall(f".//{TEI}p")
        if paragraphs:
            commentary.extend(
                value for paragraph in paragraphs if (value := element_text(paragraph))
            )
        elif value := element_text(div):
            commentary.append(value)
    return commentary


def _material(header: _etree._Element) -> str | None:
    material = header.find(
        f".//{TEI}sourceDesc/{TEI}msDesc/{TEI}physDesc/{TEI}objectDesc//{TEI}support/{TEI}material"
    )
    if material is None:
        return None
    return element_text(material) or None
