"""Hardened XML parsing for EpiDoc sources.

External entity expansion and network resolution are disabled; entity
references and malformed records are rejected with structured errors.
"""

from collections.abc import Iterable
from typing import cast

from lxml import etree

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class ParseError(Exception):
    """A source record could not be parsed safely."""


def element_text(element: etree._Element) -> str:
    """All text contained in an element, including descendants."""
    return " ".join(cast("Iterable[str]", element.itertext())).strip()


def make_safe_parser() -> etree.XMLParser:
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        recover=False,
        # Upstream EpiDoc records can repeat xml:id values; no adapter uses ID lookup.
        collect_ids=False,
    )


def parse_xml(data: bytes) -> etree._ElementTree:
    parser = make_safe_parser()
    try:
        root = etree.fromstring(data, parser=parser)
    except etree.XMLSyntaxError as error:
        raise ParseError(f"XML syntax error at line {error.lineno}: {error.msg}") from error

    if any(True for _ in root.iter(etree.Entity)):
        raise ParseError("document contains entity references; entity expansion is disabled")
    return root.getroottree()
