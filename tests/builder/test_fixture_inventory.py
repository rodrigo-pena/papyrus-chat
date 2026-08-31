"""Fixture inventory: the committed fixture set must cover SPEC 12.1's categories."""

from pathlib import Path
from typing import Any, cast

import pytest
from lxml import etree

FIXTURES = Path(__file__).parent.parent / "fixtures" / "idp.data"
INVALID_FIXTURES = Path(__file__).parent.parent / "fixtures" / "idp.data-invalid"
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def parse(path: Path) -> etree._ElementTree:
    return etree.parse(str(path))


def xpath_all(tree: etree._ElementTree, expression: str) -> list[Any]:
    """Evaluate an XPath expression; results are loosely typed by lxml-stubs."""
    return cast("list[Any]", tree.xpath(expression, namespaces=TEI_NS))


def edition_text(tree: etree._ElementTree) -> str:
    return " ".join(div.itertext() for div in xpath_all(tree, '//tei:div[@type="edition"]'))


def test_dclp_record_with_edition_text_and_uncertainty_markup() -> None:
    tree = parse(FIXTURES / "DCLP" / "23" / "23944.xml")

    editions = xpath_all(tree, '//tei:div[@type="edition"]')
    assert len(editions) == 1
    text = " ".join(editions[0].itertext())
    assert text.strip(), "edition div must contain text"

    greek = [ch for ch in text if "\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff"]
    assert greek, "edition text must contain polytonic Greek"

    assert xpath_all(tree, "//tei:supplied"), "expected supplied markup"
    assert xpath_all(tree, "//tei:unclear"), "expected unclear markup"
    assert xpath_all(tree, "//tei:gap"), "expected gap markup"
    assert xpath_all(tree, "//*[@cert]"), "expected a cert attribute"

    assert xpath_all(tree, "//tei:lb"), "expected line milestones"


def test_dclp_metadata_only_record() -> None:
    tree = parse(FIXTURES / "DCLP" / "23" / "23702.xml")

    editions = xpath_all(tree, '//tei:div[@type="edition"]')
    assert len(editions) == 1
    assert not "".join(editions[0].itertext()).strip(), "edition div must be empty"

    idno_types = xpath_all(tree, "//tei:idno/@type")
    assert {"dclp", "TM", "LDAB"} <= set(idno_types)


def test_translation_with_textparts_and_line_milestones() -> None:
    tree = parse(FIXTURES / "Translations" / "3" / "3227-1.xml")

    translation = xpath_all(tree, '//tei:div[@type="translation"]')
    assert len(translation) == 1

    textparts = xpath_all(tree, '//tei:div[@type="textpart"]')
    assert len(textparts) >= 2, "expected nested textpart structure"

    milestones = xpath_all(tree, '//tei:milestone[@unit="line"]')
    assert milestones, "expected line milestones in the translation"

    idno_types = xpath_all(tree, "//tei:idno/@type")
    assert {"TM", "HGV", "ddb-hybrid", "filename"} <= set(idno_types)


def test_translation_with_multiple_identifier_namespaces() -> None:
    tree = parse(FIXTURES / "Translations" / "3" / "3643-1.xml")

    translation = xpath_all(tree, '//tei:div[@type="translation"]')
    assert len(translation) == 1
    assert " ".join(translation[0].itertext()).strip(), "translation div must contain text"

    idno_types = xpath_all(tree, "//tei:idno/@type")
    assert {"apisid", "controlNo", "ddbdp", "HGV", "TM", "filename"} <= set(idno_types)


def test_malformed_fixture_is_not_well_formed_xml() -> None:
    broken = INVALID_FIXTURES / "DCLP" / "99" / "broken-record.xml"
    assert broken.is_file()

    with pytest.raises(etree.XMLSyntaxError):
        parse(broken)
