"""Safety tests for the shared EpiDoc XML parser."""

import pytest

from papyrus_chat.builder.xml import ParseError, parse_xml

DOCTYPE_ENTITY = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE TEI [
  <!ENTITY secret SYSTEM "file:///etc/passwd">
]>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body><ab>Hail &secret;end</ab></body></text>
</TEI>
"""

DOCTYPE_REMOTE_DTD = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE TEI SYSTEM "https://invalid.example.nonexistent/tei-epidoc.rng">
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body><ab>plain</ab></body></text>
</TEI>
"""


def test_entity_references_are_rejected() -> None:
    with pytest.raises(ParseError, match="entity"):
        parse_xml(DOCTYPE_ENTITY)


def test_remote_dtd_is_not_resolved() -> None:
    tree = parse_xml(DOCTYPE_REMOTE_DTD)

    assert tree.getroot().tag.endswith("TEI")


def test_malformed_xml_raises_parse_error_with_position() -> None:
    with pytest.raises(ParseError, match="line"):
        parse_xml(b"<TEI><teiHeader></TEI>")
