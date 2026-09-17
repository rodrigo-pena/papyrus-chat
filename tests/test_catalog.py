"""Supported collection names must be available without source XML or networking."""

from papyrus_chat.builder.pipeline import SUPPORTED_COLLECTIONS
from papyrus_chat.catalog import COLLECTION_NAMES, METADATA_SOURCE_NAMES


def test_catalog_covers_supported_collections_and_separates_metadata() -> None:
    assert set(SUPPORTED_COLLECTIONS) <= COLLECTION_NAMES.keys()
    assert not COLLECTION_NAMES.keys() & METADATA_SOURCE_NAMES.keys()
    assert "hgv" not in SUPPORTED_COLLECTIONS
