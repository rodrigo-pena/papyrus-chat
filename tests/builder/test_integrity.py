"""Pre-write integrity checks for normalized corpus records."""

import pytest

from papyrus_chat.artifact.records import (
    ComponentLinkRecord,
    ComponentRecord,
    DocumentRecord,
    PassageRecord,
    SourceReference,
)
from papyrus_chat.builder.errors import BuildError
from papyrus_chat.builder.integrity import validate_record_graph

SOURCE = SourceReference(
    repository_url="https://github.com/papyri/idp.data.git",
    commit="f" * 40,
    path="DDbDP/0/example.xml",
)


def test_reports_all_normalized_record_conflicts_before_database_writes() -> None:
    document = DocumentRecord(
        document_id="ddbdp:DDbDP/0/example.xml",
        collection="ddbdp",
        title="Example",
        languages=["grc"],
        metadata={},
        source=SOURCE,
    )
    passage = PassageRecord(
        passage_id="ddbdp:DDbDP/0/missing.xml#edition:1:edition",
        document_id="ddbdp:DDbDP/0/missing.xml",
        kind="edition",
        sequence=1,
        display_text="text",
        search_text="text",
        source=SOURCE,
    )
    component = ComponentRecord(
        component_id="hgv:HGV_meta_EpiDoc/HGV1/1.xml",
        document_id="ddbdp:DDbDP/0/missing.xml",
        kind="hgv",
        title="Example metadata",
        metadata={"subject": ("Brief", "Brief")},
        source=SOURCE.model_copy(update={"path": "HGV_meta_EpiDoc/HGV1/1.xml"}),
    )
    link = ComponentLinkRecord(
        ddbdp_component_id=component.component_id,
        hgv_component_id="hgv:HGV_meta_EpiDoc/HGV1/missing.xml",
    )

    with pytest.raises(BuildError) as excinfo:
        validate_record_graph(
            documents=[document],
            passages=[passage],
            identifiers=[],
            components=[component],
            links=[link],
        )

    message = str(excinfo.value)
    assert "duplicate metadata row" in message
    assert "passage references unknown document" in message
    assert "component references unknown document" in message
    assert "link references unknown DDbDP component" in message
    assert "link references unknown HGV component" in message
