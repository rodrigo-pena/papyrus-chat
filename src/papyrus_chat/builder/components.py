"""Immutable component models exchanged by documentary corpus adapters."""

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from papyrus_chat.artifact.records import PassageRecord, SourceReference


class ComponentIdentifier(BaseModel):
    """An identifier belonging to a source component rather than a document."""

    model_config = ConfigDict(frozen=True)

    namespace: str = Field(min_length=1)
    value: str = Field(min_length=1)


class DateInterval(BaseModel):
    """Raw HGV date bounds, preserving the source's displayed interpretation."""

    model_config = ConfigDict(frozen=True)

    not_before: str | None = None
    not_after: str | None = None
    when: str | None = None
    text: str | None = None


class ComponentRecord(BaseModel):
    """Common provenance and identifier boundary for an upstream component."""

    model_config = ConfigDict(frozen=True)

    component_id: str
    source: SourceReference
    identifiers: tuple[ComponentIdentifier, ...]


class DDbDPComponent(ComponentRecord):
    """A DDbDP transcription and its extracted edition passages."""

    kind: Literal["ddbdp"] = "ddbdp"
    title: str
    metadata: dict[str, str] = Field(default_factory=dict)
    edition_languages: tuple[str, ...]
    passages: tuple[PassageRecord, ...]
    canonical_url: str | None = None


class HGVComponent(ComponentRecord):
    """Descriptive metadata from one HGV EpiDoc component."""

    kind: Literal["hgv"] = "hgv"
    title: str
    subjects: tuple[str, ...] = ()
    commentary: tuple[str, ...] = ()
    material: str | None = None
    origins: tuple[str, ...] = ()
    dates: tuple[DateInterval, ...] = ()


class LinkedDDbDPComponent(BaseModel):
    """A DDbDP component with every HGV component matched by HGV identifier."""

    model_config = ConfigDict(frozen=True)

    component: DDbDPComponent
    hgv_components: tuple[HGVComponent, ...] = ()


def link_hgv_metadata(
    ddbdp_components: Iterable[DDbDPComponent],
    hgv_components: Iterable[HGVComponent],
) -> tuple[LinkedDDbDPComponent, ...]:
    """Join HGV components to DDbDP components without dropping unmatched rows.

    Matching uses only ``HGV`` identifiers. Duplicate HGV identifiers are kept
    in source-path order so a one-to-many link remains observable to the next
    artifact-building stage.
    """

    by_hgv: dict[str, list[HGVComponent]] = {}
    for component in hgv_components:
        for identifier in component.identifiers:
            if identifier.namespace.lower() == "hgv":
                by_hgv.setdefault(identifier.value, []).append(component)

    for matches in by_hgv.values():
        matches.sort(key=lambda component: component.source.path)

    linked: list[LinkedDDbDPComponent] = []
    for component in ddbdp_components:
        identifiers = {
            identifier.value
            for identifier in component.identifiers
            if identifier.namespace.lower() == "hgv"
        }
        matches = [match for value in sorted(identifiers) for match in by_hgv.get(value, [])]
        linked.append(
            LinkedDDbDPComponent(
                component=component,
                hgv_components=tuple(matches),
            )
        )
    return tuple(linked)
