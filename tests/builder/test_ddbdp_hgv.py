"""Tests for documentary transcriptions and linked HGV metadata."""

from pathlib import Path

from papyrus_chat.builder.collections.ddbdp import parse_record as parse_ddbdp
from papyrus_chat.builder.collections.hgv import parse_record as parse_hgv
from papyrus_chat.builder.components import link_hgv_metadata

FIXTURES = Path(__file__).parent.parent / "fixtures" / "idp.data"
REPO_URL = "https://github.com/papyri/idp.data.git"
COMMIT = "027a4a3a2d8a669bed692ed5d918892bdb7ea1b3"


def parse_ddbdp_fixture():
    path = FIXTURES / "DDbDP" / "27" / "27093.xml"
    return parse_ddbdp(
        path.read_bytes(),
        collection="ddbdp",
        source_path="DDbDP/27/27093.xml",
        repository_url=REPO_URL,
        commit=COMMIT,
    )


def parse_hgv_fixture():
    path = FIXTURES / "HGV_meta_EpiDoc" / "HGV28" / "27093.xml"
    return parse_hgv(
        path.read_bytes(),
        source_path="HGV_meta_EpiDoc/HGV28/27093.xml",
        repository_url=REPO_URL,
        commit=COMMIT,
    )


class TestDdbdpAdapter:
    parsed = parse_ddbdp_fixture()

    def test_extracts_actual_edition_languages(self) -> None:
        assert self.parsed.document.languages == ["grc"]
        assert self.parsed.component.edition_languages == ("grc",)

    def test_extracts_line_aware_edition_passage(self) -> None:
        assert len(self.parsed.passages) == 1
        passage = self.parsed.passages[0]

        assert passage.kind == "edition"
        assert passage.line_reference == "lines 1-18"
        assert passage.source.locator == "edition"
        assert "[Κλαύδιος Τερ]" in passage.display_text

    def test_keeps_documentary_identifier_and_canonical_url(self) -> None:
        identifiers = {
            (identifier.namespace, identifier.value) for identifier in self.parsed.identifiers
        }

        assert ("HGV", "27093") in identifiers
        assert ("ddb-hybrid", "p.mich;8;480") in identifiers
        assert self.parsed.document.canonical_url == "https://papyri.info/ddbdp/p.mich;8;480"


class TestHgvAdapter:
    component = parse_hgv_fixture()

    def test_extracts_identifiers_and_descriptive_metadata(self) -> None:
        identifiers = {
            (identifier.namespace, identifier.value) for identifier in self.component.identifiers
        }

        assert ("HGV", "27093") in identifiers
        assert ("TM", "27093") in identifiers
        assert self.component.title == "Terentianus to Tiberianus"
        assert self.component.subjects == (
            "Brief (privat)",
            "Terentianus an Tiberianus (Vater)",
            "Schwierigkeiten bei Registrierung von Urkunden",
            "Geld",
            "erneute Petition",
        )
        assert self.component.commentary == (
            "Gehört zum Tiberianus - Archiv. Neuedition: Strassi, L’archivio di Claudius "
            "Tiberianus, 15 (S. 60f.); zur Datierung vgl. ebenda, S. 79-97.",
        )
        assert self.component.material == "Papyrus"
        assert self.component.origins == ("Alexandria (?)",)

    def test_extracts_date_interval_attributes_and_display_text(self) -> None:
        assert len(self.component.dates) == 1
        date = self.component.dates[0]

        assert date.not_before == "0101"
        assert date.not_after == "0125"
        assert date.when is None
        assert date.text == "frühes II"


class TestHgvLinks:
    def test_preserves_missing_and_one_to_many_hgv_links(self) -> None:
        ddbdp = parse_ddbdp_fixture().component
        hgv = parse_hgv_fixture()
        duplicate = hgv.model_copy(
            update={
                "component_id": "hgv:HGV_meta_EpiDoc/HGV28/27094.xml",
                "source": hgv.source.model_copy(update={"path": "HGV_meta_EpiDoc/HGV28/27094.xml"}),
            }
        )
        missing = ddbdp.model_copy(
            update={
                "component_id": "ddbdp:DDbDP/27/missing.xml",
                "identifiers": tuple(
                    identifier.model_copy(update={"value": "missing"})
                    for identifier in ddbdp.identifiers
                    if identifier.namespace.lower() == "hgv"
                ),
            }
        )

        linked = link_hgv_metadata([ddbdp, missing], [hgv, duplicate])

        assert [item.component.component_id for item in linked] == [
            ddbdp.component_id,
            missing.component_id,
        ]
        assert linked[0].hgv_components == (hgv, duplicate)
        assert linked[1].hgv_components == ()
