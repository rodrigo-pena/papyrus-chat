from papyrus_chat.artifact.records import DocumentRecord
from papyrus_chat.builder.content.profiles import document_profile
from tests.builder.test_chunks import CharacterTokenizer, passage
from tests.builder.test_semantic import component


def test_profile_balances_source_languages_and_retains_provenance() -> None:
    greek = passage("α" * 2000)
    french = greek.model_copy(
        update={
            "passage_id": "fr",
            "kind": "translation",
            "language": "fr",
            "display_text": "é" * 2000,
        }
    )
    document = DocumentRecord(
        document_id="d",
        collection="dclp",
        title="Letter",
        languages=["grc"],
        metadata={"authority": "BOILERPLATE"},
        source=greek.source,
    )
    hgv = component("hgv", document_id=None, subjects=("Brief",))
    tokenizer = CharacterTokenizer()
    profile = document_profile(document, [greek, french], [hgv], tokenizer)
    assert profile == document_profile(document, [french, greek], [hgv], tokenizer)
    assert tokenizer.count(profile.profile_text) <= 448
    assert "BOILERPLATE" not in profile.profile_text
    assert "Brief" in profile.profile_text
    assert "α" * 50 in profile.profile_text
    assert "é" * 50 in profile.profile_text
    assert profile.component_ids == ("hgv",)
    assert {ref.passage_id for ref in profile.passages} == {"p", "fr"}
    for ref in profile.passages:
        assert ref.char_start == 0
        assert 0 < ref.char_end < 2000
    assert not profile.metadata_only


def test_metadata_only_profile_is_bounded_and_explicit() -> None:
    document = DocumentRecord(
        document_id="d",
        collection="dclp",
        title="Title " * 200,
        languages=[],
        metadata={},
        source=passage("x").source,
    )
    profile = document_profile(document, [], [], CharacterTokenizer())
    assert profile.metadata_only
    assert len(profile.profile_text) <= 128
    assert profile.passages == ()


def test_source_order_within_language_is_deterministic() -> None:
    first = passage("First")
    second = first.model_copy(update={"passage_id": "p2", "sequence": 2, "display_text": "Second"})
    document = DocumentRecord(
        document_id="d",
        collection="ddbdp",
        title="Letter",
        languages=["grc"],
        metadata={},
        source=first.source,
    )
    profile = document_profile(document, [second, first], [], CharacterTokenizer())
    assert profile.profile_text.index("First") < profile.profile_text.index("Second")
