"""Tests pinning the logical content hash canonicalization."""

from papyrus_chat.artifact.hashing import logical_content_hash
from papyrus_chat.artifact.records import (
    DocumentRecord,
    IdentifierRecord,
    PassageRecord,
    SourceReference,
)

REPO = "https://github.com/papyri/idp.data.git"
COMMIT = "04568cb5ea3775d8113bb6e7edfd9c7168cf7e88"


def doc(path: str, title: str) -> DocumentRecord:
    return DocumentRecord(
        document_id=f"dclp:{path}",
        collection="dclp",
        title=title,
        languages=["grc"],
        metadata={},
        source=SourceReference(repository_url=REPO, commit=COMMIT, path=path),
    )


def passage(document_id: str) -> PassageRecord:
    return PassageRecord(
        passage_id=f"{document_id}#edition:1:",
        document_id=document_id,
        kind="edition",
        sequence=1,
        display_text="ἔτους",
        search_text="ετουσ",
        source=SourceReference(
            repository_url=REPO, commit=COMMIT, path=document_id.removeprefix("dclp:")
        ),
    )


def full_hash(documents, passages, identifiers=()) -> str:
    return logical_content_hash(
        schema_version=1,
        builder_version="0.1.0",
        source_url=REPO,
        resolved_commit=COMMIT,
        collections=["dclp"],
        documents=documents,
        passages=passages,
        identifiers=list(identifiers),
    )


def test_hash_is_prefixed_sha256() -> None:
    documents = [doc("DCLP/23/23702.xml", "Sb. 20 14258")]

    assert full_hash(documents, []).startswith("sha256:")


def test_same_inputs_same_hash_regardless_of_order() -> None:
    documents = [doc("DCLP/23/23702.xml", "Sb. 20 14258"), doc("DCLP/23/23944.xml", "Horoscope")]
    passages = [passage("dclp:DCLP/23/23944.xml")]
    identifiers = [
        IdentifierRecord(document_id="dclp:DCLP/23/23944.xml", namespace="TM", value="23944")
    ]

    reordered = list(reversed(documents))

    assert full_hash(documents, passages, identifiers) == full_hash(
        reordered, list(reversed(passages)), list(reversed(identifiers))
    )


def test_changing_any_record_changes_hash() -> None:
    documents = [doc("DCLP/23/23702.xml", "Sb. 20 14258")]

    changed_title = [doc("DCLP/23/23702.xml", "Different Title")]

    assert full_hash(documents, []) != full_hash(changed_title, [])


def test_changing_source_commit_changes_hash() -> None:
    documents = [doc("DCLP/23/23702.xml", "Sb. 20 14258")]

    other = logical_content_hash(
        schema_version=1,
        builder_version="0.1.0",
        source_url=REPO,
        resolved_commit="1" * 40,
        collections=["dclp"],
        documents=documents,
        passages=[],
        identifiers=[],
    )

    assert other != full_hash(documents, [])


def test_canonical_form_is_pinned_by_golden_value() -> None:
    documents = [doc("DCLP/23/23702.xml", "Sb. 20 14258")]

    digest = full_hash(documents, [])

    assert digest == "sha256:be449e44441d206cfc450d2e5fe459e4d46d2c1cfc943518e6b64fa7ddd83e1f"
