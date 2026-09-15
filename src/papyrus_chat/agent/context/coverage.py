"""Deterministic coverage of source material delivered for review."""

from .evidence import EvidenceLedger
from .progress import progress_overview


def coverage_note(ledger: EvidenceLedger) -> str:
    if not any("web" not in record.tool_name for record in ledger.records):
        return ""
    counts = progress_overview(ledger.progress())
    note = (
        f"Coverage: {counts['candidate_documents']} unique candidate documents retrieved; "
        f"{counts['excerpt_documents']} documents with inspected excerpts; "
        f"{counts['full_text_documents']} documents with all stored passage text retrieved. "
        f"Result pages outstanding in {counts['searches_with_unread_results']} searches."
    )
    if unknown := counts["searches_with_unknown_coverage"]:
        note += f" Pagination coverage is unknown for {unknown} searches."
    note += " This measures material delivered for review, not exhaustive thematic discovery."
    return note
