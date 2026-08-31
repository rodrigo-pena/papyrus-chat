"""URL helpers for document deep links."""

from urllib.parse import quote


def document_url(document_id: str) -> str:
    """Percent-encoded document route, safe for path segments and hrefs."""
    return f"/documents/{quote(document_id, safe='')}"
