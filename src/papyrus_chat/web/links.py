"""Canonical external links for corpus records (SPEC 10)."""

from collections.abc import Iterable


def papyri_info_url(identifiers: Iterable[tuple[str, str]]) -> str | None:
    """Return the papyri.info canonical URL for the first TM identifier."""
    for namespace, value in identifiers:
        if namespace.lower() == "tm":
            stripped = value.strip()
            if stripped:
                return f"https://papyri.info/current/{stripped}"
    return None
