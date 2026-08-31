"""User-facing builder errors (SPEC 6.4, 11): concise, actionable."""

from __future__ import annotations


class BuildError(Exception):
    """A build failure with a concise, actionable message."""
