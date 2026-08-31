"""User-facing builder errors: concise, actionable."""

from __future__ import annotations


class BuildError(Exception):
    """A build failure with a concise, actionable message."""
