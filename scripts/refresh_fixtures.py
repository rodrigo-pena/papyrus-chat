"""Refresh the upstream-derived test fixtures from a pinned or chosen idp.data ref.

The default suite stays offline: fixtures are committed files pinned to the
commit recorded in tests/fixtures/idp.data/PROVENANCE.md. This script is an
informed-user maintenance tool (network access required, never run by tests).

Usage:
    uv run python scripts/refresh_fixtures.py            # pinned commit
    uv run python scripts/refresh_fixtures.py --ref master
    uv run python scripts/refresh_fixtures.py --ref <commit-sha>
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = REPO_ROOT / "tests" / "fixtures" / "idp.data" / "PROVENANCE.md"
DEFAULT_REF = "04568cb5ea3775d8113bb6e7edfd9c7168cf7e88"
UPSTREAM_REPO = "papyri/idp.data"

FIXTURE_PATHS = (
    "DCLP/23/23944.xml",
    "DCLP/23/23702.xml",
    "Translations/3/3227-1.xml",
    "Translations/3/3643-1.xml",
)

MARKER = "- **Pinned upstream commit:** "


def resolve_ref(ref: str) -> str:
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
        return ref.lower()
    url = f"https://api.github.com/repos/{UPSTREAM_REPO}/commits/{ref}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return str(json.load(response)["sha"])


def fetch(path: str, commit: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{commit}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def update_provenance(commit: str) -> None:
    text = PROVENANCE.read_text(encoding="utf-8")
    if MARKER not in text:
        raise SystemExit(f"Marker {MARKER!r} not found in {PROVENANCE}")
    PROVENANCE.write_text(
        text.replace(MARKER + "`" + DEFAULT_REF + "`", MARKER + f"`{commit}`", 1),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref",
        default=DEFAULT_REF,
        help="Git ref to fetch from: commit SHA, branch, or tag (default: pinned commit).",
    )
    args = parser.parse_args()

    commit = resolve_ref(args.ref)
    print(f"Fetching {len(FIXTURE_PATHS)} fixtures at {commit}")
    for path in FIXTURE_PATHS:
        destination = PROVENANCE.parent / path
        destination.write_bytes(fetch(path, commit))
        print(f"  updated {path}")
    update_provenance(commit)
    print(f"PROVENANCE.md pinned commit updated to {commit}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - CLI boundary message
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
