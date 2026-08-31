"""Optional network smoke test against the real upstream repository.

Excluded from the default suite (pytest marker `network`); run explicitly:

    uv run pytest -m network
"""

import shutil
from pathlib import Path

import pytest

from papyrus_chat.artifact.validation import validate_artifact
from papyrus_chat.builder.pipeline import build_artifact
from papyrus_chat.builder.source import RemoteGitSource

UPSTREAM = "https://github.com/papyri/idp.data.git"


@pytest.mark.network
def test_remote_build_from_real_upstream(tmp_path: Path) -> None:
    source = RemoteGitSource(UPSTREAM, cache_dir=tmp_path / "cache")
    result = build_artifact(
        ["dclp"],
        output=tmp_path / "corpus",
        source=source,
        source_url=UPSTREAM,
        requested_ref="master",
    )

    validate_artifact(result.output_dir)
    assert result.documents > 0
    assert result.logical_content_hash.startswith("sha256:")

    # Cache removal must not affect the artifact
    shutil.rmtree(tmp_path / "cache")
    validate_artifact(result.output_dir)
