"""Transport-neutral corpus service lifecycle tests."""

import sqlite3
from pathlib import Path

import pytest

from papyrus_chat.corpus import CorpusService


def test_open_binds_a_read_only_sqlite_connection(corpus_artifact: Path) -> None:
    service = CorpusService.open(corpus_artifact)

    try:
        assert service.artifact_root == corpus_artifact.resolve()
        assert service.manifest.collections == ["dclp", "translations"]
        with pytest.raises(sqlite3.OperationalError):
            service._connection.execute("CREATE TABLE should_not_exist (value TEXT)")
    finally:
        service.close()


def test_close_releases_the_service_connection(corpus_artifact: Path) -> None:
    service = CorpusService.open(corpus_artifact)

    service.close()

    with pytest.raises(sqlite3.ProgrammingError):
        service._connection.execute("SELECT 1")
