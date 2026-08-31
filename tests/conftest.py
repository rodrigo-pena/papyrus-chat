import shutil
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, NoReturn

import pytest
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

FIXTURES = Path(__file__).parent / "fixtures" / "idp.data"
INVALID_FIXTURES = Path(__file__).parent / "fixtures" / "idp.data-invalid"


@pytest.fixture(autouse=True)
def block_real_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the default suite fail fast instead of contacting an LLM provider."""

    async def blocked_request(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise AssertionError("real model requests are disabled in the test suite")

    @asynccontextmanager
    async def blocked_request_stream(*args: Any, **kwargs: Any) -> AsyncIterator[None]:
        del args, kwargs
        raise AssertionError("real model requests are disabled in the test suite")
        yield

    for model_type in (OpenAIChatModel, OpenAIResponsesModel):
        monkeypatch.setattr(model_type, "request", blocked_request)
        monkeypatch.setattr(model_type, "request_stream", blocked_request_stream)


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def make_git_repo(source: Path, target: Path) -> Path:
    """Create a git checkout from a fixture directory; returns the repo path."""
    repo = target / "idp.data"
    shutil.copytree(source, repo)
    git("init", "-b", "master", cwd=repo)
    git("config", "user.email", "test@example.com", cwd=repo)
    git("config", "user.name", "Fixture Test", cwd=repo)
    git("add", ".", cwd=repo)
    git("commit", "-m", "fixture snapshot", cwd=repo)
    return repo


@pytest.fixture(scope="session")
def fixture_git_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_git_repo(FIXTURES, tmp_path_factory.mktemp("git-fixture"))


@pytest.fixture(scope="session")
def invalid_git_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_git_repo(INVALID_FIXTURES, tmp_path_factory.mktemp("git-invalid"))


@pytest.fixture(scope="session")
def corpus_artifact(tmp_path_factory: pytest.TempPathFactory, fixture_git_repo: Path) -> Path:
    """A real corpus artifact built once per session from the fixture repo."""
    from papyrus_chat.builder.pipeline import build_artifact
    from papyrus_chat.builder.source import LocalGitSource

    output = tmp_path_factory.mktemp("corpus") / "papyrus-corpus"
    result = build_artifact(
        ["dclp", "translations"],
        output=output,
        source=LocalGitSource(fixture_git_repo),
        source_url="https://github.com/papyri/idp.data.git",
        requested_ref="master",
    )
    return result.output_dir
