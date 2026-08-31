import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "idp.data"
INVALID_FIXTURES = Path(__file__).parent / "fixtures" / "idp.data-invalid"


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
