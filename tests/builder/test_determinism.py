"""Reproducibility: identical inputs must yield identical logical hashes."""

import json
import shutil
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from papyrus_chat.builder.cli import app

runner = CliRunner()


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def repo_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "master"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def build_collections(tmp_path: Path, name: str, repo: Path) -> Path:
    result = runner.invoke(
        app,
        [
            "dclp",
            "translations",
            "--output",
            str(tmp_path / name),
            "--source",
            str(repo),
            "--ref",
            "master",
        ],
    )
    assert result.exit_code == 0, result.output
    return tmp_path / name


def manifest_hash(root: Path) -> str:
    return str(json.loads((root / "manifest.json").read_text())["logical_content_hash"])


def test_repeated_builds_produce_identical_hash(tmp_path: Path, fixture_git_repo: Path) -> None:
    first = build_collections(tmp_path, "first", fixture_git_repo)
    second = build_collections(tmp_path, "second", fixture_git_repo)

    assert manifest_hash(first) == manifest_hash(second)
    assert manifest_hash(first).startswith("sha256:")


def test_changed_input_changes_hash(tmp_path: Path, fixture_git_repo: Path) -> None:
    first = build_collections(tmp_path, "first", fixture_git_repo)

    mutated = tmp_path / "mutated-source"
    shutil.copytree(fixture_git_repo, mutated, ignore=shutil.ignore_patterns(".git"))
    git("init", "-b", "master", cwd=mutated)
    git("config", "user.email", "test@example.com", cwd=mutated)
    git("config", "user.name", "Fixture Test", cwd=mutated)
    (mutated / "DCLP" / "23" / "extra.xml").write_text(
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
        "<titleStmt><title>Extra</title></titleStmt></fileDesc></teiHeader></TEI>",
        encoding="utf-8",
    )
    git("add", ".", cwd=mutated)
    git("commit", "-m", "add extra record", cwd=mutated)

    second = build_collections(tmp_path, "second", mutated)

    assert manifest_hash(second) != manifest_hash(first)
    assert repo_sha(mutated) != repo_sha(fixture_git_repo)


def test_report_includes_logical_hash(tmp_path: Path, fixture_git_repo: Path) -> None:
    output = build_collections(tmp_path, "corpus", fixture_git_repo)

    result = runner.invoke(
        app,
        [
            "dclp",
            "translations",
            "--output",
            str(tmp_path / "again"),
            "--source",
            str(fixture_git_repo),
            "--ref",
            "master",
        ],
    )

    assert result.exit_code == 0
    assert manifest_hash(output) in result.output
