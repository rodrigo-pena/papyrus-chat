"""Reproducibility: identical inputs must yield identical logical hashes (SPEC 7.2)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from papyrus_chat.builder.cli import app

FIXTURES = Path(__file__).parent.parent / "fixtures" / "idp.data"
PINNED_COMMIT = "04568cb5ea3775d8113bb6e7edfd9c7168cf7e88"

runner = CliRunner()


def build(tmp_path: Path, name: str) -> Path:
    result = runner.invoke(
        app,
        [
            "dclp",
            "translations",
            "--output",
            str(tmp_path / name),
            "--source",
            str(FIXTURES),
            "--ref",
            PINNED_COMMIT,
        ],
    )
    assert result.exit_code == 0, result.output
    return tmp_path / name


def manifest_hash(root: Path) -> str:
    return str(json.loads((root / "manifest.json").read_text())["logical_content_hash"])


def test_repeated_builds_produce_identical_hash(tmp_path: Path) -> None:
    first = build(tmp_path, "first")
    second = build(tmp_path, "second")

    assert manifest_hash(first) == manifest_hash(second)
    assert manifest_hash(first).startswith("sha256:")


def test_changed_input_changes_hash(tmp_path: Path) -> None:
    first = build(tmp_path, "first")

    mutated_source = tmp_path / "source"
    mutated_source.mkdir()
    for item in FIXTURES.rglob("*"):
        if item.is_file():
            target = mutated_source / item.relative_to(FIXTURES)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())
    extra = mutated_source / "DCLP" / "23" / "extra.xml"
    extra.write_text(
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
        "<titleStmt><title>Extra</title></titleStmt></fileDesc></teiHeader></TEI>",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "dclp",
            "--output",
            str(tmp_path / "second"),
            "--source",
            str(mutated_source),
            "--ref",
            PINNED_COMMIT,
        ],
    )
    assert result.exit_code == 0, result.output

    assert manifest_hash(tmp_path / "second") != manifest_hash(first)


def test_report_includes_logical_hash(tmp_path: Path) -> None:
    output = build(tmp_path, "corpus")

    result = runner.invoke(
        app,
        [
            "dclp",
            "translations",
            "--output",
            str(tmp_path / "again"),
            "--source",
            str(FIXTURES),
            "--ref",
            PINNED_COMMIT,
        ],
    )

    assert result.exit_code == 0
    assert manifest_hash(output) in result.output
