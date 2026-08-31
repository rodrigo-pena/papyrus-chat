from pathlib import Path

import pytest
from typer.testing import CliRunner

import papyrus_chat.chat.cli as chat_cli_module
from papyrus_chat.builder.cli import app as build_app
from papyrus_chat.chat.cli import app as chat_app

runner = CliRunner()


class TestCorpusBuildCli:
    def test_list_collections_prints_supported_names(self) -> None:
        result = runner.invoke(build_app, ["--list-collections"])

        assert result.exit_code == 0
        assert result.output.splitlines() == ["dclp", "ddbdp", "translations"]

    def test_no_collection_fails_naming_supported_collections(self) -> None:
        result = runner.invoke(build_app, [])

        assert result.exit_code != 0
        assert "dclp" in result.output
        assert "translations" in result.output

    def test_unknown_collection_fails_naming_supported_collections(self) -> None:
        result = runner.invoke(build_app, ["apd"])

        assert result.exit_code != 0
        assert "apd" in result.output
        assert "dclp" in result.output


class TestChatCli:
    def test_help_exits_zero(self) -> None:
        result = runner.invoke(chat_app, ["--help"])

        assert result.exit_code == 0
        assert "artifact" in result.output.lower()

    def test_reports_startup_stages(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        artifact = tmp_path / "corpus"
        artifact.mkdir()
        monkeypatch.setattr(chat_cli_module, "validate_startup", lambda _artifact: None)
        monkeypatch.setattr(chat_cli_module, "load_app", lambda _artifact: object())
        monkeypatch.setattr(chat_cli_module.uvicorn, "run", lambda *args, **kwargs: None)

        result = runner.invoke(chat_app, ["--artifact", str(artifact), "--no-open"])

        assert result.exit_code == 0, result.output
        assert "Starting papyrus-chat" in result.output
        assert "Validating corpus artifact and provider configuration" in result.output
        assert "Loading chat application" in result.output
        assert "Serving papyrus-chat" in result.output
