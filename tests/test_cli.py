from typer.testing import CliRunner

from papyrus_chat.builder.cli import app as build_app
from papyrus_chat.chat.cli import app as chat_app

runner = CliRunner()


class TestCorpusBuildCli:
    def test_list_collections_prints_supported_names(self) -> None:
        result = runner.invoke(build_app, ["--list-collections"])

        assert result.exit_code == 0
        assert result.output.splitlines() == ["dclp", "translations"]

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
