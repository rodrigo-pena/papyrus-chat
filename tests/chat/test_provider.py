"""Provider configuration and OpenAI-compatible client (SPEC 9.2)."""

import socket

import pytest

from papyrus_chat.chat.provider import (
    ProviderClient,
    ProviderConfig,
    ProviderError,
    load_provider_config,
)
from tests.chat.mock_provider_server import MockProviderServer

API_KEY = "sk-test-key-12345"


class TestConfiguration:
    def test_missing_base_url_names_the_variable(self) -> None:
        with pytest.raises(ProviderError, match="LLM_BASE_URL"):
            load_provider_config({"LLM_MODEL": "m"})

    def test_missing_model_names_the_variable(self) -> None:
        with pytest.raises(ProviderError, match="LLM_MODEL"):
            load_provider_config({"LLM_BASE_URL": "https://x.example/v1"})

    def test_api_key_is_optional(self) -> None:
        config = load_provider_config({"LLM_BASE_URL": "https://x.example/v1", "LLM_MODEL": "m"})

        assert config.api_key is None

    def test_trailing_slash_is_tolerated(self) -> None:
        config = ProviderConfig(base_url="https://x.example/v1/", model="m")

        assert config.base_url == "https://x.example/v1"

    def test_error_messages_do_not_leak_secrets(self) -> None:
        config = ProviderConfig(base_url="https://x.example", model="m", api_key=API_KEY)

        assert API_KEY not in repr(config)
        assert API_KEY not in str(config)


class TestClient:
    def test_happy_path_returns_content(self) -> None:
        with MockProviderServer() as mock:
            client = ProviderClient(
                load_provider_config(
                    {
                        "LLM_BASE_URL": mock.base_url,
                        "LLM_MODEL": "test-model",
                        "LLM_API_KEY": API_KEY,
                    }
                )
            )

            answer = client.complete([{"role": "user", "content": "What is this papyrus?"}])

        assert answer == "mock answer"
        request = mock.requests[0]
        assert request["path"] == "/v1/chat/completions"
        assert request["authorization"] == f"Bearer {API_KEY}"
        assert request["body"]["model"] == "test-model"
        assert request["body"]["messages"][0]["content"].endswith("papyrus?")

    def test_no_authorization_header_without_api_key(self) -> None:
        with MockProviderServer() as mock:
            client = ProviderClient(
                load_provider_config({"LLM_BASE_URL": mock.base_url, "LLM_MODEL": "m"})
            )

            client.complete([{"role": "user", "content": "hi"}])

        assert mock.requests[0]["authorization"] is None

    def test_connection_failure_is_actionable(self) -> None:
        # Bind then close a socket to obtain a guaranteed-closed port
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
        probe.close()

        client = ProviderClient(ProviderConfig(base_url=f"http://127.0.0.1:{dead_port}", model="m"))

        with pytest.raises(ProviderError) as excinfo:
            client.complete([{"role": "user", "content": "hi"}])

        message = str(excinfo.value)
        assert "reach" in message.lower() or "connect" in message.lower()
        assert API_KEY not in message

    def test_auth_failure_is_actionable(self) -> None:
        with MockProviderServer(status=401) as mock:
            client = ProviderClient(
                load_provider_config(
                    {"LLM_BASE_URL": mock.base_url, "LLM_MODEL": "m", "LLM_API_KEY": "bad"}
                )
            )

            with pytest.raises(ProviderError, match="[Aa]uth"):
                client.complete([{"role": "user", "content": "hi"}])

    def test_server_error_is_actionable(self) -> None:
        with MockProviderServer(status=500) as mock:
            client = ProviderClient(
                load_provider_config({"LLM_BASE_URL": mock.base_url, "LLM_MODEL": "m"})
            )

            with pytest.raises(ProviderError) as excinfo:
                client.complete([{"role": "user", "content": "hi"}])

            message = str(excinfo.value)
            assert "500" in message

    def test_empty_content_is_rejected(self) -> None:
        with MockProviderServer(
            response_body={"choices": [{"message": {"content": "   "}}]}
        ) as mock:
            client = ProviderClient(
                load_provider_config({"LLM_BASE_URL": mock.base_url, "LLM_MODEL": "m"})
            )

            with pytest.raises(ProviderError, match="empty"):
                client.complete([{"role": "user", "content": "hi"}])
