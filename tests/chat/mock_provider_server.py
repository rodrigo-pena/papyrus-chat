"""A real local mock OpenAI-compatible provider for offline tests (SPEC 12.2)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockProviderServer:
    """Records requests and replies with a configurable completion response."""

    def __init__(
        self,
        status: int = 200,
        content: str = "mock answer",
        response_body: dict | None = None,
    ) -> None:
        self.status = status
        self.content = content
        self.response_body = response_body
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": body,
                    }
                )
                body = (
                    outer.response_body
                    if outer.response_body is not None
                    else {"choices": [{"message": {"role": "assistant", "content": outer.content}}]}
                )
                payload = json.dumps(body).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass  # silence stderr

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "MockProviderServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
