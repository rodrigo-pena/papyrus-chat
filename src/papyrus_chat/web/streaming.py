"""Validated final-answer streaming for the Pydantic AI web UI."""

from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.messages import TextPart, TextPartDelta
from pydantic_ai.ui._web.api import ChatRequestExtra, validate_request_options
from pydantic_ai.ui.vercel_ai import VercelAIAdapter, VercelAIEventStream
from pydantic_ai.ui.vercel_ai._event_stream import BaseChunk
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

JSON_MEDIA_TYPE = "application/json"
SDK_VERSION = 7


class ValidatedAnswerEventStream(VercelAIEventStream[Any, str]):
    """Stream tool activity immediately and text only after output validation."""

    async def handle_text_start(
        self, part: TextPart, follows_text: bool = False
    ) -> AsyncIterator[BaseChunk]:
        del part, follows_text
        return
        yield

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseChunk]:
        del delta
        return
        yield

    async def handle_text_end(
        self, part: TextPart, followed_by_text: bool = False
    ) -> AsyncIterator[BaseChunk]:
        del part, followed_by_text
        return
        yield

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[BaseChunk]:
        async for chunk in super().handle_run_result(event):
            yield chunk

        output = event.result.output
        if not isinstance(output, str):
            return
        final_part = TextPart(output)
        async for chunk in super().handle_text_start(final_part):
            yield chunk
        async for chunk in super().handle_text_end(final_part):
            yield chunk


class ValidatedVercelAIAdapter(VercelAIAdapter[Any, str]):
    """Use the validated-answer event stream for Vercel AI responses."""

    def build_event_stream(self) -> ValidatedAnswerEventStream:
        return ValidatedAnswerEventStream(
            self.run_input,
            accept=self.accept,
            sdk_version=self.sdk_version,
            server_message_id=self.server_message_id,
        )


def install_validated_chat_route(app: Any, agent: Agent[Any, str], deps: Any) -> None:
    """Replace the stock chat POST route without dropping its outer host guard."""
    api_mounts = [
        route for route in app.routes if isinstance(route, Mount) and route.path == "/api"
    ]
    if len(api_mounts) != 1:
        raise RuntimeError("Expected one stock Pydantic AI /api mount")

    api_app = api_mounts[0].app
    routes = list(api_app.routes)
    matches = [
        (index, route)
        for index, route in enumerate(routes)
        if isinstance(route, Route) and route.path == "/chat" and route.methods == {"POST"}
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one stock Pydantic AI POST /api/chat route")

    model = agent.model
    model_ids = (
        {model if isinstance(model, str) else model.model_id} if model is not None else set()
    )

    async def post_chat(request: Request) -> Response:
        media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if media_type != JSON_MEDIA_TYPE:
            return JSONResponse(
                {
                    "error": (
                        f"Expected `Content-Type: {JSON_MEDIA_TYPE}`, "
                        f"got {media_type or 'no content type'}"
                    )
                },
                status_code=415,
            )

        adapter = await ValidatedVercelAIAdapter.from_request(
            request, agent=agent, sdk_version=SDK_VERSION
        )
        extra_data = ChatRequestExtra.model_validate(adapter.run_input.__pydantic_extra__)
        if error := validate_request_options(extra_data, model_ids, set()):
            return JSONResponse({"error": error}, status_code=400)

        return await ValidatedVercelAIAdapter.dispatch_request(
            request,
            agent=agent,
            sdk_version=SDK_VERSION,
            deps=deps,
        )

    index, _route = matches[0]
    routes[index] = Route("/chat", post_chat, methods=["POST"])
    api_app.router.routes[:] = routes
