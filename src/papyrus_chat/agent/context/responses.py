"""Preserve incomplete-response finish reasons with Pydantic AI 2.36.

The pinned Responses streaming adapter maps incomplete_details only on a
ResponseCompletedEvent, although OpenAI sends ResponseIncompleteEvent. Normalize
the event class, retaining the original status, reason, output, and usage. Remove
this compatibility shim once the upstream incomplete-event branch maps reasons.
"""

from collections.abc import AsyncIterator
from typing import Any, cast

from openai.types.responses import ResponseCompletedEvent, ResponseIncompleteEvent
from pydantic_ai.models.openai import OpenAIResponsesModel


class TerminalReasonStream:
    def __init__(self, stream: Any):
        self.stream = stream

    async def __aiter__(self) -> AsyncIterator[Any]:
        async for event in self.stream:
            if isinstance(event, ResponseIncompleteEvent):
                yield ResponseCompletedEvent(
                    type="response.completed",
                    response=event.response,
                    sequence_number=event.sequence_number,
                )
            else:
                yield event

    async def close(self) -> None:
        await self.stream.close()


class RecoverableResponsesModel(OpenAIResponsesModel):
    async def _process_streamed_response(self, response: Any, *args: Any, **kwargs: Any):
        return await super()._process_streamed_response(
            cast(Any, TerminalReasonStream(response)), *args, **kwargs
        )
