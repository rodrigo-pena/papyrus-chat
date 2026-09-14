"""Conservative, provider-independent request estimates with usage anchors."""

import json
from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_core import to_jsonable_python


def estimate_text(text: str) -> int:
    """Estimate from UTF-8 bytes, avoiding the ASCII bias of chars/4 for Greek."""
    return (len(text.encode("utf-8")) + 1) // 2


def message_text(message: ModelMessage) -> str:
    return ModelMessagesTypeAdapter.dump_json([message]).decode()


def parameter_text(parameters: ModelRequestParameters) -> str:
    def tool_payload(tool: ToolDefinition) -> dict[str, object]:
        # Return schemas and framework metadata are not part of function-tool requests.
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_json_schema,
            "strict": tool.strict,
        }

    return json.dumps(
        {
            "tools": [tool_payload(tool) for tool in parameters.function_tools],
            "native_tools": to_jsonable_python(parameters.native_tools),
            "output_tools": [tool_payload(tool) for tool in parameters.output_tools],
            "output_object": to_jsonable_python(parameters.output_object),
            "instructions": [part.content for part in parameters.instruction_parts or []],
        },
        ensure_ascii=False,
    )


@dataclass
class RequestAccounting:
    prefix: tuple[str, ...] = ()
    parameters: str = ""
    input_tokens: int = 0

    def estimate(self, messages: list[ModelMessage], parameters: ModelRequestParameters) -> int:
        rendered = tuple(message_text(message) for message in messages)
        config = parameter_text(parameters)
        raw = 128 + estimate_text(config) + sum(estimate_text(text) for text in rendered)
        if (
            self.input_tokens > 0
            and config == self.parameters
            and rendered[: len(self.prefix)] == self.prefix
        ):
            anchored = self.input_tokens + sum(
                estimate_text(text) for text in rendered[len(self.prefix) :]
            )
            return max(raw, anchored)
        return raw

    def anchor(
        self, messages: list[ModelMessage], parameters: ModelRequestParameters, input_tokens: int
    ) -> None:
        self.prefix = tuple(message_text(message) for message in messages)
        self.parameters = parameter_text(parameters)
        self.input_tokens = input_tokens
