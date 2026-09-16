"""A browser transcript, deliberately independent of the model's compacted history."""

from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

NonemptyString = Annotated[str, Field(min_length=1)]

# Only strip protocol envelopes, never identically named keys inside tool evidence.
TRANSPORT_FIELDS = {"providerMetadata", "callProviderMetadata", "providerDetails", "signature"}


class MessagePart(BaseModel):
    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)

    type: NonemptyString

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        values = self.__pydantic_extra__ or {}
        if self.type in {"text", "reasoning"} and not isinstance(values.get("text"), str):
            raise ValueError("Text and reasoning parts require a text string")
        if self.type == "dynamic-tool" or self.type.startswith("tool-"):
            for field in ("toolCallId", "state"):
                if not isinstance(values.get(field), str) or not values[field]:
                    raise ValueError(f"Tool parts require {field}")
            if self.type == "dynamic-tool" and not isinstance(values.get("toolName"), str):
                raise ValueError("Dynamic tool parts require toolName")
        return self

    def shareable(self) -> dict[str, JsonValue]:
        return self.model_dump(exclude=TRANSPORT_FIELDS)


class Message(BaseModel):
    id: NonemptyString
    role: Literal["user", "assistant", "system"]
    parts: list[MessagePart]


class Conversation(BaseModel):
    id: NonemptyString
    title: NonemptyString
    messages: Annotated[list[Message], Field(min_length=1)]


class ExportRequest(BaseModel):
    format: Literal["json", "html"]
    conversation: Conversation


def export_document(conversation: Conversation) -> dict:
    """Normalize once so every download format contains the same shareable content."""
    return {
        "kind": "papyrus-chat-browser-conversation",
        "schema_version": 1,
        "exported_at": datetime.now(UTC).isoformat(),
        "conversation": {
            "id": conversation.id,
            "title": conversation.title,
            "messages": [
                {
                    "id": message.id,
                    "role": message.role,
                    "parts": [part.shareable() for part in message.parts],
                }
                for message in conversation.messages
            ],
        },
    }
