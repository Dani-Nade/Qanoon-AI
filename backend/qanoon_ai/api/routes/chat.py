"""Conversational chat endpoint (newline-delimited JSON stream)."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from qanoon_ai.api.routes.query import answer_service
from qanoon_ai.chat.service import ChatService

router = APIRouter(tags=["chat"])
# Share the backend selector and loaded retrieval models with /query.
chat_service = ChatService(retriever=answer_service)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=40)
    jurisdiction: str | None = None

    @field_validator("messages")
    @classmethod
    def last_message_is_from_user(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if messages[-1].role != "user":
            raise ValueError("the last message must be from the user")
        return messages


@router.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    messages = [message.model_dump() for message in request.messages]

    def events():
        for event in chat_service.stream(messages, request.jurisdiction or None):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
