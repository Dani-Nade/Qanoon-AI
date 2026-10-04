"""Case strength estimation endpoint (newline-delimited JSON stream)."""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from qanoon_ai.api.routes.query import answer_service
from qanoon_ai.chat.case_strength import CaseStrengthService

router = APIRouter(tags=["case-strength"])
# Share the backend selector and loaded retrieval models with /query and /chat.
case_strength_service = CaseStrengthService(retriever=answer_service)


class CaseStrengthRequest(BaseModel):
    facts: str = Field(..., min_length=15, max_length=8000)
    jurisdiction: str | None = None


@router.post("/case-strength")
def case_strength(request: CaseStrengthRequest) -> StreamingResponse:
    def events():
        for event in case_strength_service.stream(request.facts, request.jurisdiction or None):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
