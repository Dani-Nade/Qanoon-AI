"""Legal query endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from qanoon_ai.retrieval.service import LegalAnswerService
from qanoon_ai.schemas.query import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])
answer_service = LegalAnswerService()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return answer_service.answer(request)
