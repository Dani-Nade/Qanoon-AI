"""Document upload, reading status, review and deletion."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from qanoon_ai.api.routes.chat import chat_service
from qanoon_ai.documents.explain import DocumentExplainer
from qanoon_ai.documents.render import UnsupportedDocument
from qanoon_ai.documents.service import DocumentService
from qanoon_ai.documents.store import DocumentNotFound

router = APIRouter(prefix="/documents", tags=["documents"])
document_service = DocumentService()
# Shares the chat model client and the loaded retrieval models with /chat.
explainer = DocumentExplainer(chat_service)


class BlockCorrection(BaseModel):
    index: int = Field(..., ge=0)
    text: str = Field(..., max_length=20000)


class PageCorrections(BaseModel):
    blocks: list[BlockCorrection] = Field(..., min_length=1, max_length=500)


@router.post("", status_code=202)
async def upload_document(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    try:
        return document_service.upload(file.filename or "", data)
    except UnsupportedDocument as error:
        raise HTTPException(400, str(error))


@router.get("/{document_id}")
def get_document(document_id: str) -> dict:
    try:
        return document_service.document(document_id)
    except DocumentNotFound:
        raise HTTPException(404, "Document not found. It may have been deleted or expired after 24 hours.")


@router.get("/{document_id}/pages/{number}/image")
def get_page_image(document_id: str, number: int) -> FileResponse:
    try:
        path = document_service.store.page_image_path(document_id, number)
    except DocumentNotFound:
        raise HTTPException(404, "Document not found.")
    if number < 1 or not path.is_file():
        raise HTTPException(404, "Page not found.")
    return FileResponse(path, media_type="image/png")


@router.put("/{document_id}/pages/{number}")
def correct_page(document_id: str, number: int, corrections: PageCorrections) -> dict:
    try:
        return document_service.correct_page(document_id, number, {c.index: c.text for c in corrections.blocks})
    except DocumentNotFound:
        raise HTTPException(404, "Document or page not found.")
    except KeyError as error:
        raise HTTPException(400, f"Unknown {error.args[0]}.")


class ExplainRequest(BaseModel):
    language: Literal["english", "urdu", "roman_urdu"] = "english"


@router.post("/{document_id}/explain")
def explain_document(document_id: str, request: ExplainRequest) -> StreamingResponse:
    """Stream an explanation (same newline-delimited JSON events as /chat)."""
    try:
        document = document_service.document(document_id)
    except DocumentNotFound:
        raise HTTPException(404, "Document not found. It may have been deleted or expired after 24 hours.")
    if document["status"] != "ready":
        raise HTTPException(409, "The document is still being read." if document["status"] == "processing" else "The document could not be read.")

    def events():
        for event in explainer.stream(document["pages"], request.language):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str) -> Response:
    try:
        document_service.store.delete(document_id)
    except DocumentNotFound:
        raise HTTPException(404, "Document not found.")
    return Response(status_code=204)
