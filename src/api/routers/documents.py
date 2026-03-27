from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.ingestion.deduplication import content_hash

router = APIRouter(prefix="/api/v1/rag/documents", tags=["documents"])


class DocumentIngestRequest(BaseModel):
    source: str = Field(..., min_length=1)
    content: str = Field("", description="Optional raw content used for deduplication preview")
    content_type: str = Field("text")
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post("/ingest")
async def ingest_document(payload: DocumentIngestRequest, request: Request) -> dict[str, Any]:
    digest = content_hash(f"{payload.source}\n{payload.content}")
    if digest in request.app.state.document_hashes:
        raise HTTPException(status_code=409, detail="Duplicate document payload detected")

    doc_id = str(uuid4())
    request.app.state.document_catalog[doc_id] = {
        "id": doc_id,
        "source": payload.source,
        "content_hash": digest,
        "content_type": payload.content_type,
        "metadata": payload.metadata,
    }
    request.app.state.document_hashes.add(digest)
    return {"status": "accepted", "document_id": doc_id}


@router.get("")
async def list_documents(request: Request) -> dict[str, Any]:
    docs = list(request.app.state.document_catalog.values())
    return {"count": len(docs), "documents": docs}


@router.delete("/{document_id}")
async def delete_document(document_id: str, request: Request) -> dict[str, str]:
    removed = request.app.state.document_catalog.pop(document_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if removed.get("content_hash"):
        request.app.state.document_hashes.discard(removed["content_hash"])
    return {"status": "deleted", "document_id": document_id}
