from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.retrieval.sql_retriever import SQLRetriever

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

RagMode = Literal["hybrid", "multimodal", "sql", "domain"]

MODE_BACKENDS = {
    "hybrid": [
        "src.retrieval.hybrid_retriever.HybridRetriever",
        "src.generation.rag_chain.RAGChain",
    ],
    "multimodal": [
        "src.multimodal.multimodal_retriever.MultiModalRetriever",
        "src.multimodal.multimodal_rag.MultiModalRAGChain",
    ],
    "sql": [
        "src.datachat.routers.query_router.QueryRouter",
        "src.datachat.core.rag_chain.DataChatRAG",
    ],
    "domain": [
        "src.fraud_docs_rag.retrieval.hybrid_retriever.HybridRetriever",
        "src.fraud_docs_rag.generation.rag_chain.RAGChain",
    ],
}


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    mode: RagMode = Field(..., description="Retrieval mode to use")
    top_k: int = Field(5, ge=1, le=25)
    filters: dict[str, Any] = Field(default_factory=dict)
    preview_only: bool = Field(True, description="Avoid heavyweight runtime execution during bootstrap.")


class QueryResponse(BaseModel):
    mode: RagMode
    status: str
    question: str
    top_k: int
    preview_only: bool
    backends: list[str]
    routing: dict[str, Any] = Field(default_factory=dict)
    message: str


def _classify_domain(question: str) -> str:
    text = question.lower()
    if any(token in text for token in ["aml", "anti money laundering", "sar", "suspicious activity"]):
        return "aml"
    if any(token in text for token in ["kyc", "know your customer", "customer due diligence", "cdd"]):
        return "kyc"
    if any(token in text for token in ["fraud", "chargeback", "transaction monitoring", "scam"]):
        return "fraud"
    return "general"


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    routing: dict[str, Any] = {}

    if request.mode == "sql":
        try:
            routing["sql_preview"] = SQLRetriever().classify(request.question).__dict__
        except Exception as exc:
            routing["sql_preview_error"] = str(exc)
    elif request.mode == "domain":
        routing["domain_category"] = _classify_domain(request.question)
    else:
        routing["filters"] = request.filters

    return QueryResponse(
        mode=request.mode,
        status="accepted",
        question=request.question,
        top_k=request.top_k,
        preview_only=request.preview_only,
        backends=MODE_BACKENDS[request.mode],
        routing=routing,
        message=(
            "Unified routing is wired to the migrated codebase. "
            "SQL mode returns router classification today; hybrid, multimodal, and domain "
            "return backend selection and lightweight routing metadata until the heavy "
            "runtime objects are bound."
        ),
    )
