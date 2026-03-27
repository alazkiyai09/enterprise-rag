from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/rag/evaluation", tags=["evaluation"])


class EvaluationRunRequest(BaseModel):
    mode: Literal["hybrid", "multimodal", "sql", "domain"] = "hybrid"
    dataset_name: str = Field(..., min_length=1)


@router.post("/run")
async def run_evaluation(payload: EvaluationRunRequest, request: Request) -> dict[str, str]:
    evaluation_id = str(uuid4())
    request.app.state.evaluations[evaluation_id] = {
        "status": "accepted",
        "dataset_name": payload.dataset_name,
        "mode": payload.mode,
    }
    return {
        "status": "accepted",
        "evaluation_id": evaluation_id,
        "dataset_name": payload.dataset_name,
        "mode": payload.mode,
    }


@router.get("/metrics")
async def evaluation_metrics() -> dict[str, object]:
    return {
        "framework": "ragas",
        "available_metrics": ["faithfulness", "answer_relevancy", "context_precision"],
    }


@router.get("/{evaluation_id}")
async def get_evaluation(evaluation_id: str, request: Request) -> dict[str, object]:
    return request.app.state.evaluations.get(evaluation_id, {"status": "not_found"})
