from fastapi import FastAPI

from src.api.routers.documents import router as documents_router
from src.api.routers.evaluation import router as evaluation_router
from src.api.routers.query import router as query_router

app = FastAPI(
    title="enterprise-rag",
    version="0.1.0",
    description="Unified RAG platform shell combining hybrid, multimodal, SQL, and domain workflows.",
)

app.state.document_catalog = {}
app.state.document_hashes = set()
app.state.evaluations = {}

app.include_router(query_router)
app.include_router(documents_router)
app.include_router(evaluation_router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, object]:
    return {
        "status": "healthy",
        "repo": "enterprise-rag",
        "modes": ["hybrid", "multimodal", "sql", "domain"],
        "documents": len(app.state.document_catalog),
        "evaluations": len(app.state.evaluations),
    }


@app.get("/metrics", tags=["system"])
async def metrics() -> dict[str, object]:
    return {
        "repo": "enterprise-rag",
        "document_count": len(app.state.document_catalog),
        "evaluation_count": len(app.state.evaluations),
        "evaluation_backend": "ragas",
    }
