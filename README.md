# enterprise-rag

Unified RAG repo created from `enterprise-ai-systems` for the four retrieval tracks in `REWORK_PLAN.md`: hybrid, multimodal, SQL, and domain-specific retrieval.

## Included sources

- `src/generation`, `src/ingestion`, `src/retrieval`, `src/evaluation`, `src/ui` from `Enterprise-RAG`
- `src/multimodal` and `src/extraction` from `MultiModal-RAG`
- `src/datachat` from `DataChat-RAG`
- `src/fraud_docs_rag` and `frontend/` from `fraud-docs-rag`
- `src/core` and `shared/` from the monorepo shared modules

## Unified API shell

Run:

```bash
uvicorn src.api.main:app --reload
```

Key routes:

- `POST /api/v1/rag/query`
- `POST /api/v1/rag/documents/ingest`
- `GET /api/v1/rag/documents`
- `POST /api/v1/rag/evaluation/run`
- `GET /health`

The shell is intentionally lightweight during migration. It exposes the new repo layout and backend selection without forcing all heavyweight model services to initialize on startup.

