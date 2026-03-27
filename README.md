# Enterprise RAG Platform (`enterprise-rag`)

Enterprise Retrieval-Augmented Generation (RAG) platform for **hybrid search**, **multimodal document intelligence**, **SQL-assisted retrieval**, and **domain-aware question answering**. This repository is designed for production-focused AI search workloads that need high recall, traceable citations, and API-first integration.

## Why This Repository

Organizations need one system that can search text, images, and tables while supporting analytics-style questions. `enterprise-rag` unifies these capabilities behind a single service.

## Core Features

- Hybrid retrieval pipeline (sparse + dense + reranking)
- Multimodal retrieval support for document and image workflows
- SQL retrieval mode for data-backed natural-language questions
- Domain-specific retrieval flow for specialized corpora
- Unified FastAPI service with health and metrics endpoints
- Built-in evaluation surfaces for RAG quality checks

## Project Structure

- `src/api/`: unified FastAPI service and route modules
- `src/retrieval/`: hybrid, multimodal, SQL, and domain retrievers
- `src/ingestion/`: document/image/table processing and deduplication
- `src/generation/`: response orchestration and citation tracking
- `src/embedding/`: text and CLIP embedding interfaces
- `src/evaluation/`: evaluator scaffolding and metrics utilities
- `src/core/`: auth, security, error handling, rate limiting, secrets

## API Endpoints

- `POST /api/v1/rag/query`
- `POST /api/v1/rag/documents/ingest`
- `GET /api/v1/rag/documents`
- `DELETE /api/v1/rag/documents/{document_id}`
- `POST /api/v1/rag/evaluation/run`
- `GET /api/v1/rag/evaluation/metrics`
- `GET /health`
- `GET /metrics`

## Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## GLM Configuration

```bash
export GLM_API_KEY=your_glm_api_key
export GLM_BASE_URL=https://api.z.ai/api/anthropic
export GLM_MODEL=glm-5.1
export ZHIPUAI_API_KEY=$GLM_API_KEY
```

## SEO Keywords

enterprise rag, hybrid rag, multimodal rag, sql rag, retrieval augmented generation, document ai search, rag evaluation, fastapi rag service
