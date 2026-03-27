<div align="center">

# 🔍 Enterprise RAG

### Hybrid Retrieval • Multimodal Search • SQL RAG • Domain Intelligence

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0C4C97?style=flat&logo=langchain)](https://www.langchain.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-4A4A4A?style=flat)](https://www.llamaindex.ai/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker)](https://www.docker.com/)

[Overview](#-overview) • [About](#-about) • [Topics](#-topics) • [API](#-api-surfaces) • [Quick Start](#-quick-start)

---

Enterprise-grade Retrieval-Augmented Generation platform combining **hybrid search**, **multimodal understanding**, **SQL question answering**, and **domain-specific fraud/AML/KYC retrieval** in a unified service.

</div>

---

## 🎯 Overview

`enterprise-rag` consolidates four retrieval modes into one API and one deployable codebase:

- Hybrid sparse+dense retrieval with reranking
- Multimodal image/text retrieval for rich documents
- SQL-oriented retrieval for data-backed answers
- Domain-aware routing for fraud and compliance knowledge

## 📌 About

- Unified RAG backend for enterprise document and analytics workloads
- Optimized for fast integration with API-first and dashboard-driven workflows
- Includes ingestion, retrieval, generation, evaluation, and caching layers

## 🏷️ Topics

`rag` `enterprise-rag` `hybrid-search` `multimodal-rag` `sql-rag` `fastapi` `llm` `retrieval-augmented-generation` `document-intelligence`

## 🧩 Architecture

- `src/api/`: unified FastAPI app and route modules
- `src/ingestion/`: document, image, table, and dedup pipelines
- `src/retrieval/`: hybrid, multimodal, SQL, and domain retrievers
- `src/generation/`: response generation and citation tracking
- `src/evaluation/`: RAG quality and metrics surfaces
- `src/core/`: auth, security, rate-limit, errors, secrets

## 🌐 API Surfaces

- `POST /api/v1/rag/query`
- `POST /api/v1/rag/documents/ingest`
- `GET /api/v1/rag/documents`
- `DELETE /api/v1/rag/documents/{document_id}`
- `POST /api/v1/rag/evaluation/run`
- `GET /api/v1/rag/evaluation/metrics`
- `GET /health`
- `GET /metrics`

## ⚡ Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## 🔧 GLM Setup

```bash
export GLM_API_KEY=your_glm_api_key
export GLM_BASE_URL=https://api.z.ai/api/anthropic
export GLM_MODEL=glm-5.1
export ZHIPUAI_API_KEY=$GLM_API_KEY
```

## 🛠️ Tech Stack

**Core:** FastAPI, Pydantic, Uvicorn  
**RAG:** LangChain, LlamaIndex, BM25, reranking  
**Vector/Data:** ChromaDB, Qdrant, PostgreSQL, Redis  
**Multimodal:** CLIP, BLIP, OCR pipelines  
**Frontend:** Streamlit + React/Tailwind
