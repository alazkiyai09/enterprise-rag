"""Domain-mode accessors backed by the migrated fraud-docs-rag package."""

from importlib import import_module


def get_domain_retriever_class():
    module = import_module("src.fraud_docs_rag.retrieval.hybrid_retriever")
    return getattr(module, "HybridRetriever")

