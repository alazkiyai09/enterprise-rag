"""Text embedding facade for the unified repo."""

from importlib import import_module


def get_text_embedding_service_class():
    module = import_module("src.retrieval.embedding_service")
    return getattr(module, "EmbeddingService")

