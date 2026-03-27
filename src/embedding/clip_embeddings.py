"""CLIP embedding facade for the migrated multimodal stack."""

from importlib import import_module


def get_clip_embedding_backend():
    module = import_module("src.multimodal.multimodal_retriever")
    return getattr(module, "MultiModalRetriever")

