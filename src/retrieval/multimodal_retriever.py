"""Lazy access to the multimodal retriever implementation."""

from importlib import import_module


def get_multimodal_retriever_class():
    module = import_module("src.multimodal.multimodal_retriever")
    return getattr(module, "MultiModalRetriever")

