"""Lazy access to the DataChat Redis cache implementation."""

from importlib import import_module


def get_query_cache_class():
    module = import_module("src.datachat.cache.query_cache")
    return getattr(module, "QueryCache")
