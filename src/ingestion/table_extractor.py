"""Compatibility wrapper for table extraction helpers."""

from importlib import import_module


def get_table_extractor_module():
    return import_module("src.extraction.table_extractor")

