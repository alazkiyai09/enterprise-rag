"""Compatibility wrapper for the migrated multimodal image processor."""

from importlib import import_module


def get_image_processor_module():
    return import_module("src.multimodal.image_processor")

