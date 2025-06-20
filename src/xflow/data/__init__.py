"""Auto-generated API exports"""
# This file is auto-generated. Do not edit manually.

from .pipeline import BasePipeline, InMemoryPipeline
from .transform import build_transforms_from_config, ShufflePipeline, BatchPipeline

Pipeline = BasePipeline

__all__ = ['BasePipeline', 'BatchPipeline', 'InMemoryPipeline', 'Pipeline', 'ShufflePipeline', 'build_transforms_from_config']