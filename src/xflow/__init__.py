"""Auto-generated API exports"""
# This file is auto-generated. Do not edit manually.

try:
    from ._version import version as __version__
except Exception:
    __version__ = "0.0.0"

from .data.pipeline import BasePipeline, DataPipeline, InMemoryPipeline, PyTorchPipeline, TensorFlowPipeline
from .data.transform import BatchPipeline, ShufflePipeline
from .data.provider import FileProvider, SqlProvider
from .trainers.trainer import BaseTrainer
from .utils.config import ConfigManager
from .models.base import BaseModel
from .models.utils import show_model_info
from .trainers.callback import CallbackRegistry

Pipeline = BasePipeline

__all__ = ['BaseModel', 'BasePipeline', 'BaseTrainer', 'BatchPipeline', 'CallbackRegistry', 'ConfigManager', 'DataPipeline', 'FileProvider', 'InMemoryPipeline', 'Pipeline', 'PyTorchPipeline', 'ShufflePipeline', 'SqlProvider', 'TensorFlowPipeline', 'show_model_info']