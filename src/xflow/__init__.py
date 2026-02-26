"""Auto-generated API exports"""

# This file is auto-generated. Do not edit manually.

try:
    from ._version import version as __version__
except Exception:
    __version__ = "0.0.0"

from .data.core import compose, consume, flow, pipe
from .data.pipeline import (
    BasePipeline,
    DataPipeline,
    InMemoryPipeline,
    PyTorchPipeline,
    TensorFlowPipeline,
)
from .data.provider import FileProvider, SqlProvider
from .data.transform import BatchPipeline, ShufflePipeline, TransformRegistry
from .models.base import BaseModel
from .models.utils import build_model_report, shape_trace, show_model_info
from .trainers.callback import CallbackRegistry
from .trainers.trainer import BaseTrainer, TorchGANTrainer, TorchTrainer
from .utils.config import ConfigManager

Pipeline = BasePipeline

__all__ = [
    "BaseModel",
    "BasePipeline",
    "BaseTrainer",
    "BatchPipeline",
    "CallbackRegistry",
    "ConfigManager",
    "DataPipeline",
    "FileProvider",
    "InMemoryPipeline",
    "Pipeline",
    "PyTorchPipeline",
    "ShufflePipeline",
    "SqlProvider",
    "TensorFlowPipeline",
    "TorchGANTrainer",
    "TorchTrainer",
    "TransformRegistry",
    "build_model_report",
    "compose",
    "consume",
    "flow",
    "pipe",
    "shape_trace",
    "show_model_info",
]
