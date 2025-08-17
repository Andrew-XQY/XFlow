"""Lightweight model coordination layer for plugin to training frameworks."""

from abc import ABC, abstractmethod
from typing import Any
from ..utils.typing import PathLikeStr, Batch, LossOrMetrics


class InferenceModel(ABC):
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Run a forward/inference pass."""

    @abstractmethod
    def save(self, path: PathLikeStr) -> None:
        """Persist weights (and any config) to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: PathLikeStr, **kwargs) -> "InferenceModel":
        """Load model and config from disk."""


class Trainable(ABC):
    @abstractmethod
    def training_step(self, batch: Batch) -> LossOrMetrics:
        """
        Consume one batch (inputs, targets), perform an update,
        and return a loss or a metrics dict (if dict, must contain 'loss').
        """

    @abstractmethod
    def validation_step(self, batch: Batch) -> LossOrMetrics:
        """Evaluate one batch in eval mode; return loss or metrics dict."""

    @abstractmethod
    def configure_optimizers(self) -> Any:
        """Return optimizer(s)/schedulers required by the training framework."""
    
    def set_train_mode(self, training: bool = True) -> None:
        """Set model to training or evaluation mode. Override if needed."""
        pass


class BaseModel(InferenceModel, Trainable, ABC):
    """Combined abstract interface; implement a single subclass."""
