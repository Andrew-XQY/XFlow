"""Lightweight model coordination layer for plugin to training frameworks."""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from ..utils.typing import PathLikeStr


class BaseModel(ABC):
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Run a forward/inference pass."""

    @abstractmethod
    def train_step(self, batch: Tuple[Any, Any]) -> float:
        """
        Consume one batch (inputs, targets),
        do a training update, and return the batch loss.
        """

    @abstractmethod
    def validation_step(self, batch: Tuple[Any, Any]) -> float:
        """
        Consume one batch in eval mode and return the batch loss.
        """

    @abstractmethod
    def configure_optimizers(self) -> Any:
        """
        Return whatever your framework needs to step gradients
        (e.g. an optimizer instance or dict of optimizers).
        """

    @abstractmethod
    def save(self, path: PathLikeStr) -> None:
        """Persist weights (and any config) to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: PathLikeStr, **kwargs) -> "BaseModel":
        """Load the model weights to the model structure."""
