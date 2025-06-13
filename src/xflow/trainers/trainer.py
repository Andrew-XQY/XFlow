"""Ultra-minimal trainer for ML model training across frameworks."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from ..data.loader import BasePipeline
import copy

ModelType = Any # Type aliases

class BaseTrainer(ABC):
    """Abstract base trainer for ML models. Trainer is a stateless executor/orchestrator
    
    Args:
        model: Framework-specific model instance (pre-configured).
        data_pipeline: Data pipeline managing train/val/test splits (pre-configured).
        config: Configuration dictionary (single source of truth).
    """
    
    def __init__(
        self,
        model: ModelType,
        data_pipeline: BasePipeline,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        if model is None:
            raise ValueError("Model cannot be None")
        if data_pipeline is None:
            raise ValueError("Data pipeline cannot be None")
        self.model = model
        self.data_pipeline = data_pipeline
        self.config = copy.deepcopy(config) if config is not None else {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={type(self.model).__name__})"
    
    @abstractmethod
    def fit(self, **kwargs) -> Any:
        """Train the model. Returns training results/metrics."""
        ...
    
    @abstractmethod
    def predict(self, **kwargs) -> Any:
        """Generate predictions. Returns prediction results."""
        ...
    
    @abstractmethod
    def get_model(self) -> Any:
        """Return model object. Saving handled externally."""
        ...