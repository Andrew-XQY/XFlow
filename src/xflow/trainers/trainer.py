"""Ultra-minimal trainer for ML model training across frameworks."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from ..data.loader import BasePipeline
from ..utils.io import create_directory
import copy
from pathlib import Path

ModelType = Any # Type aliases

class BaseTrainer(ABC):
    """Abstract base trainer for ML models. Trainer is a stateless executor/orchestrator
    
    Args:
        model: Framework-specific model instance (pre-configured).
        data_pipeline: Data pipeline managing train/val/test splits (pre-configured).
        config: Configuration dictionary (single source of truth).
        output_dir: Optional output directory path. If None, will try to get from config.
    """
    
    def __init__(
        self,
        model: ModelType,
        data_pipeline: BasePipeline,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None
    ) -> None:
        if model is None:
            raise ValueError("Model cannot be None")
        if data_pipeline is None:
            raise ValueError("Data pipeline cannot be None")
        
        self.model = model
        self.data_pipeline = data_pipeline
        self.config = copy.deepcopy(config) if config is not None else {}
        
        if output_dir:
            self.output_dir = create_directory(output_dir)
        elif self.config.get('experiment', {}).get('output_dir'):
            self.output_dir = create_directory(self.config['experiment']['output_dir'])
        else:
            raise ValueError(
                "No output directory provided. Either:\n"
                "1. Add 'experiment.output_dir' to your config, or\n"
                "2. Pass output_dir parameter to trainer constructor"
            )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={type(self.model).__name__}, output_dir={self.output_dir})"
    
    @abstractmethod
    def fit(self, **kwargs) -> Any:
        """Train the model. Saves artifacts to self.output_dir."""
        ...
    
    @abstractmethod
    def predict(self, **kwargs) -> Any:
        """Generate predictions. Returns prediction results."""
        ...
    
    @abstractmethod
    def save_model(self, path: str, **kwargs) -> None:
        """Save model to path. Trainer knows best how to save its own model."""
        ...