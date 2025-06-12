"""Ultra-minimal trainer for ML model training across frameworks."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from ..data.loader import BasePipeline
import copy

ModelType = Any # Type aliases

class BaseTrainer(ABC):
    """Abstract base trainer for ML models.
    
    Args:
        model: Framework-specific model instance.
        data_pipeline: Data pipeline managing train/val/test splits.
        config: Configuration dictionary.
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
        self._apply_config() # Apply config during init
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={type(self.model).__name__})"

    def _apply_config(self) -> None:
        """Apply configuration to trainer. Override in subclasses."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return copy.deepcopy(self.config)

    def set_params(self, **params) -> 'BaseTrainer':
        """Set configuration parameters."""
        self.config.update(copy.deepcopy(params))
        self._apply_config() 
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get trainer metadata."""
        try:
            framework_info = self._get_framework_info()
        except Exception as e:
            framework_info = {'error': str(e)}
        return {
            'type': type(self).__name__,
            'framework_info': framework_info,
            'config_keys': list(self.config.keys())
        }
    
    @abstractmethod
    def _get_framework_info(self) -> Dict[str, Any]:
        """Get framework information."""
        ...
    
    @abstractmethod
    def fit(self, **kwargs) -> Any:
        """Train the model."""
        ...
    
    @abstractmethod
    def predict(self, **kwargs) -> Any:
        """Generate predictions."""
        ...
    
    @abstractmethod
    def save(self, **kwargs) -> Any:
        """Save model to storage."""
        ...