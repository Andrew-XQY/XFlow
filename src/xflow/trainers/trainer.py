"""
xflow.trainers.trainer
---------------------
Ultra-minimal trainer for ML model training across frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pathlib import Path
from ..data.loader import BasePipeline

# Type aliases
ModelType = Any


class BaseTrainer(ABC):
    """Abstract base class for ML model training across frameworks.
    
    Ultra-minimal trainer with single data pipeline that handles all data concerns.
    
    Args:
        model: Framework-specific model instance.
        data_pipeline: Single data pipeline that manages train/val/test splits internally.
        config: Ready-to-use configuration dictionary.
        
    Example:
        .. code-block:: python
        
            # Data pipeline handles all data logic internally
            pipeline = MyDataPipeline(data_source, train_split=0.8, val_split=0.2)
            
            # Trainer only needs to know about one data source
            trainer = TFTrainer(model, pipeline, config)
            trainer.fit()
    """
    
    def __init__(
        self,
        model: ModelType,
        data_pipeline: BasePipeline,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.model = model
        self.data_pipeline = data_pipeline
        self.config = config or {}
        self.configure(self.config)
    
    def configure(self, config: Dict[str, Any]) -> 'BaseTrainer':
        """Configure the trainer with the provided settings.

        Default no-op implementation: does nothing and returns self.
        Subclasses can override to extract and apply their own 'trainer'
        section from the config dict.

        Args:
            config: Configuration dictionary containing trainer-specific settings.

        Returns:
            Self, to allow method chaining.
        """
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Collect metadata from all components with fallback handling.
        
        Returns:
            Dict containing config and metadata from trainer, model, and data_pipeline.
        """
        metadata = {'config': self.config}
        
        components = [
            ('trainer', self),
            ('model', self.model),
            ('data_pipeline', self.data_pipeline)
        ]
        for name, component in components:
            try:
                if hasattr(component, '_get_metadata'):
                    component_metadata = component._get_metadata()
                    if isinstance(component_metadata, dict):
                        metadata[name] = component_metadata
                    else:
                        metadata[name] = {'type': type(component).__name__, 'status': 'invalid_metadata'}
                else:
                    metadata[name] = {'type': type(component).__name__, 'status': 'no_metadata_api'}
            except Exception as e:
                metadata[name] = {'type': type(component).__name__, 'status': 'metadata_error', 'error': str(e)}
        
        return metadata
    
    @abstractmethod
    def _get_metadata(self) -> Dict[str, Any]:
        """Get trainer-specific metadata.
        
        Returns:
            Dict with trainer metadata (framework, type, etc.).
        """
        ...
    
    @abstractmethod
    def fit(self, **kwargs) -> Any:
        """Train the model using data pipeline.
        
        Args:
            **kwargs: Framework-specific training parameters.
            
        Returns:
            Framework-specific training result.
        """
        ...
    
    @abstractmethod
    def predict(self, **kwargs) -> Any:
        """Generate predictions using trained model.
        
        Args:
            **kwargs: Framework-specific prediction parameters.
            
        Returns:
            Framework-native predictions.
        """
        ...
    
    @abstractmethod
    def save(self, **kwargs) -> Any:
        """Save trained model to storage.
        
        Args:
            **kwargs: Framework-specific save parameters.
            
        Returns:
            Save result (filepath, status, etc.).
        """
        ...
    
    @abstractmethod
    def load(self, **kwargs) -> 'BaseTrainer':
        """Load model from storage.
        
        Args:
            **kwargs: Framework-specific load parameters.
            
        Returns:
            Self for method chaining.
        """
        ...