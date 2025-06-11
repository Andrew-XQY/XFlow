"""
xflow.trainers.trainer
---------------------
Ultra-minimal trainer for ML model training across frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
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
            trainer.configure()  # Add this line
            trainer.fit()
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
        self.config = config.copy() if config is not None else {} 
        self._configured = False
        self._callbacks = []
        
    @property
    def is_configured(self) -> bool:
        """Check if trainer has been configured."""
        return self._configured
    
    def get_params(self) -> dict:
        """Get configuration parameters.
        
        Returns:
            Copy of current configuration parameters.
        """
        return self.config.copy()

    def set_params(self, **params) -> 'BaseTrainer':
        """Set configuration parameters.
        
        Args:
            **params: Configuration parameters to set.
            
        Returns:
            Self for method chaining.
        """
        self.config.update(params)  
        return self

    def configure(self) -> 'BaseTrainer':
        """Configure the trainer with the provided settings.

        Default no-op implementation: does nothing and returns self.
        Subclasses can override to extract and apply their own 'trainer'
        section from the config dict.

        Returns:
            Self, to allow method chaining.
        """
        self._configured = True 
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get trainer-specific metadata only.
        
        Returns:
            Dict with trainer metadata (type, framework info, config summary, etc.).
        """
        try:
            framework_info = self._get_framework_info()
        except Exception as e:
            framework_info = {'error': str(e), 'type': 'framework_error'}
        return {
            'type': type(self).__name__,
            'framework_info': framework_info,
            'config_keys': list(self.config.keys()) if self.config else []
        }
    
    @abstractmethod
    def _get_framework_info(self) -> Dict[str, Any]:
        """Get comprehensive framework information.
        
        The trainer orchestrates the training loop and knows which framework
        it's using, so it's responsible for providing framework details.
        
        Returns:
            Dict containing framework name, version, and other relevant info.
            
        Example:
            {
                'name': 'tensorflow',
                'version': '2.13.0',
                'gpu_available': True,
                'device_count': 2
            }
        """
        ...
    
    def register_callback(self, cb):
        self._callbacks.append(cb)

    def _run_hook(self, name: str, *args, **kwargs) -> None:
        """Run callback hook by name.
        
        Standard hooks: on_train_start, on_epoch_start, on_epoch_end, on_train_end
        """
        for cb in self._callbacks:
            fn = getattr(cb, name, None)
            if callable(fn): 
                try:
                    fn(self, *args, **kwargs)
                except Exception as e:
                    print(f"Callback error in {name}: {e}")
    
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