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
    
    Ultra-minimal trainer that only accepts ready-to-use configuration dict.
    No knowledge of config files, parsers, or parsing logic.
    
    Args:
        model: Framework-specific model instance.
        train_data: Training data pipeline.
        val_data: Optional validation data pipeline.
        config: Ready-to-use configuration dictionary.
        
    Example:
        .. code-block:: python
        
            # Load config externally
            from xflow.utils.config import load_config
            config = load_config('experiment.yaml')
            
            # Trainer only deals with dict
            trainer = TFTrainer(model, train_data, val_data, config)
            trainer.fit()
    """
    
    def __init__(
        self,
        model: ModelType,
        train_data: BasePipeline,
        val_data: Optional[BasePipeline] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config or {}
        
        # Distribute config to components via their APIs
        self._configure_components()
    
    def _configure_components(self):
        """Distribute config sections to components via standardized API."""
        # Configure model if it has the API
        if hasattr(self.model, 'configure') and 'model' in self.config:
            self.model.configure(self.config['model'])
        
        # Configure data pipelines if they have the API
        if 'data' in self.config:
            data_config = self.config['data']
            
            if hasattr(self.train_data, 'configure'):
                self.train_data.configure(data_config)
            
            if self.val_data and hasattr(self.val_data, 'configure'):
                self.val_data.configure(data_config)
    
    @abstractmethod
    def fit(self, **kwargs) -> Any:
        """Train the model using configuration."""
        ...
    
    @abstractmethod
    def predict(self, data: BasePipeline, **kwargs) -> Any:
        """Generate predictions."""
        ...
    
    @abstractmethod
    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save model."""
        ...
    
    @abstractmethod
    def load(self, filepath: Path) -> 'BaseTrainer':
        """Load model."""
        ...