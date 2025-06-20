"""Training callback system for monitoring and controlling training process."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
import time
import json

import numpy as np


class Callback(ABC):
    """Base callback interface."""
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch."""
        pass


class CallbackRegistry:
    """Registry for callback functions."""
    _callbacks: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(callback_class):
            cls._callbacks[name] = callback_class
            return callback_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._callbacks:
            raise ValueError(f"Callback '{name}' not found. Available: {list(cls._callbacks.keys())}")
        return cls._callbacks[name]
    
    @classmethod
    def list_callbacks(cls) -> List[str]:
        return list(cls._callbacks.keys())
    
class CallbackManager:
    """Manages multiple callbacks during training."""
    
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
    
    def _call_callbacks(self, method_name: str, *args, **kwargs) -> None:
        """Call method on all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(*args, **kwargs)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self._call_callbacks('on_train_begin', logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self._call_callbacks('on_train_end', logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._call_callbacks('on_epoch_begin', epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._call_callbacks('on_epoch_end', epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._call_callbacks('on_batch_begin', batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._call_callbacks('on_batch_end', batch, logs)


def build_callbacks_from_config(config: List[Dict[str, Any]]) -> List[Callback]:
    """Build callbacks from configuration."""
    callbacks = []
    for callback_config in config:
        name = callback_config.get('name')
        params = callback_config.get('params', {})
        
        callback_class = CallbackRegistry.get(name)
        callback = callback_class(**params)
        callbacks.append(callback)
    
    return callbacks