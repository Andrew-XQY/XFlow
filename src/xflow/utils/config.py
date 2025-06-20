"""Config Manager Module"""

import copy
from pydantic import BaseModel, Field
from typing import Dict, Any, Self, Type, Optional, List
from .parser import load_file, save_file
from .typing import PathLikeStr
from .io import copy_file


# Pydantic schemas
class BaseDataConfig(BaseModel):
    """Base data configuration schema."""
    batch_size: int = Field(..., gt=0)
    
    class Config:
        extra = "forbid"


class BaseTrainerConfig(BaseModel):
    """Base trainer configuration schema."""
    learning_rate: float = Field(..., gt=0)
    epochs: int = Field(1, gt=0)
    
    class Config:
        extra = "forbid"


class BaseModelConfig(BaseModel):
    """Base model configuration schema."""
    model_type: str = Field(..., min_length=1)
    
    class Config:
        extra = "forbid"
        

def load_validated_config(
    file_path: PathLikeStr,
    schema: Optional[Type[BaseModel]] = None
) -> Dict[str, Any]:
    """Load and optionally validate config using Pydantic schema.
    
    Args:
        file_path: Path to config file
        schema: Optional Pydantic model for validation. If None, returns raw dict
        
    Returns:
        Dict containing configuration data (validated if schema provided)
    """
    raw = load_file(file_path)
    if schema is None:
        return raw
    validated = schema(**raw)
    return validated.model_dump()


class ConfigManager:
    """In-memory config manager.

    Keeps an immutable “source of truth” (_original_config) and a mutable working copy (_config).
    """
    
    def __init__(self, initial_config: Dict[str, Any]):
        if not isinstance(initial_config, dict):
            raise TypeError("initial_config must be a dictionary")
        self._original_config = copy.deepcopy(initial_config)
        self._config = copy.deepcopy(initial_config)
        self._extra_files: List[PathLikeStr] = []

    def __repr__(self) -> str:
        return f"ConfigManager(keys={list(self._config.keys())})"
    
    def add_extra(self, *file_paths: PathLikeStr) -> Self:
        """Add extra files that are part of this experiment configuration."""
        for file_path in file_paths:
            if file_path not in self._extra_files:
                self._extra_files.append(file_path)
        return self
    
    def get(self) -> Dict[str, Any]:
        """Return a fully independent snapshot of the working config."""
        return copy.deepcopy(self._config)
        
    def reset(self) -> None:
        """Revert working config back to original."""
        self._config = copy.deepcopy(self._original_config)
        self._extra_files = []
    
    def update(self, updates: Dict[str, Any]) -> Self:
        """Recursively update in config, Nested dictionaries are merged, other values are replaced."""
        self._deep_update(self._config, updates)
        return self
        
    def validate(self, schema: Type[BaseModel]) -> Self:
        """Validate working config against provided schema. Raises Error if invalid."""
        schema(**self._config)
        return self
    
    def save(self, output_path: PathLikeStr) -> None:
        """Write the working config to disk (ext-driven format)."""
        save_file(self._config, output_path)
        # Copy extra files to same directory
        if self._extra_files:
            target_dir = output_path.parent
            for extra_file in self._extra_files:
                copy_file(extra_file, target_dir)
    
    def _deep_update(self, base: Dict[str, Any], upd: Dict[str, Any]) -> None:
        for k, v in upd.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v  
