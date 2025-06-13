"""Config Manager Module"""

import copy
from pydantic import BaseModel, Field
from typing import Dict, Any, Union, Type
from pathlib import Path
from .config import load_config, save_config


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
    filepath: Union[str, Path],
    schema: Type[BaseModel]
) -> Dict[str, Any]:
    """Load and validate config using Pydantic schema.
    
    Single validation point - all validation flows through here.
    Returns native dict for downstream consumption.
    """
    raw = load_config(filepath)
    validated = schema(**raw)
    return validated.model_dump()


class ConfigManager:
    """Simple in-memory config manager."""
    
    def __init__(self, initial_config: Dict[str, Any]):
        # keep the source of truth and a mutable working copy
        self._original = copy.deepcopy(initial_config)
        self.config = copy.deepcopy(initial_config)
    
    def get(self) -> Dict[str, Any]:
        """Return the working config (by reference)."""
        return self.config
    
    def reset(self) -> None:
        """Revert working config back to original."""
        self.config = copy.deepcopy(self._original)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Recursively merge in updates (dicts override, everything else replaces)."""
        self._deep_update(self.config, updates)
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Write the working config to disk (ext-driven format)."""
        save_config(self.config, output_path)
    
    def _deep_update(self, base: Dict[str, Any], upd: Dict[str, Any]) -> None:
        for key, val in upd.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], val)
            else:
                base[key] = val
