"""Config Manager Module"""

import copy
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

# shim Self for Python <3.11
try:
    from typing import Self
except ImportError:
    from typing_extensions import (
        Self,
    )  # ensure typing-extensions>=4.0.0 is in your deps

from .helper import deep_update
from .io import copy_file
from .parser import load_file, save_file
from .typing import PathLikeStr


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
    file_path: PathLikeStr, schema: Optional[Type[BaseModel]] = None
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

    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set config value by key."""
        self._config[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete config key."""
        del self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config

    def __iter__(self):
        """Iterate over config keys."""
        return iter(self._config)

    def __len__(self) -> int:
        """Return number of config items."""
        return len(self._config)

    def keys(self):
        """Return config keys."""
        return self._config.keys()

    def values(self):
        """Return config values."""
        return self._config.values()

    def items(self):
        """Return config items."""
        return self._config.items()

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
        deep_update(self._config, updates)
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
