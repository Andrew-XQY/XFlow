"""Config Manager Module"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

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
from .typing import PathLikeStr, Schema


def load_validated_config(
    file_path: PathLikeStr, schema: Optional[Schema] = None
) -> Dict[str, Any]:
    """Load and optionally validate config using any validation schema.

    Args:
        file_path: Path to config file
        schema: Optional schema class for validation. If None, returns raw dict

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
        self._files: List[PathLikeStr] = []

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

    def add_files(self, *file_paths: PathLikeStr) -> Self:
        """Add files that are part of this configuration."""
        for file_path in file_paths:
            if file_path not in self._files:
                self._files.append(file_path)
        return self

    def get(self) -> Dict[str, Any]:
        """Return a fully independent snapshot of the working config."""
        return copy.deepcopy(self._config)

    def reset(self) -> None:
        """Revert working config back to original."""
        self._config = copy.deepcopy(self._original_config)
        self._files = []

    def update(self, updates: Dict[str, Any]) -> Self:
        """Recursively update in config, Nested dictionaries are merged, other values are replaced."""
        deep_update(self._config, updates)
        return self

    def validate(self, schema: Schema) -> Self:
        """Validate working config against provided schema. Raises Error if invalid."""
        validated = schema(**self._config)
        validated.model_dump()  # Just to ensure it works, but we don't need the result
        return self

    def save_config(self, file_path: PathLikeStr) -> None:
        """Save the internal config to a specific file path (must include filename and extension)."""
        save_file(self._config, file_path)

    def copy_associated_files(self, target_dir: PathLikeStr) -> None:
        """Copy all associated files to the target directory."""
        if self._files:
            target_dir = Path(target_dir)
            for file_path in self._files:
                copy_file(file_path, target_dir)

    def save(self, output_dir: PathLikeStr, config_filename: Optional[str] = None) -> None:
        """Save config and copy associated files to target directory.
        
        Args:
            output_dir: Target directory path
            config_filename: Config filename with extension (e.g., 'config.yaml'). 
                           If None or empty, only copies associated files.
        """
        output_dir = Path(output_dir)
        
        # Save config only if filename is provided
        if config_filename:
            config_path = output_dir / config_filename
            self.save_config(config_path)
        
        # Always copy associated files
        self.copy_associated_files(output_dir)
