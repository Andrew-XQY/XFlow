"""
xflow.utils.config
-----------------
Pure configuration management - completely isolated from training logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from pathlib import Path
import json
import yaml


class ConfigParser(ABC):
    """Abstract base for configuration parsers."""
    
    @abstractmethod
    def parse(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Parse configuration file and return dict."""
        ...


class JSONParser(ConfigParser):
    def parse(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            return json.load(f)


class YAMLParser(ConfigParser):
    def parse(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration file and return ready-to-use dict.
    
    This is the ONLY function BaseTrainer needs to know about.
    All parser complexity is hidden here.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Ready-to-use configuration dictionary
    """
    filepath = Path(filepath)
    
    # Auto-detect and parse
    if filepath.suffix in ['.yaml', '.yml']:
        parser = YAMLParser()
    elif filepath.suffix == '.json':
        parser = JSONParser()
    else:
        raise ValueError(f"Unsupported config format: {filepath.suffix}")
    
    return parser.parse(filepath)