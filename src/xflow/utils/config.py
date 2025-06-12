"""Pure configuration management - completely isolated logic."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Type
from pathlib import Path
from pydantic import BaseModel, Field
import json
import yaml


class ConfigParser(ABC):
    """Abstract base for configuration parsers."""
    
    @abstractmethod
    def parse(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Parse configuration file and return dict."""
        ...


class JSONParser(ConfigParser):
    """JSON configuration parser."""
    
    def parse(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Parse JSON configuration file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


class YAMLParser(ConfigParser):
    """YAML configuration parser."""
    
    def parse(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Parse YAML configuration file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)


SUPPORTED_FORMATS: Dict[str, Type[ConfigParser]] = {
    '.json': JSONParser,
    '.yaml': YAMLParser, 
    '.yml': YAMLParser,
}

def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load config file. Supported formats: JSON, YAML."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    parser_cls = SUPPORTED_FORMATS.get(suffix)
    if not parser_cls:
        supported = ', '.join(SUPPORTED_FORMATS.keys())
        raise ValueError(f"Unsupported format '{suffix}'. Supported: {supported}")
    
    parser = parser_cls()
    return parser.parse(path)

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