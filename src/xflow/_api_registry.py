"""
Centralized API registry for XFlow.
This is the ONLY place you define what's publicly exposed.
"""

from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class APIItem:
    """Represents a single API item with metadata"""
    module_path: str
    class_name: str
    alias: Optional[str] = None
    deprecated: bool = False
    version_added: Optional[str] = None
    
    @property
    def public_name(self) -> str:
        return self.alias or self.class_name

# Core API - most commonly used items at package root
CORE_API = {
    # Data pipeline components
    "FileProvider": APIItem("data.provider", "FileProvider"),
    "BasePipeline": APIItem("data.pipeline", "BasePipeline"),
    "Pipeline": APIItem("data.pipeline", "BasePipeline", alias="Pipeline"),
    "InMemoryPipeline": APIItem("data.pipeline", "InMemoryPipeline"),
    "ShufflePipeline": APIItem("data.transform", "ShufflePipeline"),
    "BatchPipeline": APIItem("data.transform", "BatchPipeline"),
    "BaseTrainer": APIItem("trainers.trainer", "BaseTrainer"),
    "ConfigManager": APIItem("utils.config", "ConfigManager"),
    
    # Models
    "BaseModel": APIItem("models.base", "BaseModel"),
    
    # Utilities
    "plot_image": APIItem("utils.visualization", "plot_image"),
    "get_base_dir": APIItem("utils.helper", "get_base_dir"),
}

# Package-level API organization
PACKAGE_API = {
    "data": {
        "BasePipeline": APIItem("data.pipeline", "BasePipeline"),
        "Pipeline": APIItem("data.pipeline", "BasePipeline", alias="Pipeline"),
        "ShufflePipeline": APIItem("data.transform", "ShufflePipeline"),
        "BatchPipeline": APIItem("data.transform", "BatchPipeline"),
        "InMemoryPipeline": APIItem("data.pipeline", "InMemoryPipeline"),
    },
    "models": {
        "BaseModel": APIItem("models.base", "BaseModel"),
    },
    "utils": {
        "plot_image": APIItem("utils.visualization", "plot_image"),
    }
}

def generate_init(api_dict: Dict[str, APIItem], 
                         package_name: str = "xflow") -> str:
    """Generate __init__.py content from API dictionary"""
    imports = []
    aliases = []
    all_items = []
    
    # Group imports by module
    module_imports = {}
    for public_name, item in api_dict.items():
        if item.deprecated:
            continue  # Skip deprecated items
            
        full_module = f"{package_name}.{item.module_path}"
        if full_module not in module_imports:
            module_imports[full_module] = []
        module_imports[full_module].append((item.class_name, public_name))
    
    # Generate import statements
    for module, items in module_imports.items():
        unique_classes = list(set(item[0] for item in items))
        relative_module = module.replace(f"{package_name}.", ".")
        imports.append(f"from {relative_module} import {', '.join(unique_classes)}")
    
    # Generate aliases if needed
    for public_name, item in api_dict.items():
        if item.deprecated:
            continue
        if public_name != item.class_name:
            aliases.append(f"{public_name} = {item.class_name}")
        all_items.append(public_name)
    
    # Build content
    content = [
        '"""Auto-generated API exports"""',
        "# This file is auto-generated. Do not edit manually.",
        "",
        *imports,
        "",
        *aliases,
        "",
        f"__all__ = {sorted(all_items)}",
    ]
    
    return "\n".join(content)