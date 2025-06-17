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
    "BasePipeline": APIItem("data.loader", "BasePipeline"),
    "Pipeline": APIItem("data.loader", "BasePipeline", alias="Pipeline"),
    "ShufflePipeline": APIItem("data.transforms", "ShufflePipeline"),
    "BatchPipeline": APIItem("data.transforms", "BatchPipeline"),
    
    # Models
    "BaseModel": APIItem("models.base", "BaseModel"),
    "AutoEncoder": APIItem("models.autoencoder", "AutoEncoder"),
    
    # Utilities
    "get_logger": APIItem("utils.logger", "get_logger"),
}

# Package-level API organization
PACKAGE_API = {
    "data": {
        "BasePipeline": APIItem("data.loader", "BasePipeline"),
        "Pipeline": APIItem("data.loader", "BasePipeline", alias="Pipeline"),
        "ShufflePipeline": APIItem("data.transforms", "ShufflePipeline"),
        "BatchPipeline": APIItem("data.transforms", "BatchPipeline"),
        "InMemoryPipeline": APIItem("data.pipeline", "InMemoryPipeline"),
    },
    "models": {
        "BaseModel": APIItem("models.base", "BaseModel"),
        "AutoEncoder": APIItem("models.autoencoder", "AutoEncoder"),
    },
    "utils": {
        "get_logger": APIItem("utils.logger", "get_logger"),
    }
}

def generate_init_content(api_dict: Dict[str, APIItem], 
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