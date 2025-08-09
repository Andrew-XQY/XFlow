"""
Centralized API registry for XFlow.
This is the ONLY place you define what's publicly exposed.
"""

from dataclasses import dataclass
from typing import Dict, Optional


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


# Core API - most commonly used items at package root expose to users
CORE_API = {
    # Data pipeline components
    "BasePipeline": APIItem("data.pipeline", "BasePipeline"),
    "DataPipeline": APIItem("data.pipeline", "DataPipeline"),
    "Pipeline": APIItem("data.pipeline", "BasePipeline", alias="Pipeline"),
    "InMemoryPipeline": APIItem("data.pipeline", "InMemoryPipeline"),
    "TensorFlowPipeline": APIItem("data.pipeline", "TensorFlowPipeline"),
    "PyTorchPipeline": APIItem("data.pipeline", "PyTorchPipeline"),
    "ShufflePipeline": APIItem("data.transform", "ShufflePipeline"),
    "BatchPipeline": APIItem("data.transform", "BatchPipeline"),
    "SqlProvider": APIItem("data.provider", "SqlProvider"),
    "FileProvider": APIItem("data.provider", "FileProvider"),
    "BaseTrainer": APIItem("trainers.trainer", "BaseTrainer"),
    "ConfigManager": APIItem("utils.config", "ConfigManager"),
    # Models
    "BaseModel": APIItem("models.base", "BaseModel"),
    "show_model_info": APIItem("models.utils", "show_model_info"),
    # Utilities
    "CallbackRegistry": APIItem("trainers.callback", "CallbackRegistry"),
}

# Package-level API organization
PACKAGE_API = {
    "data": {
        "BasePipeline": APIItem("pipeline", "BasePipeline"),
        "DataPipeline": APIItem("pipeline", "DataPipeline"),
        "Pipeline": APIItem("pipeline", "BasePipeline", alias="Pipeline"),
        "InMemoryPipeline": APIItem("pipeline", "InMemoryPipeline"),
        "TensorFlowPipeline": APIItem("pipeline", "TensorFlowPipeline"),
        "PyTorchPipeline": APIItem("pipeline", "PyTorchPipeline"),
        "ShufflePipeline": APIItem("transform", "ShufflePipeline"),
        "BatchPipeline": APIItem("transform", "BatchPipeline"),
        "SqlProvider": APIItem("provider", "SqlProvider"),
        "FileProvider": APIItem("provider", "FileProvider"),
        "build_transforms_from_config": APIItem(
            "transform", "build_transforms_from_config"
        ),
    },
    "models": {
        "BaseModel": APIItem("base", "BaseModel"),
        "show_model_info": APIItem("utils", "show_model_info"),
    },
    "trainers": {
        "BaseTrainer": APIItem("trainer", "BaseTrainer"),
        "CallbackRegistry": APIItem("callback", "CallbackRegistry"),
        "build_callbacks_from_config": APIItem(
            "callback", "build_callbacks_from_config"
        ),
    },
    "utils": {
        "ConfigManager": APIItem("config", "ConfigManager"),
        "plot_image": APIItem("visualization", "plot_image"),
        "get_base_dir": APIItem("helper", "get_base_dir"),
        "load_validated_config": APIItem("config", "load_validated_config"),
    },
}


def generate_init(api_dict: Dict[str, APIItem],
                  package_name: str = "xflow",
                  include_version: bool = False) -> str:
    """Generate __init__.py content from API dictionary"""
    imports, aliases, all_items = [], [], []

    # Group imports by module
    module_imports = {}
    for public_name, item in api_dict.items():
        if item.deprecated:
            continue
        full_module = f"{package_name}.{item.module_path}"
        module_imports.setdefault(full_module, []).append((item.class_name, public_name))

    # Import statements (relative)
    for module, items in module_imports.items():
        unique_classes = sorted({cls for cls, _ in items})
        relative_module = module.replace(f"{package_name}.", ".")
        imports.append(f"from {relative_module} import {', '.join(unique_classes)}")

    # Aliases + __all__
    for public_name, item in api_dict.items():
        if item.deprecated:
            continue
        if public_name != item.class_name:
            aliases.append(f"{public_name} = {item.class_name}")
        all_items.append(public_name)

    header = [
        '"""Auto-generated API exports"""',
        "# This file is auto-generated. Do not edit manually.",
        "",
    ]

    version_block = []
    if include_version:
        version_block = [
            "try:",
            "    from ._version import version as __version__",
            "except Exception:",
            '    __version__ = "0.0.0"',
            "",
        ]

    content = header + version_block + imports + [""] + aliases + ["", f"__all__ = {sorted(all_items)}"]
    return "\n".join(content)

