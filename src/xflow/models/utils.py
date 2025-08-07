from typing import Any, Dict, Optional
from ..utils.typing import ModelLike


def get_pytorch_info(model: ModelLike) -> Dict[str, Any]:
    """Collect PyTorch model information."""
    try:
        import torch
    except ImportError:
        return {"error": "PyTorch not available"}
    
    info = {"name": model.__class__.__name__}
    
    # Device and dtype
    try:
        first_param = next(model.parameters(), None)
        info["device"] = str(first_param.device) if first_param else "no parameters"
        info["dtype"] = str(first_param.dtype) if first_param else "N/A"
    except:
        info["device"] = "unavailable"
        info["dtype"] = "N/A"
    
    # Parameters
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info["total_params"] = total_params
        info["trainable_params"] = trainable_params
        info["non_trainable_params"] = total_params - trainable_params
    except:
        info["total_params"] = "unavailable"
        info["trainable_params"] = "unavailable"
        info["non_trainable_params"] = "unavailable"
    
    # Memory size
    try:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        info["size_mb"] = (param_bytes + buffer_bytes) / (1024 ** 2)
    except:
        info["size_mb"] = "unavailable"
    
    # Module count
    try:
        info["num_modules"] = len(list(model.modules()))
    except:
        info["num_modules"] = "unavailable"
    
    return info


def get_tensorflow_info(model: ModelLike) -> Dict[str, Any]:
    """Collect TensorFlow/Keras model information."""
    try:
        import tensorflow as tf
        from tensorflow.keras import backend as K
    except ImportError:
        return {"error": "TensorFlow not available"}
    
    info = {"name": model.__class__.__name__}
    
    # Device and dtype
    try:
        first_w = next(iter(model.weights), None)
        info["device"] = str(first_w.device) if first_w else "no weights"
        info["dtype"] = first_w.dtype.name if first_w else "N/A"
    except:
        info["device"] = "unavailable"
        info["dtype"] = "N/A"
    
    # Parameters
    try:
        info["total_params"] = sum(K.count_params(w) for w in model.weights)
        info["trainable_params"] = sum(K.count_params(w) for w in model.trainable_weights)
        info["non_trainable_params"] = sum(K.count_params(w) for w in model.non_trainable_weights)
    except:
        info["total_params"] = "unavailable"
        info["trainable_params"] = "unavailable"
        info["non_trainable_params"] = "unavailable"
    
    # Memory size
    try:
        total_bytes = sum(K.count_params(w) * w.dtype.size for w in model.weights)
        info["size_mb"] = total_bytes / (1024 ** 2)
    except:
        info["size_mb"] = "unavailable"
    
    # Module count
    try:
        info["num_modules"] = len(model.submodules)
    except:
        info["num_modules"] = "unavailable"
    
    return info


# Registry for framework handlers
FRAMEWORK_HANDLERS = {
    "PyTorch": get_pytorch_info,
    "TensorFlow": get_tensorflow_info,
}


def get_model_info(model: ModelLike) -> Dict[str, Any]:
    """Collect model information as structured data."""
    framework = detect_model_framework(model)
    
    if framework in FRAMEWORK_HANDLERS:
        info = FRAMEWORK_HANDLERS[framework](model)
        info["framework"] = framework
        return info
    else:
        return {
            "framework": framework,
            "name": model.__class__.__name__,
            "error": f"No handler for framework: {framework}"
        }


def format_model_info(info: Dict[str, Any]) -> str:
    """Format model information for display."""
    if "error" in info:
        return f"Error: {info['error']}"
    
    lines = [
        f"Framework:           {info['framework']}",
        f"Model:               {info['name']}",
        f"Device / dtype:      {info['device']} / {info['dtype']}",
    ]
    
    # Format parameters
    total = info.get('total_params', 'unavailable')
    trainable = info.get('trainable_params', 'unavailable')
    non_trainable = info.get('non_trainable_params', 'unavailable')
    
    if isinstance(total, int):
        lines.append(f"Parameters:          {total:,} total")
        lines.append(f"                     {trainable:,} trainable")
        lines.append(f"                     {non_trainable:,} non-trainable")
    else:
        lines.append(f"Parameters:          {total}")
    
    # Format memory
    size_mb = info.get('size_mb', 'unavailable')
    if isinstance(size_mb, (int, float)):
        lines.append(f"Size:                {size_mb:.2f} MB")
    else:
        lines.append(f"Size:                {size_mb}")
    
    # Format modules
    num_modules = info.get('num_modules', 'unavailable')
    lines.append(f"Sub-modules:         {num_modules}")
    
    return "\n".join(lines)


def show_model_info(model: ModelLike) -> None:
    """Display model information in a user-friendly format."""
    info = get_model_info(model)
    print(f"Detected framework: {info.get('framework', 'Unknown')}")
    print(format_model_info(info))


def detect_model_framework(model: ModelLike) -> str:
    """Detect the model's framework."""
    try:
        import torch
        from torch.nn import Module as TorchModule
        if isinstance(model, TorchModule):
            return "PyTorch"
    except ImportError:
        pass

    try:
        import tensorflow as tf
        from tensorflow.keras import Model as KerasModel
        if isinstance(model, KerasModel):
            return "TensorFlow"
    except ImportError:
        pass
    
    return f"Unknown (type: {model.__class__.__name__})"
