from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        info["size_mb"] = (param_bytes + buffer_bytes) / (1024**2)
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
        info["trainable_params"] = sum(
            K.count_params(w) for w in model.trainable_weights
        )
        info["non_trainable_params"] = sum(
            K.count_params(w) for w in model.non_trainable_weights
        )
    except:
        info["total_params"] = "unavailable"
        info["trainable_params"] = "unavailable"
        info["non_trainable_params"] = "unavailable"

    # Memory size
    try:
        total_bytes = sum(K.count_params(w) * w.dtype.size for w in model.weights)
        info["size_mb"] = total_bytes / (1024**2)
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
            "error": f"No handler for framework: {framework}",
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
    total = info.get("total_params", "unavailable")
    trainable = info.get("trainable_params", "unavailable")
    non_trainable = info.get("non_trainable_params", "unavailable")

    if isinstance(total, int):
        lines.append(f"Parameters:          {total:,} total")
        lines.append(f"                     {trainable:,} trainable")
        lines.append(f"                     {non_trainable:,} non-trainable")
    else:
        lines.append(f"Parameters:          {total}")

    # Format memory
    size_mb = info.get("size_mb", "unavailable")
    if isinstance(size_mb, (int, float)):
        lines.append(f"Size:                {size_mb:.2f} MB")
    else:
        lines.append(f"Size:                {size_mb}")

    # Format modules
    num_modules = info.get("num_modules", "unavailable")
    lines.append(f"Sub-modules:         {num_modules}")

    return "\n".join(lines)


def show_model_info(model: ModelLike) -> str:
    """Return model information as a user-friendly formatted string."""
    info = get_model_info(model)
    framework = info.get("framework", "Unknown")
    return f"Detected framework: {framework}\n{format_model_info(info)}"


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


# Hooks for tracing shapes during forward pass (PyTorch specific)


def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.is_tensor(x)


def _is_tensorflow_tensor(x: Any) -> bool:
    try:
        import tensorflow as tf
    except ImportError:
        return False
    return tf.is_tensor(x)


def _shape(x: Any):
    if _is_torch_tensor(x) or _is_tensorflow_tensor(x):
        return tuple(x.shape)
    if hasattr(x, "shape") and x.shape is not None:
        try:
            return tuple(x.shape)
        except TypeError:
            pass
    if isinstance(x, (tuple, list)):
        return [_shape(v) for v in x]
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in x.items()}
    return type(x).__name__


def _attach_pytorch_shape_hooks(
    model: ModelLike, leaf_only: bool, print_fn
) -> List[Callable[[], None]]:
    if not hasattr(model, "named_modules"):
        raise TypeError(
            "Model does not expose named_modules(); expected torch.nn.Module"
        )

    cleanups = []

    def root_pre_hook(module, inputs):
        print_fn(
            f"{'<input_sample>':30s} | {module.__class__.__name__:20s} | "
            f"in={_shape(inputs)}"
        )

    cleanups.append(model.register_forward_pre_hook(root_pre_hook).remove)

    def make_hook(name):
        def hook(module, inputs, outputs):
            print_fn(
                f"{name:30s} | {module.__class__.__name__:20s} | "
                f"in={_shape(inputs)} -> out={_shape(outputs)}"
            )

        return hook

    for name, m in model.named_modules():
        if not name:
            continue
        if leaf_only and any(m.children()):
            continue
        handle = m.register_forward_hook(make_hook(name))
        cleanups.append(handle.remove)

    return cleanups


def _attach_tensorflow_shape_hooks(
    model: ModelLike, leaf_only: bool, print_fn
) -> List[Callable[[], None]]:
    if not hasattr(model, "layers"):
        raise TypeError("Model does not expose layers; expected tf.keras.Model")

    cleanups = []

    original_model_call = model.call

    def wrapped_model_call(*args, __original_call=original_model_call, **kwargs):
        model_inputs = args[0] if len(args) == 1 else args
        print_fn(
            f"{'<input_sample>':30s} | {model.__class__.__name__:20s} | "
            f"in={_shape(model_inputs)}"
        )
        return __original_call(*args, **kwargs)

    model.call = wrapped_model_call
    cleanups.append(lambda m=model, c=original_model_call: setattr(m, "call", c))

    for layer in model.layers:
        if leaf_only and hasattr(layer, "layers") and len(layer.layers) > 0:
            continue

        layer_name = layer.name
        original_call = layer.call

        def wrapped_call(
            *args,
            __original_call=original_call,
            __layer_name=layer_name,
            __layer=layer,
            **kwargs,
        ):
            outputs = __original_call(*args, **kwargs)
            layer_inputs = args[0] if len(args) == 1 else args
            print_fn(
                f"{__layer_name:30s} | {__layer.__class__.__name__:20s} | "
                f"in={_shape(layer_inputs)} -> out={_shape(outputs)}"
            )
            return outputs

        layer.call = wrapped_call
        cleanups.append(lambda l=layer, c=original_call: setattr(l, "call", c))

    return cleanups


SHAPE_HOOK_ATTACHERS = {
    "PyTorch": _attach_pytorch_shape_hooks,
    "TensorFlow": _attach_tensorflow_shape_hooks,
}


def attach_shape_hooks(
    model: ModelLike, leaf_only: bool = True, print_fn=print
) -> List[Callable[[], None]]:
    framework = detect_model_framework(model)
    if framework not in SHAPE_HOOK_ATTACHERS:
        raise TypeError(f"attach_shape_hooks does not support framework: {framework}")

    return SHAPE_HOOK_ATTACHERS[framework](model, leaf_only, print_fn)


def remove_hooks(handles: List[Callable[[], None]]) -> None:
    for cleanup in handles:
        cleanup()


@contextmanager
def shape_trace(
    model: ModelLike, enabled: bool = True, leaf_only: bool = True, print_fn=print
):
    handles = (
        attach_shape_hooks(model, leaf_only=leaf_only, print_fn=print_fn)
        if enabled
        else []
    )
    try:
        yield
    finally:
        remove_hooks(handles)


def build_model_report(
    model: ModelLike,
    run_forward: Callable[[], Any],
    leaf_only: bool = True,
) -> str:
    trace_lines: List[str] = []
    with shape_trace(
        model, enabled=True, leaf_only=leaf_only, print_fn=trace_lines.append
    ):
        out = run_forward()

    input_sample_lines = [
        line for line in trace_lines if line.lstrip().startswith("<input_sample>")
    ]
    non_input_lines = [
        line for line in trace_lines if not line.lstrip().startswith("<input_sample>")
    ]

    lines: List[str] = [show_model_info(model), "", "-- Shape Trace --"]
    if input_sample_lines:
        lines.append(input_sample_lines[0])
    lines.extend(non_input_lines)
    lines += ["", f"Final output shape: {_shape(out)}"]
    return "\n".join(lines) + "\n"
