"""Type definitions and aliases for XFlow.

Provides common type aliases and optional dependency type hints without
importing heavy libraries at runtime.
"""

from __future__ import annotations

from os import PathLike
from typing import (
    TYPE_CHECKING,
    Type,
    Any,
    Callable,
    Dict,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

# shim TypeAlias for Python <3.10
try:
    # 3.10+
    from typing import TypeAlias
except ImportError:
    # backport
    from typing_extensions import (  # make sure typing-extensions>=4.0.0 is in your deps
        TypeAlias,
    )


# Only import heavy libraries for type checking
if TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf
    import torch
    from numpy.typing import NDArray
    from PIL.Image import Image as PILImage


# Protocol for any object that can be converted to an ndarray
class ArrayLike(Protocol):
    """Protocol for objects that can be converted to numpy arrays."""

    def __array__(self, dtype: type[Any] | None = None) -> NDArray[Any]: ...

# Protocol for validation schemas that can dump their data
class SupportsValidation(Protocol):
    """Protocol for any validation schema."""
    
    def __init__(self, **data: Any) -> None: ...
    def model_dump(self) -> Dict[str, Any]: ...
    # Could also support: to_dict(), dict(), as_dict(), etc.

ModelT = TypeVar("ModelT", bound=SupportsValidation)
Schema: TypeAlias = Type[ModelT]                    # a schema class (e.g., Pydantic model)
ValidatorFn: TypeAlias = Callable[[Mapping[str, Any], Schema], Dict[str, Any]]

# Common type aliases
try:
    # Py 3.11+
    PathLikeStr: TypeAlias = Union[str, PathLike[str]]
except TypeError:
    # older Pythons
    PathLikeStr: TypeAlias = Union[str, PathLike]
MetaHook: TypeAlias = Callable[[Mapping[str, Any]], Dict[str, Any]]
ModelType: TypeAlias = Any
T = TypeVar("T")  # Generic type

# Numeric types
Numeric: TypeAlias = Union[int, float, complex]
# Shape: sequence of ints
try:
    # Python 3.9+
    Shape: TypeAlias = Union[Sequence[int], tuple[int, ...]]
except TypeError:
    # Python <3.9
    Shape: TypeAlias = Union[Sequence[int], Tuple[int, ...]]

# Image-like types: PIL, NumPy arrays, ArrayLike objects, and tensors
ImageLike: TypeAlias = Union[
    "PILImage",  # PIL.Image.Image
    "NDArray[Any]",  # numpy arrays
    ArrayLike,  # any __array__-compatible object
    "tf.Tensor",  # TensorFlow tensor
    "torch.Tensor",  # PyTorch tensor
]

# Tensor-like types for ML backends
TensorLike: TypeAlias = Union[
    "NDArray[Any]",  # numpy arrays
    "tf.Tensor",  # TensorFlow tensor
    "torch.Tensor",  # PyTorch tensor
]

# Model-like types for ML frameworks
ModelLike: TypeAlias = Union[
    "tf.keras.Model",  # TensorFlow/Keras model
    "torch.nn.Module",  # PyTorch model
    "Any",  # Any other custom model type
]