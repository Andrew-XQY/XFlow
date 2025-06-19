"""Type definitions and aliases for XFlow.

Provides common type aliases and optional dependency type hints without
importing heavy libraries at runtime.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, Union, Callable, Mapping, Any, Dict, TypeAlias, Sequence
from os import PathLike

# Only import heavy libraries for type checking
if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from PIL.Image import Image as PILImage
    import tensorflow as tf
    import torch

# Protocol for any object that can be converted to an ndarray
class ArrayLike(Protocol):
    """Protocol for objects that can be converted to numpy arrays."""
    def __array__(self, dtype: type[Any] | None = None) -> NDArray[Any]: ...

# Common type aliases
PathLikeStr: TypeAlias = Union[str, PathLike[str]]
MetaHook: TypeAlias = Callable[[Mapping[str, Any]], Dict[str, Any]]
ModelType: TypeAlias = Any

# Numeric types
Numeric: TypeAlias = Union[int, float, complex]
Shape: TypeAlias = Union[Sequence[int], tuple[int, ...]]

# Image-like types: PIL, NumPy arrays, ArrayLike objects, and tensors
ImageLike: TypeAlias = Union[
    "PILImage",           # PIL.Image.Image
    "NDArray[Any]",       # numpy arrays
    ArrayLike,            # any __array__-compatible object
    "tf.Tensor",          # TensorFlow tensor
    "torch.Tensor",       # PyTorch tensor
]

# Tensor-like types for ML backends
TensorLike: TypeAlias = Union[
    "NDArray[Any]",       # numpy arrays
    "tf.Tensor",          # TensorFlow tensor
    "torch.Tensor",       # PyTorch tensor
]