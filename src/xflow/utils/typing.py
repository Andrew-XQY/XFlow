"""Type definitions and aliases for XFlow.

Provides common type aliases and optional dependency type hints without
importing heavy libraries at runtime.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, Union, Callable, Mapping, Any, Dict, Sequence, TypeVar, Tuple
from os import PathLike

# shim TypeAlias for Python <3.10
try:
    # 3.10+
    from typing import TypeAlias
except ImportError:
    # backport
    from typing_extensions import TypeAlias  # make sure typing-extensions>=4.0.0 is in your deps


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
try:
    # Py 3.11+
    PathLikeStr: TypeAlias = Union[str, PathLike[str]]
except TypeError:
    # older Pythons
    PathLikeStr: TypeAlias = Union[str, PathLike]
MetaHook: TypeAlias = Callable[[Mapping[str, Any]], Dict[str, Any]]
ModelType: TypeAlias = Any
T = TypeVar('T')  # Generic type

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