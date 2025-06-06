# src/xflow/data/transforms.py

from __future__ import annotations
from typing import Callable, TypeVar, Sequence, Generic

SampleT = TypeVar("SampleT")
PreprocessFn = Callable[[SampleT], SampleT]


class PreprocessPipeline(Generic[SampleT]):
    """
    A generic preprocessing pipeline that applies a sequence of transforms
    to a data sample. Each transform is a callable that takes the sample as input
    and returns a modified sample. The pipeline does not assume any specific
    structure for the sample, only that each transform knows how to handle it.

    Example:
        >>> from src.xflow.data.transforms import PreprocessPipeline
        >>>
        >>> # Suppose each sample is a dict with keys "path" (combined image),
        >>> # and transforms know how to split into "x" and "y":
        >>>
        >>> def split_concat(sample: dict[str, str]) -> dict[str, Any]:
        ...     # Read the image from disk, split left/right, attach numpy arrays.
        ...     import cv2
        ...     import numpy as np
        ...     path = sample["path"]
        ...     img = cv2.imread(path)  # BGR uint8
        ...     height, width = img.shape[:2]
        ...     half = width // 2
        ...     x = img[:, :half, :]
        ...     y = img[:, half:, :]
        ...     sample["x"] = x  # numpy array
        ...     sample["y"] = y  # numpy array
        ...     return sample
        >>>
        >>> def normalize_x(sample: dict[str, Any]) -> dict[str, Any]:
        ...     x = sample["x"].astype("float32") / 255.0
        ...     sample["x"] = x
        ...     return sample
        >>>
        >>> def normalize_y(sample: dict[str, Any]) -> dict[str, Any]:
        ...     y = sample["y"].astype("float32") / 255.0
        ...     sample["y"] = y
        ...     return sample
        >>>
        >>> # Build a pipeline that first splits, then normalizes x and y separately
        >>> pipeline = PreprocessPipeline([
        ...     split_concat,
        ...     normalize_x,
        ...     normalize_y,
        ... ])
        >>>
        >>> # In your data loader:
        >>> sample = {"path": "/data/combined/img01.png"}
        >>> processed = pipeline(sample)
        >>> # processed now has keys "path", "x", "y", where x and y are float32 numpy arrays.
    """

    def __init__(self, transforms: Sequence[PreprocessFn[SampleT]]) -> None:
        """
        Args:
            transforms: A sequence of callables, each with signature
                        (sample: SampleT) -> SampleT. Each transform
                        should accept the entire sample, mutate or
                        copy fields as needed, and return the updated sample.

        Raises:
            ValueError: If `transforms` is empty.
        """
        if not transforms:
            raise ValueError("`transforms` must contain at least one callable")
        self.transforms: list[PreprocessFn[SampleT]] = list(transforms)

    def __call__(self, sample: SampleT) -> SampleT:
        """
        Apply each transform in sequence to the sample.

        Args:
            sample: An instance of SampleT (e.g., dict, tuple, custom object).
                    Each transform must know how to handle this sample type.

        Returns:
            The transformed sample after applying every function in order.

        Raises:
            Any exception raised by an individual transform.
        """
        for fn in self.transforms:
            sample = fn(sample)
        return sample
