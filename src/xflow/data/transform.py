"""Pipeline transformation utilities for data preprocessing."""

import itertools
import logging
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..utils.decorator import with_progress
from ..utils.typing import ImageLike, PathLikeStr, TensorLike
from .pipeline import BasePipeline


def _copy_pipeline_attributes(target: "BasePipeline", source: BasePipeline) -> None:
    """Helper function to copy essential attributes from source to target pipeline.

    This ensures all pipeline wrappers maintain the same interface as BasePipeline.
    """
    target.data_provider = source.data_provider
    target.transforms = source.transforms
    target.logger = source.logger
    target.skip_errors = source.skip_errors
    target.error_count = source.error_count


@with_progress
def apply_transforms_to_dataset(
    data: Iterable[Any],
    transforms: List[Callable],
    *,
    logger: Optional[logging.Logger] = None,
    skip_errors: bool = True,
) -> Tuple[List[Any], int]:
    """Apply sequential transforms to dataset items."""
    logger = logger or logging.getLogger(__name__)
    processed_items = []
    error_count = 0

    for item in data:
        try:
            for transform in transforms:
                item = transform(item)
            processed_items.append(item)
        except Exception as e:
            error_count += 1
            logger.warning(f"Failed to process item: {e}")
            if not skip_errors:
                raise

    return processed_items, error_count


class ShufflePipeline(BasePipeline):
    """Memory-efficient shuffle using reservoir sampling."""

    def __init__(self, base: BasePipeline, buffer_size: int) -> None:
        _copy_pipeline_attributes(self, base)
        self.base = base
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Any]:
        it = self.base.__iter__()
        buf = list(itertools.islice(it, self.buffer_size))
        random.shuffle(buf)

        for x in buf:
            yield x

        for x in it:
            buf[random.randrange(self.buffer_size)] = x
            random.shuffle(buf)
            yield buf.pop()

    def __len__(self) -> int:
        return len(self.base)

    def sample(self, n: int = 5) -> List[Any]:
        """Return up to n preprocessed items for inspection."""
        return list(itertools.islice(self.__iter__(), n))

    def reset_error_count(self) -> None:
        """Reset the error count to zero."""
        self.error_count = 0
        self.base.reset_error_count()

    def to_framework_dataset(self) -> Any:
        return self.base.to_framework_dataset().shuffle(self.buffer_size)


class BatchPipeline(BasePipeline):
    """Groups items into fixed-size batches."""

    def __init__(self, base: BasePipeline, batch_size: int) -> None:
        _copy_pipeline_attributes(self, base)
        self.base = base
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[Any]]:
        it = self.base.__iter__()
        while True:
            batch = list(itertools.islice(it, self.batch_size))
            if not batch:
                break
            yield batch

    def __len__(self) -> int:
        return (len(self.base) + self.batch_size - 1) // self.batch_size

    def sample(self, n: int = 5) -> List[Any]:
        """Return up to n preprocessed items for inspection."""
        return list(itertools.islice(self.__iter__(), n))

    def reset_error_count(self) -> None:
        """Reset the error count to zero."""
        self.error_count = 0
        self.base.reset_error_count()

    def unbatch(self) -> BasePipeline:
        """Return the underlying pipeline yielding individual items (no batch dimension)."""
        return self.base

    def batch(self, batch_size: int) -> "BatchPipeline":
        """Return a new BatchPipeline with the specified batch size."""
        return BatchPipeline(self, batch_size)

    def to_framework_dataset(self) -> Any:
        return self.base.to_framework_dataset().batch(self.batch_size)


class TransformRegistry:
    """Registry for all available transforms."""

    _transforms: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._transforms[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._transforms:
            raise ValueError(
                f"Transform '{name}' not found. Available: {list(cls._transforms.keys())}"
            )
        return cls._transforms[name]

    @classmethod
    def list_transforms(cls) -> List[str]:
        return list(cls._transforms.keys())


# Core transforms
@TransformRegistry.register("load_image")
def load_image(path: PathLikeStr) -> Image.Image:
    """Load image from file path."""
    return Image.open(Path(path))


@TransformRegistry.register("to_narray")
def to_numpy_array(image: ImageLike) -> np.ndarray:
    """Convert image to numpy array."""
    if hasattr(image, "numpy"):  # TensorFlow tensor
        return image.numpy()
    elif isinstance(image, Image.Image):  # PIL Image
        return np.array(image)
    elif hasattr(image, "__array__"):  # Array-like objects
        return np.asarray(image)
    else:
        raise ValueError(f"Cannot convert {type(image)} to numpy array")


@TransformRegistry.register("to_grayscale")
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale using channel averaging."""
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        return np.mean(image, axis=2).astype(image.dtype)
    elif len(image.shape) == 4:
        if image.shape[2] == 4:  # RGBA format (H, W, 4)
            return np.mean(image[:, :, :3], axis=2).astype(image.dtype)
        elif image.shape[3] == 4:  # RGBA format (H, W, 1, 4)
            return np.mean(image[:, :, 0, :3], axis=2).astype(image.dtype)
        else:
            return np.mean(image.reshape(image.shape[:2] + (-1,)), axis=2).astype(
                image.dtype
            )
    else:
        spatial_dims = image.shape[:2]
        flattened = image.reshape(spatial_dims + (-1,))
        return np.mean(flattened, axis=2).astype(image.dtype)


@TransformRegistry.register("remap_range")
def remap_range(
    image: np.ndarray,
    current_min: float = 0.0,
    current_max: float = 255.0,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> np.ndarray:
    """Remap pixel values from [current_min, current_max] to [target_min, target_max]."""
    image = image.astype(np.float32)
    denominator = current_max - current_min
    if denominator == 0:
        return np.full_like(image, target_min, dtype=np.float32)
    normalized = (image - current_min) / denominator
    remapped = normalized * (target_max - target_min) + target_min
    return remapped.astype(np.float32)


@TransformRegistry.register("resize")
def resize(
    image: np.ndarray, size: Tuple[int, int], interpolation: str = "lanczos"
) -> np.ndarray:
    """Resize image using OpenCV."""
    import cv2

    target_height, target_width = size

    interp_map = {
        "lanczos": cv2.INTER_LANCZOS4,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }

    cv_interpolation = interp_map.get(interpolation, cv2.INTER_LANCZOS4)
    return cv2.resize(
        image, (target_width, target_height), interpolation=cv_interpolation
    )


@TransformRegistry.register("expand_dims")
def expand_dims(image: np.ndarray, axis: int = -1) -> np.ndarray:
    """Add a dimension of size 1 at the specified axis."""
    return np.expand_dims(image, axis=axis)


@TransformRegistry.register("squeeze")
def squeeze(image: np.ndarray, axis: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Remove dimensions of size 1 from the array."""
    return np.squeeze(image, axis=axis)


@TransformRegistry.register("split_width")
def split_width(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split image at width midpoint."""
    height, width = image.shape[:2]
    mid_point = width // 2
    return image[:, :mid_point], image[:, mid_point:]


# TensorFlow transforms
@TransformRegistry.register("tf_read_file")
def tf_read_file(file_path: str) -> TensorLike:
    """Read file contents as bytes using TensorFlow. tf only supports string paths."""
    import tensorflow as tf

    return tf.io.read_file(file_path)


@TransformRegistry.register("tf_decode_image")
def tf_decode_image(
    image_bytes: TensorLike, channels: int = 1, expand_animations: bool = False
) -> TensorLike:
    """Decode image bytes to tensor with specified channels."""
    import tensorflow as tf

    return tf.image.decode_image(
        image_bytes, channels=channels, expand_animations=expand_animations
    )


@TransformRegistry.register("tf_convert_image_dtype")
def tf_convert_image_dtype(image: TensorLike, dtype=None) -> TensorLike:
    """Convert image to specified dtype. and normalize to [0, 1] range."""
    import tensorflow as tf

    return tf.image.convert_image_dtype(image, tf.float32 if not dtype else dtype)


@TransformRegistry.register("tf_remap_range")
def tf_remap_range(
    image: TensorLike,
    current_min: float = 0.0,
    current_max: float = 255.0,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> TensorLike:
    """Remap pixel values from [current_min, current_max] to [target_min, target_max] using TensorFlow."""
    import tensorflow as tf

    image = tf.cast(image, tf.float32)
    # Avoid division by zero
    denominator = tf.where(
        tf.equal(current_max, current_min),
        tf.ones_like(current_max),
        current_max - current_min,
    )
    normalized = (image - current_min) / denominator
    remapped = normalized * (target_max - target_min) + target_min
    return remapped


@TransformRegistry.register("tf_resize")
def tf_resize(image: TensorLike, size: List[int]) -> TensorLike:
    """Resize image using TensorFlow."""
    import tensorflow as tf

    return tf.image.resize(image, size)


@TransformRegistry.register("tf_to_grayscale")
def tf_to_grayscale(image: TensorLike) -> TensorLike:
    """Convert image to grayscale, handling RGB, RGBA, and single-channel images."""
    import tensorflow as tf

    # Handle dynamic shapes properly
    rank = tf.rank(image)
    image = tf.cond(tf.equal(rank, 2), lambda: tf.expand_dims(image, -1), lambda: image)
    ch = tf.shape(image)[-1]

    def rgb_branch():
        rgb = image[..., :3]
        return tf.image.rgb_to_grayscale(rgb)

    def gray_branch():
        return image

    return tf.cond(tf.equal(ch, 1), gray_branch, rgb_branch)


@TransformRegistry.register("tf_split_width")
def tf_split_width(
    image: TensorLike, swap: bool = False
) -> Tuple[TensorLike, TensorLike]:
    """Split image at width midpoint using TensorFlow."""
    import tensorflow as tf

    width = tf.shape(image)[1]
    mid_point = width // 2
    left_half = image[:, :mid_point]
    right_half = image[:, mid_point:]

    if swap:
        return right_half, left_half
    return left_half, right_half


@TransformRegistry.register("tf_expand_dims")
def tf_expand_dims(image: TensorLike, axis: int = -1) -> TensorLike:
    """Add dimension to tensor."""
    import tensorflow as tf

    return tf.expand_dims(image, axis)


@TransformRegistry.register("tf_squeeze")
def tf_squeeze(image: TensorLike, axis: List[int] = None) -> TensorLike:
    """Remove dimensions of size 1."""
    import tensorflow as tf

    return tf.squeeze(image, axis)


def build_transforms_from_config(
    config: List[Dict[str, Any]], name_key: str = "name", params_key: str = "params"
) -> List[Callable]:
    """Build transform pipeline from configuration."""
    transforms = []
    for transform_config in config:
        if name_key not in transform_config:
            raise ValueError(
                f"Transform config missing '{name_key}' key: {transform_config}"
            )
        name = transform_config[name_key]
        params = transform_config.get(params_key, {})
        transform_fn = TransformRegistry.get(name)
        if params:
            transform_fn = partial(transform_fn, **params)
        transforms.append(transform_fn)
    return transforms


class DatasetOperationRegistry:
    """Registry for dataset-level operations."""

    _operations: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn):
            cls._operations[name] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._operations:
            raise ValueError(f"Unknown dataset operation: {name}")
        return cls._operations[name]

    @classmethod
    def list_operations(cls):
        return list(cls._operations.keys())


# Dataset operations (applied to entire dataset)
@DatasetOperationRegistry.register("tf_batch")
def tf_batch(dataset, batch_size: int, drop_remainder: bool = False):
    """Group dataset elements into batches."""
    return dataset.batch(batch_size, drop_remainder=drop_remainder)


@DatasetOperationRegistry.register("tf_prefetch")
def tf_prefetch(dataset, buffer_size: int = None):
    """Prefetch data for better performance."""
    import tensorflow as tf

    if buffer_size is None:
        buffer_size = tf.data.AUTOTUNE
    return dataset.prefetch(buffer_size)


@DatasetOperationRegistry.register("tf_shuffle")
def tf_shuffle(dataset, buffer_size: int, seed: int = 42):
    """Randomly shuffle dataset elements."""
    return dataset.shuffle(buffer_size, seed=seed)


@DatasetOperationRegistry.register("tf_repeat")
def tf_repeat(dataset, count: int = None):
    """Repeat dataset for multiple epochs."""
    return dataset.repeat(count)


@DatasetOperationRegistry.register("tf_cache")
def tf_cache(dataset, filename: str = ""):
    """Cache dataset in memory or disk."""
    return dataset.cache(filename)


@DatasetOperationRegistry.register("tf_take")
def tf_take(dataset, count: int):
    """Take first count elements from dataset."""
    return dataset.take(count)


@DatasetOperationRegistry.register("tf_skip")
def tf_skip(dataset, count: int):
    """Skip first count elements from dataset."""
    return dataset.skip(count)


def apply_dataset_operations_from_config(
    dataset: Any,
    operations_config: List[Dict[str, Any]],
    name_key: str = "name",
    params_key: str = "params",
) -> Any:
    """Apply dataset operations from configuration."""
    for op_config in operations_config:
        if name_key not in op_config:
            raise ValueError(f"Operation config missing '{name_key}' key: {op_config}")
        name = op_config[name_key]
        params = op_config.get(params_key, {})
        operation = DatasetOperationRegistry.get(name)
        dataset = operation(dataset, **params)
    return dataset


# Text processing transforms
@TransformRegistry.register("add_prefix")
def add_prefix(text: str, prefix: str, separator: str = "") -> str:
    """Add prefix to text with optional separator."""
    return prefix + separator + text


@TransformRegistry.register("add_suffix")
def add_suffix(text: str, suffix: str, separator: str = "") -> str:
    """Add suffix to text with optional separator."""
    return text + separator + suffix


@TransformRegistry.register("to_uppercase")
def to_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


@TransformRegistry.register("to_lowercase")
def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


@TransformRegistry.register("strip_whitespace")
def strip_whitespace(text: str, chars: str = None) -> str:
    """Strip whitespace or specified characters from both ends."""
    return text.strip(chars)


@TransformRegistry.register("replace_text")
def replace_text(text: str, old: str, new: str, count: int = -1) -> str:
    """Replace occurrences of old substring with new substring."""
    return text.replace(old, new, count)


@TransformRegistry.register("split_text")
def split_text(text: str, separator: str = None, maxsplit: int = -1) -> List[str]:
    """Split text into list of strings."""
    return text.split(separator, maxsplit)


@TransformRegistry.register("join_text")
def join_text(text_list: List[str], separator: str = "") -> str:
    """Join list of strings into single string."""
    return separator.join(text_list)


# TensorFlow native text transforms
@TransformRegistry.register("tf_add_prefix")
def tf_add_prefix(text: TensorLike, prefix: str, separator: str = "") -> TensorLike:
    """Add prefix to text tensor using TensorFlow."""
    import tensorflow as tf

    prefix_tensor = tf.constant(prefix + separator)
    return tf.strings.join([prefix_tensor, text])


@TransformRegistry.register("tf_add_suffix")
def tf_add_suffix(text: TensorLike, suffix: str, separator: str = "") -> TensorLike:
    """Add suffix to text tensor using TensorFlow."""
    import tensorflow as tf

    suffix_tensor = tf.constant(separator + suffix)
    return tf.strings.join([text, suffix_tensor])


@TransformRegistry.register("tf_to_uppercase")
def tf_to_uppercase(text: TensorLike) -> TensorLike:
    """Convert text tensor to uppercase using TensorFlow."""
    import tensorflow as tf

    return tf.strings.upper(text)


@TransformRegistry.register("tf_to_lowercase")
def tf_to_lowercase(text: TensorLike) -> TensorLike:
    """Convert text tensor to lowercase using TensorFlow."""
    import tensorflow as tf

    return tf.strings.lower(text)


@TransformRegistry.register("tf_strip_whitespace")
def tf_strip_whitespace(text: TensorLike) -> TensorLike:
    """Strip whitespace from text tensor using TensorFlow."""
    import tensorflow as tf

    return tf.strings.strip(text)


@TransformRegistry.register("tf_replace_text")
def tf_replace_text(text: TensorLike, old: str, new: str) -> TensorLike:
    """Replace substring in text tensor using TensorFlow."""
    import tensorflow as tf

    return tf.strings.regex_replace(text, old, new)


@TransformRegistry.register("tf_split_text")
def tf_split_text(text: TensorLike, separator: str = " ") -> TensorLike:
    """Split text tensor into tokens using TensorFlow."""
    import tensorflow as tf

    return tf.strings.split(text, separator)


@TransformRegistry.register("tf_join_text")
def tf_join_text(text_tokens: TensorLike, separator: str = "") -> TensorLike:
    """Join text tokens into single string using TensorFlow."""
    import tensorflow as tf

    return tf.strings.reduce_join(text_tokens, separator=separator)


@TransformRegistry.register("tf_string_length")
def tf_string_length(text: TensorLike) -> TensorLike:
    """Get length of text tensor using TensorFlow."""
    import tensorflow as tf

    return tf.strings.length(text)


@TransformRegistry.register("tf_substring")
def tf_substring(text: TensorLike, start: int, length: int) -> TensorLike:
    """Extract substring from text tensor using TensorFlow."""
    import tensorflow as tf

    return tf.strings.substr(text, start, length)
