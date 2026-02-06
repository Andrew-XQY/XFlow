"""Pipeline transformation utilities for data preprocessing."""

import itertools
import logging
import random
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from PIL import Image

from ..utils.decorator import with_progress
from ..utils.io import resolve_save_path
from ..utils.typing import ImageLike, PathLikeStr, TensorLike
from ..utils.visualization import to_numpy_image
from .pipeline import BasePipeline, Transform

# Only for type checkers; won't import torch at runtime
if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset  # noqa: F401

# Runtime-safe base: real Dataset if available, else a stub so this module imports fine
try:
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore
except Exception:

    class _TorchDataset:  # minimal stub
        pass


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


@TransformRegistry.register("join_image")
def join_image(images: Iterable[ImageLike], layout: Tuple[int, int]) -> ImageLike:
    """Tile images using (rows, cols) layout and return merged image."""

    img_list = list(images)
    if not img_list:
        raise ValueError("join_image expects at least one image")

    rows, cols = layout
    if rows <= 0 or cols <= 0:
        raise ValueError("Layout values must be positive")
    if rows * cols != len(img_list):
        raise ValueError(
            f"Layout {layout} must match number of images ({len(img_list)})"
        )

    np_imgs = [to_numpy_image(img) for img in img_list]
    first_shape = np_imgs[0].shape
    if any(img.shape != first_shape for img in np_imgs[1:]):
        raise ValueError("All images must share the same shape")

    row_blocks = []
    for r in range(rows):
        start = r * cols
        row_blocks.append(np.concatenate(np_imgs[start : start + cols], axis=1))
    mosaic = np.concatenate(row_blocks, axis=0)

    first = img_list[0]
    if isinstance(first, Image.Image):
        if mosaic.dtype != np.uint8:
            mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
        return Image.fromarray(mosaic)
    if isinstance(first, np.ndarray):
        return mosaic
    return mosaic


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


@TransformRegistry.register("tf_crop_area")
def tf_crop_area(image: TensorLike, points: Sequence[Tuple[int, int]]) -> TensorLike:
    """Crop a rectangular area from image tensor defined by two corner points.

    Args:
        image: Input tensor with shape (H, W, C) or (H, W) (TensorFlow format)
        points: Two corner points as [(x1, y1), (x2, y2)] or ((x1, y1), (x2, y2))
                where x is column index, y is row index. Can be any iterable of two points.

    Returns:
        Cropped tensor preserving the original format

    Examples:
        >>> # Crop region from (10, 20) to (100, 150) from HWC tensor
        >>> image = tf.random.normal([224, 224, 3])
        >>> cropped = tf_crop_area(image, [(10, 20), (100, 150)])
        >>> # Result shape: (130, 90, 3) - preserves C dimension

        >>> # Works with grayscale too
        >>> image = tf.random.normal([224, 224, 1])
        >>> cropped = tf_crop_area(image, [[50, 50], [150, 150]])
        >>> # Result shape: (100, 100, 1)
    """
    import tensorflow as tf

    point1, point2 = points
    x1, y1 = point1
    x2, y2 = point2

    # Ensure coordinates are in correct order (top-left to bottom-right)
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # TensorFlow uses (H, W, C) format
    # Slicing: image[y_start:y_end, x_start:x_end, :]
    rank = tf.rank(image)

    # Handle 2D (H, W) or 3D (H, W, C)
    cropped = tf.cond(
        tf.equal(rank, 2),
        lambda: image[y_min:y_max, x_min:x_max],
        lambda: image[y_min:y_max, x_min:x_max, :],
    )

    return cropped


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

        # Check if params has 'transforms' list (multi-branch pattern)
        if "transforms" in params and isinstance(params["transforms"], list):
            processed_params = params.copy()
            nested_transforms = []

            for nested_config in params["transforms"]:
                if nested_config is None:
                    nested_transforms.append(None)
                else:
                    nested_transforms.append(
                        build_transform_closure(nested_config, name_key, params_key)
                    )

            processed_params["transforms"] = nested_transforms
            transform_fn = partial(TransformRegistry.get(name), **processed_params)
        else:
            # Regular transform - original behavior preserved
            transform_fn = TransformRegistry.get(name)
            if params:
                transform_fn = partial(transform_fn, **params)

        transforms.append(_wrap_transform_callable(transform_fn, name))
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


@TransformRegistry.register("add_parent_dir")
def add_parent_dir(path: PathLikeStr, parent_dir: PathLikeStr) -> str:
    """Prepend parent directory to file path using pathlib for cross-platform safety.

    Args:
        path: Relative or absolute file path
        parent_dir: Parent directory to prepend

    Returns:
        Full path as string

    Examples:
        >>> add_parent_dir("image.jpg", "/data/images")
        '/data/images/image.jpg'

        >>> add_parent_dir("train/img.jpg", "C:\\\\data")
        'C:\\\\data\\\\train\\\\img.jpg'  # Windows
    """
    return str(Path(parent_dir) / path)


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


# PyTorch/torchvision transforms
@TransformRegistry.register("torch_load_image")
def torch_load_image(path: PathLikeStr) -> TensorLike:
    """Load image from file path using torchvision."""
    try:
        import torchvision.io

        return torchvision.io.read_image(str(path))
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_load_image_with_meta")
def torch_load_image_with_meta(
    path: PathLikeStr,
    meta_fn: Optional[Callable[[PathLikeStr], Dict[str, Any]]] = None,
) -> Tuple[TensorLike, Dict[str, Any]]:
    """Load image and return (image, metadata) tuple."""
    try:
        import torchvision.io as io

        p = Path(path)
        image = io.read_image(str(p))

        if meta_fn is not None:
            meta = meta_fn(path)
        else:
            meta = {"path": str(p), "filename": p.stem}

        return (image, meta)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_to_tensor")
def torch_to_tensor(image: ImageLike) -> TensorLike:
    """Convert image to PyTorch tensor."""
    try:
        import torch
        import torchvision.transforms.functional as F
        from PIL import Image

        if isinstance(image, Image.Image):
            return F.to_tensor(image)
        elif isinstance(image, np.ndarray):
            return torch.from_numpy(image).float()
        elif hasattr(image, "__array__"):
            return torch.from_numpy(np.asarray(image)).float()
        else:
            raise ValueError(f"Cannot convert {type(image)} to PyTorch tensor")
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_to_pil")
def torch_to_pil(tensor: TensorLike) -> Image.Image:
    """Convert PyTorch tensor to PIL Image."""
    try:
        import torchvision.transforms.functional as F

        return F.to_pil_image(tensor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_flatten")
def torch_flatten(
    tensor: TensorLike,
    start_dim: int = 1,
    end_dim: int = -1,
    make_contiguous: bool = True,
) -> TensorLike:
    """Flatten tensor dimensions for vectorization (e.g., image serialization).

    This is the standard PyTorch approach for converting multi-dimensional tensors
    into vectors while preserving batch dimensions or other specified dimensions.
    Commonly used for:
    - Image vectorization: (B, C, H, W) -> (B, C*H*W)
    - Feature flattening: (B, H, W, C) -> (B, H*W*C)
    - Complete flattening: (H, W, C) -> (H*W*C,)

    Args:
        tensor: Input PyTorch tensor to flatten
        start_dim: First dimension to flatten (inclusive). Default: 1 (preserve batch)
        end_dim: Last dimension to flatten (inclusive). Default: -1 (last dimension)
        make_contiguous: Whether to ensure output is contiguous in memory for better performance

    Returns:
        Flattened tensor with dimensions from start_dim to end_dim collapsed into a single dimension

    Examples:
        >>> # Image vectorization preserving batch: (32, 3, 224, 224) -> (32, 150528)
        >>> images = torch.randn(32, 3, 224, 224)
        >>> flattened = torch_flatten(images)  # start_dim=1 by default

        >>> # Complete flattening: (3, 224, 224) -> (150528,)
        >>> image = torch.randn(3, 224, 224)
        >>> vector = torch_flatten(image, start_dim=0)

        >>> # Flatten spatial dimensions only: (32, 256, 7, 7) -> (32, 256, 49)
        >>> features = torch.randn(32, 256, 7, 7)
        >>> spatial_flat = torch_flatten(features, start_dim=2)

        >>> # Flatten everything except last dim: (32, 256, 7, 7) -> (114688, 7)
        >>> flattened = torch_flatten(features, start_dim=0, end_dim=2)
    """
    try:
        import torch

        # Use torch.flatten which is the standard and most efficient approach
        flattened = torch.flatten(tensor, start_dim=start_dim, end_dim=end_dim)

        # Ensure contiguous memory layout for better performance if requested
        if make_contiguous and not flattened.is_contiguous():
            flattened = flattened.contiguous()

        return flattened

    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_remap_range")
def torch_remap_range(
    tensor: TensorLike,
    current_min: float = 0.0,
    current_max: float = 255.0,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> TensorLike:
    """Remap tensor values from [current_min, current_max] to [target_min, target_max] using PyTorch."""
    try:
        import torch

        tensor = tensor.float()
        denominator = current_max - current_min
        if denominator == 0:
            return torch.full_like(tensor, target_min, dtype=torch.float32)
        normalized = (tensor - current_min) / denominator
        remapped = normalized * (target_max - target_min) + target_min
        return remapped
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_resize")
def torch_resize(
    tensor: TensorLike, size: List[int], interpolation: str = "bilinear"
) -> TensorLike:
    """Resize tensor using torchvision."""
    try:
        import torchvision.transforms.functional as F
        from torchvision.transforms import InterpolationMode

        interp_map = {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
            "lanczos": InterpolationMode.LANCZOS,
        }

        interp_mode = interp_map.get(interpolation, InterpolationMode.BILINEAR)
        return F.resize(tensor, size, interpolation=interp_mode)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_center_crop")
def torch_center_crop(tensor: TensorLike, size: List[int]) -> TensorLike:
    """Center crop tensor using torchvision."""
    try:
        import torchvision.transforms.functional as F

        return F.center_crop(tensor, size)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_random_crop")
def torch_random_crop(tensor: TensorLike, size: List[int]) -> TensorLike:
    """Random crop tensor using torchvision."""
    try:
        import torchvision.transforms as T

        transform = T.RandomCrop(size)
        return transform(tensor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_crop_area")
def torch_crop_area(
    tensor: TensorLike, points: Sequence[Tuple[int, int]]
) -> TensorLike:
    """Crop a rectangular area from tensor defined by two corner points.

    Args:
        tensor: Input tensor with shape (..., C, H, W) or (H, W, C) or (H, W)
        points: Two corner points as [(x1, y1), (x2, y2)] or ((x1, y1), (x2, y2))
                where x is column index, y is row index. Can be any iterable of two points.

    Returns:
        Cropped tensor preserving the original format

    Examples:
        >>> # Crop region from (10, 20) to (100, 150) from CHW tensor
        >>> tensor = torch.randn(3, 224, 224)
        >>> cropped = torch_crop_area(tensor, [(10, 20), (100, 150)])
        >>> # Result shape: (3, 130, 90) - preserves C dimension

        >>> # Works with batched tensors too
        >>> tensor = torch.randn(32, 3, 224, 224)
        >>> cropped = torch_crop_area(tensor, ((50, 50), (150, 150)))
        >>> # Result shape: (32, 3, 100, 100)
    """
    try:
        import torch

        point1, point2 = points
        x1, y1 = point1
        x2, y2 = point2

        # Ensure coordinates are in correct order (top-left to bottom-right)
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Handle different tensor formats
        if tensor.dim() == 2:
            # (H, W) format
            return tensor[y_min:y_max, x_min:x_max]
        elif tensor.dim() == 3:
            # Could be (C, H, W) or (H, W, C)
            # Assume (C, H, W) format (PyTorch standard)
            return tensor[:, y_min:y_max, x_min:x_max]
        elif tensor.dim() >= 4:
            # Batched: (..., C, H, W)
            return tensor[..., :, y_min:y_max, x_min:x_max]
        else:
            raise ValueError(f"Unexpected tensor dimension: {tensor.dim()}")

    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_horizontal_flip")
def torch_horizontal_flip(tensor: TensorLike) -> TensorLike:
    """Horizontally flip tensor using torchvision."""
    try:
        import torchvision.transforms.functional as F

        return F.hflip(tensor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_vertical_flip")
def torch_vertical_flip(tensor: TensorLike) -> TensorLike:
    """Vertically flip tensor using torchvision."""
    try:
        import torchvision.transforms.functional as F

        return F.vflip(tensor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_random_horizontal_flip")
def torch_random_horizontal_flip(tensor: TensorLike, p: float = 0.5) -> TensorLike:
    """Randomly horizontally flip tensor using torchvision."""
    try:
        import torchvision.transforms as T

        transform = T.RandomHorizontalFlip(p=p)
        return transform(tensor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_random_vertical_flip")
def torch_random_vertical_flip(tensor: TensorLike, p: float = 0.5) -> TensorLike:
    """Randomly vertically flip tensor using torchvision."""
    try:
        import torchvision.transforms as T

        transform = T.RandomVerticalFlip(p=p)
        return transform(tensor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_rotation")
def torch_rotation(
    tensor: TensorLike, angle: float, interpolation: str = "bilinear"
) -> TensorLike:
    """Rotate tensor by angle using torchvision."""
    try:
        import torchvision.transforms.functional as F
        from torchvision.transforms import InterpolationMode

        interp_map = {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
        }

        interp_mode = interp_map.get(interpolation, InterpolationMode.BILINEAR)
        return F.rotate(tensor, angle, interpolation=interp_mode)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_random_rotation")
def torch_random_rotation(tensor: TensorLike, degrees: List[float]) -> TensorLike:
    """Randomly rotate tensor using torchvision."""
    try:
        import torchvision.transforms as T

        transform = T.RandomRotation(degrees)
        return transform(tensor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_to_grayscale")
def torch_to_grayscale(tensor: TensorLike, num_output_channels: int = 1) -> TensorLike:
    """Convert tensor to grayscale, handling tensors shaped (..., C, H, W) or (H, W). Supports 1/3/4 channels."""
    import torch
    import torchvision.transforms.functional as F

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels must be 1 or 3.")

    # Normalize to have a channel dim
    if tensor.dim() == 2:  # (H, W)
        tensor = tensor.unsqueeze(0)  # (1, H, W)

    if tensor.dim() < 3:
        raise ValueError("Expected at least 3D tensor with channel dimension.")

    C = tensor.shape[-3]

    if C == 1:
        y = tensor
    elif C == 3:
        y = F.rgb_to_grayscale(tensor, num_output_channels=1)
    elif C == 4:
        y = F.rgb_to_grayscale(tensor[..., :3, :, :], num_output_channels=1)
    else:
        # Fallback: simple mean across channels
        y = (
            tensor.float().mean(dim=-3, keepdim=True).to(tensor.dtype)
            if tensor.is_floating_point()
            else tensor.mean(dim=-3, keepdim=True)
        )

    if num_output_channels == 3:
        y = y.repeat_interleave(3, dim=-3)

    return y


@TransformRegistry.register("torch_split_width")
def torch_split_width(
    tensor: TensorLike, swap: bool = False, width_dim: int = -1
) -> Tuple[TensorLike, TensorLike]:
    """Split tensor at width midpoint along specified dimension.

    Args:
        tensor: Input tensor to split
        swap: If True, return (right_half, left_half) instead of (left_half, right_half)
        width_dim: Dimension to split along (0, 1, 2, 3, etc. or -1 for last)

    Returns:
        Tuple of (left_half, right_half) or (right_half, left_half) if swap=True
    """
    try:
        import torch

        width = tensor.shape[width_dim]
        mid_point = width // 2

        left_half = torch.split(tensor, mid_point, dim=width_dim)[0]
        right_half = torch.split(tensor, mid_point, dim=width_dim)[1]

        if swap:
            return right_half, left_half
        return left_half, right_half
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_adjust_brightness")
def torch_adjust_brightness(tensor: TensorLike, brightness_factor: float) -> TensorLike:
    """Adjust brightness of tensor using torchvision."""
    try:
        import torchvision.transforms.functional as F

        return F.adjust_brightness(tensor, brightness_factor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_adjust_contrast")
def torch_adjust_contrast(tensor: TensorLike, contrast_factor: float) -> TensorLike:
    """Adjust contrast of tensor using torchvision."""
    try:
        import torchvision.transforms.functional as F

        return F.adjust_contrast(tensor, contrast_factor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_adjust_saturation")
def torch_adjust_saturation(tensor: TensorLike, saturation_factor: float) -> TensorLike:
    """Adjust saturation of tensor using torchvision."""
    try:
        import torchvision.transforms.functional as F

        return F.adjust_saturation(tensor, saturation_factor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_adjust_hue")
def torch_adjust_hue(tensor: TensorLike, hue_factor: float) -> TensorLike:
    """Adjust hue of tensor using torchvision."""
    try:
        import torchvision.transforms.functional as F

        return F.adjust_hue(tensor, hue_factor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_gaussian_blur")
def torch_gaussian_blur(
    tensor: TensorLike, kernel_size: List[int], sigma: List[float] = None
) -> TensorLike:
    """Apply Gaussian blur to tensor using torchvision."""
    try:
        import torchvision.transforms.functional as F

        return F.gaussian_blur(tensor, kernel_size, sigma)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_pad")
def torch_pad(
    tensor: TensorLike,
    padding: List[int],
    fill: float = 0,
    padding_mode: str = "constant",
) -> TensorLike:
    """Pad tensor using torchvision."""
    try:
        import torchvision.transforms.functional as F

        return F.pad(tensor, padding, fill=fill, padding_mode=padding_mode)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_random_crop_resize")
def torch_random_crop_resize(
    tensor: TensorLike,
    size: List[int],
    scale: List[float] = (0.8, 1.0),
    ratio: List[float] = (0.75, 1.33),
) -> TensorLike:
    """Random crop and resize tensor using torchvision."""
    try:
        import torchvision.transforms as T

        transform = T.RandomResizedCrop(size, scale=scale, ratio=ratio)
        return transform(tensor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_color_jitter")
def torch_color_jitter(
    tensor: TensorLike,
    brightness: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    hue: float = 0,
) -> TensorLike:
    """Apply color jitter to tensor using torchvision."""
    try:
        import torchvision.transforms as T

        transform = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        return transform(tensor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_permute")
def torch_permute(
    tensor, dims=None, format_from="BHWC", format_to="BCHW", make_contiguous=False
):
    """
    Permute tensor dims either explicitly (dims) or via format strings.
    Examples:
      torch_permute(x, dims=[0,3,1,2])           # BHWC -> BCHW
      torch_permute(x, format_from="BHWC", format_to="BCHW")
      torch_permute(x, format_from="HWC",  format_to="CHW")
    """
    import torch

    rank = tensor.dim()

    if dims is not None:
        if len(dims) != rank:
            raise ValueError(f"dims length {len(dims)} != tensor rank {rank}")
        if sorted(dims) != list(range(rank)):
            raise ValueError(f"dims must be a permutation of 0..{rank-1}, got {dims}")
        out = tensor.permute(*dims)
        return out.contiguous() if make_contiguous else out

    # Normalize format strings
    fr = "".join(format_from.split()).upper()
    to = "".join(format_to.split()).upper()

    if len(fr) != len(to):
        raise ValueError(f"format lengths differ: {fr} vs {to}")
    if len(fr) != rank:
        raise ValueError(f"format length {len(fr)} != tensor rank {rank}")
    if len(set(fr)) != len(fr) or len(set(to)) != len(to):
        raise ValueError("format chars must be unique (e.g., no repeated 'H')")
    if set(fr) != set(to):
        raise ValueError(f"formats must contain same symbols: {fr} vs {to}")

    # Build permutation: for each target char, find its index in source
    idx = [fr.index(ch) for ch in to]
    out = tensor.permute(*idx)
    return out.contiguous() if make_contiguous else out


@TransformRegistry.register("torch_squeeze")
def torch_squeeze(tensor: TensorLike, dim: Optional[int] = None) -> TensorLike:
    """Remove dimensions of size 1 from PyTorch tensor.

    This is the PyTorch equivalent of the numpy squeeze function, designed to
    handle image channel squeezing operations like:
    - (256, 256, 1) -> (256, 256)  # Remove trailing single channel
    - (1, 256, 256) -> (256, 256)  # Remove leading single channel
    - (1, 1, 256, 256) -> (256, 256)  # Remove multiple single dimensions

    Args:
        tensor: Input PyTorch tensor
        dim: If given, only removes dimensions of size 1 at the specified dimension.
             If None, removes all dimensions of size 1.

    Returns:
        Squeezed tensor with single-size dimensions removed

    Examples:
        >>> # Remove trailing channel dimension: (H, W, 1) -> (H, W)
        >>> tensor = torch.randn(256, 256, 1)
        >>> squeezed = torch_squeeze(tensor, dim=2)  # or dim=-1

        >>> # Remove leading batch/channel dimension: (1, H, W) -> (H, W)
        >>> tensor = torch.randn(1, 256, 256)
        >>> squeezed = torch_squeeze(tensor, dim=0)

        >>> # Remove all single dimensions automatically
        >>> tensor = torch.randn(1, 256, 256, 1)
        >>> squeezed = torch_squeeze(tensor)  # -> (256, 256)
    """
    try:
        import torch

        if dim is not None:
            # Only squeeze the specified dimension if it has size 1
            if tensor.size(dim) == 1:
                return torch.squeeze(tensor, dim=dim)
            else:
                return tensor  # Return unchanged if dimension is not size 1
        else:
            # Squeeze all dimensions of size 1
            return torch.squeeze(tensor)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_unsqueeze")
def torch_unsqueeze(tensor: TensorLike, dim: int) -> TensorLike:
    """Add dimension to PyTorch tensor at specified position.

    This is the PyTorch equivalent of numpy's expand_dims function, useful for:
    - Adding batch dimension: (H, W, C) -> (1, H, W, C)
    - Adding channel dimension: (H, W) -> (H, W, 1)
    - Preparing tensors for operations that require specific dimensionality

    Args:
        tensor: Input PyTorch tensor
        dim: Position where the new axis is placed

    Returns:
        Tensor with an additional dimension of size 1 inserted at the specified position

    Examples:
        >>> # Add batch dimension at the beginning: (H, W, C) -> (1, H, W, C)
        >>> tensor = torch.randn(256, 256, 3)
        >>> batched = torch_unsqueeze(tensor, dim=0)

        >>> # Add channel dimension at the end: (H, W) -> (H, W, 1)
        >>> tensor = torch.randn(256, 256)
        >>> with_channel = torch_unsqueeze(tensor, dim=-1)

        >>> # Add dimension for broadcasting: (N,) -> (N, 1)
        >>> tensor = torch.randn(256, 256)
        >>> batched = torch_unsqueeze(tensor, dim=0)
    """
    try:
        import torch

        return torch.unsqueeze(tensor, dim=dim)
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")


@TransformRegistry.register("torch_debug_shape")
def torch_debug_shape(
    tensor: TensorLike,
    label: str = "tensor",
    show_stats: bool = False,
    blocking: bool = False,
) -> TensorLike:
    """Debug utility that prints tensor shape and passes data through unchanged.

    Useful for inspecting data flow in transform pipelines without modifying the data.
    Can be inserted anywhere in a pipeline to understand tensor dimensions.

    Args:
        tensor: Input PyTorch tensor (passed through unchanged)
        label: Descriptive label for the tensor (default: "tensor")
        show_stats: Whether to show additional statistics (mean, std, min, max)
        blocking: If True, waits for user input before continuing (useful for step-by-step debugging)

    Returns:
        The input tensor unchanged

    Examples:
        >>> # Basic shape debugging
        >>> x = torch.randn(32, 3, 224, 224)
        >>> x = torch_debug_shape(x, "after_loading")
        # Prints: "[DEBUG] after_loading: torch.Size([32, 3, 224, 224]) | dtype: float32"

        >>> # With statistics and blocking
        >>> x = torch_debug_shape(x, "normalized", show_stats=True, blocking=True)
        # Prints: "[DEBUG] normalized: torch.Size([32, 3, 224, 224]) | dtype: float32 | μ=0.02 σ=1.0 [min=-2.1, max=2.3]"
        # Waits: "Press Enter to continue..."

        >>> # Step-by-step pipeline debugging
        >>> x = torch_debug_shape(x, "critical_point", blocking=True)
        # Pauses execution to examine this specific step
    """
    try:
        import torch

        # Basic info
        shape_str = f"[DEBUG] {label}: {tensor.shape} | dtype: {tensor.dtype}"

        if show_stats and tensor.numel() > 0:
            if tensor.is_floating_point():
                mean_val = tensor.mean().item()
                std_val = tensor.std().item() if tensor.numel() > 1 else 0.0
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                shape_str += f" | μ={mean_val:.2f} σ={std_val:.2f} [min={min_val:.1f}, max={max_val:.1f}]"
            else:
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                shape_str += f" | range=[{min_val}, {max_val}]"

        print(shape_str)

        if blocking:
            input("Press Enter to continue...")

        return tensor

    except ImportError:
        print(f"[DEBUG] {label}: <transform failed, please check the source code>")
        if blocking:
            input("Press Enter to continue...")
        return tensor


@TransformRegistry.register("torch_shape")
def torch_shape(tensor: TensorLike, label: str = "") -> TensorLike:
    """Minimal shape debug utility - just prints shape and passes through.

    Ultra-simple version for quick debugging. Just prints the shape with
    optional label and returns the tensor unchanged.

    Args:
        tensor: Input tensor (unchanged)
        label: Optional prefix label

    Returns:
        Input tensor unchanged

    Examples:
        >>> x = torch_shape(torch.randn(3, 224, 224), "input")
        # Prints: "input: (3, 224, 224)"

        >>> x = torch_shape(x)  # No label
        # Prints: "(3, 224, 224)"
    """
    try:
        import torch

        if label:
            print(f"{label}: {tuple(tensor.shape)}")
        else:
            print(f"{tuple(tensor.shape)}")
        return tensor
    except ImportError:
        print(
            f"{label}: <transform failed, please check the source code>"
            if label
            else "<transform failed, please check the source code>"
        )
        return tensor


@TransformRegistry.register("identity")
def identity(x):
    """Identity transform - passes input through unchanged.

    Args:
        x: Any input (tensor, image, text, tuple, etc.)

    Returns:
        The input unchanged
    """
    return x


# PyTorch dataset operations
@DatasetOperationRegistry.register("torch_batch")
def torch_batch(
    dataset: "_TorchDataset",
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    collate_fn: Optional[Any] = None,
    pin_memory_device: str = "",
    worker_init_fn=None,
    prefetch_factor: Optional[int] = None,
    seed: Optional[int] = None,  # <--- new
):
    """Wrap a dataset in a PyTorch DataLoader for batching with optional seed."""
    import torch
    from torch.utils.data import DataLoader

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        if worker_init_fn is None and num_workers > 0:

            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            worker_init_fn = seed_worker

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        generator=generator,  # <--- key for deterministic shuffle
    )


@DatasetOperationRegistry.register("torch_subset")
def torch_subset(dataset: "_TorchDataset", indices: List[int]):
    """Create a subset of a dataset using specified indices."""
    try:
        from torch.utils.data import Subset  # lazy import
    except Exception:
        raise RuntimeError("Transform failed, please check the source code")
    return Subset(dataset, indices)


@DatasetOperationRegistry.register("torch_concat")
def torch_concat(datasets: List["_TorchDataset"]):
    """Concatenate multiple datasets into one."""
    try:
        from torch.utils.data import ConcatDataset  # lazy import
    except Exception:
        raise RuntimeError("Transform failed, please check the source code")
    return ConcatDataset(datasets)


@DatasetOperationRegistry.register("torch_random_split")
def torch_random_split(
    dataset: "_TorchDataset", lengths: Sequence[int], generator=None
):
    """Randomly split a dataset into non-overlapping subsets."""
    try:
        from torch.utils.data import random_split  # lazy import
    except Exception:
        raise RuntimeError("Transform failed, please check the source code")
    return random_split(dataset, lengths, generator=generator)


@TransformRegistry.register("multi_transform")
def multi_transform(inputs, transforms):
    """Apply different transforms to multiple inputs.

    Args:
        inputs: tuple/list of inputs (e.g., from split operations)
        transforms: list of transform functions, one per input

    Returns:
        tuple of transformed outputs
    """
    if not isinstance(inputs, (tuple, list)):
        raise ValueError("inputs must be tuple or list")

    if len(inputs) != len(transforms):
        raise ValueError(
            f"Number of inputs ({len(inputs)}) must match transforms ({len(transforms)})"
        )

    results = []
    for inp, transform in zip(inputs, transforms):
        if transform is not None:  # Allow None to mean "no transform"
            results.append(transform(inp))
        else:
            results.append(inp)

    return tuple(results)


def build_transform_closure(transform_config, name_key="name", params_key="params"):
    """Build a single transform function with preset parameters.

    Args:
        transform_config: dict with transform name and params

    Returns:
        Callable transform function with parameters bound
    """
    if isinstance(transform_config, str):
        # Simple case: just transform name, no params
        return TransformRegistry.get(transform_config)

    if name_key not in transform_config:
        raise ValueError(
            f"Transform config missing '{name_key}' key: {transform_config}"
        )

    name = transform_config[name_key]
    params = transform_config.get(params_key, {})
    transform_fn = TransformRegistry.get(name)

    if params:
        transform_fn = partial(transform_fn, **params)
    return _wrap_transform_callable(transform_fn, name)


def _wrap_transform_callable(fn: Callable, name: str) -> Transform:
    """Ensure callable has Transform wrapper so pipeline can report names."""
    if isinstance(fn, Transform):
        return fn
    wrapped = Transform(fn, name)
    return wrapped


@TransformRegistry.register("tuple_select")
def tuple_select(inputs, index=0):
    """Select specific item from tuple/list (useful after multi_transform)."""
    return inputs[index]


class TorchDataset(_TorchDataset):
    """Map-style Dataset wrapper for an indexable pipeline."""

    def __init__(self, pipeline):
        self.pipeline = pipeline  # expects __len__ and __getitem__

    def __len__(self):
        """Return number of samples in the pipeline."""
        return len(self.pipeline)

    def __getitem__(self, idx):
        """Return a single sample by index."""
        return self.pipeline[idx]


@TransformRegistry.register("save_image")
def save_image(
    data: TensorLike,
    directory: Optional[str] = None,
    filename: Optional[str] = None,
    extension: Optional[str] = None,
) -> TensorLike:
    """Save tensor/array as image file with flexible path resolution.

    Supports multiple input patterns:
    - Full path: save_image(img, directory="/output/result.png")
    - Dir + filename: save_image(img, directory="/output", filename="result.png")
    - Dir + filename + ext: save_image(img, directory="/output", filename="result", extension=".png")
    - Tuple input: save_image((img, "/output/result.png"))  # for pipeline use

    Creates output directory (recursively) if it doesn't exist.
    Uses file extension to determine format (png, tiff, jpg, etc.).

    Args:
        data: Image tensor/array, OR tuple of (tensor, full_path) for pipeline
        directory: Output directory, OR full file path if contains extension
        filename: Output filename (optional, with or without extension)
        extension: File extension if filename lacks one (e.g., '.png' or 'png')

    Returns:
        Original input unchanged (tensor or tuple)

    Raises:
        ValueError: If path cannot be resolved

    Examples:
        >>> save_image(tensor, directory="/output/result.png")
        >>> save_image(tensor, directory="/output", filename="result.png")
        >>> save_image((tensor, "/output/result.png"))  # tuple input for pipeline
    """
    original_data = data

    # Handle tuple input: (tensor, full_path)
    if isinstance(data, (tuple, list)) and len(data) == 2:
        tensor, path_from_tuple = data
        if directory is None:
            directory = str(path_from_tuple)
    else:
        tensor = data

    output_path = resolve_save_path(directory, filename, extension)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array = to_numpy_image(tensor)

    # Normalize dtype for saving
    if array.dtype in (np.float32, np.float64):
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    elif array.max() <= 1.0 and array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)

    Image.fromarray(array).save(output_path)

    return original_data


@TransformRegistry.register("apply")
def apply(data: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply an arbitrary function to the data.

    Useful for quick custom transformations without registration.

    Args:
        data: Input data
        fn: Function to apply

    Returns:
        Result of fn(data)

    Examples:
        >>> # In config (with a lambda reference or registered function)
        >>> apply(tensor, lambda x: x * 2)
    """
    return fn(data)


@TransformRegistry.register("flatten_nested")
def flatten_nested(data: Any) -> List[Any]:
    """Recursively flatten nested lists/tuples into a single flat list.

    Args:
        data: Input data (can be nested lists/tuples or single item)

    Returns:
        Flat list of all leaf elements

    Examples:
        >>> flatten_nested(((a, b), (c, d)))  # -> [a, b, c, d]
        >>> flatten_nested([a, [b, [c]]])     # -> [a, b, c]
        >>> flatten_nested(tensor)            # -> [tensor]
    """
    if not isinstance(data, (list, tuple)):
        return [data]

    result = []
    for item in data:
        if isinstance(item, (list, tuple)):
            result.extend(flatten_nested(item))
        else:
            result.append(item)
    return result


@TransformRegistry.register("collect")
def collect(data: Any, remove_none: bool = True, unwrap_single: bool = True) -> Any:
    """Collect results, optionally removing None values and unwrapping single items.

    Args:
        data: Input data (can be single item, tuple, or list)
        remove_none: If True, filter out None values
        unwrap_single: If True, return single item directly instead of list

    Returns:
        - None if all values are None (after filtering)
        - Single item if only one non-None value remains and unwrap_single=True
        - List of items otherwise

    Examples:
        >>> collect((tensor1, None, tensor2))  # -> [tensor1, tensor2]
        >>> collect((None, tensor1, None))     # -> tensor1
        >>> collect((None, None))              # -> None
        >>> collect(tensor)                    # -> tensor (passthrough)
    """
    # Handle non-iterable inputs
    if not isinstance(data, (list, tuple)):
        return None if (remove_none and data is None) else data

    items = list(data)

    # Remove None values if requested
    if remove_none:
        items = [x for x in items if x is not None]

    # Return based on result count
    if len(items) == 0:
        return None
    elif len(items) == 1 and unwrap_single:
        return items[0]
    else:
        return items


@TransformRegistry.register("discard")
def discard(data: Any) -> None:
    """Discard input and return None.

    Shorthand for constant(data, value=None). Use with collect() to remove.

    Examples:
        >>> # Discard first branch, keep second
        >>> multi_transform((a, b), [discard, identity])  # -> (None, b)
        >>> collect((None, b))  # -> b
    """
    return None


@TransformRegistry.register("raise_if_none")
def raise_if_none(data: Any, message: str = "Discarded sample") -> Any:
    """Raise exception if data is None, allowing pipeline to skip it.

    Use with skip_errors=True in pipe_each to filter out None samples.

    Args:
        data: Input data
        message: Error message for logging

    Returns:
        data unchanged if not None

    Raises:
        ValueError: If data is None

    Examples:
        >>> # In pipeline with skip_errors=True
        >>> results = list(pipe_each(
        ...     items,
        ...     some_transform,  # might return None
        ...     T.get("raise_if_none"),  # converts None to skip
        ...     skip_errors=True,
        ... ))
    """
    if data is None:
        raise ValueError(message)
    return data


@TransformRegistry.register("debug_print")
def debug_print(data: Any, label: str = "") -> Any:
    """Print data and pass through unchanged."""
    if label:
        print(f"{label}: {data}")
    else:
        print(data)
    return data


@TransformRegistry.register("extract_filename")
def extract_filename(path: PathLikeStr, with_extension: bool = True) -> str:
    """Extract filename from a file path.

    Args:
        path: File path (string or Path object)
        with_extension: If True, include extension. If False, return stem only.

    Returns:
        Filename string

    Examples:
        >>> extract_filename("/data/images/photo.jpg")
        'photo.jpg'

        >>> extract_filename("/data/images/photo.jpg", with_extension=False)
        'photo'

        >>> extract_filename("C:\\\\Users\\\\data\\\\image.png")
        'image.png'
    """
    p = Path(path)
    return p.name if with_extension else p.stem


@TransformRegistry.register("extract_stem")
def extract_stem(path: PathLikeStr) -> str:
    """Extract filename without extension from a file path.

    Shorthand for extract_filename(path, with_extension=False).

    Args:
        path: File path

    Returns:
        Filename without extension

    Examples:
        >>> extract_stem("/data/images/photo.jpg")
        'photo'
    """
    return Path(path).stem


@TransformRegistry.register("extract_parent")
def extract_parent(path: PathLikeStr) -> str:
    """Extract parent directory from a file path.

    Args:
        path: File path

    Returns:
        Parent directory as string

    Examples:
        >>> extract_parent("/data/images/photo.jpg")
        '/data/images'
    """
    return str(Path(path).parent)


@TransformRegistry.register("extract_extension")
def extract_extension(path: PathLikeStr, include_dot: bool = True) -> str:
    """Extract file extension from a file path.

    Args:
        path: File path
        include_dot: If True, include the dot (e.g., '.jpg'). If False, just 'jpg'.

    Returns:
        File extension

    Examples:
        >>> extract_extension("/data/photo.jpg")
        '.jpg'

        >>> extract_extension("/data/photo.jpg", include_dot=False)
        'jpg'
    """
    ext = Path(path).suffix
    return ext if include_dot else ext.lstrip(".")


@TransformRegistry.register("reorder")
def reorder(data: Tuple[Any, ...], order: Sequence[int]) -> Tuple[Any, ...]:
    """Reorder elements in a tuple by index.

    General-purpose transform for changing element order in broadcast pipelines.

    Args:
        data: Input tuple
        order: New order as indices (e.g., [1, 0, 2] swaps first two)

    Returns:
        Reordered tuple

    Raises:
        IndexError: If any index is out of range
        ValueError: If order length doesn't match data length

    Examples:
        >>> reorder((a, b, c), [2, 0, 1])
        (c, a, b)

        >>> reorder((img1, img2, meta), [1, 0, 2])  # Swap images, keep meta
        (img2, img1, meta)

        >>> # In pipeline before join_image
        >>> pipe(
        ...     (left, right, meta),
        ...     [consume(2, partial(reorder, order=[1, 0])), None],  # Swap images
        ...     [consume(2, lambda x: join_image(x, layout=(1,2))), None],
        ... )
    """
    if not isinstance(data, (tuple, list)):
        raise TypeError(f"reorder expects tuple/list, got {type(data)}")

    if len(order) != len(data):
        raise ValueError(
            f"order length ({len(order)}) must match data length ({len(data)})"
        )

    return tuple(data[i] for i in order)


@TransformRegistry.register("swap")
def swap(data: Tuple[Any, Any]) -> Tuple[Any, Any]:
    """Swap two elements in a pair. Shorthand for reorder(data, [1, 0]).

    Args:
        data: Tuple of exactly 2 elements

    Returns:
        Swapped tuple (b, a)

    Examples:
        >>> swap((left, right))
        (right, left)

        >>> # In pipeline
        >>> pipe((img1, img2), swap)
        (img2, img1)
    """
    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise ValueError(f"swap expects pair (2 elements), got {len(data)} elements")

    return (data[1], data[0])


@TransformRegistry.register("reverse")
def reverse(data: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Reverse order of all elements in tuple.

    Args:
        data: Input tuple

    Returns:
        Reversed tuple

    Examples:
        >>> reverse((a, b, c))
        (c, b, a)
    """
    if not isinstance(data, (tuple, list)):
        raise TypeError(f"reverse expects tuple/list, got {type(data)}")

    return tuple(reversed(data))


@TransformRegistry.register("l2_normalize")
def l2_normalize(
    image: np.ndarray,
    eps: float = 1e-12,
    axis: Optional[Tuple[int, ...]] = None,
    keep_dtype: bool = False,
) -> np.ndarray:
    """L2-normalize an image/array by its own energy.

    This is the standard "make this basis component unit-norm" operation:
        x_hat = x / (||x||_2 + eps)

    Args:
        image: Input array. Typically (H, W) or (H, W, C) or batched.
        eps: Small constant to avoid division by zero.
        axis: Axes to reduce over when computing the norm.
              - None: normalize over all elements (typical for a basis image)
              - For (H, W, C): use axis=(0, 1) to normalize each channel separately
              - For (B, H, W): use axis=(1, 2) to normalize each sample in the batch
        keep_dtype: If True, cast back to original dtype. Otherwise returns float32.

    Returns:
        L2-normalized array (float32 by default).
    """
    x = np.asarray(image)
    x_float = x.astype(np.float32, copy=False)

    # Default: normalize over all elements
    reduce_axis = axis

    # Compute L2 norm with stable eps guard
    sq = np.square(x_float)
    denom = np.sqrt(np.sum(sq, axis=reduce_axis, keepdims=True))
    denom = np.maximum(denom, eps)

    y = x_float / denom
    return y.astype(x.dtype, copy=False) if keep_dtype else y


@TransformRegistry.register("torch_l2_normalize")
def torch_l2_normalize(
    tensor: TensorLike,
    eps: float = 1e-12,
    dims: Optional[Tuple[int, ...]] = None,
) -> TensorLike:
    """L2-normalize a tensor by its own energy.

    Typical for a basis image:
      - CHW: dims=(0,1,2)
      - HW:  dims=(0,1)

    For batched BCHW:
      - per-sample normalize: dims=(1,2,3)

    Args:
        tensor: Input torch tensor.
        eps: Small constant to avoid division by zero.
        dims: Dimensions to reduce over. If None, reduce over all dims.

    Returns:
        Float32 normalized tensor.
    """
    try:
        import torch

        x = tensor.float()
        if dims is None:
            dims = tuple(range(x.dim()))
        denom = torch.sqrt(torch.sum(x * x, dim=dims, keepdim=True)).clamp_min(eps)
        return x / denom
    except ImportError:
        raise RuntimeError("Transform failed, please check the source code")
