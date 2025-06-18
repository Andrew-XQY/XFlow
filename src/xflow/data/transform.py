"""Pipeline transformation utilities for data preprocessing."""
from typing import Iterator, List, Any, Optional, Tuple, Iterable, Callable, Dict
from functools import partial
from pathlib import Path
import itertools
import logging
import random

import numpy as np
from PIL import Image

from .pipeline import BasePipeline
from ..utils.decorator import with_progress


@with_progress
def apply_transforms_to_dataset(
    data: Iterable[Any],
    transforms: List[Callable],
    *,
    logger: Optional[logging.Logger] = None,
    skip_errors: bool = True
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
        self._copy_base_attributes(base)
        self.base = base
        self.buffer_size = buffer_size

    def _copy_base_attributes(self, base: BasePipeline) -> None:
        """Copy essential attributes from base pipeline."""
        self.data_provider = base.data_provider
        self.preprocess_fns = base.preprocess_fns
        self.logger = base.logger
        self.on_error = base.on_error
        self.error_handler = base.error_handler
        self.cache = False
        self._cached_data = None

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

    def to_framework_dataset(self) -> Any:
        return self.base.to_framework_dataset().shuffle(self.buffer_size)


class BatchPipeline(BasePipeline):
    """Groups items into fixed-size batches."""
    
    def __init__(self, base: BasePipeline, batch_size: int) -> None:
        self._copy_base_attributes(base)
        self.base = base
        self.batch_size = batch_size

    def _copy_base_attributes(self, base: BasePipeline) -> None:
        self.data_provider = base.data_provider
        self.preprocess_fns = base.preprocess_fns
        self.logger = base.logger
        self.on_error = base.on_error
        self.error_handler = base.error_handler
        self.cache = False
        self._cached_data = None

    def __iter__(self) -> Iterator[List[Any]]:
        it = self.base.__iter__()
        while True:
            batch = list(itertools.islice(it, self.batch_size))
            if not batch:
                break
            yield batch

    def __len__(self) -> int:
        return (len(self.base) + self.batch_size - 1) // self.batch_size

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
            raise ValueError(f"Transform '{name}' not found. Available: {list(cls._transforms.keys())}")
        return cls._transforms[name]
    
    @classmethod
    def list_transforms(cls) -> List[str]:
        return list(cls._transforms.keys())


# Core transforms
@TransformRegistry.register("load_image")
def load_image(path) -> Image.Image:
    """Load image from file path."""
    return Image.open(Path(path))


@TransformRegistry.register("to_narray")
def to_numpy_array(image) -> np.ndarray:
    """Convert image to numpy array."""
    if hasattr(image, 'numpy'):  # TensorFlow tensor
        return image.numpy()
    elif isinstance(image, Image.Image):  # PIL Image
        return np.array(image)
    elif hasattr(image, '__array__'):  # Array-like objects
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
            return np.mean(image.reshape(image.shape[:2] + (-1,)), axis=2).astype(image.dtype)
    else:
        spatial_dims = image.shape[:2]
        flattened = image.reshape(spatial_dims + (-1,))
        return np.mean(flattened, axis=2).astype(image.dtype)


@TransformRegistry.register("remap_range")
def remap_range(image: np.ndarray, target_min: float = 0.0, target_max: float = 1.0) -> np.ndarray:
    """Remap pixel values to target range."""
    current_min = np.min(image)
    current_max = np.max(image)
    
    if current_max == current_min:
        return np.full_like(image, target_min, dtype=np.float32)
    
    normalized = (image - current_min) / (current_max - current_min)
    remapped = normalized * (target_max - target_min) + target_min
    
    return remapped.astype(np.float32)


@TransformRegistry.register("resize")
def resize(image: np.ndarray, size: Tuple[int, int], interpolation: str = "lanczos") -> np.ndarray:
    """Resize image using OpenCV."""
    import cv2
    
    target_height, target_width = size
    
    interp_map = {
        "lanczos": cv2.INTER_LANCZOS4,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST
    }
    
    cv_interpolation = interp_map.get(interpolation, cv2.INTER_LANCZOS4)
    return cv2.resize(image, (target_width, target_height), interpolation=cv_interpolation)

@TransformRegistry.register("expand_dims")
def expand_dims(image: np.ndarray, axis: int = -1) -> np.ndarray:
    """Add a dimension of size 1 at the specified axis."""
    return np.expand_dims(image, axis=axis)

@TransformRegistry.register("squeeze")
def squeeze(image: np.ndarray, axis: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Remove dimensions of size 1 from the array."""
    return np.squeeze(image, axis=axis)

@TransformRegistry.register("split_width")
def split_width(image: np.ndarray) -> List[np.ndarray]:
    """Split image at width midpoint."""
    height, width = image.shape[:2]
    mid_point = width // 2
    
    return [image[:, :mid_point], image[:, mid_point:]]


# TensorFlow transforms
@TransformRegistry.register("tf_read_file")
def tf_read_file(filepath):
    """Read file contents as bytes using TensorFlow."""
    import tensorflow as tf
    return tf.io.read_file(filepath)

@TransformRegistry.register("tf_decode_image")
def tf_decode_image(image_bytes):
    """Decode image bytes to tensor using TensorFlow."""
    import tensorflow as tf
    return tf.image.decode_image(image_bytes)

@TransformRegistry.register("tf_normalize")
def tf_normalize(image, mean: float = 0.0, std: float = 1.0):
    import tensorflow as tf
    return tf.cast(image, tf.float32) / 255.0 * std + mean

@TransformRegistry.register("tf_resize")
def tf_resize(image, size: List[int]):
    """Resize image using TensorFlow."""
    import tensorflow as tf
    return tf.image.resize(image, size)

@TransformRegistry.register("tf_to_grayscale")
def tf_to_grayscale(image):
    """Convert to grayscale using TensorFlow."""
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(image)

@TransformRegistry.register("tf_split_width")
def tf_split_width(image):
    """Split image at width midpoint using TensorFlow."""
    import tensorflow as tf
    width = tf.shape(image)[1]
    mid_point = width // 2
    left_half = image[:, :mid_point]
    right_half = image[:, mid_point:]
    return left_half, right_half

@TransformRegistry.register("tf_expand_dims")
def tf_expand_dims(image, axis: int = -1):
    """Add dimension to tensor."""
    import tensorflow as tf
    return tf.expand_dims(image, axis)

@TransformRegistry.register("tf_squeeze")
def tf_squeeze(image, axis: List[int] = None):
    """Remove dimensions of size 1."""
    import tensorflow as tf
    return tf.squeeze(image, axis)

def build_transforms_from_config(
    config: List[Dict[str, Any]], 
    name_key: str = "name",
    params_key: str = "params"
) -> List[Callable]:
    """Build transform pipeline from configuration."""
    transforms = []
    for transform_config in config:
        if name_key not in transform_config:
            raise ValueError(f"Transform config missing '{name_key}' key: {transform_config}")
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
def tf_shuffle(dataset, buffer_size: int, seed: int = None):
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
    params_key: str = "params"
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

                