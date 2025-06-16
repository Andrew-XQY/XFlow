"""Pipeline transformation utilities for data preprocessing."""
from typing import Iterator, List, Any, Optional, Tuple, Iterable, Callable, Dict
from .pipeline import BasePipeline
from ..utils.decorators import with_progress
import random
import logging
import itertools


@with_progress
def apply_transforms_to_dataset(
    data: Iterable[Any],
    transforms: List[Any],  # Transform objects or callables
    *,
    logger: Optional[logging.Logger] = None,
    skip_errors: bool = True
) -> Tuple[List[Any], int]:
    """Apply transforms to all items in an iterable.
    
    Args:
        data: Iterable of raw data items
        transforms: List of transform functions to apply sequentially
        logger: Optional logger for error reporting
        skip_errors: Whether to skip failed items or raise errors
    
    Returns:
        Tuple of (processed_items, error_count)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
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
    """Pipeline that shuffles items using reservoir sampling.
    
    Applies shuffle transformation to a base pipeline using a fixed-size buffer
    for memory-efficient shuffling of large datasets.
    
    Args:
        base: Base pipeline to shuffle.
        buffer_size: Size of the shuffle buffer for reservoir sampling.
    """
    
    def __init__(self, base: BasePipeline, buffer_size: int) -> None:
        # Copy essential attributes from base pipeline
        self.data_provider = base.data_provider
        self.preprocess_fns = base.preprocess_fns
        self.logger = base.logger
        self.on_error = base.on_error
        self.error_handler = base.error_handler
        self.cache = False  # Disable caching for transforms
        self._cached_data = None
        
        # Transform-specific attributes
        self.base = base
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Any]:
        """Iterate over shuffled items using reservoir sampling."""
        it = self.base.__iter__()
        buf = list(itertools.islice(it, self.buffer_size))
        random.shuffle(buf)
        
        # Yield initial buffer
        for x in buf:
            yield x
            
        # Reservoir sampling for remaining items
        for x in it:
            buf[random.randrange(self.buffer_size)] = x
            random.shuffle(buf)
            yield buf.pop()

    def __len__(self) -> int:
        """Return the number of items (same as base pipeline)."""
        return len(self.base)

    def to_framework_dataset(self) -> Any:
        """Convert to framework-native dataset with shuffle applied."""
        return self.base.to_framework_dataset().shuffle(self.buffer_size)


class BatchPipeline(BasePipeline):
    """Pipeline that batches items into lists of specified size.
    
    Groups items from a base pipeline into batches for efficient processing.
    The last batch may be smaller if the dataset size is not divisible by batch_size.
    
    Args:
        base: Base pipeline to batch.
        batch_size: Number of items per batch.
    """
    
    def __init__(self, base: BasePipeline, batch_size: int) -> None:
        # Copy essential attributes from base pipeline
        self.data_provider = base.data_provider
        self.preprocess_fns = base.preprocess_fns
        self.logger = base.logger
        self.on_error = base.on_error
        self.error_handler = base.error_handler
        self.cache = False  # Disable caching for transforms
        self._cached_data = None
        
        # Transform-specific attributes
        self.base = base
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[Any]]:
        """Iterate over batched items."""
        it = self.base.__iter__()
        while True:
            batch = list(itertools.islice(it, self.batch_size))
            if not batch:
                break
            yield batch

    def __len__(self) -> int:
        """Return the number of batches."""
        return (len(self.base) + self.batch_size - 1) // self.batch_size

    def to_framework_dataset(self) -> Any:
        """Convert to framework-native dataset with batching applied."""
        return self.base.to_framework_dataset().batch(self.batch_size)


class TransformRegistry:
    """Registry for all available transforms."""
    _transforms: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register transforms."""
        def decorator(func):
            cls._transforms[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Callable:
        """Get transform by name."""
        if name not in cls._transforms:
            raise ValueError(f"Transform '{name}' not found. Available: {list(cls._transforms.keys())}")
        return cls._transforms[name]
    
    @classmethod
    def list_transforms(cls) -> list:
        """List all registered transforms."""
        return list(cls._transforms.keys())


# Register transforms
@TransformRegistry.register("load_image")
def load_image(path: str):
    from PIL import Image
    return Image.open(path)

@TransformRegistry.register("resize")
def resize(image, size: tuple):
    return image.resize(size)

# Framework-specific transforms
@TransformRegistry.register("tf_decode_image")
def tf_decode_image(image_bytes):
    import tensorflow as tf
    return tf.image.decode_image(image_bytes)

@TransformRegistry.register("tf_resize")
def tf_resize(image, size: list):
    import tensorflow as tf
    return tf.image.resize(image, size)