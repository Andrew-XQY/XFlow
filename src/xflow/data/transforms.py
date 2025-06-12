"""Pipeline transformation utilities for data preprocessing."""

from __future__ import annotations
import random
import itertools
from typing import Iterator, List, Any
from .loader import BasePipeline, TData


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

    def __iter__(self) -> Iterator[TData]:
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

    def __iter__(self) -> Iterator[List[TData]]:
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