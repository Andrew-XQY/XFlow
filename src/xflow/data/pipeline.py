"""Core abstractions for building reusable, named preprocessing pipelines:"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Any, Optional, List, Union
from .provider import DataProvider
import logging
import itertools

@dataclass
class Transform:
    """Wrapper for a preprocessing function with metadata. (like len)"""
    fn: Callable[[Any], Any]
    name: str

    def __call__(self, item: Any) -> Any:
        return self.fn(item)

    def __repr__(self) -> str:
        return self.name
    
    
class BasePipeline(ABC):
    """Base class for data pipelines in scientific machine learning.

    Provides a simple interface for data sources with preprocessing pipelines,
    yielding preprocessed items for ML training.

    Args:
        data_provider: DataProvider instance that yields raw data items.
        transforms: List of functions (Transform-wrapped or named) applied sequentially.
        logger: Optional logger for debugging and error tracking.
        skip_errors: Whether to skip items that fail preprocessing vs. raise errors.

    Example:
        >>> # Using Transform wrapper for clear metadata
        >>> transforms = [
        ...     Transform(lambda path: np.loadtxt(path, delimiter=","), "load_csv"),
        ...     Transform(lambda data: (data[:-1], data[-1]), "split_features_target"),
        ...     Transform(lambda x: (normalize(x[0]), x[1]), "normalize_features")
        ... ]
        >>> 
        >>> files = ListProvider(["data1.csv", "data2.csv"])
        >>> pipeline = MyPipeline(files, transforms)
        >>> 
        >>> # Clear, meaningful metadata
        >>> print(pipeline.get_metadata())
        >>> # {"pipeline_type": "MyPipeline", "dataset_size": 2,
        >>> #  "preprocessing_functions": ["load_csv", "split_features_target", "normalize_features"]}
    """
        
    def __init__(
        self,
        data_provider: DataProvider, 
        transforms: Optional[List[Union[Callable[[Any], Any], Transform]]] = None,
        *,
        logger: Optional[logging.Logger] = None,
        skip_errors: bool = True,
    ) -> None:
        self.data_provider = data_provider
        self.transforms = [
            fn if isinstance(fn, Transform) else Transform(fn, getattr(fn, '__name__', 'unknown'))
            for fn in (transforms or [])
        ]
        self.logger = logger or logging.getLogger(__name__)
        self.skip_errors = skip_errors
        self.error_count = 0

    def __iter__(self) -> Iterator[Any]:
        """Iterate over preprocessed items."""
        for raw_item in self.data_provider():
            try:
                item = raw_item
                for fn in self.transforms:
                    item = fn(item)
                yield item
            except Exception as e:
                self.error_count += 1
                self.logger.warning(f"Failed to preprocess item: {e}")
                if not self.skip_errors:
                    raise

    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        return len(self.data_provider)
        
    def sample(self, n: int = 5) -> List[Any]:
        """Return up to n preprocessed items for inspection."""
        return list(itertools.islice(self.__iter__(), n))
        
    def shuffle(self, buffer_size: int) -> 'BasePipeline':
        """Return a new pipeline that shuffles items with a reservoir buffer."""
        from .transforms import ShufflePipeline
        return ShufflePipeline(self, buffer_size)

    def batch(self, batch_size: int) -> 'BasePipeline':
        """Return a new pipeline that batches items into lists."""
        from .transforms import BatchPipeline
        return BatchPipeline(self, batch_size)
    
    def reset_error_count(self) -> None:
        """Reset the error count to zero."""
        self.error_count = 0
    
    @abstractmethod
    def to_framework_dataset(self) -> Any:
        """Convert pipeline to framework-native dataset."""
        ...
        
class InMemoryPipeline(BasePipeline):
    """In-memory pipeline that processes all data upfront."""
    
    def __init__(
        self,
        data_provider: DataProvider, 
        transforms: Optional[List[Union[Callable[[Any], Any], Transform]]] = None,
        *,
        logger: Optional[logging.Logger] = None,
        skip_errors: bool = True,
    ) -> None:
        super().__init__(data_provider, transforms, logger=logger, skip_errors=skip_errors)
        
        from .transforms import apply_transforms_to_dataset
        self.dataset, self.error_count = apply_transforms_to_dataset(
            self.data_provider(),
            self.transforms,
            logger=self.logger,
            skip_errors=self.skip_errors
        )
    
    def __iter__(self) -> Iterator[Any]:
        return iter(self.dataset)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]
    
    def to_framework_dataset(self) -> Any:
        pass