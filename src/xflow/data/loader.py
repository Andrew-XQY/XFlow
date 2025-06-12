"""pipeline.py
Core abstractions for building reusable, named preprocessing pipelines:
- DataProvider / ListProvider
- Transform (named transform)
- BasePipeline
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Any, Optional, List, Dict, Union
import logging
import itertools


@dataclass
class Transform:
    """Wrapper for a preprocessing function with metadata."""
    fn: Callable[[Any], Any]
    name: str

    def __call__(self, item: Any) -> Any:
        return self.fn(item)

    def __repr__(self) -> str:
        return self.name

class DataProvider(ABC):
    """Minimal wrapper to add attributes to data sources."""
    
    @abstractmethod
    def __call__(self) -> Iterable[Any]:
        """Return iterable of data items."""
        ...
    
    @abstractmethod  
    def __len__(self) -> int:
        """Return number of items."""
        ...

class ListProvider(DataProvider):
    """Wrap a list/array as a data provider."""
    def __init__(self, data: List[Any]):
        self._data = data
    
    def __call__(self) -> Iterable[Any]:
        return self._data
    
    def __len__(self) -> int:
        return len(self._data)
    
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
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(items={len(self)})"

    def __iter__(self) -> Iterator[Any]:
        """Iterate over preprocessed items."""
        for raw_item in self.data_provider():
            try:
                item = raw_item
                for fn in self.transforms:
                    item = fn(item)
                yield item
            except Exception as e:
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
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metadata for experiment reproducibility."""
        metadata = {
            "pipeline_type": self.__class__.__name__,
            "preprocessing_functions": [str(fn) for fn in self.transforms]
        }
        try:
            metadata["dataset_size"] = len(self)
        except (TypeError, NotImplementedError):
            metadata["dataset_size"] = "unknown"
        return metadata
    
    @abstractmethod
    def to_framework_dataset(self) -> Any:
        """Convert pipeline to framework-native dataset."""
        ...