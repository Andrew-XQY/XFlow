"""
xflow.data.loader
"""

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Iterator, TypeVar, Any, Optional, List
import logging
import itertools

TData = TypeVar("TData")  # Preprocessed item (e.g., NumPy array tuple)
TRaw = TypeVar("TRaw")    # Raw item from data_provider

class BasePipeline(ABC):
    """Abstract base class for data pipelines in scientific machine learning.

    Provides a flexible interface for complex data sources (experiment files, sensor 
    streams) with preprocessing pipelines, yielding preprocessed items like 
    ``(input, label)`` NumPy arrays for ML training.

    **Key Features:**
    
    * Iterator support (``__iter__``) for TensorFlow/PyTorch compatibility
    * Optional indexing (``__getitem__``) for cached datasets  
    * Framework-native conversion (``to_framework_dataset``)
    * Optional caching for small datasets
    * Robust error handling for noisy experimental data

    Args:
        data_provider: Callable returning fresh iterable of raw items (file paths, 
            database records). Must return new iterable on each call.
        preprocess_fns: List of transform functions. Each takes previous output 
            and returns transformed item. At least one required.
        logger: Optional logger. Defaults to module logger.
        on_error: Error handling - "skip" (default), "log", or "raise".
        error_handler: Optional callable for custom error handling.
        cache: If True, cache preprocessed items in memory during init.

    Raises:
        ValueError: If ``preprocess_fns`` is empty.

    Example:
        .. code-block:: python

            def data_provider():
                return ["/data/file1.csv", "/data/file2.csv"]
            
            preprocess_fns = [
                lambda path: np.loadtxt(path, delimiter=","),
                lambda data: data / np.max(data),  # normalize
                lambda data: (data[:-1], data[-1])  # (input, label)
            ]
            
            pipeline = MyPipeline(data_provider, preprocess_fns, cache=True)
            model.fit(pipeline, epochs=10)  # Direct use with TensorFlow
    """
    
    def __init__(
        self,
        data_provider: Callable[[], Iterable[TRaw]],
        preprocess_fns: List[Callable[[Any], Any]],
        *,
        logger: Optional[logging.Logger] = None,
        on_error: str = "skip",
        error_handler: Optional[Callable[[TRaw, Exception], None]] = None,
        cache: bool = False
    ) -> None:
        if not preprocess_fns:
            raise ValueError("At least one preprocess_fn is required to ensure valid output.")
        self.data_provider = data_provider
        self.preprocess_fns = preprocess_fns
        self.logger = logger or logging.getLogger(__name__)
        self.on_error = on_error
        self.error_handler = error_handler
        self.cache = cache
        self._cached_data: Optional[List[TData]] = None
        if cache:
            self._cached_data = list(self._iter_uncached())

    def _iter_uncached(self) -> Iterator[TData]:
        """Internal iterator for non-cached data processing.

        Yields preprocessed items by applying `preprocess_fns` to each raw item from
        `data_provider`. Handles errors according to `on_error` and `error_handler`.

        Yields:
            TData: Preprocessed item (e.g., `(input, label)` NumPy array tuple).

        Raises:
            Exception: If `on_error="raise"`, propagates preprocessing exceptions.
        """
        for raw_item in self.data_provider():
            try:
                item = raw_item
                for fn in self.preprocess_fns:
                    item = fn(item)
                yield item
            except Exception as e:
                if self.on_error == "raise":
                    raise
                if self.on_error == "log":
                    self.logger.warning(f"Failed to preprocess {raw_item!r}: {e!s}")
                if self.error_handler:
                    try:
                        self.error_handler(raw_item, e)
                    except Exception as handler_e:
                        self.logger.error(f"Error handler failed for {raw_item!r}: {handler_e!s}")

    def __iter__(self) -> Iterator[TData]:
        """Iterate over preprocessed items.

        If `cache=True`, yields items from the cached list. Otherwise, processes items
        on-the-fly using `_iter_uncached`. Suitable for direct use with ML frameworks
        (e.g., TensorFlow's `.fit()` or PyTorch's `DataLoader`).

        Yields:
            TData: Preprocessed item (e.g., `(input, label)` NumPy array tuple).
        """
        if self.cache and self._cached_data is not None:
            return iter(self._cached_data)
        return self._iter_uncached()

    def __getitem__(self, idx: int) -> TData:
        """Fetch a preprocessed item by index.

        Only supported if `cache=True` or overridden by a subclass. Useful for random
        access in small datasets or for frameworks requiring indexed datasets (e.g.,
        PyTorch's `DataLoader`).

        Args:
            idx: Index of the item to fetch.

        Returns:
            TData: Preprocessed item at the specified index.

        Raises:
            NotImplementedError: If indexing is not supported (e.g., `cache=False` and
                not overridden by a subclass).
        """
        if self.cache and self._cached_data is not None:
            return self._cached_data[idx]
        raise NotImplementedError("Indexing requires caching or subclass override.")

    def sample(self, n: int = 5) -> List[TData]:
        """Return up to n preprocessed items for debugging or inspection.

        Uses cached data if available; otherwise, processes items on-the-fly.

        Args:
            n: Maximum number of items to return (default: 5).

        Returns:
            List[TData]: List of up to n preprocessed items.
        """
        if self.cache and self._cached_data is not None:
            return self._cached_data[:n]
        return list(itertools.islice(self.__iter__(), n))

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset.

        Subclasses must implement this for finite datasets (e.g., number of files or
        records). For streaming or unbounded datasets, raise `NotImplementedError`.
        Computationally expensive operations (e.g., counting files) should be cached by
        the subclass.

        Returns:
            int: Number of items in the dataset.

        Raises:
            NotImplementedError: If the dataset size is unknown or streaming.
        """
        ...

    @abstractmethod
    def to_framework_dataset(self) -> Any:
        """Convert the pipeline to a framework-native dataset.

        Subclasses implement this to return framework-specific datasets (e.g.,
        `tf.data.Dataset` for TensorFlow, `torch.utils.data.Dataset` for PyTorch).
        Enables high-performance features like batching, prefetching, and distributed
        training.

        Returns:
            Any: Framework-native dataset object.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        ...
        
    def shuffle(self, buffer_size: int):
        """Return a new pipeline that shuffles items with a reservoir buffer.
        
        Args:
            buffer_size: Size of the shuffle buffer for reservoir sampling.
            
        Returns:
            ShufflePipeline: A new pipeline instance with shuffling applied.
        """
        from .transforms import ShufflePipeline  # Local import
        return ShufflePipeline(self, buffer_size)

    def batch(self, batch_size: int):
        """Return a new pipeline that batches items into lists.
        
        Args:
            batch_size: Number of items per batch.
            
        Returns:
            BatchPipeline: A new pipeline instance with batching applied.
        """
        from .transforms import BatchPipeline  # Local import
        return BatchPipeline(self, batch_size)