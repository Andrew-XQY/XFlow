"""Core abstractions for building reusable, named preprocessing pipelines:"""

import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Union

from ..utils.decorator import with_progress
from .provider import DataProvider

if TYPE_CHECKING:
    import tensorflow as tf


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
            (
                fn
                if isinstance(fn, Transform)
                else Transform(fn, getattr(fn, "__name__", "unknown"))
            )
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
                if item is not None:
                    yield item
                else:
                    self.error_count += 1
                    self.logger.warning("Preprocessed item is None, skipping.")
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

    def shuffle(self, buffer_size: int) -> "BasePipeline":
        """Return a new pipeline that shuffles items with a reservoir buffer."""
        from .transform import ShufflePipeline

        return ShufflePipeline(self, buffer_size)

    def batch(self, batch_size: int) -> "BasePipeline":
        """Return a new pipeline that batches items into lists."""
        from .transform import BatchPipeline

        return BatchPipeline(self, batch_size)

    def prefetch(self) -> "BasePipeline":
        """Return a new pipeline that prefetches items in background."""
        # TODO: Implement prefetching logic

    def reset_error_count(self) -> None:
        """Reset the error count to zero."""
        self.error_count = 0

    @abstractmethod
    def to_framework_dataset(self) -> Any:
        """Convert pipeline to framework-native dataset."""
        ...

    def to_numpy(self):
        """
        Convert the pipeline to NumPy arrays.
        If each item is a tuple, returns a tuple of arrays (one per component).
        If each item is a single array, returns a single array.
        """
        import numpy as np
        from IPython.display import clear_output
        from tqdm.auto import tqdm 

        items = []
        pbar = tqdm(
            self, desc="Converting to numpy", leave=False, miniters=1, position=0
        )
        for x in pbar:
            items.append(x)
        pbar.close()
        clear_output(wait=True) 

        if not items:
            return None
        first = items[0]
        if isinstance(first, (tuple, list)):
            return tuple(np.stack(c) for c in zip(*items))
        return np.stack(items)


class DataPipeline(BasePipeline):
    """Simple pipeline that processes data lazily without storing in memory."""

    def to_framework_dataset(self) -> Any:
        """Not supported for lazy processing."""
        raise NotImplementedError(
            "DataPipeline doesn't support framework conversion. "
            "Use InMemoryPipeline or TensorFlowPipeline instead."
        )


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
        super().__init__(
            data_provider, transforms, logger=logger, skip_errors=skip_errors
        )

        from .transform import apply_transforms_to_dataset

        self.dataset, self.error_count = apply_transforms_to_dataset(
            self.data_provider(),
            self.transforms,
            logger=self.logger,
            skip_errors=self.skip_errors,
        )

    def __iter__(self) -> Iterator[Any]:
        return iter(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def to_framework_dataset(
        self, framework: str = "tensorflow", dataset_ops: List[Dict] = None
    ) -> Any:
        """Convert to framework-native dataset using already processed data."""
        if framework.lower() == "tensorflow":
            try:
                import tensorflow as tf
                dataset = tf.data.Dataset.from_tensor_slices(self.dataset)
                if dataset_ops:
                    from .transform import apply_dataset_operations_from_config
                    dataset = apply_dataset_operations_from_config(dataset, dataset_ops)
                return dataset
            except ImportError:
                raise RuntimeError("TensorFlow not available")
        elif framework.lower() in ("pytorch", "torch"):
            try:
                from .transform import TorchDataset, apply_dataset_operations_from_config
                
                torch_dataset = TorchDataset(self)
                if dataset_ops:
                    torch_dataset = apply_dataset_operations_from_config(torch_dataset, dataset_ops)
                return torch_dataset
            except ImportError:
                raise RuntimeError("PyTorch not available")
        else:
            raise NotImplementedError(f"Framework {framework} not implemented")


class TensorFlowPipeline(BasePipeline):
    """Pipeline that uses TensorFlow-native transforms without preprocessing."""

    def to_framework_dataset(
        self, framework: str = "tensorflow", dataset_ops: List[Dict] = None
    ):
        """Convert to TensorFlow dataset."""
        if framework.lower() != "tensorflow":
            raise ValueError(
                f"TensorFlowPipeline only supports tensorflow, got {framework}"
            )

        try:
            import tensorflow as tf

            file_paths = list(self.data_provider())
            dataset = tf.data.Dataset.from_tensor_slices(file_paths)

            for transform in self.transforms:
                dataset = dataset.map(transform.fn, num_parallel_calls=tf.data.AUTOTUNE)

            if dataset_ops:
                from .transform import apply_dataset_operations_from_config

                dataset = apply_dataset_operations_from_config(dataset, dataset_ops)

            return dataset

        except ImportError:
            raise RuntimeError("TensorFlow not available")


class PyTorchPipeline(BasePipeline):
    """Pipeline that uses PyTorch-native transforms without preprocessing."""

    def to_framework_dataset(
        self, framework: str = "pytorch", dataset_ops: List[Dict] = None
    ):
        """Convert to PyTorch dataset."""
        if framework.lower() not in ("pytorch", "torch"):
            raise ValueError(
                f"PyTorchPipeline only supports pytorch/torch, got {framework}"
            )

        try:
            from .transform import TorchDataset, apply_dataset_operations_from_config
            
            # Create a PyTorch-compatible dataset that applies transforms on-the-fly
            class PyTorchTransformDataset(TorchDataset):
                def __init__(self, data_provider, transforms):
                    self.data_provider = data_provider
                    self.transforms = transforms
                    self._file_paths = list(data_provider())
                    
                def __len__(self):
                    return len(self._file_paths)
                    
                def __getitem__(self, idx):
                    item = self._file_paths[idx]
                    for transform in self.transforms:
                        item = transform.fn(item)
                    return item
            
            dataset = PyTorchTransformDataset(self.data_provider, self.transforms)
            
            if dataset_ops:
                dataset = apply_dataset_operations_from_config(dataset, dataset_ops)
                
            return dataset

        except ImportError:
            raise RuntimeError("PyTorch not available")
