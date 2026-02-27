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
        pre_transform_hook: Optional callable applied before transforms for each item.
                            Signature: (item, item_id) -> item
        post_transform_hook: Optional callable applied after transforms for each item.
                             Signature: (item, item_id) -> item
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
        pre_transform_hook: Optional[Callable[[Any, Any], Any]] = None,
        post_transform_hook: Optional[Callable[[Any, Any], Any]] = None,
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
        self.in_memory_sample_count: Optional[int] = None
        self._pre_transform_hook = pre_transform_hook
        self._post_transform_hook = post_transform_hook

    def __iter__(self) -> Iterator[Any]:
        """Iterate over preprocessed items."""
        for idx, raw_item in enumerate(self.data_provider()):
            current_transform: Optional[Transform] = None
            try:
                item = raw_item
                if self._pre_transform_hook is not None:
                    item = self._pre_transform_hook(item, idx)
                for fn in self.transforms:
                    current_transform = fn
                    item = fn(item)
                if self._post_transform_hook is not None:
                    item = self._post_transform_hook(item, idx)
                if item is not None:
                    yield item
                else:
                    self.error_count += 1
                    self.logger.warning("Preprocessed item is None, skipping.")
            except Exception as e:
                self.error_count += 1
                if current_transform is not None:
                    transform_name = getattr(
                        current_transform,
                        "name",
                        getattr(current_transform, "__name__", repr(current_transform)),
                    )
                    self.logger.warning(
                        "Failed to preprocess item in transform '%s': %s",
                        transform_name,
                        e,
                    )
                else:
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
        pre_transform_hook: Optional[Callable[[Any, Any], Any]] = None,
        post_transform_hook: Optional[Callable[[Any, Any], Any]] = None,
        logger: Optional[logging.Logger] = None,
        skip_errors: bool = True,
    ) -> None:
        super().__init__(
            data_provider,
            transforms,
            pre_transform_hook=pre_transform_hook,
            post_transform_hook=post_transform_hook,
            logger=logger,
            skip_errors=skip_errors,
        )

        from .transform import apply_transforms_to_dataset

        self.dataset, self.error_count = apply_transforms_to_dataset(
            self.data_provider(),
            self.transforms,
            pre_transform_hook=self._pre_transform_hook,
            post_transform_hook=self._post_transform_hook,
            logger=self.logger,
            skip_errors=self.skip_errors,
        )
        self.in_memory_sample_count = len(self.dataset)

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
                from .transform import (
                    TorchDataset,
                    apply_dataset_operations_from_config,
                )

                torch_dataset = TorchDataset(self)
                if dataset_ops:
                    torch_dataset = apply_dataset_operations_from_config(
                        torch_dataset, dataset_ops
                    )
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

    def to_memory_dataset(self, dataset_ops: List[Dict] = None):
        """
        Load and process ALL data samples into memory as PyTorch TensorDataset.
        Only use this for datasets that fit comfortably in your available RAM.

        This method:
        1. Processes all data samples through the complete transform pipeline
        2. Converts results to PyTorch tensors
        3. Stores everything in memory for ultra-fast O(1) random access
        4. Returns a native PyTorch TensorDataset

        Benefits:
        - Eliminates file I/O during training (much faster)
        - Enables efficient shuffling and random sampling
        - Optimized for GPU transfer

        Args:
            dataset_ops: Optional list of dataset operations to apply after loading

        Returns:
            torch.utils.data.TensorDataset: In-memory dataset with all pre-processed tensors

        Example:
            >>> pipeline = PyTorchPipeline(provider, transforms)
            >>> # Load entire dataset into memory (use carefully!)
            >>> memory_dataset = pipeline.load_all_into_memory()
            >>> dataloader = DataLoader(memory_dataset, batch_size=32, shuffle=True)
        """
        try:
            import torch
            from IPython.display import clear_output
            from torch.utils.data import Dataset as TorchDataset
            from torch.utils.data import TensorDataset
            from tqdm.auto import tqdm

            from .transform import apply_dataset_operations_from_config

            class _MemoryListDataset(TorchDataset):
                """Fallback dataset that returns preprocessed samples without stacking."""

                def __init__(self, data: List[Any]) -> None:
                    self._data = data

                def __len__(self) -> int:
                    return len(self._data)

                def __getitem__(self, idx: int) -> Any:
                    return self._data[idx]

            def _to_tensor_or_keep(value: Any) -> Any:
                """Convert value to tensor when possible, otherwise leave unchanged."""

                if isinstance(value, torch.Tensor):
                    return value
                if isinstance(value, (str, bytes)):
                    return value
                try:
                    return torch.as_tensor(value)
                except (TypeError, ValueError):
                    return value

            # Process all data through pipeline and collect results
            processed_data = []
            pbar = tqdm(
                self,
                desc="Loading data into memory",
                leave=False,
                miniters=1,
                position=0,
            )

            for item in pbar:
                # Convert to tensor if not already
                if not isinstance(item, torch.Tensor):
                    if isinstance(item, (tuple, list)):
                        # Handle multiple outputs (e.g., features, labels)
                        item = tuple(_to_tensor_or_keep(x) for x in item)
                    else:
                        item = _to_tensor_or_keep(item)
                processed_data.append(item)

            pbar.close()
            clear_output(wait=True)

            if not processed_data:
                raise ValueError("No data was processed from the pipeline")

            self.in_memory_sample_count = len(processed_data)

            # Handle the case where each item is a tuple/list (multiple tensors)
            first_item = processed_data[0]
            dataset: TorchDataset
            try:
                if isinstance(first_item, (tuple, list)):
                    tensors = []
                    for i in range(len(first_item)):
                        component_values = [item[i] for item in processed_data]
                        if not all(
                            isinstance(x, torch.Tensor) for x in component_values
                        ):
                            raise TypeError("Non-tensor component detected")
                        tensors.append(torch.stack(component_values))
                    dataset = TensorDataset(*tensors)
                else:
                    if not all(isinstance(x, torch.Tensor) for x in processed_data):
                        raise TypeError("Non-tensor samples detected")
                    stacked_tensor = torch.stack(processed_data)
                    dataset = TensorDataset(stacked_tensor)
            except (TypeError, ValueError) as err:
                self.logger.warning(
                    "Falling back to list dataset because tensors could not be stacked: %s",
                    err,
                )
                dataset = _MemoryListDataset(processed_data)

            # Apply any additional dataset operations
            if dataset_ops:
                dataset = apply_dataset_operations_from_config(dataset, dataset_ops)

            return dataset

        except ImportError:
            raise RuntimeError("PyTorch not available")
