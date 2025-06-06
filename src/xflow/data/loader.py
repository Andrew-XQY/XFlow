# src/xflow/data/loader.py
"""
xflow.data.loader
-----------------
Defines a minimal, framework-agnostic data-pipeline API.
"""

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Iterator, TypeVar, Any, Optional
import logging

TData = TypeVar("TData")   # “thing” produced by preprocess_fn
TRaw = TypeVar("TRaw")     # raw item from data_provider


class BasePipeline(ABC):
    """
    Abstract blueprint for any ML data pipeline.

    Core ideas:
      1. data_provider(): returns an iterable of raw items (e.g. file paths, stream tokens).
      2. preprocess_fn(raw) -> item: turns each raw item into 
         a final “thing” (e.g. for supervised: (input, label), 
         or for generative: just input, etc.).

    Two core methods:
      • get_dataset(): yields each item (of type TData) as a plain Python iterator.
      • to_framework_dataset(): wraps get_dataset() into a framework-native dataset; 
        implemented in subclasses.

    Args:
        data_provider: callable → Iterable[TRaw]
        preprocess_fn: callable(raw: TRaw) → TData
    """
    def __init__(
        self,
        data_provider: Callable[[], Iterable[TRaw]],
        preprocess_fn: Callable[[TRaw], TData],
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.data_provider = data_provider
        self.preprocess_fn = preprocess_fn
        self.logger = logger

    def __getitem__(self, idx: int) -> TData:
        """
        Optional: Fetch a single preprocessed item by index.
        Only valid if the data_provider returns an indexable collection.

        Args:
            idx (int): Index of the item.

        Returns:
            TData: Preprocessed item.

        Raises:
            NotImplementedError: By default, indexing is not supported.
        """
        raise NotImplementedError("This pipeline does not support indexing. Override if needed.")

    def __iter__(self) -> Iterator[TData]:
        """
        Each call to __iter__ fetches a fresh iterable from data_provider().
        Wrap preprocess_fn(raw) in try/except. If a logger was passed in, log failures; otherwise, skip silently.
        """
        for raw_item in self.data_provider():
            try:
                yield self.preprocess_fn(raw_item)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to preprocess {raw_item!r}: {e!s}")

    @abstractmethod
    def __len__(self) -> int:
        """
        Total number of items. 
        - If finite, return that count (e.g. len(list_of_paths)).
        - If streaming/unknown, raise NotImplementedError().
        """
        ...
        
    @abstractmethod
    def to_framework_dataset(self) -> Any:
        """
        Wrap get_dataset() as a framework-native dataset (e.g. tf.data.Dataset, torch.utils.data.Dataset).
        """
        ...
