# src/xflow/data/loader.py
"""
xflow.data.loader
-----------------
Defines a minimal, framework-agnostic dataset API.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, Tuple, TypeVar

TInput = TypeVar("TInput")
TLabel = TypeVar("TLabel")


class BaseDataset(ABC, Generic[TInput, TLabel]):
    """
    Abstract base class for all datasets in XFlow.

    Each dataset must return a pair (input, label) for a given index.
    Subclasses should implement:
      - __len__(): total number of samples.
      - __getitem__(): load and return (input, label) at a given index.
    """

    def __init__(self, root_dir: Path, transform: Optional[Any] = None) -> None:
        """
        Args:
            root_dir (Path):
                Path to the dataset’s root directory.
            transform (Optional[Any]):
                A callable taking (input, label) and returning transformed
                (input, label). If None, no transform is applied.
        """
        self.root_dir: Path = root_dir
        self.transform: Optional[Any] = transform

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[TInput, TLabel]:
        """
        Load and return a single sample pair.

        Args:
            index (int): Index of the sample (0 ≤ index < len(self)).

        Returns:
            Tuple[TInput, TLabel]:
                - input: model input (e.g., an image array)
                - label: corresponding ground truth (e.g., a mask or target image)

        Raises:
            IndexError: If `index` is out of bounds.
            FileNotFoundError: If expected files for this index are missing.
        """
        ...
