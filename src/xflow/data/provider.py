from typing import Any, Iterable, List, Union, Optional, Tuple, Literal
from abc import ABC, abstractmethod
from pathlib import Path
from ..utils.typing import PathLikeStr
from ..utils.helper import subsample_sequence, split_sequence
from ..utils.io import scan_files
    
    
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
    
    @abstractmethod
    def subsample(self, n_samples: Optional[int] = None, fraction: Optional[float] = None, 
                  seed: int = None, strategy: str = None) -> 'DataProvider':
        """
        Create a subsampled version of this provider.
        
        Args:
            n_samples: Exact number of samples to take
            fraction: Fraction of total samples (0.0 to 1.0)
            seed: Random seed for reproducible subsampling
            strategy: "random", "first", "last", "every_nth"
            
        Returns:
            New provider with subsampled data
        """
        pass
    
    def split(self, train_ratio: float = None, seed: int = None) -> Tuple['DataProvider', 'DataProvider']:
        """
        Split provider into train/val providers.
        
        Args:
            train_ratio: Portion for training set (0.0 to 1.0)
            seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_provider, val_provider)
            
        Raises:
            NotImplementedError: If provider doesn't support splitting
        """
        raise NotImplementedError(f"{self.__class__.__name__} doesn't support splitting. Create separate providers manually.")
    
    
class FileProvider(DataProvider):
    """Data provider that scans directories for files with specified extensions."""
    
    def __init__(
        self,
        root_paths: Union[PathLikeStr, List[PathLikeStr]], 
        extensions: Optional[Union[str, List[str]]] = None,
        path_type: Literal["string", "str", "path", "Path"] = "path"
        ):
        """
        Args:
            root_paths: Single path (string or Path) or list of paths
            extensions: Single extension string or list of extensions (e.g., '.jpg' or ['.jpg', '.png']).
                       If None, returns all files.
            path_type: Whether to return paths as "string" or "path" objects.
                      Use "string" for TensorFlow compatibility, "path" for rich Path API.
        """
        # Convert to list and ensure all are Path objects
        if isinstance(root_paths, (str, Path)):
            self.root_paths = [Path(root_paths)]
        else:
            self.root_paths = [Path(p) for p in root_paths]
        
        # Convert extensions to list of lowercase strings, or None for all files
        if extensions is None:
            self.extensions = None
        elif isinstance(extensions, str):
            self.extensions = [extensions.lower()]
        else:
            self.extensions = [ext.lower() for ext in extensions]
        
        self.path_type = path_type
        self._file_paths = self._scan_files()
    
    def _scan_files(self) -> Union[List[str], List[Path]]:
        """Scan all root paths for files with specified extensions."""
        return scan_files(
            self.root_paths,
            extensions=self.extensions,
            return_type=self.path_type,
            recursive=True 
        )
    
    def __call__(self) -> Union[List[str], List[Path]]:
        """Return the list of file paths in the configured type."""
        return self._file_paths
    
    def __len__(self):
        """Return number of files found."""
        return len(self._file_paths)
    
    @classmethod
    def _from_file_list(
        cls, file_paths: Union[List[str], List[Path]],
        extensions: Optional[List[str]] = None, 
        path_type: Literal["string", "path"] = "path"
        ) -> 'FileProvider':
        """Create FileProvider from explicit file list (internal helper)."""
        instance = cls.__new__(cls)
        instance.root_paths = []  # Not used when created from file list
        instance.extensions = extensions
        instance.path_type = path_type
        instance._file_paths = sorted(file_paths)
        return instance
    
    def split(self, train_ratio: float = 0.8, seed: int = 42) -> Tuple['FileProvider', 'FileProvider']:
        """
        Split files into train/val providers.
        
        Args:
            train_ratio: Portion for training set (0.0 to 1.0)
            seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_provider, val_provider)
        """
        train_files, val_files = split_sequence(
            self._file_paths, 
            split_ratio=train_ratio, 
            seed=seed
        )
        # Create new providers with same extensions and path_type
        train_provider = self._from_file_list(train_files, self.extensions, self.path_type)
        val_provider = self._from_file_list(val_files, self.extensions, self.path_type)
        return train_provider, val_provider
    
    def subsample(self, n_samples: Optional[int] = None, fraction: Optional[float] = None,
                  seed: int = None, strategy: str = "random") -> 'FileProvider':
        """
        Create a subsampled version of this provider.
        
        Args:
            n_samples: Exact number of samples to take
            fraction: Fraction of total samples (0.0 to 1.0)
            seed: Random seed for reproducible subsampling
            strategy: "random", "first", "last", "stride", or "reservoir".
            
        Returns:
            New provider with subsampled data
        """
        sampled_files = subsample_sequence(
            self._file_paths, 
            n_samples=n_samples, 
            fraction=fraction, 
            strategy=strategy, 
            seed=seed
        )
        return self._from_file_list(sampled_files, self.extensions, self.path_type)