from typing import Any, Iterable, List, Union, Optional
from abc import ABC, abstractmethod
from pathlib import Path
    
    
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
    
    
class FileProvider(DataProvider):
    """Data provider that scans directories for files with specified extensions."""
    
    def __init__(self, root_paths: Union[str, Path, List[Union[str, Path]]], extensions: Optional[Union[str, List[str]]] = None):
        """
        Args:
            root_paths: Single path (string or Path) or list of paths
            extensions: Single extension string or list of extensions (e.g., '.jpg' or ['.jpg', '.png']).
                       If None, returns all files.
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
        
        self._file_paths = self._scan_files()
    
    def _scan_files(self):
        """Scan all root paths for files with specified extensions."""
        file_paths = []
        for root_path in self.root_paths:
            for file_path in root_path.rglob("*"):
                if file_path.is_file():
                    # If no extensions specified, include all files
                    if self.extensions is None or file_path.suffix.lower() in self.extensions:
                        file_paths.append(str(file_path))
        return sorted(file_paths)
    
    def __call__(self):
        """Return the list of file paths."""
        return self._file_paths
    
    def __len__(self):
        """Return number of files found."""
        return len(self._file_paths)