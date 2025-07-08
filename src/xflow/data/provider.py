from typing import Any, Iterable, List, Union, Optional, Tuple, Literal, TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from ..utils.typing import PathLikeStr
from ..utils.helper import subsample_sequence, split_sequence
from ..utils.io import scan_files

if TYPE_CHECKING:
    from ..utils.sql import Database
    
    
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
    
    def split(self, *args, **kwargs) -> Union[Tuple['DataProvider', 'DataProvider'], List['DataProvider']]:
        """
        Split provider into multiple providers.
        
        Default implementation supports train/val split with ratio.
        Subclasses can override for different splitting strategies.
        
        Args:
            *args: Variable arguments (implementation-specific)
            **kwargs: Keyword arguments (implementation-specific)
            
        Returns:
            Tuple of (train_provider, val_provider) or List of providers
            
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
    
class SqlProvider(DataProvider):
    """Data provider that uses SQL queries to fetch data from database."""
    
    def __init__(self, 
                 db: 'Database',
                 base_query: str,
                 where_conditions: Optional[List[str]] = None):
        """
        Args:
            db: Database instance (SQLiteDB, PostgreSQLDB, etc.)
            base_query: Base SQL query (without WHERE clause)
            where_conditions: List of WHERE conditions to append
        """
        self.db = db
        self.base_query = base_query.strip()
        self.where_conditions = where_conditions or []
        self._data = None
        self._count = None
    
    @classmethod
    def from_sqlite(cls, db_path: str, base_query: str, where_conditions: Optional[List[str]] = None):
        """Convenience constructor for SQLite databases."""
        from ..utils.sql import SQLiteDB
        db = SQLiteDB(db_path)
        return cls(db, base_query, where_conditions)
    
    def _build_query(self, limit: Optional[int] = None, offset: Optional[int] = None) -> str:
        """Build final SQL query with WHERE conditions and optional LIMIT/OFFSET."""
        query = self.base_query
        
        # Add WHERE conditions
        if self.where_conditions:
            # Check if base query already has WHERE clause
            if 'WHERE' in query.upper():
                # Append with AND
                where_clause = ' AND ' + ' AND '.join(self.where_conditions)
            else:
                # Add new WHERE clause
                where_clause = ' WHERE ' + ' AND '.join(self.where_conditions)
            query += where_clause
        
        # Add LIMIT/OFFSET for pagination
        if limit is not None:
            query += f' LIMIT {limit}'
            if offset is not None:
                query += f' OFFSET {offset}'
        
        return query
    
    def _count_query(self) -> str:
        """Build count query to get total number of records."""
        # Extract FROM clause from base query
        from_part = self.base_query.upper().split('FROM')[1].split('ORDER BY')[0].split('GROUP BY')[0].strip()
        count_query = f"SELECT COUNT(*) FROM {from_part}"
        
        # Add WHERE conditions
        if self.where_conditions:
            count_query += ' WHERE ' + ' AND '.join(self.where_conditions)
        
        return count_query
    
    def __call__(self) -> pd.DataFrame:
        """Return pandas DataFrame with query results."""
        if self._data is None:
            query = self._build_query()
            self._data = self.db.query(query)
        return self._data
    
    def __len__(self) -> int:
        """Return number of records."""
        if self._count is None:
            count_query = self._count_query()
            result = self.db.query(count_query)
            self._count = result.iloc[0, 0]
        return self._count
    
    def subsample(self, n_samples: Optional[int] = None, fraction: Optional[float] = None,
                  seed: int = None, strategy: str = "random") -> 'SqlProvider':
        """
        Create subsampled provider using SQL LIMIT with random sampling.
        
        Args:
            n_samples: Exact number of samples to take
            fraction: Fraction of total samples (0.0 to 1.0)
            seed: Random seed for reproducible subsampling
            strategy: "random", "first", "last"
        """
        # Calculate target sample size
        if n_samples is None and fraction is None:
            raise ValueError("Either n_samples or fraction must be provided")
        
        total_size = len(self)
        if n_samples is None:
            n_samples = int(total_size * fraction)
        n_samples = min(n_samples, total_size)
        
        # Create subsampled query using your SQL class
        if strategy == "random":
            # Use ORDER BY RANDOM() LIMIT for random sampling
            subquery = f"({self._build_query()}) ORDER BY RANDOM() LIMIT {n_samples}"
            new_base_query = f"SELECT * FROM ({subquery})"
        elif strategy == "first":
            # Use LIMIT for first N records
            new_base_query = f"({self._build_query()}) LIMIT {n_samples}"
        elif strategy == "last":
            # Use ORDER BY DESC + LIMIT for last N records
            subquery = f"({self._build_query()}) ORDER BY rowid DESC LIMIT {n_samples}"
            new_base_query = f"SELECT * FROM ({subquery}) ORDER BY rowid ASC"
        else:
            raise ValueError(f"Strategy '{strategy}' not supported")
        
        # Create new provider with modified base query and no additional conditions
        return SqlProvider(self.db, new_base_query, [])
    
    def split(self, *conditions: Union[str, List[str]]) -> List['SqlProvider']:
        """
        Split data by creating multiple providers with different WHERE conditions.
        
        Args:
            *conditions: Variable number of WHERE condition strings, or a single list of conditions
            
        Returns:
            List of SqlProvider instances, one for each condition
            
        Example:
            # Individual arguments
            train, val, test = sql_provider.split(
                "status = 'train'",
                "status = 'val'", 
                "status = 'test'"
            )
            
            # List argument
            train, test = sql_provider.split(["purpose = 'training'", "purpose = 'testing'"])
        """
        # Handle case where a list is passed as first argument
        if len(conditions) == 1 and isinstance(conditions[0], list):
            conditions = conditions[0]
        
        if not conditions:
            raise ValueError("At least one WHERE condition must be provided")
        
        providers = []
        for condition in conditions:
            if not isinstance(condition, str):
                raise ValueError(f"Each condition must be a string, got {type(condition)}")
            
            # Create new provider with base conditions + split condition
            new_conditions = self.where_conditions + [condition]
            provider = SqlProvider(self.db, self.base_query, new_conditions)
            providers.append(provider)
        
        return providers
    
    def close(self):
        """Close database connection."""
        if hasattr(self.db, 'close'):
            self.db.close()