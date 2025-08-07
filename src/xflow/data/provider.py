from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import pandas as pd

from ..utils.helper import split_sequence, subsample_sequence
from ..utils.io import scan_files
from ..utils.typing import PathLikeStr


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
    def subsample(
        self,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: int = None,
        strategy: str = None,
    ) -> "DataProvider":
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

    def merge(self, other: "DataProvider") -> "DataProvider":
        """
        Merge with another provider of the same type.

        Args:
            other: Another provider to merge with

        Returns:
            New provider containing combined data from both providers

        Raises:
            NotImplementedError: If provider doesn't support merging
            TypeError: If providers are not compatible types
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support merging. Use framework-specific concatenation."
        )

    def split(
        self, *args, **kwargs
    ) -> Union[Tuple["DataProvider", "DataProvider"], List["DataProvider"]]:
        """
        Split provider into multiple providers.

        Default implementation supports ratio-based split.
        Subclasses can override for different splitting strategies.

        Args:
            *args: Variable arguments (implementation-specific)
            **kwargs: Keyword arguments (implementation-specific)

        Returns:
            Tuple of (provider_1, provider_2) or List of providers

        Raises:
            NotImplementedError: If provider doesn't support splitting
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support splitting. Create separate providers manually."
        )


class FileProvider(DataProvider):
    """Data provider that scans directories for files with specified extensions."""

    def __init__(
        self,
        root_paths: Union[PathLikeStr, List[PathLikeStr]],
        extensions: Optional[Union[str, List[str]]] = None,
        path_type: Literal["string", "str", "path", "Path"] = "path",
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
            recursive=True,
        )

    def __call__(self) -> Union[List[str], List[Path]]:
        """Return the list of file paths in the configured type."""
        return self._file_paths.copy()

    def __len__(self):
        """Return number of files found."""
        return len(self._file_paths)

    @classmethod
    def _from_file_list(
        cls,
        file_paths: Union[List[str], List[Path]],
        extensions: Optional[List[str]] = None,
        path_type: Literal["string", "path"] = "path",
    ) -> "FileProvider":
        """Create FileProvider from explicit file list (internal helper)."""
        instance = cls.__new__(cls)
        instance.root_paths = []  # Not used when created from file list
        instance.extensions = extensions
        instance.path_type = path_type
        instance._file_paths = sorted(file_paths)
        return instance

    def split(
        self, ratio: float = 0.8, seed: int = 42
    ) -> Tuple["FileProvider", "FileProvider"]:
        """
        Split files into two providers.

        Args:
            ratio: Portion for first provider (0.0 to 1.0)
            seed: Random seed for reproducible splits

        Returns:
            Tuple of (provider_1, provider_2)
        """
        first_files, second_files = split_sequence(
            self._file_paths, split_ratio=ratio, seed=seed
        )
        # Create new providers with same extensions and path_type
        provider_1 = self._from_file_list(first_files, self.extensions, self.path_type)
        provider_2 = self._from_file_list(second_files, self.extensions, self.path_type)
        return provider_1, provider_2

    def subsample(
        self,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: int = None,
        strategy: str = "random",
    ) -> "FileProvider":
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
            seed=seed,
        )
        return self._from_file_list(sampled_files, self.extensions, self.path_type)

    def merge(self, other: "FileProvider") -> "FileProvider":
        """
        Merge with another FileProvider.

        Args:
            other: Another FileProvider to merge with

        Returns:
            New FileProvider containing files from both providers

        Raises:
            TypeError: If other is not a FileProvider
        """
        if not isinstance(other, FileProvider):
            raise TypeError(f"Cannot merge FileProvider with {type(other).__name__}")

        # Combine file lists and remove duplicates while preserving order
        combined_files = list(dict.fromkeys(self._file_paths + other._file_paths))

        # Use the more restrictive settings (intersection of extensions)
        if self.extensions is None:
            merged_extensions = other.extensions
        elif other.extensions is None:
            merged_extensions = self.extensions
        else:
            merged_extensions = list(set(self.extensions) & set(other.extensions))

        # Use the path_type of the first provider
        return self._from_file_list(combined_files, merged_extensions, self.path_type)


class SqlProvider(DataProvider):
    """Data provider that unifies data from SQL database sources into a DataFrame."""

    def __init__(
        self,
        sources: Union[List[Dict[str, Any]], Dict[str, Any], None] = None,
        output_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            sources: Source configuration(s). Can be:
                - List of dicts: [{"connection": "path.db", "sql": "SELECT ..."}]
                - Single dict: {"connection": "path.db", "sql": "SELECT ..."
                - None: Creates empty provider
            output_config: Optional dict controlling output behavior:
                - {"list": "<column_name>"} returns that column as a Python list
                - {} or None returns the full unified DataFrame
        """
        self._unified_df = pd.DataFrame()
        self._output_config = output_config or {"operation": "dataframe"}
        if sources is not None:
            # Wrap single dict into list
            if isinstance(sources, dict):
                sources = [sources]

            # Process each source
            for source in sources:
                self._add_source(source)

    def _add_source(self, source: Dict[str, Any]) -> None:
        """Add data from a source config to unified DataFrame."""
        from ..utils.dataframe import concat_dataframes
        from ..utils.sql import (
            create_database_instance,
            find_connection,
            find_db_type,
            find_sql,
        )

        connection = find_connection(source)
        sql = find_sql(source)
        db_type = find_db_type(source, connection)

        # Create database and get data
        db = create_database_instance(db_type, connection)
        try:
            df = db.query(sql)
            if not df.empty:
                self._unified_df = concat_dataframes([self._unified_df, df])
        finally:
            db.close()

    def __call__(self) -> Any:
        """Return DataFrame or transformed data based on output_config."""
        data = self._unified_df.copy()

        # interpret output_config directly: key is operation type
        config = self._output_config or {}
        # if 'list' operation, return specified column as Python list
        if "list" in config.keys():
            column_name = config["list"]
            return data[column_name].tolist()
        # default: return full DataFrame
        return data

    def __len__(self) -> int:
        """Return total number of rows."""
        return len(self._unified_df)

    def set_output_config(self, config: Dict[str, Any]) -> None:
        """Set the output configuration."""
        self._output_config = config

    def subsample(
        self,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: int = None,
        strategy: str = "random",
    ) -> "SqlProvider":
        """Create subsampled provider."""
        from ..utils.dataframe import subsample_dataframe

        sampled_df = subsample_dataframe(
            self._unified_df,
            n_samples=n_samples,
            fraction=fraction,
            seed=seed,
            strategy=strategy,
        )

        new_provider = SqlProvider()
        new_provider._unified_df = sampled_df.copy()
        new_provider._output_config = self._output_config.copy()
        return new_provider

    def split(
        self, ratio: float = None, seed: int = 42, filters: List[str] = None
    ) -> Union[Tuple["SqlProvider", "SqlProvider"], List["SqlProvider"]]:
        """Split unified DataFrame into multiple providers.

        Args:
            ratio: For ratio-based split (0.0 to 1.0). If provided, returns (provider_1, provider_2).
            seed: Random seed for ratio-based splits
            filters: List of pandas query strings for filter-based split. Returns list of providers.
                    Example: ["age > 30", "category == 'A'", "score >= 0.8"]

        Returns:
            Tuple of (provider_1, provider_2) for ratio split
            List of providers for filter split
        """
        if filters is not None:
            # Filter-based split
            from ..utils.dataframe import split_dataframe_by_filters

            filtered_dfs = split_dataframe_by_filters(self._unified_df, filters)
            providers = []
            for df in filtered_dfs:
                provider = SqlProvider()
                provider._unified_df = df.copy()
                provider._output_config = self._output_config.copy()
                providers.append(provider)
            return providers

        elif ratio is not None:
            # Ratio-based split 
            from ..utils.dataframe import split_dataframe_by_ratio

            first_df, second_df = split_dataframe_by_ratio(
                self._unified_df, ratio, seed
            )

            first_provider = SqlProvider()
            first_provider._unified_df = first_df.copy()
            first_provider._output_config = self._output_config.copy()

            second_provider = SqlProvider()
            second_provider._unified_df = second_df.copy()
            second_provider._output_config = self._output_config.copy()

            return first_provider, second_provider

        else:
            raise ValueError("Either 'ratio' or 'filters' must be provided")

    def merge(self, other: "SqlProvider") -> "SqlProvider":
        """
        Merge with another SqlProvider.

        Args:
            other: Another SqlProvider to merge with

        Returns:
            New SqlProvider containing combined DataFrame from both providers

        Raises:
            TypeError: If other is not a SqlProvider
        """
        if not isinstance(other, SqlProvider):
            raise TypeError(f"Cannot merge SqlProvider with {type(other).__name__}")

        from ..utils.dataframe import concat_dataframes

        combined_df = concat_dataframes([self._unified_df, other._unified_df])

        new_provider = SqlProvider()
        new_provider._unified_df = combined_df
        new_provider._output_config = self._output_config.copy()
        return new_provider

    @classmethod
    def get_supported_db_types(cls) -> List[str]:
        """Get list of supported database types."""
        from ..utils.sql import get_supported_db_types

        return get_supported_db_types()
