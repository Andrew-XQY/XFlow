"""Generic DataFrame utilities for common operations."""

from typing import List, Optional, Tuple, Union

import pandas as pd


class DataFrameTransformRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry[name]


@DataFrameTransformRegistry.register("subsample_dataframe")
def subsample_dataframe(
    df: pd.DataFrame,
    n_samples: Optional[int] = None,
    fraction: Optional[float] = None,
    seed: Optional[int] = None,
    strategy: str = "random",
) -> pd.DataFrame:
    """Subsample a DataFrame using various strategies.

    Args:
        df: DataFrame to subsample
        n_samples: Exact number of samples to take
        fraction: Fraction of total samples (0.0 to 1.0)
        seed: Random seed for reproducible subsampling
        strategy: Sampling strategy ("random", "first", "last")

    Returns:
        Subsampled DataFrame

    Raises:
        ValueError: If neither n_samples nor fraction provided, or invalid strategy
    """
    if n_samples is None and fraction is None:
        raise ValueError("Either n_samples or fraction must be provided")

    total_size = len(df)
    if total_size == 0:
        return df.copy()

    # Calculate target sample size
    if n_samples is None:
        n_samples = int(total_size * fraction)
    n_samples = min(n_samples, total_size)

    # Apply sampling strategy
    if strategy == "random":
        return df.sample(n=n_samples, random_state=seed)
    elif strategy == "first":
        return df.head(n_samples)
    elif strategy == "last":
        return df.tail(n_samples)
    else:
        raise ValueError(
            f"Strategy '{strategy}' not supported. Use 'random', 'first', or 'last'"
        )


@DataFrameTransformRegistry.register("split_dataframe_by_ratio")
def split_dataframe_by_ratio(
    df: pd.DataFrame, ratio: float = 0.8, seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into two parts by ratio.

    Args:
        df: DataFrame to split
        ratio: Fraction for first part (0.0 to 1.0)
        seed: Random seed for shuffling before split

    Returns:
        Tuple of (first_part, second_part) DataFrames
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Ratio must be between 0.0 and 1.0, got {ratio}")

    if len(df) == 0:
        return df.copy(), df.copy()

    # Shuffle if seed provided
    if seed is not None:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Calculate split point and split
    split_idx = int(len(df) * ratio)
    first_part = df.iloc[:split_idx]
    second_part = df.iloc[split_idx:]

    return first_part.copy(), second_part.copy()


@DataFrameTransformRegistry.register("split_dataframe_by_filters")
def split_dataframe_by_filters(
    df: pd.DataFrame, filters: List[str]
) -> List[pd.DataFrame]:
    """Split DataFrame into multiple parts using pandas query filters.

    Args:
        df: DataFrame to split
        filters: List of pandas query strings
                Example: ["age > 30", "category == 'A'", "score >= 0.8"]

    Returns:
        List of filtered DataFrames

    Raises:
        ValueError: If any filter is invalid
    """
    if not filters:
        raise ValueError("At least one filter must be provided")

    results = []
    for filter_expr in filters:
        try:
            filtered_df = df.query(filter_expr)
            results.append(filtered_df.copy())
        except Exception as e:
            raise ValueError(f"Invalid filter '{filter_expr}': {e}")

    return results


@DataFrameTransformRegistry.register("concat_dataframes")
def concat_dataframes(
    dataframes: List[pd.DataFrame], ignore_index: bool = True, sort: bool = False
) -> pd.DataFrame:
    """Concatenate multiple DataFrames with error handling.

    Args:
        dataframes: List of DataFrames to concatenate
        ignore_index: Whether to ignore index in result
        sort: Whether to sort columns

    Returns:
        Concatenated DataFrame
    """
    # Filter out empty DataFrames
    non_empty_dfs = [df for df in dataframes if not df.empty]

    if not non_empty_dfs:
        return pd.DataFrame()

    if len(non_empty_dfs) == 1:
        return non_empty_dfs[0].copy()

    return pd.concat(non_empty_dfs, ignore_index=ignore_index, sort=sort)
