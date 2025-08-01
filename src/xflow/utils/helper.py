import inspect
import itertools
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import __main__

from .typing import T

# =============================================================================
# Path helpers
# =============================================================================


def print_caller_directory():
    """
    Prints the directory path of the script that called this function.

    Useful for debugging or logging script origin in multi-file projects.
    """
    caller_frame = inspect.stack()[1]
    caller_file = os.path.abspath(caller_frame.filename)
    print("Caller script directory:", os.path.dirname(caller_file))


def get_base_dir() -> Path:
    """
    Returns the directory path of the calling context with cross-environment compatibility.
    """
    # 1. Check if running in Jupyter notebook
    try:
        # Check for IPython/Jupyter environment
        if "ipykernel" in sys.modules or "IPython" in sys.modules:
            # Try to get notebook directory from IPython
            try:
                from IPython import get_ipython

                ipython = get_ipython()
                if ipython is not None:
                    # Get the current working directory in Jupyter
                    return Path(os.getcwd()).resolve()
            except ImportError:
                pass
    except Exception:
        pass

    # 2. Direct script execution: __main__.__file__ exists
    try:
        main_file = getattr(__main__, "__file__", None)
        if main_file and os.path.exists(main_file):
            return Path(main_file).parent.resolve()
    except Exception:
        pass

    # 3. Check if running as frozen executable
    try:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).parent.resolve()
    except Exception:
        pass

    # 4. Fallback: inspect stack for first external caller (skip IPython frames)
    try:
        current_file = Path(__file__).resolve()

        for frame_info in inspect.stack()[1:]:  # skip current frame
            filename = frame_info.filename
            # Skip interactive frames, this module, IPython/Jupyter internals, and built-ins
            if (
                filename.startswith("<")
                or filename.startswith("[")
                or "IPython" in filename  # some REPLs use brackets
                or "ipykernel" in filename
                or "jupyter" in filename
                or Path(filename).resolve() == current_file
            ):
                continue

            file_path = Path(filename)
            if file_path.exists():
                return file_path.parent.resolve()

    except Exception:
        pass

    # 5. Try sys.argv[0] if available
    try:
        if sys.argv and sys.argv[0]:
            script_path = Path(sys.argv[0])
            if script_path.exists() and script_path.is_file():
                return script_path.parent.resolve()
    except Exception:
        pass

    # 6. Ultimate fallback: current working directory
    return Path(os.getcwd()).resolve()


# =============================================================================
# Iterable/Sequence helpers
# =============================================================================


def subsample_sequence(
    items: Sequence[T],
    n_samples: Optional[int] = None,
    fraction: Optional[float] = None,
    strategy: str = "random",
    seed: Optional[int] = 42,
) -> List[T]:
    """
    Subsampling function for any Sequence.

    Args:
        items: Any sequence (list, tuple, etc.) of type T.
        n_samples: Exact number to sample.
        fraction: Fraction to sample (0.0 to 1.0).
        strategy: "random", "first", "last", "stride", or "reservoir".
        seed: Random seed for reproducibility.

    Returns:
        List of sampled items of type T.
    """
    # Validate parameters
    if n_samples is not None and fraction is not None:
        raise ValueError("Specify exactly one of n_samples or fraction, not both.")

    length = len(items)
    if n_samples is None and fraction is None:
        return list(items)

    if fraction is not None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be between 0.0 and 1.0")
        n_samples = int(length * fraction)

    # Clamp n_samples to [0, length]
    n_samples = max(0, min(n_samples, length))

    # Random sampling
    if strategy == "random":
        rng = random.Random(seed)
        return rng.sample(list(items), n_samples)
    # First n samples
    elif strategy == "first":
        return list(items[:n_samples])
    # Last n samples
    elif strategy == "last":
        return list(items[-n_samples:])
    # Stride sampling (lazy with islice)
    elif strategy == "stride":
        if n_samples == 0:
            return []
        step = max(1, length // n_samples)
        return list(itertools.islice(items, 0, None, step))[:n_samples]
    # Reservoir sampling for true iterators
    elif strategy == "reservoir":
        rng = random.Random(seed)
        reservoir: List[T] = []
        for i, elem in enumerate(items):
            if i < n_samples:
                reservoir.append(elem)
            else:
                j = rng.randint(0, i)
                if j < n_samples:
                    reservoir[j] = elem
        return reservoir
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def split_sequence(
    items: Sequence[T], split_ratio: float = 0.8, seed: int = 42, shuffle: bool = True
) -> Tuple[List[T], List[T]]:
    """
    Split a sequence into two parts.

    Args:
        items: Any sequence (list, tuple, etc.) of type T.
        split_ratio: Ratio for first part (0.0 to 1.0).
        seed: Random seed for reproducibility.
        shuffle: Whether to shuffle before splitting.

    Returns:
        Tuple of (first_part, second_part) as lists of type T.
    """
    if not 0.0 <= split_ratio <= 1.0:
        raise ValueError(f"split_ratio must be between 0.0 and 1.0, got {split_ratio}")

    items_list = list(items)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items_list)

    split_idx = int(len(items_list) * split_ratio)
    first_part = items_list[:split_idx]
    second_part = items_list[split_idx:]

    return first_part, second_part


# =============================================================================
# Dictionary helpers
# =============================================================================


def deep_update(base: MutableMapping[str, Any], updates: Dict[str, Any]) -> None:
    """Recursively update a dictionary with another dictionary.

    Nested dictionaries are merged, other values are replaced.
    Modifies base dictionary in-place.

    Args:
        base: Dictionary to update (modified in-place)
        updates: Dictionary with updates to apply

    Example:
        >>> base = {"a": {"x": 1, "y": 2}, "b": 3}
        >>> updates = {"a": {"x": 10, "z": 3}, "c": 4}
        >>> deep_update(base, updates)
        >>> base
        {"a": {"x": 10, "y": 2, "z": 3}, "b": 3, "c": 4}
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def deep_merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries recursively, returning a new dictionary.

    Args:
        *dicts: Dictionaries to merge (left-to-right precedence)

    Returns:
        New merged dictionary
    """
    if not dicts:
        return {}

    result = {}
    for d in dicts:
        deep_update(result, d)
    return result
