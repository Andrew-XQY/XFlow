from functools import wraps
from typing import Callable

from tqdm import tqdm


def with_progress(func: Callable) -> Callable:
    """Decorator to add progress bar to functions that process iterables."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find the iterable argument (usually first positional arg)
        if args and hasattr(args[0], "__iter__"):
            iterable = args[0]
            # Wrap with tqdm
            progress_iterable = tqdm(iterable)
            # Replace first argument with progress-wrapped version
            new_args = (progress_iterable,) + args[1:]
            return func(*new_args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def print_separator(symbol: str = "-", length: int = 80):
    """
    Decorator to print separator lines before and after function execution.

    Args:
        symbol: Character to use (default: "-")
        length: Length of separator (default: 80)

    Usage:
        @print_separator()          # default: - with length 80
        @print_separator("=")       # = with length 80
        @print_separator("*", 50)   # * with length 50
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(symbol * length)
            result = func(*args, **kwargs)
            print(symbol * length)
            return result

        return wrapper

    return decorator
