from functools import wraps
from typing import Callable
from tqdm import tqdm

def with_progress(desc: str = "Processing", unit: str = "item"):
    """Decorator to add progress bar to functions that process iterables."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find the iterable argument (usually first positional arg)
            if args and hasattr(args[0], '__iter__'):
                iterable = args[0]
                # Wrap with tqdm
                progress_iterable = tqdm(iterable, desc=desc, unit=unit)
                return func(progress_iterable, *args[1:], **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator