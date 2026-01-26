"""Pure data transformation primitives.

No ML frameworks, no config, no context - just composable functions.
"""

from typing import Any, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")


def pipe(sample: T, *transforms: Callable[[Any], Any]) -> Any:
    """Apply transforms sequentially to a single sample.

    >>> pipe(5, lambda x: x * 2, lambda x: x + 1)
    11
    """
    result = sample
    for fn in transforms:
        result = fn(result)
    return result


def pipe_each(samples: Iterable[T], *transforms: Callable[[Any], Any]) -> Iterator[Any]:
    """Apply transforms to each sample in an iterable (lazy).

    >>> list(pipe_each([1, 2, 3], lambda x: x * 2))
    [2, 4, 6]
    """
    for sample in samples:
        yield pipe(sample, *transforms)


def compose(*transforms: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose transforms into a single callable.

    >>> double_plus_one = compose(lambda x: x * 2, lambda x: x + 1)
    >>> double_plus_one(5)
    11
    """

    def composed(sample: Any) -> Any:
        return pipe(sample, *transforms)

    return composed
