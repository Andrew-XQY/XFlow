"""Pure data transformation primitives.

No ML frameworks, no config, no context - just composable functions.
"""

from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union

T = TypeVar("T")


def pipe(
    sample: T,
    *transforms: Union[Callable[[Any], Any], Sequence[Callable[[Any], Any]]],
) -> Any:
    """Apply transforms sequentially to a single sample.

    If sample is tuple/list and transform is a list/tuple of callables,
    applies each transform to the corresponding part.

    >>> pipe(5, lambda x: x * 2, lambda x: x + 1)
    11
    >>> pipe((1, 2), [lambda x: x * 10, lambda x: x * 100])
    (10, 200)
    """
    result = sample
    for fn in transforms:
        if isinstance(fn, (list, tuple)) and isinstance(result, (list, tuple)):
            parts = []
            for i, part in enumerate(result):
                if i < len(fn) and fn[i] is not None:
                    parts.append(fn[i](part))
                else:
                    parts.append(part)
            result = tuple(parts)
        else:
            result = fn(result)
    return result


def pipe_each(
    samples: Iterable[T],
    *transforms: Union[Callable[[Any], Any], Sequence[Callable[[Any], Any]]],
    progress: bool = False,
    desc: str = "Processing",
    skip_errors: bool = False,
    on_error: Optional[Callable[[Exception, T], None]] = None,
) -> Iterator[Any]:
    """Apply pipe() to each sample in an iterable (lazy).

    >>> list(pipe_each([1, 2, 3], lambda x: x * 2))
    [2, 4, 6]
    """
    if progress:
        try:
            from tqdm import tqdm

            samples = tqdm(samples, desc=desc)
        except ImportError:
            pass

    for sample in samples:
        if skip_errors:
            try:
                yield pipe(sample, *transforms)
            except Exception as e:
                if on_error:
                    on_error(e, sample)
        else:
            yield pipe(sample, *transforms)


def compose(
    *transforms: Union[Callable[[Any], Any], Sequence[Callable[[Any], Any]]],
) -> Callable[[Any], Any]:
    """Compose transforms into a single callable (uses pipe internally)."""

    def composed(sample: Any) -> Any:
        return pipe(sample, *transforms)

    return composed
