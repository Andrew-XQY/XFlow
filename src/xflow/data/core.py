"""Pure data transformation primitives.

No ML frameworks, no config, no context - just composable functions
with data flow (distribution) and transformation mapping.
"""

from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union

T = TypeVar("T")


# --- Markers for special operations ---


class _Consume:
    """Marker: consume N inputs, feed to one transform."""

    __slots__ = ("n", "fn")

    def __init__(self, n: int, fn: Callable):
        if n < 1:
            raise ValueError(f"consume() requires n >= 1, got {n}")
        self.n = n
        self.fn = fn

    def __repr__(self):
        return f"consume({self.n}, {self.fn})"


class _Rest:
    """Marker: consume all remaining inputs, feed to one transform."""

    __slots__ = ("fn",)

    def __init__(self, fn: Callable):
        self.fn = fn

    def __repr__(self):
        return f"rest({self.fn})"


def consume(n: int, fn: Callable) -> _Consume:
    """Consume exactly N inputs and feed as tuple to transform.

    Use when a transform needs to JOIN multiple inputs into one.

    Args:
        n: Number of inputs to consume (must be >= 1)
        fn: Transform receiving tuple of n items

    Returns:
        Marker for pipe() to interpret

    Examples:
        >>> # Join 2 images into 1, pass meta through
        >>> pipe((img1, img2, meta), [consume(2, join_images), None])
        (joined_image, meta)

        >>> # Explicit: take 3 inputs
        >>> pipe((a, b, c, d), [consume(3, combine), None])
        (combine((a, b, c)), d)
    """
    return _Consume(n, fn)


def rest(fn: Callable) -> _Rest:
    """Consume ALL remaining inputs and feed as tuple to transform.

    Use when you want to collect everything left in the stream.

    Args:
        fn: Transform receiving tuple of all remaining items

    Returns:
        Marker for pipe() to interpret

    Examples:
        >>> # Collect all into list, regardless of count
        >>> pipe((a, b, c, d), [rest(list)])
        ([a, b, c, d],)

        >>> # Process first, collect rest
        >>> pipe((a, b, c), [transform_a, rest(combine_rest)])
        (transformed_a, combined_bc)
    """
    return _Rest(fn)


# --- Core implementation ---


def _apply_broadcast(
    parts: tuple,
    transforms: Sequence[Union[Callable, _Consume, _Rest, None]],
    flatten: bool,
) -> tuple:
    """Apply transforms to parts with positional + marker semantics.

    Rules (in order of processing):
        - None: pass through one input unchanged
        - _Consume(n, fn): take n inputs, apply fn to tuple
        - _Rest(fn): take ALL remaining inputs, apply fn to tuple
        - Callable: take one input, apply fn (may split if returns tuple)

    Remaining inputs after all transforms are passed through.
    """
    results = []
    idx = 0
    n_parts = len(parts)

    for tfm in transforms:
        if idx >= n_parts:
            break

        if tfm is None:
            # Identity: pass one through
            results.append(parts[idx])
            idx += 1

        elif isinstance(tfm, _Rest):
            # Consume all remaining
            remaining = parts[idx:]
            output = tfm.fn(remaining)
            if flatten and isinstance(output, tuple):
                results.extend(output)
            else:
                results.append(output)
            idx = n_parts  # All consumed
            break

        elif isinstance(tfm, _Consume):
            # Consume exactly n
            end = idx + tfm.n
            if end > n_parts:
                raise ValueError(
                    f"consume({tfm.n}) needs {tfm.n} inputs, "
                    f"but only {n_parts - idx} remain at index {idx}."
                )
            chunk = parts[idx:end]
            output = tfm.fn(chunk)
            if flatten and isinstance(output, tuple):
                results.extend(output)
            else:
                results.append(output)
            idx = end

        else:
            # Regular callable: consume one
            output = tfm(parts[idx])
            if flatten and isinstance(output, tuple):
                results.extend(output)
            else:
                results.append(output)
            idx += 1

    # Pass through any remaining inputs
    while idx < n_parts:
        results.append(parts[idx])
        idx += 1

    return tuple(results)


def pipe(
    sample: T,
    *transforms: Union[Callable, Sequence[Union[Callable, _Consume, _Rest, None]]],
    flatten: bool = True,
) -> Any:
    """Apply transforms sequentially to a sample.

    Supports broadcast mode with splits and joins:
    - List of transforms: apply positionally to tuple inputs
    - None in list: identity (pass-through)
    - consume(n, fn): join n inputs into one
    - rest(fn): join all remaining inputs
    - Transform returning tuple: split into multiple (when flatten=True)

    Args:
        sample: Input (single value or tuple)
        *transforms: Transforms or broadcast lists
        flatten: If True, tuple outputs expand the stream (enables splits)

    Returns:
        Transformed sample

    Examples:
        >>> # Basic
        >>> pipe(5, lambda x: x * 2)
        10

        >>> # Broadcast with pass-through
        >>> pipe((img, meta), [resize, None])
        (resized_img, meta)

        >>> # Split (transform returns tuple)
        >>> pipe((img, meta), [split_lr, None])
        (left, right, meta)

        >>> # Join with consume()
        >>> pipe((left, right, meta), [consume(2, join), None])
        (joined, meta)

        >>> # Full pipeline
        >>> pipe(
        ...     (path, meta),
        ...     [load, None],           # (img, meta)
        ...     [split_width, None],    # (L, R, meta)
        ...     [crop_a, crop_b, None], # (L', R', meta)
        ...     [consume(2, join), None], # (joined, meta)
        ... )
    """
    result = sample

    for fn in transforms:
        if isinstance(fn, (list, tuple)):
            # Broadcast mode
            if not isinstance(result, (list, tuple)):
                result = (result,)
            result = _apply_broadcast(tuple(result), fn, flatten=flatten)
        elif fn is not None:
            # Single transform
            output = fn(result)
            # Flatten only if input wasn't already a tuple (avoid double-wrap)
            if flatten and isinstance(output, tuple) and not isinstance(result, tuple):
                result = output
            else:
                result = output

    return result


def pipe_each(
    samples: Iterable[T],
    *transforms: Union[Callable, Sequence[Union[Callable, _Consume, _Rest, None]]],
    progress: bool = False,
    desc: str = "Processing",
    skip_errors: bool = False,
    on_error: Optional[Callable[[Exception, T], None]] = None,
    flatten: bool = True,
) -> Iterator[Any]:
    """Apply pipe() to each sample in an iterable (lazy).

    Args:
        samples: Iterable of samples
        *transforms: Transforms or broadcast lists
        progress: Show tqdm progress bar
        desc: Progress description
        skip_errors: Continue on errors
        on_error: Error callback(exception, sample)
        flatten: Enable split operations

    Yields:
        Transformed samples

    Examples:
        >>> list(pipe_each([1, 2, 3], lambda x: x * 2))
        [2, 4, 6]

        >>> # With broadcast and join
        >>> list(pipe_each(
        ...     [(p, {}) for p in paths],
        ...     [load, None],
        ...     [split_width, None],
        ...     [consume(2, join), None],
        ... ))
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
                yield pipe(sample, *transforms, flatten=flatten)
            except Exception as e:
                if on_error:
                    on_error(e, sample)
        else:
            yield pipe(sample, *transforms, flatten=flatten)


def compose(
    *transforms: Union[Callable, Sequence[Union[Callable, _Consume, _Rest, None]]],
    flatten: bool = True,
) -> Callable[[Any], Any]:
    """Compose transforms into a single callable.

    Examples:
        >>> process = compose([load, None], [resize, None])
        >>> result = process((path, meta))
    """

    def composed(sample: Any) -> Any:
        return pipe(sample, *transforms, flatten=flatten)

    return composed
