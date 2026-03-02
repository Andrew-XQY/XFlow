"""Utilities for composing pipeline hook callables."""

from typing import Any, Callable, Iterable, Optional, Union

from ..utils.typing import HookFn, HookInput


def compose_hooks(hooks: HookInput) -> Optional[HookFn]:
    """Compose one or many item hooks into a single callable.

    Args:
        hooks: Either:
            - None
            - a single hook callable with signature ``(item, item_id) -> item``
            - an iterable of hook callables with the same signature

    Returns:
        A single hook callable or ``None``.
    """
    if hooks is None:
        return None

    if callable(hooks):
        return hooks

    if isinstance(hooks, (str, bytes)):
        raise TypeError("hooks must be a callable or iterable of callables")

    try:
        hook_list = list(hooks)
    except TypeError as exc:
        raise TypeError("hooks must be a callable or iterable of callables") from exc

    if not hook_list:
        return None

    for index, hook in enumerate(hook_list):
        if not callable(hook):
            raise TypeError(f"hook at index {index} is not callable: {type(hook)}")

    def _composed(item: Any, item_id: Any) -> Any:
        for hook in hook_list:
            item = hook(item, item_id)
        return item

    return _composed


def as_hook(transform_fn: Callable[[Any], Any]) -> HookFn:
    """Adapt a standard transform ``(item) -> item`` to hook contract.

    Args:
        transform_fn: Callable that accepts only ``item`` and returns transformed item.

    Returns:
        Hook callable with signature ``(item, item_id) -> item``.
    """
    if not callable(transform_fn):
        raise TypeError("transform_fn must be callable")

    def _hook(item: Any, item_id: Any) -> Any:
        return transform_fn(item)

    return _hook
