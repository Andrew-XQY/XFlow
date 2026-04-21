from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from ..utils.typing import DeviceLike, ModelLike


def _get_torch():
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise ImportError(
            "xflow.evaluation.runner requires PyTorch at runtime. Install xflow-py[ml_torch] or torch."
        ) from exc


# ----------------------------
# Contract objects
# ----------------------------
@dataclass
class EvalBatch:
    """
    Per-batch record passed to hooks.

    Tensors are detached to CPU by default.
    Hooks that need a different representation should provide a custom
    `forward_fn` or their own hook-side handling.

    `targets` is None for unsupervised runs. That's the only difference —
    the runner does not branch on task type.
    """

    index: int
    inputs: Any
    predictions: Any
    targets: Any = None
    metadata: Any = None
    raw: Any = None


@dataclass
class EvalContext:
    """
    Run-level context shared by all hooks.

    Hooks can:
    - read all fields
    - write to `extras` for cross-hook data sharing
    - set `stop = True` to halt before the next batch
    """

    model: ModelLike
    device: DeviceLike
    total_batches: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    seen_batches: int = 0
    seen_samples: int = 0
    stop: bool = False
    extras: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)  # list[tuple[str, Exception]]


# ----------------------------
# Hook protocol
# ----------------------------
class EvalHook(Protocol):
    def on_start(self, ctx: EvalContext) -> None: ...
    def on_batch(self, ctx: EvalContext, batch: EvalBatch) -> None: ...
    def on_end(self, ctx: EvalContext) -> None: ...


class BaseEvalHook:
    """No-op base class. Inherit and override only what you need."""

    def on_start(self, ctx: EvalContext) -> None:
        pass

    def on_batch(self, ctx: EvalContext, batch: EvalBatch) -> None:
        pass

    def on_end(self, ctx: EvalContext) -> None:
        pass


# ----------------------------
# Defaults
# ----------------------------
Unpacked = Tuple[
    Any, Any, Any
]  # (inputs, targets, metadata); targets/metadata may be None


def default_unpack_batch(raw: Any) -> Unpacked:
    """
    Canonical batch shapes:
    - dict: {"inputs": ..., "targets": ..., "metadata": ...}  (targets/metadata optional)
    - tuple/list: (x,) or (x, y)

    Anything more exotic: write your own unpacker and pass it via `unpack_batch=`.
    """
    if isinstance(raw, Mapping):
        if "inputs" not in raw:
            raise ValueError("Mapping batch must contain key 'inputs'.")
        return raw["inputs"], raw.get("targets"), raw.get("metadata")

    if isinstance(raw, (tuple, list)):
        if len(raw) == 1:
            return raw[0], None, None
        if len(raw) == 2:
            return raw[0], raw[1], None
        raise ValueError(
            f"Tuple/list batch must have length 1 or 2, got {len(raw)}. "
            "For richer shapes, supply a custom unpack_batch."
        )

    raise TypeError(f"Unsupported batch type: {type(raw).__name__}")


def default_forward(model: ModelLike, inputs: Any) -> Any:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


# ----------------------------
# Small tensor utilities
# ----------------------------
def _tree_map(fn: Callable[[Any], Any], x: Any) -> Any:
    torch = _get_torch()
    if isinstance(x, torch.Tensor):
        return fn(x)
    if isinstance(x, dict):
        return {k: _tree_map(fn, v) for k, v in x.items()}
    if isinstance(x, tuple):
        mapped = [_tree_map(fn, v) for v in x]
        return type(x)(*mapped) if hasattr(x, "_fields") else type(x)(mapped)
    if isinstance(x, list):
        return [_tree_map(fn, v) for v in x]
    return x


def _to_device(value: Any, device: DeviceLike) -> Any:
    return _tree_map(lambda tensor: tensor.to(device), value)


def _detach_cpu(value: Any) -> Any:
    return _tree_map(lambda tensor: tensor.detach().cpu(), value)


def _batch_size_from_inputs(inputs: Any) -> int:
    """Infer batch size from inputs only. Assumption: first tensor's dim 0 is batch."""
    torch = _get_torch()
    if isinstance(inputs, torch.Tensor):
        return 1 if inputs.ndim == 0 else int(inputs.shape[0])
    if isinstance(inputs, Mapping):
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                return 1 if v.ndim == 0 else int(v.shape[0])
    if isinstance(inputs, (list, tuple)):
        for v in inputs:
            if isinstance(v, torch.Tensor):
                return 1 if v.ndim == 0 else int(v.shape[0])
    return 1


def slice_sample(value: Any, i: int) -> Any:
    """
    Index the i-th sample from a batched pytree. Returns views (no clone).
    Exposed because hooks commonly need it.
    """

    torch = _get_torch()

    def pick(t):
        if isinstance(t, torch.Tensor):
            return t if t.ndim == 0 else t[i]
        return t

    return _tree_map(pick, value)


# ----------------------------
# Core runner
# ----------------------------
def run_evaluation(
    model: ModelLike,
    dataset: Iterable[Any],
    device: DeviceLike,
    hooks: Optional[Sequence[EvalHook]] = None,
    unpack_batch: Callable[[Any], Unpacked] = default_unpack_batch,
    forward_fn: Callable[[ModelLike, Any], Any] = default_forward,
    max_batches: Optional[int] = None,
    strict_hook_errors: bool = True,
) -> EvalContext:
    """
    Run a simple model inference/evaluation loop.

    Use this when you already have:
    - a model
    - an iterable dataset or dataloader
    - optional hooks for progress, saving, metrics, or custom post-processing

    Accepted batch shapes by default:
    - `{"inputs": x, "targets": y, "metadata": meta}`
    - `(x,)`
    - `(x, y)`

    Hooks receive:
    - `ctx`: run-level state such as device, counters, and stop flag
    - `batch`: one batch with `inputs`, `predictions`, optional `targets`, and metadata

    Typical usage:
    - `ctx = run_evaluation(model, loader, "cuda")`
    - `ctx = run_evaluation(model, loader, "cuda", hooks=[TqdmHook(), InMemoryCollector()])`

    For custom batch formats, pass your own `unpack_batch`.
    For custom model calls, pass your own `forward_fn`.
    """
    torch = _get_torch()
    hooks = list(hooks or [])
    device = torch.device(device)
    total_batches = len(dataset) if hasattr(dataset, "__len__") else None  # type: ignore[arg-type]
    ctx = EvalContext(model=model, device=device, total_batches=total_batches)

    def fire(method: str, *args) -> None:
        for h in hooks:
            fn = getattr(h, method, None)
            if not callable(fn):
                continue
            try:
                fn(*args)
            except Exception as e:
                if strict_hook_errors:
                    raise
                ctx.errors.append((f"{h.__class__.__name__}.{method}", e))

    was_training = model.training
    model.to(device)
    model.eval()

    try:
        fire("on_start", ctx)

        with torch.inference_mode():
            for i, raw in enumerate(dataset):
                if ctx.stop:
                    break
                if max_batches is not None and i >= max_batches:
                    break

                inputs, targets, metadata = unpack_batch(raw)
                device_inputs = _to_device(inputs, device)
                predictions = forward_fn(model, device_inputs)

                batch = EvalBatch(
                    index=i,
                    inputs=_detach_cpu(inputs),
                    predictions=_detach_cpu(predictions),
                    targets=_detach_cpu(targets),
                    metadata=metadata,
                    raw=raw,
                )

                ctx.seen_batches += 1
                ctx.seen_samples += _batch_size_from_inputs(batch.inputs)

                fire("on_batch", ctx, batch)

        return ctx
    finally:
        ctx.finished_at = time.time()
        try:
            fire("on_end", ctx)
        finally:
            model.train(was_training)


# ----------------------------
# Built-in hooks
# ----------------------------
class TqdmHook(BaseEvalHook):
    """Progress bar as a hook. Silent failure if tqdm unavailable."""

    def __init__(self, desc: str = "Evaluation", unit: str = "batch") -> None:
        self.desc = desc
        self.unit = unit
        self._bar = None

    def on_start(self, ctx: EvalContext) -> None:
        try:
            tqdm = importlib.import_module("tqdm.auto").tqdm
            self._bar = tqdm(total=ctx.total_batches, desc=self.desc, unit=self.unit)
        except Exception:
            self._bar = None

    def on_batch(self, ctx: EvalContext, batch: EvalBatch) -> None:
        if self._bar is not None:
            self._bar.update(1)

    def on_end(self, ctx: EvalContext) -> None:
        if self._bar is not None:
            self._bar.close()


class InMemoryCollector(BaseEvalHook):
    """Collect per-sample records. Suitable for small validation sets."""

    def __init__(self) -> None:
        self.samples: list = []

    def on_batch(self, ctx: EvalContext, batch: EvalBatch) -> None:
        b = _batch_size_from_inputs(batch.inputs)
        for i in range(b):
            row = {
                "batch_index": batch.index,
                "sample_index": i,
                "inputs": slice_sample(batch.inputs, i),
                "predictions": slice_sample(batch.predictions, i),
            }
            if batch.targets is not None:
                row["targets"] = slice_sample(batch.targets, i)
            if batch.metadata is not None:
                row["metadata"] = slice_sample(batch.metadata, i)
            self.samples.append(row)
