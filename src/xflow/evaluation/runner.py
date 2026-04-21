from __future__ import annotations

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

import torch
from torch.utils._pytree import tree_map


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

    model: torch.nn.Module
    device: torch.device
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


def default_forward(model: torch.nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


# ----------------------------
# Small tensor utilities
# ----------------------------
def _to_device(value: Any, device: torch.device) -> Any:
    return tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, value)


def _detach_cpu(value: Any) -> Any:
    return tree_map(
        lambda t: t.detach().cpu() if isinstance(t, torch.Tensor) else t, value
    )


def _batch_size_from_inputs(inputs: Any) -> int:
    """Infer batch size from inputs only. Assumption: first tensor's dim 0 is batch."""
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

    def pick(t):
        if isinstance(t, torch.Tensor):
            return t if t.ndim == 0 else t[i]
        return t

    return tree_map(pick, value)


# ----------------------------
# Core runner
# ----------------------------
def run_evaluation(
    model: torch.nn.Module,
    dataset: Iterable[Any],
    device: Union[str, torch.device],
    hooks: Optional[Sequence[EvalHook]] = None,
    unpack_batch: Callable[[Any], Unpacked] = default_unpack_batch,
    forward_fn: Callable[[torch.nn.Module, Any], Any] = default_forward,
    max_batches: Optional[int] = None,
    strict_hook_errors: bool = True,
) -> EvalContext:
    """
    Minimal evaluation/inference workflow.

    Batch contract:
    - `{"inputs": ..., "targets": ..., "metadata": ...}`
    - `(x,)`
    - `(x, y)`

    Hook contract:
    - `on_start(ctx)`
    - `on_batch(ctx, batch)`
    - `on_end(ctx)`

    Example:
    - `ctx = run_evaluation(model, loader, "cuda", hooks=[TqdmHook(), InMemoryCollector()])`

    Responsibilities:
    - set eval mode, move model to device, restore training mode on exit
    - iterate batches, unpack, forward, build EvalBatch
    - fire hooks
    - track basic stats

    Non-responsibilities (use a hook):
    - progress bars, saving, metrics, plotting, domain-specific extraction
    """
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
            from tqdm.auto import tqdm

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
