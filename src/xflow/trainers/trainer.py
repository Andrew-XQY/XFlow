"""trainer with unified callbacks and delegated model I/O across frameworks."""

from __future__ import annotations

import collections
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING

from ..utils.io import create_directory
from ..utils.typing import ModelType, PathLikeStr

if TYPE_CHECKING:
    import torch
# ============================== Callback core ==============================


@dataclass
class CallbackContext:
    trainer: "BaseTrainer"
    model: Any
    optimizer: Any = None
    scheduler: Any = None
    device: Any = None
    # progress
    epochs: int = 0  # NEW: total epochs
    epoch: int = 0
    batch_idx: int = -1
    batch: int = 0  # NEW: alias for PyTorch-style callbacks
    global_step: int = 0
    total_batches: int = 0
    phase: str = "train"
    logs: Dict[str, float] = field(default_factory=dict)
    request_stop: bool = False


class Callback:
    # Subclass and override the hooks you need. Every method receives only `ctx`.
    def on_train_begin(self, ctx: CallbackContext): ...
    def on_train_end(self, ctx: CallbackContext): ...
    def on_epoch_begin(self, ctx: CallbackContext): ...
    def on_epoch_end(self, ctx: CallbackContext): ...
    def on_batch_begin(self, ctx: CallbackContext): ...
    def on_batch_end(self, ctx: CallbackContext): ...

    # Optional validation-specific hooks:
    def on_val_epoch_begin(self, ctx: CallbackContext): ...
    def on_val_epoch_end(self, ctx: CallbackContext): ...
    def on_val_batch_begin(self, ctx: CallbackContext): ...
    def on_val_batch_end(self, ctx: CallbackContext): ...


class CallbackDispatcher:
    def __init__(self, callbacks):
        self.cbs = callbacks or []

    def call(self, name, ctx):
        import inspect

        kw = asdict(ctx)
        for cb in self.cbs:
            fn = getattr(cb, name, None)
            if not callable(fn):
                continue
            try:
                params = inspect.signature(fn).parameters
                usable = {k: v for k, v in kw.items() if k in params}
                if usable:
                    fn(**usable)  # works with your current PyTorch callbacks
                else:
                    fn(ctx)  # also supports modern def on_*(self, ctx)
            except TypeError:
                fn()  # last fallback

    @property
    def should_stop(self) -> bool:
        return any(getattr(cb, "should_stop", False) for cb in self.cbs)


# ============================== Model I/O (optional) ==============================


class ModelIO:
    """Tiny adapter so the trainer can save models when the model has no save_model()."""

    def save(self, model: Any, path: str, extra: Optional[Dict[str, Any]] = None):
        # 1) If model exposes save_model, use it (preferred).
        if hasattr(model, "save_model") and callable(getattr(model, "save_model")):
            model.save_model(path)
            return
        # 2) Try tf.keras
        try:
            import tensorflow as tf  # type: ignore

            if isinstance(model, tf.keras.Model):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                model.save(path)
                return
        except Exception:
            pass
        # 3) Try PyTorch state_dict
        try:
            import torch  # type: ignore

            if hasattr(model, "state_dict"):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                ckpt = {"model_state": model.state_dict()}
                if extra:
                    ckpt.update(extra)
                if not (path.endswith(".pt") or path.endswith(".pth")):
                    path = path + ".pt"
                torch.save(ckpt, path)
                return
        except Exception:
            pass
        raise NotImplementedError(
            "No known way to save this model. Implement model.save_model(path) or provide a custom ModelIO."
        )


# ============================== Base trainer ==============================


class BaseTrainer(ABC):
    """
    Thin orchestrator: runs loops, dispatches callbacks, collects history.
    Creation/compilation of model/optimizer is outside; inject everything in.
    """

    def __init__(
        self,
        model: Any,
        data_pipeline: Any,
        output_dir: str,
        *,
        callbacks: Optional[List[Callback]] = None,
        model_io: Optional[ModelIO] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if model is None:
            raise ValueError("model cannot be None")
        if data_pipeline is None:
            raise ValueError("data_pipeline cannot be None")
        if not output_dir:
            raise ValueError("output_dir is required")

        os.makedirs(output_dir, exist_ok=True)
        self.model = model
        self.data = data_pipeline
        self.output_dir = output_dir
        self.cb = CallbackDispatcher(callbacks)
        self.model_io = model_io or ModelIO()
        self.config = dict(config or {})
        self.history: Dict[str, List[Any]] = collections.defaultdict(list)

    def save_history(self, path: Optional[str] = None):
        path = path or os.path.join(self.output_dir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f)

    def save_model(self, path: Optional[str] = None, **extra):
        path = path or os.path.join(self.output_dir, "model.pt")
        self.model_io.save(self.model, path, extra=extra or {})

    # Loader resolution helper (optional; keeps user code short)
    def _resolve_loaders(self, train_loader, val_loader):
        if train_loader is None:
            for name in ("train_loader", "train", "get_train_loader"):
                cand = getattr(self.data, name, None)
                train_loader = cand() if callable(cand) else cand
                if train_loader is not None:
                    break
        if val_loader is None:
            for name in ("val_loader", "val", "get_val_loader"):
                cand = getattr(self.data, name, None)
                val_loader = cand() if callable(cand) else cand
                if val_loader is not None:
                    break
        if train_loader is None:
            raise ValueError("No train_loader provided and data_pipeline has none.")
        return train_loader, val_loader

    @abstractmethod
    def fit(
        self,
        *,
        epochs: int,
        train_loader: Optional[Iterable] = None,
        val_loader: Optional[Iterable] = None,
    ) -> Dict[str, List[Any]]: ...

    @abstractmethod
    def predict(self, loader: Iterable, **kwargs) -> Any: ...


# ============================== PyTorch trainer ==============================


class TorchTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        model: Any,
        data_pipeline: Any,
        output_dir: str,
        optimizer: Any,
        criterion: Any,
        device: Any = None,
        callbacks: Optional[List[Callback]] = None,
        model_io: Optional[ModelIO] = None,
        config: Optional[Dict[str, Any]] = None,
        val_metrics: Optional[List[Callable[[Any, Any], Dict[str, float]]]] = None,
    ):
        super().__init__(
            model,
            data_pipeline,
            output_dir,
            callbacks=callbacks,
            model_io=model_io,
            config=config,
        )
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.val_metrics = val_metrics or []

    # ---- micro-steps (override if needed) ----
    def _to_device(self, batch):
        import torch

        x, y = batch[:2]
        return x.to(self.device), y.to(self.device)

    def train_step(self, batch) -> Dict[str, float]:
        x, y = self._to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)
        out = self.model(x)
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item())}

    def val_step(self, batch) -> Dict[str, float]:
        import torch

        with torch.no_grad():
            x, y = self._to_device(batch)
            out = self.model(x)
            logs = {"val_loss": float(self.criterion(out, y).item())}
            for fn in self.val_metrics:
                extra = fn(out, y)  # must return dict[str, float]
                if extra:
                    logs.update(extra)
            return logs

    # ---- main loop ----
    def fit(
        self,
        *,
        epochs: int,
        train_loader: Optional[Iterable] = None,
        val_loader: Optional[Iterable] = None,
    ) -> Dict[str, List[Any]]:
        import torch

        self.device = self.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        if hasattr(self, "discriminator") and self.discriminator is not None:
            self.discriminator.to(self.device)
            
        train_loader, val_loader = self._resolve_loaders(train_loader, val_loader)

        ctx = CallbackContext(
            trainer=self,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            total_batches=len(train_loader),
            logs={},
        )
        ctx.epochs = epochs  # NEW: make epochs visible to callbacks
        self.cb.call("on_train_begin", ctx)

        global_step = 0
        for epoch in range(epochs):
            ctx.epoch, ctx.phase, ctx.logs = epoch, "train", {}
            self.cb.call("on_epoch_begin", ctx)

            # -------- train --------
            self.model.train()
            if hasattr(self, "discriminator") and self.discriminator is not None:
                self.discriminator.train()
            sum_loss = 0.0
            for i, batch in enumerate(train_loader):
                ctx.batch_idx = i
                ctx.batch = i  # NEW: provide PyTorch-style 'batch'
                self.cb.call("on_batch_begin", ctx)
                logs = self.train_step(batch)
                logs["train_loss"] = float(logs.get("loss", 0.0))   # pass train_loss
                sum_loss += logs.get("loss", 0.0)
                global_step += 1
                ctx.logs = logs
                ctx.global_step = global_step
                self.cb.call("on_batch_end", ctx)
                if ctx.request_stop:
                    break

            avg_train = sum_loss / max(1, len(train_loader))

            # -------- validate --------
            val_logs_epoch = {}
            if val_loader is not None and not ctx.request_stop:
                ctx.phase, ctx.logs = "val", {}
                self.cb.call("on_val_epoch_begin", ctx)
                self.model.eval()
                if hasattr(self, "discriminator") and self.discriminator is not None:
                    self.discriminator.eval()
                acc = collections.defaultdict(float)
                for j, batch in enumerate(val_loader):
                    ctx.batch_idx = j
                    ctx.batch = j
                    self.cb.call("on_val_batch_begin", ctx)
                    logs = self.val_step(batch)
                    for k, v in logs.items():
                        acc[k] += float(v)
                    ctx.logs = logs
                    self.cb.call("on_val_batch_end", ctx)
                    if ctx.request_stop:
                        break
                val_logs_epoch = {k: acc[k] / max(1, len(val_loader)) for k in acc}
                ctx.logs = val_logs_epoch
                self.cb.call("on_val_epoch_end", ctx)

            # -------- end epoch --------
            epoch_logs = {"train_loss": avg_train, **val_logs_epoch}
            for k, v in epoch_logs.items():
                self.history[k].append(v)
            ctx.phase, ctx.logs = "train", epoch_logs
            self.cb.call("on_epoch_end", ctx)

            if ctx.request_stop or self.cb.should_stop:
                break

        self.cb.call("on_train_end", ctx)
        return dict(self.history)

    def predict(self, loader: Iterable, **_) -> List[Any]:
        self.model.eval()
        preds = []
        for batch in loader:
            x, _ = self._to_device(batch)
            preds.append(self.model(x))
        return preds

class TorchGANTrainer(TorchTrainer):
    def __init__(
        self,
        *,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        losses,  # Pix2PixLosses(lambda_l1=100)
        data_pipeline: Any,
        output_dir: str,
        device: Any = None,
        callbacks: Optional[List[Any]] = None,
        model_io: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        val_metrics: Optional[List[Any]] = None,
    ):
        # keep base fields happy: treat "model" as the generator
        super().__init__(
            model=generator,
            data_pipeline=data_pipeline,
            output_dir=output_dir,
            optimizer=optimizer_g,
            criterion=losses,  # stored; we won’t call base train_step
            device=device,
            callbacks=callbacks,
            model_io=model_io,
            config=config,
            val_metrics=val_metrics or [],
        )
        self.discriminator = discriminator
        self.optimizer_d = optimizer_d
        self.losses = losses  # Pix2PixLosses

    def train_step(self, batch) -> Dict[str, float]:
        import torch
        
        x, y = self._to_device(batch)

        # ---- D step ----
        self.optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = self.model(x)
        d_real = self.discriminator(x, y)
        d_fake = self.discriminator(x, fake)
        d_loss = self.losses.discriminator_loss(d_real, d_fake)
        d_loss.backward()
        self.optimizer_d.step()

        # ---- G step ----
        self.optimizer.zero_grad(set_to_none=True)  # optimizer for G
        fake = self.model(x)
        d_fake_for_g = self.discriminator(x, fake)
        g_total, g_gan, g_l1 = self.losses.generator_loss(d_fake_for_g, fake, y)
        g_total.backward()
        self.optimizer.step()

        return {
            "loss": float(g_total.item()),
            "g_gan": float(g_gan.item()),
            "g_l1": float(g_l1.item()),
            "d_loss": float(d_loss.item()),
        }

    def val_step(self, batch) -> Dict[str, float]:
        import torch
        
        with torch.no_grad():
            x, y = self._to_device(batch)
            fake = self.model(x)
            d_fake = self.discriminator(x, fake)
            d_real = self.discriminator(x, y)
            g_total, g_gan, g_l1 = self.losses.generator_loss(d_fake, fake, y)
            d_loss = self.losses.discriminator_loss(d_real, d_fake)
            logs = {
                "val_loss": float(g_total.item()),
                "val_g_gan": float(g_gan.item()),
                "val_g_l1": float(g_l1.item()),
                "val_d_loss": float(d_loss.item()),
            }
            for fn in self.val_metrics:
                extra = fn(fake, y)  # same signature as your current metrics
                if extra:
                    logs.update(extra)
            return logs