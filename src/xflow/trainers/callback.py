from typing import Any, Callable, Dict, List, Optional, Tuple, Any, Dict

import yaml

# Map unified event names to framework-specific hook method names
event_map = {
    "train_start": {"tf": "on_train_begin", "pl": "on_train_start"},
    "train_end": {"tf": "on_train_end", "pl": "on_train_end"},
    "epoch_start": {"tf": "on_epoch_begin", "pl": "on_train_epoch_start"},
    "epoch_end": {"tf": "on_epoch_end", "pl": "on_train_epoch_end"},
    "batch_start": {"tf": "on_train_batch_begin", "pl": "on_train_batch_start"},
    "batch_end": {"tf": "on_train_batch_end", "pl": "on_train_batch_end"},
}


class CallbackRegistry:
    """Registry for callback handlers or factories."""

    _handlers: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable):
            cls._handlers[name] = func
            return func

        return decorator

    @classmethod
    def get_handler(cls, name: str) -> Callable:
        if name not in cls._handlers:
            raise ValueError(f"Handler '{name}' not found")
        return cls._handlers[name]

    @classmethod
    def list_handlers(cls) -> List[str]:
        return list(cls._handlers.keys())


def make_tf_callback(handlers: Dict[str, List[Callable]]):
    from tensorflow.keras.callbacks import Callback

    methods = {}
    for event, fns in handlers.items():
        if not isinstance(fns, (list, tuple)):
            fns = [fns]
        hook_name = event_map[event]["tf"]

        def _make_hook(fns):
            def _hook(self, *args, **kwargs):
                for fn in fns:
                    fn(self, *args, **kwargs)

            return _hook

        methods[hook_name] = _make_hook(fns)
    return type("UnifiedTFCallback", (Callback,), methods)()


def make_pl_callback(handlers: Dict[str, List[Callable]]):
    import pytorch_lightning as pl

    methods = {}
    for event, fns in handlers.items():
        if not isinstance(fns, (list, tuple)):
            fns = [fns]
        hook_name = event_map[event]["pl"]

        def _make_hook(fns):
            def _hook(self, trainer, pl_module, *args, **kwargs):
                for fn in fns:
                    fn(self, trainer, pl_module, *args, **kwargs)

            return _hook

        methods[hook_name] = _make_hook(fns)
    return type("UnifiedPLCallback", (pl.Callback,), methods)()


def build_callbacks_from_config(
    config: List[Dict[str, Any]],
    framework: str,
    name_key: str = "name",
    params_key: str = "params",
) -> List[Any]:
    """
    Build a list of callbacks (native or unified) from a config list.

    Args:
        config: List of callback config dicts. Each dict should have at least a 'name' key and optionally a 'params' dict.
        framework: Which framework to use ('tf', 'pl', or 'torch').
        name_key: Key in each config dict for the callback/factory name (default: 'name').
        params_key: Key in each config dict for the callback/factory parameters (default: 'params').

    Returns:
        List of instantiated callback objects.

    Each config entry may either:
    1) Define only 'name' + 'params' → handler must return a Callback instance (native/factory style)
    2) Define 'events' (list of {event, name, params}) → use unified wrapper for event-based callbacks
    """
    callbacks = []
    for cb in config:
        if name_key not in cb:
            raise ValueError(f"Callback config missing '{name_key}' key: {cb}")
        name = cb[name_key]
        params = cb.get(params_key, {}) or {}
        handler = CallbackRegistry.get_handler(name)

        # 1) Native callback factory: no events = direct instance
        if not cb.get("events"):
            instance = handler(**params)
            callbacks.append(instance)
            continue

        # 2) Unified hook functions
        handlers: Dict[str, List[Callable]] = {}
        for evt in cb["events"]:
            evt_name = evt["event"]
            evt_handler_name = evt[name_key]
            evt_handler = CallbackRegistry.get_handler(evt_handler_name)
            evt_params = evt.get(params_key, {})
            fn = evt_handler(**evt_params) if evt_params else evt_handler
            handlers.setdefault(evt_name, []).append(fn)

        if framework in ("tf", "tensorflow"):
            callbacks.append(make_tf_callback(handlers))
        elif framework in ("pl", "pytorch_lightning"):
            callbacks.append(make_pl_callback(handlers))
        elif framework in ("torch", "pytorch"):
            callbacks.append(make_torch_callback(handlers))
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    return callbacks


# --- Handlers & Factories, Tensorflow native ---
@CallbackRegistry.register("tf_early_stopping")
def make_early_stopping(monitor: str = "val_loss", patience: int = 3, **kwargs):
    from tensorflow.keras.callbacks import EarlyStopping

    return EarlyStopping(monitor=monitor, patience=patience, **kwargs)


@CallbackRegistry.register("tf_model_checkpoint")
def make_model_checkpoint(
    filepath: str, monitor: str = "val_loss", save_best_only: bool = True, **kwargs
):
    from tensorflow.keras.callbacks import ModelCheckpoint

    return ModelCheckpoint(
        filepath=filepath, monitor=monitor, save_best_only=save_best_only, **kwargs
    )


@CallbackRegistry.register("tf_model_checkpoint")
def make_tf_model_checkpoint(
    filepath: str, monitor: str = "val_loss", save_best_only: bool = True, **kwargs
):
    from tensorflow.keras.callbacks import ModelCheckpoint

    return ModelCheckpoint(
        filepath=filepath, monitor=monitor, save_best_only=save_best_only, **kwargs
    )


@CallbackRegistry.register("tf_eta")
def make_eta_callback():
    import time

    import numpy as np
    import tensorflow as tf

    class ETACallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.times = []

        def on_epoch_begin(self, epoch, logs=None):
            self.start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.time() - self.start
            self.times.append(elapsed)
            avg = np.mean(self.times[-5:])  # smooth over last 5
            remaining = (self.params["epochs"] - epoch - 1) * avg
            if remaining > 3600:
                hrs = remaining // 3600
                mins = (remaining % 3600) // 60
                print(f"Estimated time left: {hrs:.0f}h {mins:.0f}m")
            else:
                print(f"Estimated time left: {remaining:.1f}s")

    return ETACallback()


# --- PyTorch (vanilla) Callback System ---


class PyTorchCallback:
    """Base class for PyTorch callbacks following common callback patterns."""

    def on_train_begin(self, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch, **kwargs):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch, logs=None, **kwargs):
        pass


def make_torch_callback(handlers: Dict[str, List[Callable]]):
    """Create a unified PyTorch callback from event handlers."""

    class UnifiedTorchCallback(PyTorchCallback):
        def __init__(self):
            super().__init__()
            self.handlers = handlers

        def _call_handlers(self, event, *args, **kwargs):
            """Call all handlers for a given event."""
            if event in self.handlers:
                for handler in self.handlers[event]:
                    handler(self, *args, **kwargs)

        def on_train_begin(self, **kwargs):
            self._call_handlers("train_start", **kwargs)

        def on_train_end(self, **kwargs):
            self._call_handlers("train_end", **kwargs)

        def on_epoch_begin(self, epoch, **kwargs):
            self._call_handlers("epoch_start", epoch, **kwargs)

        def on_epoch_end(self, epoch, logs=None, **kwargs):
            self._call_handlers("epoch_end", epoch, logs=logs, **kwargs)

        def on_batch_begin(self, batch, **kwargs):
            self._call_handlers("batch_start", batch, **kwargs)

        def on_batch_end(self, batch, logs=None, **kwargs):
            self._call_handlers("batch_end", batch, logs=logs, **kwargs)

    return UnifiedTorchCallback()


# --- PyTorch Native Callback Implementations ---


@CallbackRegistry.register("torch_eta")
def make_torch_eta_callback(total_epochs=None, smoothing=5, sink=None):
    """Create PyTorch ETA callback to estimate remaining training time."""
    import time

    class TorchETACallback(PyTorchCallback):
        def __init__(self, total_epochs, smoothing, sink):
            super().__init__()
            self.total_epochs = total_epochs
            self.smoothing = max(1, int(smoothing))
            self.sink = sink if callable(sink) else print
            self.times = []
            self.start_time = None

        def on_train_begin(self, epochs=None, **kwargs):
            self.times.clear()
            if epochs is not None:
                self.total_epochs = epochs

        def on_epoch_begin(self, epoch, **kwargs):
            self.start_time = time.time()

        def on_epoch_end(self, epoch, **kwargs):
            if self.start_time is None:
                return
            elapsed = time.time() - self.start_time
            self.times.append(elapsed)

            window = self.times[-self.smoothing :]
            avg_time = sum(window) / len(window)

            if self.total_epochs is None:
                self.sink(f"Avg epoch: {avg_time:.2f}s | done {epoch+1}")
                return

            remaining = (self.total_epochs - (epoch + 1)) * avg_time
            h, rem = divmod(int(remaining + 0.5), 3600)
            m, s = divmod(rem, 60)
            if h:
                self.sink(f"ETA: {h}h {m}m")
            elif m:
                self.sink(f"ETA: {m}m {s}s")
            else:
                self.sink(f"ETA: {s}s")

    return TorchETACallback(total_epochs, smoothing, sink)


@CallbackRegistry.register("torch_early_stopping")
def make_torch_early_stopping(
    monitor: str = "val_loss",
    patience: int = 20,
    min_delta: float = 0.0,
    restore_best: bool = True,
    mode: str = "min",
):
    """Create PyTorch EarlyStopping callback."""
    import copy

    class TorchEarlyStopping(PyTorchCallback):
        def __init__(self):
            super().__init__()
            self.monitor = monitor
            self.patience = patience
            self.min_delta = min_delta
            self.restore_best = restore_best
            self.mode = mode

            self.best = float("inf") if mode == "min" else float("-inf")
            self.wait = 0
            self.best_state = None
            self.should_stop = False

        def _is_better(self, current, best):
            """Check if current metric is better than best."""
            if self.mode == "min":
                return current < best - self.min_delta
            else:
                return current > best + self.min_delta

        def on_epoch_end(self, epoch, logs=None, model=None, **kwargs):
            """Check if we should stop training."""
            if logs is None or self.monitor not in logs:
                return

            current = logs[self.monitor]

            if self._is_better(current, self.best):
                self.best = current
                self.wait = 0
                if self.restore_best and model is not None:
                    try:
                        import torch

                        self.best_state = copy.deepcopy(model.state_dict())
                    except ImportError:
                        pass
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.should_stop = True
                    print(f"Early stopping triggered after {epoch + 1} epochs")

        def on_train_end(self, model=None, **kwargs):
            """Restore best weights if requested."""
            if self.restore_best and self.best_state is not None and model is not None:
                try:
                    model.load_state_dict(self.best_state)
                    print(f"Restored best weights with {self.monitor}={self.best:.4f}")
                except Exception as e:
                    print(f"Warning: Could not restore best weights: {e}")

    return TorchEarlyStopping()


@CallbackRegistry.register("torch_model_checkpoint")
def make_torch_model_checkpoint(
    filepath: str,
    monitor: str = "val_loss",
    save_best_only: bool = True,
    mode: str = "min",
    save_weights_only: bool = False,
):
    """Create PyTorch ModelCheckpoint callback."""
    import os

    class TorchModelCheckpoint(PyTorchCallback):
        def __init__(self):
            super().__init__()
            self.filepath = filepath
            self.monitor = monitor
            self.save_best_only = save_best_only
            self.mode = mode
            self.save_weights_only = save_weights_only

            self.best = float("inf") if mode == "min" else float("-inf")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        def _is_better(self, current, best):
            """Check if current metric is better than best."""
            if self.mode == "min":
                return current < best
            else:
                return current > best

        def on_epoch_end(self, epoch, logs=None, model=None, **kwargs):
            """Save model checkpoint if conditions are met."""
            if model is None:
                return

            should_save = True

            if self.save_best_only and logs is not None and self.monitor in logs:
                current = logs[self.monitor]
                if self._is_better(current, self.best):
                    self.best = current
                    should_save = True
                else:
                    should_save = False

            if should_save:
                try:
                    import torch

                    # Format filepath with epoch number
                    formatted_path = self.filepath.format(epoch=epoch)

                    if self.save_weights_only:
                        torch.save(model.state_dict(), formatted_path)
                    else:
                        torch.save(model, formatted_path)

                    print(f"Saved model checkpoint to {formatted_path}")

                except ImportError:
                    print("Warning: PyTorch not available for saving checkpoint")
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")

    return TorchModelCheckpoint()


@CallbackRegistry.register("torch_lr_scheduler")
def make_torch_lr_scheduler(scheduler_class: str = "StepLR", **scheduler_kwargs):
    """Create PyTorch learning rate scheduler callback."""

    class TorchLRScheduler(PyTorchCallback):
        def __init__(self):
            super().__init__()
            self.scheduler_class = scheduler_class
            self.scheduler_kwargs = scheduler_kwargs
            self.scheduler = None

        def on_train_begin(self, optimizer=None, **kwargs):
            """Initialize scheduler with optimizer."""
            if optimizer is None:
                print("Warning: No optimizer provided to LR scheduler")
                return

            try:
                import torch.optim.lr_scheduler as lr_scheduler

                scheduler_cls = getattr(lr_scheduler, self.scheduler_class)
                self.scheduler = scheduler_cls(optimizer, **self.scheduler_kwargs)
                print(f"Initialized {self.scheduler_class} scheduler")

            except ImportError:
                print("Warning: PyTorch not available for LR scheduling")
            except AttributeError:
                print(f"Warning: Unknown scheduler class: {self.scheduler_class}")
            except Exception as e:
                print(f"Warning: Could not initialize scheduler: {e}")

        def on_epoch_end(self, epoch, logs=None, **kwargs):
            """Step the learning rate scheduler."""
            if self.scheduler is not None:
                try:
                    # Some schedulers need validation loss
                    if hasattr(self.scheduler, "step") and logs is not None:
                        if "val_loss" in logs and hasattr(
                            self.scheduler, "_step_count"
                        ):
                            # ReduceLROnPlateau needs metric
                            if "ReduceLR" in self.scheduler_class:
                                self.scheduler.step(logs["val_loss"])
                            else:
                                self.scheduler.step()
                        else:
                            self.scheduler.step()

                    # Log current learning rate
                    if hasattr(self.scheduler, "get_last_lr"):
                        current_lr = self.scheduler.get_last_lr()[0]
                        print(f"Learning rate: {current_lr:.6f}")

                except Exception as e:
                    print(f"Warning: Error stepping scheduler: {e}")

    return TorchLRScheduler()


@CallbackRegistry.register("torch_progress_bar")
def make_torch_progress_bar(desc: str = "Training"):
    """Create a simple progress bar callback for PyTorch."""

    class TorchProgressBar(PyTorchCallback):
        def __init__(self):
            super().__init__()
            self.desc = desc
            self.total_epochs = None

        def on_train_begin(self, epochs=None, **kwargs):
            """Initialize progress tracking."""
            self.total_epochs = epochs
            print(f"Starting {self.desc}")

        def on_epoch_end(self, epoch, logs=None, **kwargs):
            """Update progress after each epoch."""
            progress = f"Epoch {epoch + 1}"
            if self.total_epochs:
                progress += f"/{self.total_epochs}"

            if logs:
                metrics = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                progress += f" - {metrics}"

            print(progress)

        def on_train_end(self, **kwargs):
            """Finish progress tracking."""
            print(f"Completed {self.desc}")

    return TorchProgressBar()


@CallbackRegistry.register("torch_batch_progress_bar")
def make_torch_batch_progress_bar(
    desc: str = "Training",
    update_freq: int = 1,
    show_metrics: bool = True,
    bar_width: int = 30,
    only_keys=None,  # e.g. ["train_loss", "val_loss"]
    hide_keys=None,  # e.g. ["val_accuracy"]
):
    """Create a batch-level progress bar callback for PyTorch with detailed progress tracking.

    Use:
        make_torch_batch_progress_bar(only_keys=["train_loss", "val_loss"])
        # or
        make_torch_batch_progress_bar(hide_keys=["beam_param_metric"])
    """
    import time

    class TorchBatchProgressBar(PyTorchCallback):
        def __init__(self):
            super().__init__()
            self.desc = desc
            self.update_freq = max(1, update_freq)
            self.show_metrics = show_metrics
            self.bar_width = max(10, bar_width)

            # filtering
            self.only_keys = set(only_keys) if only_keys else None
            self.hide_keys = set(hide_keys or [])

            # Training state
            self.total_epochs = None
            self.current_epoch = 0
            self.total_batches = None
            self.current_batch = 0
            self.epoch_start_time = None
            self.batch_times = []

        # ------------------------ helpers ------------------------
        def _format_metrics(self, logs):
            if not logs or not self.show_metrics:
                return ""
            items = list(logs.items())
            if self.only_keys is not None:
                items = [(k, v) for k, v in items if k in self.only_keys]
            if self.hide_keys:
                items = [(k, v) for k, v in items if k not in self.hide_keys]
            if not items:
                return ""

            def _fmt_val(v):
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return str(v)

            return " - " + " - ".join(f"{k}: {_fmt_val(v)}" for k, v in items)

        # ------------------------ lifecycle ------------------------
        def on_train_begin(self, epochs=None, **kwargs):
            """Initialize progress tracking."""
            self.total_epochs = epochs
            print(f"Starting {self.desc}")
            if self.total_epochs:
                print(f"Total epochs: {self.total_epochs}")

        def on_epoch_begin(self, epoch, total_batches=None, **kwargs):
            """Start epoch progress tracking."""
            self.current_epoch = epoch
            self.total_batches = total_batches
            self.current_batch = 0
            self.epoch_start_time = time.time()
            self.batch_times.clear()

            epoch_info = f"Epoch {epoch + 1}"
            if self.total_epochs:
                epoch_info += f"/{self.total_epochs}"
            if self.total_batches:
                epoch_info += f" - {self.total_batches} batches"
            print(f"\n{epoch_info}")

        def on_batch_begin(self, batch=None, batch_idx=None, **kwargs):
            """Track batch start (supports either arg name)."""
            b = batch if batch is not None else batch_idx
            if b is not None:
                self.current_batch = b

        def on_batch_end(self, batch=None, batch_idx=None, logs=None, **kwargs):
            """Update progress bar after each batch."""
            b = batch if batch is not None else batch_idx
            if b is None:
                b = self.current_batch
            self.current_batch = b + 1

            # Update every N batches or at the end
            should_update = ((b + 1) % self.update_freq == 0) or (
                self.total_batches and (b + 1) == self.total_batches
            )
            if should_update:
                self._update_progress_bar(logs)

        def on_epoch_end(self, epoch, logs=None, **kwargs):
            """Finalize epoch progress."""
            # Ensure final update
            self._update_progress_bar(logs, force_complete=True)

            # Show epoch summary
            epoch_time = (
                time.time() - self.epoch_start_time if self.epoch_start_time else 0
            )
            summary = f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s"
            summary += self._format_metrics(logs)
            print(summary)

        def on_train_end(self, **kwargs):
            """Finish progress tracking."""
            print(f"\n{self.desc} completed!")

        # ------------------------ rendering ------------------------
        def _update_progress_bar(self, logs=None, force_complete=False):
            """Update the progress bar display."""
            if self.total_batches is None:
                # Simple counter if total unknown
                progress = f"\rBatch {self.current_batch}"
                progress += self._format_metrics(logs)
                print(progress, end="", flush=True)
                return

            # Calculate progress
            if force_complete:
                progress_ratio = 1.0
                current = self.total_batches
            else:
                progress_ratio = min(self.current_batch / self.total_batches, 1.0)
                current = self.current_batch

            # Create progress bar (TensorFlow style)
            filled_width = int(self.bar_width * progress_ratio)
            if progress_ratio < 1.0 and filled_width > 0:
                bar = (
                    "=" * (filled_width - 1)
                    + ">"
                    + "." * (self.bar_width - filled_width)
                )
            elif progress_ratio >= 1.0:
                bar = "=" * self.bar_width
            else:
                bar = "." * self.bar_width

            # Percentage + ETA
            percentage = progress_ratio * 100
            if self.epoch_start_time and self.current_batch > 0:
                elapsed = time.time() - self.epoch_start_time
                if progress_ratio > 0:
                    total_estimated = elapsed / progress_ratio
                    remaining = max(0, total_estimated - elapsed)
                    eta = (
                        f"{remaining/60:.1f}m"
                        if remaining > 60
                        else f"{remaining:.0f}s"
                    )
                else:
                    eta = "?"
            else:
                eta = "?"

            # Build line
            progress_str = f"\r[{bar}] {current}/{self.total_batches} ({percentage:5.1f}%) - ETA: {eta}"
            progress_str += self._format_metrics(logs)

            # Print with spacing to clear previous line
            print(f"{progress_str:<120}", end="", flush=True)

    return TorchBatchProgressBar()

@CallbackRegistry.register("torch_total_training_time")
def make_torch_training_time_only(save_dir: str):
    import os, json, time

    class TorchTrainingTimeOnly(PyTorchCallback):
        def __init__(self):
            super().__init__()
            self.save_dir = save_dir
            self._t0 = None

        def on_train_begin(self, **kwargs):
            self._t0 = time.time()

        def on_train_end(self, **kwargs):
            elapsed = 0.0 if self._t0 is None else (time.time() - self._t0)
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, "training_meta.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"framework": "torch", "total_wall_time_sec": round(elapsed, 4)}, f, indent=2)
            print(f"[torch_training_time_only] Saved {path}")

    return TorchTrainingTimeOnly()



@CallbackRegistry.register("torch_training_meta_info")
def make_torch_training_meta_info(
    save_dir: str,                           # core arg required
    example_input: Any = None,               # e.g. a torch.Tensor or tuple of tensors
    input_shape: Optional[Tuple[int, ...]] = None,  # used to synthesize a dummy batch, core arg
    method: str = "auto",                    # "auto" | "profiler" | "thop"
    backward_factor: float = 2.0,            # used when fallback to thop forward-only estimate
    grad_accum_steps: int = 1,
    model_name: Optional[str] = None,
):
    import os, json, time, datetime, traceback
    """
    PyTorch callback that records total wall-time and estimates training FLOPs,
    then writes JSON to <save_dir>/training_meta.json.

    Notes:
      - If neither example_input nor input_shape is provided (and batch tensors
        are not passed in kwargs), FLOPs may be null but timing/metadata still saved.
      - FLOPs estimation is performed once (lazily) using a single train step,
        then scaled by total steps.
    """

    class TorchTrainingMetaInfo(PyTorchCallback):
        def __init__(self):
            super().__init__()
            self.save_dir = save_dir
            self.example_input = example_input
            self.input_shape = input_shape
            self.method = method
            self.backward_factor = float(backward_factor)
            self.grad_accum_steps = max(1, int(grad_accum_steps))
            self.model_name = model_name

            # time bookkeeping
            self._train_start_ts = None
            self._train_end_ts = None
            self._wall_time_sec = 0.0

            # progress bookkeeping (best-effort; will be inferred if not provided)
            self.epochs_declared = None
            self.epochs_seen = 0
            self.total_batches_declared = None
            self.batches_seen_this_epoch = 0
            self.total_batches_seen = 0
            self.batch_size_seen = None

            # flops bookkeeping
            self.per_step_flops = None
            self.per_forward_flops = None
            self.per_epoch_flops = None
            self.total_training_flops = None
            self.flops_method = None
            self.flops_notes = None

            # model info snapshot
            self.param_count = None
            self.device = None
            self.world_size = None
            self.local_rank = None

            # lazily profiled?
            self._did_profile_once = False

        # ------------------- helpers -------------------
        def _atomic_write_json(self, path: str, payload: Dict):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, path)

        def _now_iso(self):
            return datetime.datetime.now().isoformat(timespec="seconds")

        def _count_params(self, model):
            try:
                import torch
                return int(sum(p.numel() for p in model.parameters()))
            except Exception:
                return None

        def _infer_device(self, model):
            try:
                import torch
                p = next(model.parameters(), None)
                if p is not None:
                    return str(p.device)
            except Exception:
                pass
            return None

        def _dist_info(self):
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    return dist.get_world_size(), dist.get_rank()
            except Exception:
                pass
            return None, None

        def _make_dummy_from_shape(self, shape, device, dtype=None):
            try:
                import torch
                if dtype is None:
                    dtype = torch.float32
                return torch.randn(*shape, device=device, dtype=dtype)
            except Exception:
                return None

        def _as_tuple(self, x):
            return x if isinstance(x, (tuple, list)) else (x,)

        def _extract_batch_from_kwargs(self, **kwargs):
            # Best-effort: look for a tensor or tuple of tensors in common keys
            candidates = []
            for key in ["inputs", "batch", "data", "x", "batch_data"]:
                if key in kwargs:
                    candidates.append(kwargs[key])
            for c in candidates:
                # accept tensor, or (tensor, y) pairs, or dict with 'inputs'
                try:
                    import torch
                    if isinstance(c, torch.Tensor):
                        return (c,)
                    if isinstance(c, (tuple, list)) and len(c) > 0:
                        return self._as_tuple(c[0])  # assume first is input
                    if isinstance(c, dict) and "inputs" in c:
                        return self._as_tuple(c["inputs"])
                except Exception:
                    pass
            return None

        def _ensure_example_input(self, model, **kwargs):
            if self.example_input is not None:
                return self._as_tuple(self.example_input)
            # try from kwargs
            kw_batch = self._extract_batch_from_kwargs(**kwargs)
            if kw_batch is not None:
                self._maybe_set_batch_size(kw_batch)
                return self._as_tuple(kw_batch)
            # try from input_shape
            if self.input_shape is not None:
                dev = self.device or self._infer_device(model) or "cpu"
                dummy = self._make_dummy_from_shape(self.input_shape, dev)
                return (dummy,) if dummy is not None else None
            return None

        def _maybe_set_batch_size(self, inputs_tuple):
            try:
                import torch
                x0 = inputs_tuple[0]
                if isinstance(x0, torch.Tensor) and x0.dim() >= 1:
                    b = int(x0.shape[0])
                    if b > 0:
                        self.batch_size_seen = b
            except Exception:
                pass

        def _profile_one_train_step(self, model, optimizer=None, **kwargs):
            """
            Try to measure FLOPs of a full train step (fwd+backward+step) once.
            Sets per_step_flops and per_forward_flops on success.
            """
            if self._did_profile_once:
                return
            inputs = self._ensure_example_input(model, **kwargs)
            if inputs is None:
                self.flops_notes = "No example input available; FLOPs not estimated."
                return

            # shallow copy model to avoid altering caller's state
            m = model
            per_step = None
            fwd = None
            notes = []
            used_method = None

            # Prefer profiler if available
            if self.method in ("auto", "profiler"):
                try:
                    import torch
                    import torch.profiler as prof

                    activities = [prof.ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        activities.append(prof.ProfilerActivity.CUDA)

                    m_train = m.train()
                    for p in m_train.parameters():
                        p.requires_grad_(True)

                    with torch.enable_grad():
                        with prof.profile(
                            activities=activities,
                            record_shapes=True,
                            with_flops=True,
                            profile_memory=False,
                        ) as p:
                            out = m_train(*inputs)
                            # generic scalar loss; if not scalar, sum it
                            if isinstance(out, (tuple, list)):
                                out = out[0]
                            loss = out.sum() if hasattr(out, "sum") else (out if out.ndim == 0 else None)
                            if loss is None:
                                raise RuntimeError("Cannot reduce model output to scalar for backward()")
                            loss.backward()
                            if optimizer is not None:
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)

                    ka = p.key_averages()
                    # sum flops across ops
                    total_flops = 0
                    for evt in ka:
                        # PyTorch profiler exposes flops via .flops (may be None)
                        val = getattr(evt, "flops", None)
                        if val is not None:
                            total_flops += int(val)
                    if total_flops > 0:
                        used_method = "torch.profiler"
                        per_step = int(total_flops)
                        # We also try a forward-only pass to estimate per-forward if desired
                        # Fallback: assume forward ≈ per_step / (1 + backward_factor)
                        fwd = int(round(per_step / max(1.0, 1.0 + self.backward_factor)))
                    else:
                        notes.append("Profiler returned zero flops; falling back.")
                except Exception as e:
                    notes.append(f"Profiler failed: {e}")

            # Fallback: THOP forward-only
            if per_step is None and self.method in ("auto", "thop"):
                try:
                    import torch
                    from thop import profile as thop_profile

                    m_eval = m.eval()
                    with torch.no_grad():
                        macs, _params = thop_profile(m_eval, inputs=inputs, verbose=False)
                    fwd = int(macs * 2)  # MACs→FLOPs
                    used_method = "thop (forward-only)"
                    per_step = int(round(fwd * (1.0 + self.backward_factor)))
                except Exception as e:
                    notes.append(f"THOP failed: {e}")

            if per_step is None:
                self.flops_notes = " | ".join(notes) if notes else "FLOPs estimation unavailable."
                return

            self.per_step_flops = per_step
            self.per_forward_flops = fwd
            self.flops_method = used_method
            self.flops_notes = "Estimated from a single train step."

            self._did_profile_once = True

        def _finalize_flops_totals(self):
            if self.per_step_flops is None:
                return
            steps = max(1, self.total_batches_seen // self.grad_accum_steps)
            self.total_training_flops = int(self.per_step_flops * steps)
            # Approximate per-epoch if we know per-epoch batches
            if self.total_batches_declared:
                epoch_steps = max(1, self.total_batches_declared // self.grad_accum_steps)
                self.per_epoch_flops = int(self.per_step_flops * epoch_steps)

        def _build_payload(self, torch_module=None):
            versions = {}
            try:
                import torch
                versions = {
                    "torch": getattr(torch, "__version__", None),
                    "cuda": getattr(torch.version, "cuda", None),
                    "cudnn": getattr(getattr(torch.backends, "cudnn", None), "version", lambda: None)(),
                }
            except Exception:
                pass

            payload = {
                "framework": "torch",
                "timestamp": self._now_iso(),
                "model_name": self.model_name,
                "device": self.device,
                "ddp": {"world_size": self.world_size, "local_rank": self.local_rank},
                "params": self.param_count,
                "epochs": self.epochs_declared or self.epochs_seen or None,
                "epochs_seen": self.epochs_seen,
                "batches_per_epoch": self.total_batches_declared,
                "total_batches_seen": self.total_batches_seen,
                "batch_size": self.batch_size_seen,
                "grad_accum_steps": self.grad_accum_steps,
                "total_wall_time_sec": round(self._wall_time_sec, 4),
                "flops": {
                    "method": self.flops_method,
                    "per_forward_pass": self.per_forward_flops,
                    "per_train_step": self.per_step_flops,
                    "per_epoch": self.per_epoch_flops,
                    "total_training": self.total_training_flops,
                    "notes": self.flops_notes,
                },
                "versions": versions,
            }
            return payload

        # ------------------- lifecycle -------------------
        def on_train_begin(self, model=None, epochs=None, total_batches=None, **kwargs):
            self._train_start_ts = time.time()
            if epochs is not None:
                self.epochs_declared = int(epochs)
            if total_batches is not None:
                self.total_batches_declared = int(total_batches)
            if model is not None:
                self.param_count = self._count_params(model)
                self.device = self._infer_device(model)
                self.world_size, self.local_rank = self._dist_info()

        def on_epoch_begin(self, epoch, total_batches=None, **kwargs):
            if total_batches is not None:
                self.total_batches_declared = int(total_batches)
            self.batches_seen_this_epoch = 0

        def on_batch_begin(self, batch, **kwargs):
            # Capture batch size if tensors are provided
            inputs = self._extract_batch_from_kwargs(**kwargs)
            if inputs is not None:
                self._maybe_set_batch_size(inputs)

        def on_batch_end(self, batch, model=None, optimizer=None, **kwargs):
            self.batches_seen_this_epoch += 1
            self.total_batches_seen += 1
            # lazily profile on the very first step if possible
            if not self._did_profile_once and model is not None:
                try:
                    self._profile_one_train_step(model, optimizer=optimizer, **kwargs)
                except Exception:
                    # Keep going; timing will still be recorded
                    pass

        def on_epoch_end(self, epoch, **kwargs):
            self.epochs_seen = max(self.epochs_seen, epoch + 1)

        def on_train_end(self, model=None, **kwargs):
            self._train_end_ts = time.time()
            self._wall_time_sec = float(self._train_end_ts - (self._train_start_ts or self._train_end_ts))
            # finalize FLOPs totals
            self._finalize_flops_totals()

            # write JSON
            target = os.path.join(self.save_dir, "training_meta.json")
            try:
                payload = self._build_payload()
                self._atomic_write_json(target, payload)
                print(f"[torch_training_meta_info] Saved {target}")
            except Exception as e:
                print(f"[torch_training_meta_info] Failed to write meta: {e}")
                try:
                    # last resort: dump traceback to a sidecar
                    side = target + ".error.txt"
                    with open(side, "w", encoding="utf-8") as f:
                        f.write("Exception while saving training_meta.json:\n")
                        traceback.print_exc(file=f)
                    print(f"[torch_training_meta_info] Error details saved to {side}")
                except Exception:
                    pass

    return TorchTrainingMetaInfo()


@CallbackRegistry.register("torch_grad_monitor")
def make_torch_grad_monitor(
    save_dir: str,
    track_per_param: bool = False,   # False = per-layer summary only (recommended)
    log_every_n_batches: int = 1,    # collect every n batches (increase to reduce overhead)
    clip_large_values_at: float = 0, # 0 = no clipping; else clip abs grads for robust stats
):
    """
    Gradient-flow monitor (safe & robust).
    - Registers .register_hook() on each requires_grad parameter.
    - For each batch, gathers:
        * global_grad_norm (L2)
        * any_nan / any_inf flags
        * per-layer {count, l2, l1, max_abs, mean_abs, std, zero_frac, finite_frac}
      (per-parameter detail can be enabled but increases JSON size)
    - Saves one JSON per epoch: grads_epoch_{epoch+1}.json
    """
    import torch
    from collections import defaultdict
    import os, json, math, traceback
    class TorchGradMonitor(PyTorchCallback):
        def __init__(self):
            super().__init__()
            self.save_dir = save_dir
            self.track_per_param = bool(track_per_param)
            self.n = max(1, int(log_every_n_batches))
            self.clip = float(clip_large_values_at)
            self._hooks = []
            self._epoch = -1
            self._batch = -1

            # per-batch scratch (filled by hooks during backward)
            self._batch_global_sumsq = 0.0
            self._batch_any_nan = False
            self._batch_any_inf = False

            # per-batch per-layer accumulators
            self._batch_layer = defaultdict(lambda: {
                "count": 0, "l2": 0.0, "l1": 0.0, "max_abs": 0.0,
                "mean_abs_sum": 0.0, "std_sum": 0.0,
                "zeros": 0, "numel": 0, "finite": 0
            })
            # per-epoch aggregated stats
            self._epoch_batches = 0
            self._epoch_global_norms = []  # sample per logged batch
            self._epoch_any_nan = 0
            self._epoch_any_inf = 0
            self._epoch_layer = defaultdict(lambda: {
                "count": 0, "l2": 0.0, "l1": 0.0, "max_abs": 0.0,
                "mean_abs_sum": 0.0, "std_sum": 0.0,
                "zeros": 0, "numel": 0, "finite": 0
            })

        # ----------------- helpers -----------------
        def _atomic_write_json(self, path: str, payload: Dict[str, Any]):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)

        def _safe_stats(self, g: torch.Tensor):
            # Work on a view; no grad change
            t = g.detach()
            if self.clip > 0:
                t = t.clamp(min=-self.clip, max=self.clip)

            isfinite = torch.isfinite(t)
            finite = t[isfinite]
            numel = t.numel()
            zeros = (t == 0).sum().item()

            if finite.numel() == 0:
                return {
                    "l2": 0.0, "l1": 0.0, "max_abs": 0.0,
                    "mean_abs": 0.0, "std": 0.0,
                    "zeros": zeros, "numel": numel, "finite": 0,
                    "any_nan": True, "any_inf": True
                }

            absf = finite.abs()
            l2 = float(torch.linalg.vector_norm(finite, ord=2).item())
            l1 = float(absf.sum().item())
            max_abs = float(absf.max().item())
            mean_abs = float(absf.mean().item())
            std = float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0
            any_nan = bool(torch.isnan(t).any().item())
            any_inf = bool(torch.isinf(t).any().item())

            return {
                "l2": l2, "l1": l1, "max_abs": max_abs,
                "mean_abs": mean_abs, "std": std,
                "zeros": int(zeros), "numel": int(numel),
                "finite": int(finite.numel()),
                "any_nan": any_nan, "any_inf": any_inf,
            }

        def _attach_hooks(self, model):
            # Remove old hooks if any
            for h in self._hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            self._hooks.clear()

            # Name parameters for per-layer aggregation
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue

                def make_hook(pname):
                    def _hook(grad):
                        s = self._safe_stats(grad)
                        # global
                        self._batch_global_sumsq += (s["l2"] ** 2)
                        self._batch_any_nan |= s["any_nan"]
                        self._batch_any_inf |= s["any_inf"]
                        # per-layer (group by module/param prefix)
                        layer = pname.rsplit(".", 1)[0] if "." in pname else pname
                        L = self._batch_layer[layer]
                        L["count"] += 1
                        L["l2"] += s["l2"]
                        L["l1"] += s["l1"]
                        L["max_abs"] = max(L["max_abs"], s["max_abs"])
                        L["mean_abs_sum"] += s["mean_abs"]
                        L["std_sum"] += s["std"]
                        L["zeros"] += s["zeros"]
                        L["numel"] += s["numel"]
                        L["finite"] += s["finite"]

                        if self.track_per_param:
                            # store minimal per-param details lazily
                            PP = self._batch_layer.setdefault(f"{layer}::{pname}", {
                                "count": 0, "l2": 0.0, "l1": 0.0, "max_abs": 0.0,
                                "mean_abs_sum": 0.0, "std_sum": 0.0,
                                "zeros": 0, "numel": 0, "finite": 0
                            })
                            PP["count"] += 1
                            PP["l2"] += s["l2"]
                            PP["l1"] += s["l1"]
                            PP["max_abs"] = max(PP["max_abs"], s["max_abs"])
                            PP["mean_abs_sum"] += s["mean_abs"]
                            PP["std_sum"] += s["std"]
                            PP["zeros"] += s["zeros"]
                            PP["numel"] += s["numel"]
                            PP["finite"] += s["finite"]
                    return _hook

                self._hooks.append(p.register_hook(make_hook(name)))

        def _reset_batch_accum(self):
            self._batch_global_sumsq = 0.0
            self._batch_any_nan = False
            self._batch_any_inf = False
            self._batch_layer.clear()

        def _merge_batch_into_epoch(self):
            if self._batch_global_sumsq > 0.0:
                gnorm = math.sqrt(self._batch_global_sumsq)
                self._epoch_global_norms.append(gnorm)
            if self._batch_any_nan:
                self._epoch_any_nan += 1
            if self._batch_any_inf:
                self._epoch_any_inf += 1

            for k, v in self._batch_layer.items():
                E = self._epoch_layer[k]
                E["count"] += v["count"]
                E["l2"] += v["l2"]
                E["l1"] += v["l1"]
                E["max_abs"] = max(E["max_abs"], v["max_abs"])
                E["mean_abs_sum"] += v["mean_abs_sum"]
                E["std_sum"] += v["std_sum"]
                E["zeros"] += v["zeros"]
                E["numel"] += v["numel"]
                E["finite"] += v["finite"]

        def _epoch_payload(self, epoch_idx: int):
            # Build per-layer means
            layer_stats = {}
            for k, v in self._epoch_layer.items():
                c = max(1, v["count"])
                layer_stats[k] = {
                    "count": v["count"],
                    "l2_sum": round(v["l2"], 6),
                    "l1_sum": round(v["l1"], 6),
                    "max_abs": round(v["max_abs"], 6),
                    "mean_abs_mean": round(v["mean_abs_sum"] / c, 6),
                    "std_mean": round(v["std_sum"] / c, 6),
                    "zero_frac": round((v["zeros"] / v["numel"]) if v["numel"] else 0.0, 6),
                    "finite_frac": round((v["finite"] / v["numel"]) if v["numel"] else 0.0, 6),
                }

            return {
                "epoch": epoch_idx + 1,
                "batches_logged": self._epoch_batches,
                "global_grad_norm": {
                    "num_samples": len(self._epoch_global_norms),
                    "mean": round(float(sum(self._epoch_global_norms) / max(1, len(self._epoch_global_norms))), 6),
                    "max": round(float(max(self._epoch_global_norms)) if self._epoch_global_norms else 0.0, 6),
                    "min": round(float(min(self._epoch_global_norms)) if self._epoch_global_norms else 0.0, 6),
                },
                "any_nan_batches": self._epoch_any_nan,
                "any_inf_batches": self._epoch_any_inf,
                "layers": layer_stats,
            }

        # --------------- lifecycle ----------------
        def on_train_begin(self, model=None, **kwargs):
            if model is not None:
                self._attach_hooks(model)

        def on_epoch_begin(self, epoch, **kwargs):
            self._epoch = epoch
            self._batch = -1
            self._epoch_batches = 0
            self._epoch_any_nan = 0
            self._epoch_any_inf = 0
            self._epoch_global_norms.clear()
            self._epoch_layer.clear()

        def on_batch_begin(self, batch, **kwargs):
            self._batch = batch
            self._reset_batch_accum()

        def on_batch_end(self, batch, **kwargs):
            # Only log every n batches to reduce overhead/size
            if ((batch + 1) % self.n) == 0:
                self._merge_batch_into_epoch()
                self._epoch_batches += 1
                self._reset_batch_accum()  # reset for next accumulation window

        def on_epoch_end(self, epoch, **kwargs):
            # Flush any remainder accumulation
            self._merge_batch_into_epoch()
            payload = self._epoch_payload(epoch_idx=epoch)
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                path = os.path.join(self.save_dir, f"grads_epoch_{epoch+1:04d}.json")
                self._atomic_write_json(path, payload)
                print(f"[torch_grad_monitor] Saved {path}")
            except Exception as e:
                print(f"[torch_grad_monitor] Failed to save epoch grads: {e}")
                try:
                    with open(os.path.join(self.save_dir, f"grads_epoch_{epoch+1:04d}.error.txt"), "w") as f:
                        traceback.print_exc(file=f)
                except Exception:
                    pass

        def on_train_end(self, **kwargs):
            # Remove hooks
            for h in self._hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            self._hooks.clear()

    return TorchGradMonitor()


# Update the event map to include PyTorch (vanilla) support
event_map.update(
    {
        "train_start": {**event_map["train_start"], "torch": "on_train_begin"},
        "train_end": {**event_map["train_end"], "torch": "on_train_end"},
        "epoch_start": {**event_map["epoch_start"], "torch": "on_epoch_begin"},
        "epoch_end": {**event_map["epoch_end"], "torch": "on_epoch_end"},
        "batch_start": {**event_map["batch_start"], "torch": "on_batch_begin"},
        "batch_end": {**event_map["batch_end"], "torch": "on_batch_end"},
    }
)
