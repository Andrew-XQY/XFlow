from typing import Any, Callable, Dict, List

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
):
    """Create a batch-level progress bar callback for PyTorch with detailed progress tracking."""
    import sys
    import time

    class TorchBatchProgressBar(PyTorchCallback):
        def __init__(self):
            super().__init__()
            self.desc = desc
            self.update_freq = max(1, update_freq)
            self.show_metrics = show_metrics
            self.bar_width = max(10, bar_width)

            # Training state
            self.total_epochs = None
            self.current_epoch = 0
            self.total_batches = None
            self.current_batch = 0
            self.epoch_start_time = None
            self.batch_times = []

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

        def on_batch_begin(self, batch, **kwargs):
            """Track batch start."""
            self.current_batch = batch

        def on_batch_end(self, batch, logs=None, **kwargs):
            """Update progress bar after each batch."""
            self.current_batch = batch + 1

            # Update every N batches or at the end
            should_update = (batch + 1) % self.update_freq == 0 or (
                self.total_batches and batch + 1 == self.total_batches
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

            if logs and self.show_metrics:
                metrics = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                summary += f" - {metrics}"

            print(summary)

        def on_train_end(self, **kwargs):
            """Finish progress tracking."""
            print(f"\n{self.desc} completed!")

        def _update_progress_bar(self, logs=None, force_complete=False):
            """Update the progress bar display."""
            if self.total_batches is None:
                # Simple counter if total unknown
                progress = f"\rBatch {self.current_batch}"
                if logs and self.show_metrics:
                    metrics = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                    progress += f" - {metrics}"
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
                # Show progress with = and > like TensorFlow
                bar = (
                    "=" * (filled_width - 1)
                    + ">"
                    + "." * (self.bar_width - filled_width)
                )
            elif progress_ratio >= 1.0:
                # Complete bar
                bar = "=" * self.bar_width
            else:
                # Empty bar
                bar = "." * self.bar_width

            # Calculate percentage
            percentage = progress_ratio * 100

            # Estimate time remaining
            if self.epoch_start_time and self.current_batch > 0:
                elapsed = time.time() - self.epoch_start_time
                if progress_ratio > 0:
                    total_estimated = elapsed / progress_ratio
                    remaining = max(0, total_estimated - elapsed)

                    if remaining > 60:
                        eta = f"{remaining/60:.1f}m"
                    else:
                        eta = f"{remaining:.0f}s"
                else:
                    eta = "?"
            else:
                eta = "?"

            # Build progress string
            progress_str = f"\r[{bar}] {current}/{self.total_batches} "
            progress_str += f"({percentage:5.1f}%) - ETA: {eta}"

            # Add metrics if available
            if logs and self.show_metrics:
                metrics = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                progress_str += f" - {metrics}"

            # Print with proper spacing to clear previous line
            print(f"{progress_str:<100}", end="", flush=True)

    return TorchBatchProgressBar()


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
