import yaml
from typing import Dict, Callable, Any, List

# Map unified event names to framework-specific hook method names
event_map = {
    "train_start":  {"tf": "on_train_begin",         "pl": "on_train_start"},
    "train_end":    {"tf": "on_train_end",           "pl": "on_train_end"},
    "epoch_start":  {"tf": "on_epoch_begin",         "pl": "on_train_epoch_start"},
    "epoch_end":    {"tf": "on_epoch_end",           "pl": "on_train_epoch_end"},
    "batch_start":  {"tf": "on_train_batch_begin",   "pl": "on_train_batch_start"},
    "batch_end":    {"tf": "on_train_batch_end",     "pl": "on_train_batch_end"},
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
    name_key: str = "handler",
    params_key: str = "params"
) -> List[Any]:
    """
    Build a list of callbacks (native or unified) from a config list.

    Each config entry may either:
    1) Define 'events' + 'handler' (or parameterized factory) → use unified wrapper
    2) Define only 'handler' + 'params' → handler must return a Callback instance
    """
    callbacks = []
    for cb in config:
        if name_key not in cb:
            raise ValueError(f"Callback config missing '{name_key}' key: {cb}")
        name = cb[name_key]
        params = cb.get(params_key, {}) or {}
        handler = CallbackRegistry.get_handler(name)

        # 1) Native callback factory: no events = direct instance
        if not cb.get('events'):
            instance = handler(**params)
            callbacks.append(instance)
            continue

        # 2) Unified hook functions
        handlers: Dict[str, List[Callable]] = {}
        for evt in cb['events']:
            evt_handler = CallbackRegistry.get_handler(evt['handler'])
            evt_params = evt.get('params', {})
            fn = evt_handler(**evt_params) if evt_params else evt_handler
            handlers.setdefault(evt['event'], []).append(fn)

        if framework in ('tf', 'tensorflow'):
            callbacks.append(make_tf_callback(handlers))
        elif framework in ('pl', 'pytorch_lightning'):
            callbacks.append(make_pl_callback(handlers))
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    return callbacks

# --- Handlers & Factories ---
@CallbackRegistry.register("tf_early_stopping")
def make_early_stopping(monitor: str = 'val_loss', patience: int = 3):
    from tensorflow.keras.callbacks import EarlyStopping
    return EarlyStopping(monitor=monitor, patience=patience)

@CallbackRegistry.register("tf_model_checkpoint")
def make_model_checkpoint(filepath: str, monitor: str = 'val_loss', save_best_only: bool = True, **kwargs):
    from tensorflow.keras.callbacks import ModelCheckpoint
    return ModelCheckpoint(filepath=filepath, monitor=monitor, save_best_only=save_best_only, **kwargs)

@CallbackRegistry.register("tf_epoch_end")
def on_epoch_end_tf(self, epoch, logs=None):
    print(f"[TF] Epoch {epoch}, loss={logs.get('loss')}")

@CallbackRegistry.register("tf_batch_end")
def on_batch_end_tf(self, batch, logs=None):
    print(f"[TF] Batch {batch}, loss={logs.get('loss')}")

@CallbackRegistry.register("save_preds")
def make_save_preds(output_dir: str, val_data: Any):
    def closure(self, epoch, logs=None):
        preds = self.model.predict(val_data)
        # save preds to output_dir...
    return closure
