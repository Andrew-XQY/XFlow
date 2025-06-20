# callbacks_registry.py

import yaml
from typing import Dict, Callable, Any, List


EVENT_MAP = {
    "train_start":  {"tf": "on_train_begin",         "pl": "on_train_start"},
    "train_end":    {"tf": "on_train_end",           "pl": "on_train_end"},
    "epoch_start":  {"tf": "on_epoch_begin",         "pl": "on_train_epoch_start"},
    "epoch_end":    {"tf": "on_epoch_end",           "pl": "on_train_epoch_end"},
    "batch_start":  {"tf": "on_train_batch_begin",   "pl": "on_train_batch_start"},
    "batch_end":    {"tf": "on_train_batch_end",     "pl": "on_train_batch_end"},
}


class CallbackRegistry:
    """Registry for callback handlers (or factories)."""
    _handlers: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a callback handler or factory."""
        def decorator(func: Callable):
            cls._handlers[name] = func
            return func
        return decorator
    
    @classmethod
    def get_handler(cls, name: str) -> Callable:
        """Get a registered handler (or factory) by name."""
        if name not in cls._handlers:
            raise ValueError(f"Handler '{name}' not found in registry")
        return cls._handlers[name]
    
    @classmethod
    def list_handlers(cls) -> List[str]:
        """List all registered handler names."""
        return list(cls._handlers.keys())


def make_tf_callback(handlers: Dict[str, Callable]):
    """Factory for a TensorFlow Callback."""
    from tensorflow.keras.callbacks import Callback
    
    methods = {}
    for event_name, fn in handlers.items():
        hook = EVENT_MAP[event_name]["tf"]
        methods[hook] = fn
    return type("UnifiedTFCallback", (Callback,), methods)()


def make_pl_callback(handlers: Dict[str, Callable]):
    """Factory for a PyTorch Lightning Callback."""
    import pytorch_lightning as pl
    
    methods = {}
    for event_name, fn in handlers.items():
        hook = EVENT_MAP[event_name]["pl"]
        methods[hook] = fn
    return type("UnifiedPLCallback", (pl.Callback,), methods)()


def build_callbacks_from_config(config_path: str, framework: str):
    """Build callbacks from YAML configuration.

    Args:
        config_path: Path to YAML config file
        framework: Either 'tf' or 'pl'

    Returns:
        List of callback objects for the specified framework
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    callbacks = []
    for cb_config in config.get('callbacks', []):
        if cb_config.get('framework') != framework:
            continue

        # collect hook functions for this callback
        handlers: Dict[str, Callable] = {}
        for event_config in cb_config.get('events', []):
            event_name   = event_config['event']
            handler_name = event_config['handler']
            handler      = CallbackRegistry.get_handler(handler_name)

            # if params provided, invoke factory to get actual hook
            params = event_config.get('params', {})
            if params:
                hook_fn = handler(**params)
            else:
                hook_fn = handler

            handlers[event_name] = hook_fn

        # build the actual callback object
        if framework in ('tf', 'tensorflow'):
            callback = make_tf_callback(handlers)
        elif framework in ('pl', 'pytorch_lightning'):
            callback = make_pl_callback(handlers)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        callbacks.append(callback)

    return callbacks


# --- Registered callback handlers & factories ---

@CallbackRegistry.register("tf_epoch_start")
def on_epoch_start_tf(self, epoch, logs=None):
    print(f"[TF] Starting epoch {epoch}")

@CallbackRegistry.register("tf_epoch_end")
def on_epoch_end_tf(self, epoch, logs=None):
    print(f"[TF] Finished epoch {epoch}, loss={logs.get('loss', 'N/A'):.4f}")

@CallbackRegistry.register("pl_epoch_start")
def on_epoch_start_pl(self, trainer, pl_module):
    print(f"[PL] Starting epoch {trainer.current_epoch}")

@CallbackRegistry.register("pl_epoch_end")
def on_epoch_end_pl(self, trainer, pl_module):
    print(f"[PL] Finished epoch {trainer.current_epoch}")

@CallbackRegistry.register("tf_train_start")
def on_train_start_tf(self, logs=None):
    print("[TF] Training started")

@CallbackRegistry.register("pl_train_start")
def on_train_start_pl(self, trainer, pl_module):
    print("[PL] Training started")

@CallbackRegistry.register("save_preds")
def make_save_preds_callback(output_dir: str, val_data: Any):
    """Factory that returns a tf-style on_epoch_end hook."""
    def closure(self, epoch, logs=None):
        preds = self.model.predict(val_data)
        # …save preds to output_dir…
    return closure
