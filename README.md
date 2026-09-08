# XFlow

[Documentation](https://andrew-xqy.github.io/XFlow/) · [Issues](https://github.com/Andrew-XQY/XFlow/issues) · [MIT license](LICENSE)

## Overview

XFlow is a Python library for scientific machine learning. It connects data
sources, preprocessing functions, training loops, and evaluation hooks. It grew
out of physics research and keeps application-specific work in ordinary Python
functions and classes.

The design separates responsibilities: a **provider** selects raw data, a
**pipeline** transforms each sample, and a **trainer** runs the model on batches.
You supply the model, optimizer, and loss. Use the pieces independently or join
them into a workflow.

For the PyTorch example below, use Python 3.12 and install:

```bash
python -m pip install "xflow-py[ml_torch]"
```

For this checkout, use `python -m pip install -e ".[ml_torch]"` from the repository
root. The base install, `pip install xflow-py`, provides the data and configuration
tools. TensorFlow dataset adapters are available with the `ml_tf` extra; the
concrete general training loop is `TorchTrainer`.

This example creates a tiny dataset, learns `y = 2x + 1`, and saves model weights
and training history:

```python
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from xflow import FileProvider, PyTorchPipeline, TorchTrainer

# Make a small regression dataset: one CSV row (input, target) per file.
data_dir = Path("xflow-demo-data")
data_dir.mkdir(exist_ok=True)
for i, x in enumerate(np.linspace(-1, 1, 64)):
    np.savetxt(data_dir / f"{i:03d}.csv", [[x, 2 * x + 1]], delimiter=",")

def load_sample(path):
    row = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return torch.from_numpy(row[:1]), torch.from_numpy(row[1:])

provider = FileProvider(data_dir, extensions=".csv")
train_source, val_source = provider.split(ratio=0.8, seed=42)
train = PyTorchPipeline(train_source, transforms=[load_sample], skip_errors=False)
val = PyTorchPipeline(val_source, transforms=[load_sample], skip_errors=False)
train_loader = DataLoader(train.to_framework_dataset(), batch_size=8, shuffle=True)
val_loader = DataLoader(val.to_framework_dataset(), batch_size=8)

model = torch.nn.Linear(1, 1)
trainer = TorchTrainer(
    model=model,
    data_pipeline=train,
    output_dir="xflow-demo-run",
    optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    criterion=torch.nn.MSELoss(),
    device="cpu",
)
history = trainer.fit(epochs=10, train_loader=train_loader, val_loader=val_loader)
trainer.save_history()
trainer.save_model()
print(history["val_loss"][-1])
```

For your own dataset, replace the data creation and `load_sample` function.
Each transformed sample here is an `(input_tensor, target_tensor)` pair.
`FileProvider` selects paths; `PyTorchPipeline` loads samples on access;
`DataLoader` batches them; `fit()` runs training and validation. The example uses
the default `num_workers=0`.

Extend XFlow with a callable in `transforms`, a provider implementing the
`DataProvider` interface, or a callback with hooks such as `on_epoch_end(ctx)`.
For configuration-based transforms, register a callable with
`TransformRegistry.register("name")` and pass a list of `{"name": ..., "params": ...}`
entries to `build_transforms_from_config`. Import the module containing the
registration first. Keep custom extensions in your own package; the repository's
`xflow.extensions` modules are excluded from published wheels. See the
[overview](https://andrew-xqy.github.io/XFlow/quickstart.html) for a short example.

## Core API

| Module | Main interfaces | Responsibility |
| --- | --- | --- |
| `xflow.data` | `FileProvider`, `SqlProvider`, `DataPipeline`, `InMemoryPipeline`, `PyTorchPipeline`, `TensorFlowPipeline` | Select data and transform samples. |
| `xflow.data.core` | `pipe`, `flow`, `compose`, `consume` | Compose preprocessing, including tuple branches and joins. |
| `xflow.models` | `BaseModel` | Optional abstract model interface; `TorchTrainer` accepts a native `torch.nn.Module`. |
| `xflow.trainers` | `BaseTrainer`, `TorchTrainer`, `CallbackRegistry` | Train, validate, record history, and dispatch callbacks. |
| `xflow.evaluation` | `run_evaluation`, `BaseEvalHook`, `InMemoryCollector` | Run PyTorch inference and process predictions. |
| `xflow.utils` | `ConfigManager`, `load_validated_config` | Load and manage configuration dictionaries. |

The [core API reference](https://andrew-xqy.github.io/XFlow/api/index.html)
documents the main contracts and methods.
