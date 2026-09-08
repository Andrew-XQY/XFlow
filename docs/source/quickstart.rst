Overview
========

XFlow connects data sources, preprocessing, training, and evaluation for
scientific machine learning. It grew out of physics research. Application logic
stays in Python functions and classes, with each part of the workflow responsible
for one job:

* **Provider:** select raw files or database records.
* **Pipeline:** turn each raw item into a sample using ordered callables.
* **Trainer:** run a supplied model, optimizer, and loss over batches.
* **Evaluation:** run inference and pass predictions to hooks.

Use these parts independently or combine them. A pipeline does not choose a
model, and a trainer does not decide how to load files. Native framework objects
remain accessible.

Install
-------

Use Python 3.12 for this example:

.. code-block:: bash

   python -m pip install "xflow-py[ml_torch]"

For a local checkout, run ``python -m pip install -e ".[ml_torch]"`` from the
repository root. ``pip install xflow-py`` installs the data and configuration
tools without a training backend. The ``ml_tf`` extra provides TensorFlow
dataset adapters. XFlow's concrete trainer is ``TorchTrainer``, which uses PyTorch.

Provider to training
--------------------

This runnable example creates 64 CSV files, each containing one input and one
target. It fits a linear model to ``y = 2x + 1`` and writes a checkpoint and
history to ``xflow-demo-run``.

.. code-block:: python

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

For an existing dataset, replace the creation loop and ``load_sample``.
``FileProvider`` returns paths; ``PyTorchPipeline`` applies the loader on access;
the native ``DataLoader`` shuffles and batches the resulting tensor pairs.
Pass loaders explicitly to ``fit()``. This example uses the default
``num_workers=0``.

``DataPipeline`` offers lazy Python iteration; ``InMemoryPipeline`` processes
samples once and keeps them in memory. ``BaseModel`` and ``BaseTrainer`` are
abstract interfaces. A native ``torch.nn.Module`` is enough for ``TorchTrainer``.

Compose and extend
------------------

Transforms are callables. ``pipe`` processes one sample; ``flow`` applies the same
steps lazily to an iterable; ``compose`` packages steps into a reusable callable.
A list of transforms applies positionally to tuple components, with ``None``
passing a component through:

.. code-block:: python

   from xflow import compose, consume, flow, pipe

   scale_input = compose([lambda x: x / 255.0, None])
   assert scale_input((255.0, "label")) == (1.0, "label")
   assert pipe((2, 3, "label"), [consume(2, sum), None]) == (5, "label")
   assert list(flow([1, 2], lambda x: x * 2)) == [2, 4]

Add a custom transform directly to ``transforms`` or register it for use in
configuration. Import its module before building the configured pipeline:

.. code-block:: python

   from xflow.data.transform import TransformRegistry, build_transforms_from_config

   @TransformRegistry.register("scale_pair")
   def scale_pair(sample, factor=1.0):
       x, y = sample
       return x * factor, y

   transforms = build_transforms_from_config([
       {"name": "scale_pair", "params": {"factor": 0.5}}
   ])

For a new data source, implement ``DataProvider`` from ``xflow.data.provider``:
``__call__()`` returns items, ``__len__()`` reports their count, and ``subsample()``
returns a subset provider. A training callback can subclass ``Callback`` from
``xflow.trainers.trainer`` and implement hooks such as ``on_epoch_end(ctx)``.
Keep custom extensions in your own package. The repository's
``xflow.extensions`` modules are excluded from published wheels.

See :doc:`api/index` for the core contracts.
