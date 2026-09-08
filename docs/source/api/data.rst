Data
====

Providers
---------

Import from ``xflow.data.provider``. Providers select raw items; transforms load
and process their contents.

.. py:class:: xflow.data.provider.FileProvider(root_paths, extensions=None, path_type="path")

   Scan one or more directories recursively. Extensions include the dot, such
   as ``".csv"``. Calling the provider returns a copy of its path list.
   ``len(provider)`` returns the number of files.

   ``split(ratio=0.8, seed=42)`` returns two providers.
   ``subsample(n_samples=..., seed=...)`` selects a smaller source.

.. py:class:: xflow.data.provider.SqlProvider(sources=None, output_config=None)

   Read queries into a combined pandas DataFrame. A source is a dictionary such
   as ``{"connection": "data.db", "sql": "SELECT * FROM samples"}``.
   Calling the provider returns the DataFrame, or a column list when
   ``output_config={"list": "column_name"}`` is supplied. A list output is useful
   when a pipeline consumes paths stored in a database.

   ``split(ratio=0.8, seed=42)`` returns two providers.

Pipelines
---------

Import from ``xflow.data.pipeline``. The concrete pipelines share the
constructor ``(data_provider, transforms=None, *, seed=None,
pre_transform_hook=None, post_transform_hook=None, logger=None, skip_errors=True)``.
Transforms receive one item and return the next representation.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Behavior
   * - ``DataPipeline``
     - Process samples lazily through Python iteration.
   * - ``InMemoryPipeline``
     - Process once at construction; keep the results for iteration and indexing.
   * - ``PyTorchPipeline``
     - Convert to a PyTorch Dataset that applies transforms on access.
   * - ``TensorFlowPipeline``
     - Convert to a TensorFlow Dataset using TensorFlow-compatible transforms.

``sample(n=5)`` returns processed items for inspection. ``len(pipeline)`` counts
source items for lazy pipelines and stored items for ``InMemoryPipeline``.
During Python iteration, ``skip_errors=False`` raises preprocessing exceptions;
the default logs and skips them.

``PyTorchPipeline.to_framework_dataset()`` and
``TensorFlowPipeline.to_framework_dataset()`` use their respective backends.
``InMemoryPipeline.to_framework_dataset("pytorch")`` adapts stored samples.
``DataPipeline`` has no framework conversion. Use native loaders for batching.

Pre/post hooks have signature ``(item, item_id) -> item``. They run during Python
iteration and in-memory preprocessing; the current framework conversion paths
on ``PyTorchPipeline`` and ``TensorFlowPipeline`` apply only ``transforms``.
Use ``num_workers=0`` with the Dataset returned by ``PyTorchPipeline`` on Windows.

Composition
-----------

Import from ``xflow.data.core``.

.. py:function:: xflow.data.core.pipe(sample, *transforms, flatten=True)

   Process one sample. A callable receives the whole current value; a list or
   tuple of callables processes components positionally. ``None`` passes a
   component through. With ``flatten=True``, tuple results from a positional
   transform expand into separate components.

.. py:function:: xflow.data.core.flow(samples, *transforms, progress=False, desc="Processing", skip_errors=False, on_error=None, flatten=True)

   Yield ``pipe`` results lazily. When errors are skipped, ``on_error`` can
   receive ``(exception, sample)``.

.. py:function:: xflow.data.core.compose(*transforms, flatten=True)

   Return a callable that applies the same ``pipe`` steps to each input.

.. py:function:: xflow.data.core.consume(n, fn)

   Inside a positional transform list, pass the next ``n`` components as a
   tuple to ``fn``. This joins branches of a sample.

For configuration-based transforms, use ``TransformRegistry.register(name)``
and ``build_transforms_from_config(config)`` from ``xflow.data.transform``.
The config is a list of ``{"name": "registered_name", "params": {...}}`` entries.
