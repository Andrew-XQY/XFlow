Training
========

Import ``TorchTrainer`` and ``BaseTrainer`` from ``xflow.trainers``.

.. py:class:: xflow.trainers.trainer.TorchTrainer(*, model, data_pipeline, output_dir, optimizer, criterion, device=None, callbacks=None, model_io=None, config=None, val_metrics=None, scheduler=None, scheduler_step_per_batch=False)

   Run a supervised PyTorch training loop. Supply a native model, optimizer,
   loss function, and an output directory. Batches must begin with input and
   target tensors. Move the model to its intended device before constructing an
   optimizer when preparing an accelerator workflow.

   .. py:method:: fit(*, epochs, train_loader=None, val_loader=None)

      Train and optionally validate. Pass finite loaders with ``len()``.
      Returns a dictionary of per-epoch lists including ``train_loss`` and,
      when validation is supplied, ``val_loss``. Extra ``val_metrics`` functions
      receive ``(predictions, targets)`` and return dictionaries of numbers.

      If loaders are omitted, the trainer looks for ``train_loader`` / ``val_loader``,
      ``train`` / ``val``, or ``get_train_loader`` / ``get_val_loader`` on
      ``data_pipeline``. Ordinary XFlow pipelines require explicit loaders.

   .. py:method:: save_history(path=None)

      Save history as JSON; defaults to ``output_dir/history.json``.

   .. py:method:: save_model(path=None, **extra)

      Delegate to ``ModelIO``. For a native PyTorch model, the default checkpoint
      is ``output_dir/model.pt`` with a ``model_state`` dictionary and any
      supplied extra values. This does not automatically save optimizer state.

``BaseTrainer`` is abstract. Custom trainers implement ``fit`` and ``predict``
while reusing history, callback, and model-saving support. For inference without
gradient recording, see :doc:`evaluation`.

Callbacks
---------

Import ``Callback`` and ``CallbackContext`` from ``xflow.trainers.trainer``.
Subclass ``Callback`` and override the hooks needed by your application:
``on_train_begin/end``, ``on_epoch_begin/end``, ``on_batch_begin/end``, or the
validation hooks ``on_val_epoch_begin/end`` and ``on_val_batch_begin/end``.
Each hook receives ``ctx`` with model, optimizer, zero-based epoch and batch
indices, logs, and ``request_stop``.

``CallbackRegistry`` and ``build_callbacks_from_config`` are available from
``xflow.trainers.callback``. The builder takes a list of
``{"name": "registered_factory", "params": {...}}`` entries and an explicit
``framework="torch"``. Import the module registering custom factories first.
