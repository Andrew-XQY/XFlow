Models
======

``TorchTrainer`` accepts a native ``torch.nn.Module``. Construct its optimizer
and loss before creating the trainer; a basic supervised batch is ``(x, y)``
and the trainer calls ``model(x)`` and ``criterion(predictions, y)``.

.. py:class:: xflow.models.base.BaseModel

   Optional abstract interface for an application-defined model. A subclass
   implements all of these methods:

   * ``predict(inputs, **kwargs)``: run inference.
   * ``save(path)`` and classmethod ``load(path, **kwargs)``: persist and restore.
   * ``training_step(batch)``: perform one update and return a loss or metrics.
   * ``validation_step(batch)``: return validation loss or metrics.
   * ``configure_optimizers()``: return application-specific optimizer objects.

   ``set_train_mode(training=True)`` is an optional override. A metrics
   dictionary returned by ``training_step`` must contain ``"loss"``.

``BaseModel`` cannot be instantiated directly. ``TorchTrainer`` uses the native
PyTorch interface and does not dispatch to these abstract step methods or
``BaseModel.save``. Use a custom trainer or ``ModelIO`` adapter when integrating
that model contract.
