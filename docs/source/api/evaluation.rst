Evaluation
==========

Import from ``xflow.evaluation``. The runner uses PyTorch inference mode and
restores the model's previous training/evaluation mode after the run.

.. py:function:: xflow.evaluation.runner.run_evaluation(model, dataset, device, hooks=None, unpack_batch=default_unpack_batch, forward_fn=default_forward, max_batches=None, strict_hook_errors=True)

   Run inference over an iterable of batches. Default batch forms are ``(x,)``,
   ``(x, y)``, or a dictionary containing ``inputs`` and optional ``targets``
   and ``metadata``. Supply ``unpack_batch`` for a different format.
   Returns an ``EvalContext`` containing sample/batch counts and hook state.

   A ``BaseEvalHook`` subclass can implement ``on_start(ctx)``,
   ``on_batch(ctx, batch)``, and ``on_end(ctx)``. ``EvalBatch`` contains inputs,
   predictions, targets, and metadata; tensor outputs are detached to CPU.
   Set ``ctx.stop = True`` to stop processing further batches.

For small datasets, ``InMemoryCollector`` collects per-sample results:

.. code-block:: python

   from xflow.evaluation import InMemoryCollector, run_evaluation

   collector = InMemoryCollector()
   context = run_evaluation(model, val_loader, "cpu", hooks=[collector])
   predictions = [sample["predictions"] for sample in collector.samples]

This example continues from the model and loader in :doc:`../quickstart`.
