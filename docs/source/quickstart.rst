Quickstart Guide
================

Installation
------------

Install XFlow using pip:

.. code-block:: bash

   pip install xflow-py

Basic Usage
-----------

XFlow provides a simple and intuitive API for building machine learning pipelines:

1. **Data Pipeline**

   .. code-block:: python

      from xflow import BasePipeline, InMemoryPipeline

      # Create a basic pipeline
      pipeline = BasePipeline()

      # Or use in-memory pipeline for small datasets
      data_pipeline = InMemoryPipeline(data)

2. **Model Creation**

   .. code-block:: python

      from xflow import BaseModel

      # Create a model
      model = BaseModel()

3. **Training**

   .. code-block:: python

      from xflow import BaseTrainer

      # Create and configure trainer
      trainer = BaseTrainer(model=model, data=pipeline)

      # Start training
      trainer.train()

4. **Data Transforms (PyTorch Support)**

   XFlow now supports PyTorch/torchvision transforms alongside TensorFlow transforms:

   .. code-block:: python

      from xflow.data.transform import TransformRegistry, build_transforms_from_config

      # Use individual PyTorch transforms
      to_tensor = TransformRegistry.get("torch_to_tensor")
      resize = TransformRegistry.get("torch_resize")
      normalize = TransformRegistry.get("torch_normalize")

      # Or build from configuration
      config = [
          {"name": "torch_to_tensor"},
          {"name": "torch_resize", "params": {"size": [224, 224]}},
          {"name": "torch_normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
      ]
      transforms = build_transforms_from_config(config)

      # Convert pipeline to PyTorch dataset
      torch_dataset = pipeline.to_framework_dataset("pytorch")

5. **Training Callbacks (PyTorch Support)**

   XFlow now supports PyTorch callbacks for training monitoring and control:

   .. code-block:: python

      from xflow.trainers.callback import CallbackRegistry, build_callbacks_from_config

      # Use individual PyTorch callbacks
      early_stopping = CallbackRegistry.get_handler("torch_early_stopping")(
          monitor="val_loss", patience=10, restore_best=True
      )
      
      model_checkpoint = CallbackRegistry.get_handler("torch_model_checkpoint")(
          filepath="best_model.pth", monitor="val_loss", save_best_only=True
      )

      # Or build from configuration
      callback_config = [
          {"name": "torch_early_stopping", "params": {"patience": 10}},
          {"name": "torch_progress_bar", "params": {"desc": "Training"}},
          {"name": "torch_lr_scheduler", "params": {"scheduler_class": "StepLR", "step_size": 10}}
      ]
      callbacks = build_callbacks_from_config(callback_config, framework="torch")

6. **Configuration Management**

   .. code-block:: python

      from xflow import ConfigManager

      # Load configuration
      config = ConfigManager.load_config('config.yaml')

      # Access configuration values
      learning_rate = config.training.learning_rate

Next Steps
----------

- Check out the :doc:`api/index` for detailed API documentation
- See :doc:`examples/basic_usage` for more comprehensive examples
- Explore the core modules: :doc:`api/data`, :doc:`api/models`, :doc:`api/trainers`, :doc:`api/utils`
