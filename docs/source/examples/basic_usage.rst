Basic Usage Examples
====================

This page provides comprehensive examples of using XFlow for common machine learning tasks.

Data Pipeline Example
---------------------

Here's how to create and use data pipelines:

.. code-block:: python

   from xflow import BasePipeline, InMemoryPipeline, ShufflePipeline, BatchPipeline
   import numpy as np

   # Create sample data
   data = np.random.rand(1000, 784)  # 1000 samples, 784 features
   labels = np.random.randint(0, 10, 1000)  # 10 classes

   # Method 1: Using InMemoryPipeline for small datasets
   pipeline = InMemoryPipeline(data)

   # Method 2: Building a pipeline with transforms
   pipeline = BasePipeline()
   pipeline = ShufflePipeline(pipeline)  # Add shuffling
   pipeline = BatchPipeline(pipeline, batch_size=32)  # Add batching

   # Use the pipeline
   for batch in pipeline:
       # Process each batch
       print(f"Batch shape: {batch.shape}")

Configuration Management
------------------------

XFlow provides robust configuration management:

.. code-block:: python

   from xflow import ConfigManager
   from xflow.utils import load_validated_config

   # Load configuration from YAML
   config = ConfigManager.load_config('config.yaml')

   # Access nested configuration values
   learning_rate = config.training.learning_rate
   batch_size = config.data.batch_size

   # Load and validate configuration
   validated_config = load_validated_config('config.yaml', schema='training_schema.json')

Model Training Example
----------------------

Complete example of setting up and training a model:

.. code-block:: python

   from xflow import BaseModel, BaseTrainer, BasePipeline
   from xflow.trainers import build_callbacks_from_config

   # Create model
   model = BaseModel()

   # Create data pipeline
   train_pipeline = BasePipeline()
   val_pipeline = BasePipeline()

   # Build callbacks from configuration
   callbacks = build_callbacks_from_config({
       'early_stopping': {'patience': 10},
       'model_checkpoint': {'filepath': 'best_model.h5'}
   })

   # Create trainer
   trainer = BaseTrainer(
       model=model,
       train_data=train_pipeline,
       val_data=val_pipeline,
       callbacks=callbacks
   )

   # Start training
   history = trainer.train(epochs=100)

Visualization Example
---------------------

Using XFlow's visualization utilities:

.. code-block:: python

   from xflow.utils import plot_image
   import matplotlib.pyplot as plt

   # Plot an image
   image = np.random.rand(28, 28)
   plot_image(image, title="Sample Image", cmap='gray')
   plt.show()

Configuration File Example
---------------------------

Example YAML configuration file (``config.yaml``):

.. code-block:: yaml

   # Model configuration
   model:
     type: "autoencoder"
     input_shape: [784]
     latent_dim: 128

   # Training configuration
   training:
     learning_rate: 0.001
     batch_size: 32
     epochs: 100

   # Data configuration
   data:
     train_path: "data/train.csv"
     val_path: "data/val.csv"
     transforms:
       - name: "normalize"
         params:
           mean: 0.5
           std: 0.5

   # Callbacks configuration
   callbacks:
     early_stopping:
       patience: 10
       monitor: "val_loss"
     model_checkpoint:
       filepath: "models/best_model.h5"
       save_best_only: true
