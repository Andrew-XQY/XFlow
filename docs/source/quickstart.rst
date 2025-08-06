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

4. **Configuration Management**

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
