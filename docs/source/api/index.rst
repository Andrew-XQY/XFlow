API Reference
=============

XFlow API is organized into four main modules that provide the building blocks for machine learning workflows.

Modules
-------

.. toctree::
   :maxdepth: 2

   data
   models
   trainers
   utils

Overview
--------

The XFlow package is structured around four main modules:

- **Data Module** (:doc:`data`) - Data pipelines, providers, and transformations
- **Models Module** (:doc:`models`) - Machine learning model base classes  
- **Trainers Module** (:doc:`trainers`) - Training utilities and callback management
- **Utils Module** (:doc:`utils`) - Configuration management and utility functions

Quick Start
-----------

Access the main components from their respective modules:

.. code-block:: python

   # Data processing
   from xflow.data import BasePipeline, InMemoryPipeline, BatchPipeline
   from xflow.data import FileProvider, SqlProvider
   
   # Models
   from xflow.models import BaseModel
   
   # Training
   from xflow.trainers import BaseTrainer, CallbackRegistry
   
   # Utilities  
   from xflow.utils import ConfigManager, get_base_dir, load_validated_config, plot_image

* **xflow.data** - All data-related functionality
* **xflow.models** - Model definitions and implementations
* **xflow.trainers** - Training loops, callbacks, and utilities
* **xflow.utils** - Configuration, visualization, and helper functions
