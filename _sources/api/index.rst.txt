API Reference
=============

XFlow API is organized into core modules that provide the building blocks for machine learning workflows.

Overview
--------

The XFlow package is structured around four main modules:

- **Data Module** (:doc:`data`) - Data loading, processing, and pipeline management
- **Models Module** (:doc:`models`) - Machine learning model implementations
- **Trainers Module** (:doc:`trainers`) - Training utilities and callbacks
- **Utils Module** (:doc:`utils`) - Helper functions and utilities

Core API
--------

The most commonly used classes are available directly from the package root:

.. code-block:: python

   from xflow import BasePipeline, BaseModel, BaseTrainer, ConfigManager

See :doc:`core` for detailed documentation of the core API.

Module Documentation
--------------------

.. toctree::
   :maxdepth: 2

   core
   data
   models
   trainers
   utils

Package Structure
-----------------

XFlow follows a clear modular structure where each module has a specific responsibility:

* **xflow.data** - All data-related functionality
* **xflow.models** - Model definitions and implementations
* **xflow.trainers** - Training loops, callbacks, and utilities
* **xflow.utils** - Configuration, visualization, and helper functions
