XFlow Documentation
===================

**XFlow** is a lightweight modular machine-learning framework with a clear high-level structure.

.. toctree::
   :maxdepth: 2

   quickstart
   api/index
   examples/basic_usage

Core Modules
============

XFlow is organized into four main modules:

:doc:`api/data`
   Data loading, processing, and pipeline management

:doc:`api/models`
   Machine learning model implementations

:doc:`api/trainers`
   Training utilities and callbacks

:doc:`api/utils`
   Helper functions and utilities

Quick Example
=============

.. code-block:: python

   from xflow.data import BasePipeline
   from xflow.models import BaseModel

   # Create a data pipeline
   pipeline = BasePipeline()

   # Create and configure model
   model = BaseModel()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
