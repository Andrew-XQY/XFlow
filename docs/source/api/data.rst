Data Module
===========

The data module provides pipeline and transformation utilities for data processing.

.. currentmodule:: xflow.data

Pipeline Classes
----------------

.. autoclass:: BasePipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: InMemoryPipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DataPipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TensorFlowPipeline
   :members:
   :undoc-members:
   :show-inheritance:

Transform Classes
-----------------

.. autoclass:: ShufflePipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BatchPipeline
   :members:
   :undoc-members:
   :show-inheritance:

Provider Classes
----------------

.. autoclass:: FileProvider
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SqlProvider
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: build_transforms_from_config
