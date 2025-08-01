Data Module
===========

The data module provides pipeline components for loading and processing data.

.. currentmodule:: xflow.data

Pipeline Classes
----------------

.. autoclass:: BasePipeline
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Pipeline
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: InMemoryPipeline
   :members:
   :show-inheritance:
   :no-index:

Transform Components
--------------------

.. autoclass:: ShufflePipeline
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: BatchPipeline
   :members:
   :show-inheritance:
   :no-index:

Utility Functions
-----------------

.. autofunction:: build_transforms_from_config
