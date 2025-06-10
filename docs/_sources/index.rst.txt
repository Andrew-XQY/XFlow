XFlow Documentation
===================

Welcome to XFlow - a comprehensive machine learning framework.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   xflow

Installation
============

.. code-block:: bash

   pip install xflow

Quick Start
===========

.. code-block:: python

   from xflow.data import loader
   from xflow.models import autoencoder
   
   # Load and process data
   data = loader.load_data("your_data.csv")
   
   # Create and train model
   model = autoencoder.AutoEncoder()
   model.fit(data)

API Documentation
=================

The complete API documentation is organized by modules:

- :doc:`xflow.data <xflow.data>` - Data loading and preprocessing
- :doc:`xflow.models <xflow.models>` - Machine learning models  
- :doc:`xflow.trainers <xflow.trainers>` - Training utilities
- :doc:`xflow.evaluation <xflow.evaluation>` - Evaluation metrics
- :doc:`xflow.utils <xflow.utils>` - Utility functions

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`