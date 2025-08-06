Utils Module
============

The utils module provides configuration management and utility functions.

.. currentmodule:: xflow.utils

Configuration Classes
---------------------

.. autoclass:: ConfigManager
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: get_base_dir

.. autofunction:: load_validated_config

.. autofunction:: plot_image

.. autoclass:: BaseTrainerConfig
   :members:
   :show-inheritance:

.. autoclass:: BaseModelConfig
   :members:
   :show-inheritance:

Helper Functions (Internal)
---------------------------

.. currentmodule:: xflow.utils.helper

.. autofunction:: split_sequence

.. autofunction:: subsample_sequence

.. autofunction:: deep_update

IO Functions (Internal)
-----------------------

.. currentmodule:: xflow.utils.io

.. autofunction:: scan_files

.. autofunction:: copy_file

.. currentmodule:: xflow.utils.parser

.. autofunction:: load_file

.. autofunction:: save_file
