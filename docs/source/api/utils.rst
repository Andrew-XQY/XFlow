Configuration
=============

Import from ``xflow.utils``.

.. py:function:: xflow.utils.config.load_validated_config(file_path, schema=None)

   Load a configuration dictionary. With a schema, instantiate it with the
   loaded values and return ``model_dump()``. YAML and JSON are supported.
   A schema is a class, not a path to a schema file.

.. py:class:: xflow.utils.config.ConfigManager(initial_config)

   Keep an original configuration and an editable working copy.
   Access nested values as dictionary keys, for example
   ``config["training"]["epochs"]``.

   ``update(updates)`` merges nested dictionaries and returns the manager.
   ``get()`` returns an independent snapshot; ``reset()`` restores the original.
   ``save_config(file_path)`` writes the current configuration.

.. code-block:: python

   from xflow.utils import ConfigManager

   config = ConfigManager({"training": {"epochs": 10}})
   config.update({"training": {"epochs": 20}})
   assert config["training"]["epochs"] == 20

To load a file, use ``ConfigManager(load_validated_config("config.yaml"))``.
