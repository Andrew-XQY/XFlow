import os
import sys

# Add your source code to the Python path
sys.path.insert(0, os.path.abspath('../../src'))

# Project information
project = 'XFlow'
copyright = '2025, Andrew Xu'
author = 'Andrew Xu'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
]

# Theme
html_theme = 'sphinx_rtd_theme'

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autosummary
autosummary_generate = True

# HTML options
html_static_path = ['_static']
html_show_sourcelink = True