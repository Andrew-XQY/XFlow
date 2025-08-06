import os
import sys

# Add your source code to the Python path
sys.path.insert(0, os.path.abspath("../../src"))

# Mock heavy dependencies for docs (common practice)
autodoc_mock_imports = [
    'numpy',
    'pandas',
    'matplotlib', 
    'cv2',        # opencv-python
    'tqdm',
    'pydantic',
    'yaml',
    'tensorflow',
    'keras'
]

# Project information
project = "XFlow"
copyright = "2025, Andrew Xu"
author = "Andrew Xu"

# Get version from setuptools_scm (single source of truth)
try:
    from setuptools_scm import get_version
    release = get_version(root='../..', relative_to=__file__)
    version = '.'.join(release.split('.')[:2])  # Major.minor version
except Exception:
    # Fallback version if setuptools_scm fails
    release = "0.1.0"
    version = "0.1"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

# Theme configuration
html_theme = "furo"
html_title = f"{project} Documentation"

# Furo theme options - minimal and clean
html_theme_options = {
    "sidebar_hide_name": True,  # Hide project name in sidebar for cleaner look
    "navigation_with_keys": True,  # Keyboard navigation
    "top_of_page_buttons": ["view", "edit"],  # Simple top buttons
}

# Static files (Furo handles its own styling)
html_static_path = ["_static"]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "inherited-members": True,
    "show-inheritance": True,
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Clean up the sidebar
html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True
