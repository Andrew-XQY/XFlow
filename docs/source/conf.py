import os
import sys

# Add your source code to the Python path
# Tell Sphinx where to find the src/ directory
sys.path.insert(
    0,
    os.path.abspath(os.path.join(__file__, '..', '..', '..', 'src'))
)

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
    'keras',
    'typing_extensions',  # For Python <3.11 compatibility
    'setuptools_scm',     # Version management
    'PIL',                # Pillow
    'Pillow',             # Alternative import name
]

# Project information
project = "XFlow"
copyright = "2025, Andrew Xu"
author = "Andrew Xu"

# Override the default title pattern
html_short_title = "Documentation"

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

# Suppress warnings about duplicate object descriptions
# suppress_warnings = ['autodoc.duplicate_object']

# Theme configuration
html_theme = "furo"
html_title = "Documentation"

# Project logo
html_logo = "https://raw.githubusercontent.com/Andrew-XQY/XFlow/9feba3930f468ca95b35401232a6febd66f2432c/images/logo.png"

# Furo theme options - minimal and clean
html_theme_options = {
    "sidebar_hide_name": True,  # Hide project name in sidebar for cleaner look
    "navigation_with_keys": True,  # Keyboard navigation
    "top_of_page_buttons": ["view", "edit"],  # Simple top buttons
    "source_repository": "https://github.com/Andrew-XQY/XFlow",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# Static files and templates
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_js_files = ["clickable-logo.js"]

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
autosummary_generate = True  # Generate individual files for better GitHub Pages compatibility
autosummary_imported_members = True

# GitHub Pages configuration
html_baseurl = "https://andrew-xqy.github.io/XFlow/"

# Ensure all pages have proper content for GitHub Pages
html_copy_source = True
html_show_sourcelink = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Clean up the sidebar
html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True
