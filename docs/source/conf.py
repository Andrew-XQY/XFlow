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
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

# Theme - modern RTD with customization
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'style_nav_header_background': '#2980B9',
}

# Custom CSS for dark mode option
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Clean up the sidebar
html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True
