# XFlow Documentation

This directory contains the Sphinx documentation for XFlow.

## Building the Documentation

1. Install dependencies:
   ```bash
   pip install sphinx sphinx-rtd-theme
   ```

2. Build the documentation:
   ```bash
   python build.py
   ```

   Or manually:
   ```bash
   cd docs
   sphinx-build -b html source build/html
   ```

3. Open `build/html/index.html` in your browser.

## Structure

- `source/` - Documentation source files (RST format)
- `source/api/` - API reference documentation
- `source/examples/` - Usage examples
- `source/_static/` - Static files (CSS, images)
- `build/` - Generated HTML files (created after building)

## Customization

- `source/conf.py` - Sphinx configuration
- `source/_static/custom.css` - Custom styling
- Theme: sphinx_rtd_theme with custom styling for dark mode support

The documentation structure follows the API organization defined in `src/xflow/_api_registry.py`.
