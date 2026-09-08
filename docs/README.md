# Documentation

From the repository root, using Python 3.12:

```bash
python -m pip install -e ".[docs]"
python docs/build.py
```

Open `docs/build/html/index.html`. To treat documentation warnings as errors:

```bash
python -m sphinx -W --keep-going -b html docs/source docs/build/html
```

The reader documentation has two parts: `source/quickstart.rst` (overview and a
small training example) and `source/api/` (core interfaces). Keep the reference
focused on common workflows. The site uses Sphinx with Furo; its configuration
is in `source/conf.py`.
