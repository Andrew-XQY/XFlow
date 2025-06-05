import xflow.data.tf_pipeline as t
from pathlib import Path

# Suppose BASE_DIR is something you set in a config or env var
BASE_DIR = Path("/path/to/your/project")
data_dir = BASE_DIR / "data" / "images"
t