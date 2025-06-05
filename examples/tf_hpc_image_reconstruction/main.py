from xflow.data.tf_pipeline import test
from pathlib import Path

# Suppose BASE_DIR is something you set in a config or env var
BASE_DIR = Path("/path/to/your/project")
data_dir = BASE_DIR / "data" / "images"
test()