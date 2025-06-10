import tensorflow as tf
import pandas as pd
import cv2
import numpy as np

from xflow.data.tf_loader import TFPipeline
from pathlib import Path

# Suppose BASE_DIR is something you set in a config or env var
BASE_DIR = Path("/path/to/your/project")
data_dir = BASE_DIR / "data" / "images"

# 1 Preprocessing: read, normalize, split
def load_and_process_image(path: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    w = tf.shape(img)[1] // 2
    label = img[:, :w, :]
    inp = img[:, w:, :]
    return inp, label

# 2 Data provider: SQL → list of paths
def data_provider() -> list[str]:
    sql = """
        SELECT image_path
        FROM mmf_dataset_metadata
        WHERE is_calibration = 0
          AND purpose = 'testing'
          AND comments IS NULL
    """
    df = DB.sql_select(sql)
    paths = [ABS_DIR + p for p in df["image_path"].to_list()]
    return paths[::5]

# 3 Build tf.data.Dataset
def build_dataset(batch_size: int) -> tf.data.Dataset:
    pipeline = TFPipeline(
        data_provider=data_provider,
        map_fn=load_and_process_image
    )
    return pipeline.get_dataset(batch_size=batch_size, shuffle=False)

# 4 Example: evaluate a model
if __name__ == "__main__":
    val_ds = build_dataset(batch_size=32)
    model = ...  # your tf.keras.Model
    model.compile(optimizer="adam", loss="mse")
    model.evaluate(val_ds)


