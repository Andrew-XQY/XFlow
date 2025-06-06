# src/xflow/data/tf_pipeline.py
"""
xflow.data.tf_pipeline
----------------------
TensorFlow-compatible dataset pipeline.
"""

import tensorflow as tf
from typing import Callable, List, Tuple

# Type aliases for clarity
XYPair = Tuple[str, str]
MapFn = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


class TFPipeline:
    """
    Builds a tf.data.Dataset from a list of (x_path, y_path) pairs by applying
    a user‐provided `map_fn`. The `map_fn` can be either a pure-TF function
    or one that wraps a NumPy-based pipeline via tf.py_function.

    Example usage:
        def data_provider() -> List[XYPair]:
            return [
                ("/data/img1.png", "/data/img1.png"),  # combined image
                ("/data/img2_x.png", "/data/img2_y.png"),
                # ...
            ]

        def tf_map_fn(x_path: tf.Tensor, y_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            # Example pure‐TF decode + resize
            raw = tf.io.read_file(x_path)
            img = tf.image.decode_image(raw, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            # split if needed, resize, normalize...
            return img, img

        pipeline = TFPipeline(data_provider, tf_map_fn)
        ds = pipeline.get_dataset(batch_size=16, shuffle=True)
        model.fit(ds, epochs=10)
    """

    def __init__(
        self,
        data_provider: Callable[[], List[XYPair]],
        map_fn: MapFn
    ) -> None:
        """
        Args:
            data_provider: A zero‐argument callable returning a List of (x_path, y_path) tuples.
                           If x_path == y_path, it indicates a concatenated [x|y] image.
            map_fn:        A function that accepts two tf.Tensor (tf.string) arguments
                           and returns two tf.Tensor images (both tf.float32). This can be
                           a pure‐TF function or one that wraps a NumPy‐based pipeline
                           via tf.py_function.
        """
        self.data_provider = data_provider
        self.map_fn = map_fn

    def get_dataset(
        self,
        batch_size: int,
        shuffle: bool = True,
        buffer_size: int = 1000,
        num_parallel_calls: int = tf.data.AUTOTUNE
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset that yields batches of (x, y) image tensors.

        Args:
            batch_size:          Number of samples per batch.
            shuffle:             If True, shuffle the list of (x_path, y_path) pairs.
            buffer_size:         Buffer size for shuffling.
            num_parallel_calls:  Number of parallel calls for `map`.

        Returns:
            A `tf.data.Dataset` yielding tuples (x_tensor, y_tensor), each batch of shape
            (batch_size, H, W, C) and dtype tf.float32.
        """
        # 1) Retrieve the list of (x_path, y_path) pairs
        xy_list = self.data_provider()  # type: List[XYPair]

        # 2) Create a Dataset of string‐pairs
        ds = tf.data.Dataset.from_tensor_slices(xy_list)

        # 3) Optionally shuffle
        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size)

        # 4) Apply the user‐provided map function (decoding / preprocessing)
        ds = ds.map(self.map_fn, num_parallel_calls=num_parallel_calls)

        # 5) Batch & prefetch for performance
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds
