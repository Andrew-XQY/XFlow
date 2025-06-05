# src/xflow/data/tf_pipeline.py
"""
xflow.data.tf_pipeline
----------------------
TensorFlow-compatible datasets
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import tensorflow as tf

class ImageToImageDataset:
    """
    Wraps two parallel folders of images (inputs and targets) into a tf.data.Dataset. for image-to-image tasks.

    Args:
        input_dir (Union[str, Path]):
            Path to the folder containing input images.
        target_dir (Union[str, Path]):
            Path to the folder containing target images (filenames must match inputs).
        batch_size (int):
            Number of samples per batch.
        shuffle (bool):
            Whether to shuffle the dataset each epoch.
        buffer_size (int):
            Buffer size for shuffling (only used if shuffle=True).
        preprocess_fn (Optional[Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]):
            Optional function taking (x, y) tensors and returning (x, y) tensors.
            Use for resizing, normalization, augmentation, etc.
        extensions (Tuple[str, ...]):
            Valid file extensions (lowercase) to include (e.g. (".png", ".jpg")).
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        target_dir: Union[str, Path],
        batch_size: int,
        shuffle: bool = True,
        buffer_size: int = 1000,
        preprocess_fn: Optional[
            Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
        ] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ) -> None:
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.preprocess_fn = preprocess_fn
        self.extensions = extensions

        self._validate_directories()
        self.input_paths, self.target_paths = self._gather_paths()
        self.dataset = self._build_dataset()

    def _validate_directories(self) -> None:
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.target_dir.is_dir():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

    def _gather_paths(self) -> Tuple[List[str], List[str]]:
        """
        Collects and sorts matching file paths from input_dir and target_dir.

        Returns:
            Tuple[List[str], List[str]]: Sorted lists of file paths.
        """
        input_list = sorted(
            [
                str(p)
                for p in self.input_dir.iterdir()
                if p.suffix.lower() in self.extensions
            ]
        )
        target_list = sorted(
            [
                str(p)
                for p in self.target_dir.iterdir()
                if p.suffix.lower() in self.extensions
            ]
        )
        if len(input_list) == 0:
            raise FileNotFoundError(f"No images found in input directory: {self.input_dir}")
        if len(input_list) != len(target_list):
            raise ValueError(
                "Number of input images and target images must match. "
                f"Found {len(input_list)} inputs and {len(target_list)} targets."
            )
        return input_list, target_list

    def _load_pair(self, input_path: tf.Tensor, target_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Reads and decodes an input-target image pair.

        Args:
            input_path (tf.Tensor): Path to an input image.
            target_path (tf.Tensor): Path to a target image.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (input_image, target_image), both float32 [0,1].
        """
        x = tf.io.read_file(input_path)
        x = tf.image.decode_image(x, channels=3)
        x = tf.image.convert_image_dtype(x, tf.float32)

        y = tf.io.read_file(target_path)
        y = tf.image.decode_image(y, channels=3)
        y = tf.image.convert_image_dtype(y, tf.float32)

        if self.preprocess_fn:
            x, y = self.preprocess_fn(x, y)
        return x, y

    def _build_dataset(self) -> tf.data.Dataset:
        """
        Builds a tf.data.Dataset yielding (input, target) image pairs.

        Returns:
            tf.data.Dataset[Tuple[tf.Tensor, tf.Tensor]]: Batched dataset.
        """
        ds = tf.data.Dataset.from_tensor_slices((self.input_paths, self.target_paths))
        ds = ds.map(self._load_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle:
            ds = ds.shuffle(self.buffer_size)

        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def as_dataset(self) -> tf.data.Dataset:
        """
        Returns:
            tf.data.Dataset[Tuple[tf.Tensor, tf.Tensor]]:
                Dataset yielding batches of (input, target) image pairs.
        """
        return self.dataset
