import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .typing import ImageLike


def to_numpy_image(img: ImageLike) -> np.ndarray:
    """
    Convert various image formats to a 2D numpy array suitable for display.

    Args:
        img: PIL Image, numpy array, TensorFlow tensor, or PyTorch tensor

    Returns:
        numpy.ndarray: 2D array representing the image
    """
    if hasattr(img, "numpy"):  # TF tensor
        arr = img.numpy()
    elif hasattr(img, "detach"):  # PyTorch tensor
        arr = img.detach().cpu().numpy()
    elif isinstance(img, Image.Image):  # PIL
        arr = np.array(img)
    elif isinstance(img, np.ndarray):  # already numpy
        arr = img
    else:
        arr = np.array(img)  # fallback

    if arr.ndim == 4:  # If batch dim, take first sample
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # If channel-first, move channels last
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:  # If single channel, squeeze
        arr = arr[:, :, 0]
    return arr


def plot_image(
    img: ImageLike, cmap: str = None, title: str = None, figsize: tuple = None
) -> None:
    """
    Plot an image using matplotlib.

    Args:
        img: Image in any supported format (will be converted automatically)
        cmap: Colormap to use (auto-detected if None)
        title: Plot title
        figsize: Figure size tuple
    """
    arr = to_numpy_image(img)
    if cmap is None:
        cmap = "gray" if arr.ndim == 2 else None
    if figsize:
        plt.figure(figsize=figsize)
    plt.imshow(arr, cmap=cmap)
    plt.xlabel("X (pixel index)")
    plt.ylabel("Y (pixel index)")
    plt.colorbar(label="Pixel value")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
