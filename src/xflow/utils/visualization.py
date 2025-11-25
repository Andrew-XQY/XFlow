from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .typing import ImageLike


def to_numpy_image(img: ImageLike) -> np.ndarray:
    """
    Convert various image formats to a 2D/3D numpy array suitable for display.
    Works for CPU/GPU PyTorch tensors, TF eager tensors, PIL, and numpy arrays.
    """
    import torch

    if isinstance(img, torch.Tensor):  # PyTorch tensor
        arr = img.detach().cpu().numpy()
    elif hasattr(img, "numpy"):  # TF tensor
        arr = img.numpy()
    elif isinstance(img, Image.Image):  # PIL
        arr = np.array(img)
    elif isinstance(img, np.ndarray):  # already numpy
        arr = img
    else:  # fallback
        arr = np.array(img)

    # Normalize shape for display
    if arr.ndim == 4:  # (B, C, H, W) or (B, H, W, C) → take first
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # channel-first → channel-last
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:  # single channel → squeeze
        arr = arr[..., 0]
    return arr


def plot_image(
    img: ImageLike,
    cmap: str = None,
    title: str = None,
    figsize: tuple = None,
    vmin: float = None,
    vmax: float = None,
    colorbar: bool = True,
) -> None:
    """
    Plot an image using matplotlib.

    Args:
        img: Image in any supported format (will be converted automatically)
        cmap: Colormap to use (auto-detected if None)
        title: Plot title
        figsize: Figure size tuple
        vmin: Minimum pixel value for color scaling (auto if None)
        vmax: Maximum pixel value for color scaling (auto if None)
        colorbar: Whether to show the colorbar (default True)
    """
    arr = to_numpy_image(img)
    if cmap is None:
        cmap = "gray" if arr.ndim == 2 else None
    if figsize:
        plt.figure(figsize=figsize)
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("X (pixel index)")
    plt.ylabel("Y (pixel index)")
    if colorbar:
        plt.colorbar(label="Pixel value")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def save_image(
    img: ImageLike,
    path: str,
    cmap: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 4),
    dpi: int = 150,
) -> None:
    """
    Same as plot_image, but saves to disk instead of showing.
    Defaults ensure it runs without extra args.
    """
    arr = to_numpy_image(img)
    if cmap is None:
        cmap = "gray" if arr.ndim == 2 else None

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap=cmap)
    ax.set_xlabel("X (pixel index)")
    ax.set_ylabel("Y (pixel index)")
    # Colorbar can fail for RGB; keep behavior but make it safe
    try:
        fig.colorbar(im, ax=ax, label="Pixel value")
    except Exception:
        pass
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
