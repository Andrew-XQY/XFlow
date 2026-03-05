import importlib
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image
except Exception:  # optional dependency
    Image = None

from .typing import ImageLike


def _to_2d_feature_array(X: Any) -> np.ndarray:
    """Validate and normalize embedding input to a 2D float numpy array."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected X to be 2D (n_samples, n_features), got shape {arr.shape}."
        )
    if arr.shape[0] < 2:
        raise ValueError("Expected at least 2 samples for visualization.")
    return arr.astype(np.float64, copy=False)


class DimReducer:
    """
    Minimal unified contract for dimensionality reduction visualization.

    Contract:
    - Input X is treated as ndarray-like with shape (n_samples, n_features)
    - fit_transform(X) -> ndarray (n_samples, n_components)
    - transform(X) is available only when supports_transform is True
    """

    def __init__(
        self,
        method: str = "pca",
        n_components: int = 2,
        random_state: int | None = 42,
        **kwargs: Any,
    ) -> None:
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = self._build_model()

    def _build_model(self):
        if self.method == "pca":
            from sklearn.decomposition import PCA

            return PCA(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
        if self.method == "tsne":
            from sklearn.manifold import TSNE

            return TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
        if self.method == "umap":
            try:
                umap = importlib.import_module("umap")
            except Exception as exc:
                raise ImportError(
                    "UMAP is not installed. Install with `pip install umap-learn`."
                ) from exc
            return umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
        raise ValueError(f"Unknown method: {self.method}. Use one of: pca, tsne, umap.")

    @property
    def supports_transform(self) -> bool:
        return self.method in {"pca", "umap"}

    def fit_transform(self, X: Any) -> np.ndarray:
        X_arr = _to_2d_feature_array(X)
        return self.model.fit_transform(X_arr)

    def transform(self, X: Any) -> np.ndarray:
        if not self.supports_transform:
            raise NotImplementedError(f"{self.method} does not support transform().")
        X_arr = _to_2d_feature_array(X)
        return self.model.transform(X_arr)


def plot_embedding(
    coords: Any,
    labels: Any | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> None:
    """Plot 2D embedding coordinates."""
    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected coords shape (n_samples, >=2), got {arr.shape}.")

    plt.figure(figsize=figsize)
    if labels is None:
        plt.scatter(arr[:, 0], arr[:, 1], s=12)
    else:
        plt.scatter(arr[:, 0], arr[:, 1], c=np.asarray(labels), s=12, cmap="tab10")
        plt.colorbar()
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def to_numpy_image(img: ImageLike) -> np.ndarray:
    """
    Convert various image formats to a 2D/3D numpy array suitable for display.
    Works for CPU/GPU PyTorch tensors, TF eager tensors, PIL, and numpy arrays.
    """
    torch_tensor_type = ()
    try:
        import torch

        torch_tensor_type = (torch.Tensor,)
    except Exception:
        pass

    if torch_tensor_type and isinstance(img, torch_tensor_type):  # PyTorch tensor
        arr = img.detach().cpu().numpy()
    elif hasattr(img, "numpy"):  # TF tensor
        arr = img.numpy()
    elif Image is not None and isinstance(img, Image.Image):  # PIL
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


def stack_log_remap(images: Iterable[ImageLike], eps: float = 1e-12) -> np.ndarray:
    """
    Pixel-wise sum (no clipping), then log1p + min-max remap to uint8 [0,255].
    Returns a single 2D uint8 image.
    """
    # optional progress bar
    try:
        from tqdm import tqdm

        it = tqdm(images)
    except Exception:
        it = images

    stack = None
    for img in it:
        im = to_numpy_image(img).astype(np.float64, copy=False)
        if stack is None:
            stack = np.zeros_like(im, dtype=np.float64)
        stack += im

    if stack is None:
        raise ValueError("Empty images iterable.")

    log_img = np.log1p(stack)
    denom = (log_img.max() - log_img.min()) + eps
    out = (log_img - log_img.min()) / denom
    return (out * 255).astype(np.uint8)


def stack_linear_clip(images: Iterable[ImageLike], max_val: int = 255) -> np.ndarray:
    """
    Pixel-wise sum, then clip to max_val. Returns a single 2D uint8 image.

    Args:
        images: Iterable of images to stack
        max_val: Maximum value to clip to (default 255)

    Returns:
        Stacked image as uint8 array clipped to [0, max_val]
    """
    try:
        from tqdm import tqdm

        it = tqdm(images)
    except Exception:
        it = images

    stack = None
    for img in it:
        im = to_numpy_image(img).astype(np.float64, copy=False)
        if stack is None:
            stack = np.zeros_like(im, dtype=np.float64)
        stack += im

    if stack is None:
        raise ValueError("Empty images iterable.")

    return np.clip(stack, 0, max_val).astype(np.uint8)
