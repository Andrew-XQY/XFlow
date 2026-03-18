import importlib
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

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


def _resolve_point_colors(
    n_samples: int,
    labels: Any | None,
    properties: Mapping[str, Any] | None,
    color_by: str | None,
) -> np.ndarray | None:
    """Resolve point-wise color values from labels or named properties.

    Contract: values are index-aligned with coords; value[i] maps to coords[i].
    """
    values = labels
    if color_by is not None:
        if properties is None:
            raise ValueError("`properties` must be provided when `color_by` is set.")
        if color_by not in properties:
            raise ValueError(
                f"Unknown property '{color_by}'. Available: {list(properties.keys())}."
            )
        values = properties[color_by]

    if values is None:
        return None

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1D point metadata for colors, got shape {arr.shape}."
        )
    if arr.shape[0] != n_samples:
        raise ValueError(
            f"Color metadata length mismatch: expected {n_samples}, got {arr.shape[0]}."
        )
    return arr


def _resolve_point_groups(
    n_samples: int,
    labels: Any | None,
    properties: Mapping[str, Any] | None,
    group_by: str | None,
) -> np.ndarray | None:
    """Resolve point-wise grouping values used for envelope drawing.

    Contract: values are index-aligned with coords; value[i] maps to coords[i].
    """
    values = labels
    if group_by is not None:
        if properties is None:
            raise ValueError("`properties` must be provided when `group_by` is set.")
        if group_by not in properties:
            raise ValueError(
                f"Unknown property '{group_by}'. Available: {list(properties.keys())}."
            )
        values = properties[group_by]

    if values is None:
        return None

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1D point metadata for groups, got shape {arr.shape}."
        )
    if arr.shape[0] != n_samples:
        raise ValueError(
            f"Group metadata length mismatch: expected {n_samples}, got {arr.shape[0]}."
        )
    return arr


def _scatter_3d_with_color(
    fig: Any,
    ax: Any,
    arr: np.ndarray,
    color_values: np.ndarray | None,
    cmap: str,
    point_size: float,
    legend_title: str | None,
) -> None:
    if color_values is None:
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=point_size)
        return

    if np.issubdtype(color_values.dtype, np.number):
        sc = ax.scatter(
            arr[:, 0], arr[:, 1], arr[:, 2], c=color_values, s=point_size, cmap=cmap
        )
        fig.colorbar(sc, ax=ax)
        return

    categories = np.unique(color_values.astype(str))
    cmap_obj = plt.get_cmap(cmap, len(categories))
    for idx, category in enumerate(categories):
        mask = color_values.astype(str) == category
        ax.scatter(
            arr[mask, 0],
            arr[mask, 1],
            arr[mask, 2],
            s=point_size,
            color=cmap_obj(idx),
            label=category,
        )
    ax.legend(title=legend_title or "Category")


def _draw_3d_group_envelopes(
    ax: Any,
    arr: np.ndarray,
    group_values: np.ndarray,
    cmap: str,
    envelope_alpha: float,
    envelope_linewidth: float,
) -> None:
    """Draw convex-hull surfaces for each group in 3D."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull, QhullError

    groups = group_values.astype(str)
    categories = np.unique(groups)
    cmap_obj = plt.get_cmap(cmap, len(categories))
    for idx, category in enumerate(categories):
        mask = groups == category
        pts = arr[mask, :3]
        if pts.shape[0] < 4:
            continue
        try:
            hull = ConvexHull(pts)
        except QhullError:
            continue
        faces = [pts[simplex] for simplex in hull.simplices]
        poly = Poly3DCollection(
            faces,
            facecolor=cmap_obj(idx),
            edgecolor=cmap_obj(idx),
            linewidths=envelope_linewidth,
            alpha=envelope_alpha,
        )
        ax.add_collection3d(poly)


class Embedding3DPlot:
    """Stateful Plotly-native 3D embedding plot object.

    Contract:
    - `coords` is 2D with shape (n_samples, >=3)
    - Point metadata (`labels`, `properties[key]`) is index-aligned with `coords`
    - `get_plot(...)` builds/returns a Plotly Figure for automation workflows
    - `show(...)` is interactive display only (no backend switching)
    - This object owns the cached Plotly figure until `close()`
    """

    def __init__(
        self,
        coords: Any,
        labels: Any | None = None,
        properties: Mapping[str, Any] | None = None,
        color_by: str | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (7, 6),
        cmap: str = "tab10",
        point_size: float = 12,
        legend_title: str | None = None,
        envelope: bool = False,
        envelope_by: str | None = None,
        envelope_alpha: float = 0.12,
        envelope_linewidth: float = 0.8,
    ) -> None:
        try:
            import plotly.express as px
        except Exception as exc:
            raise ImportError(
                "Embedding3DPlot requires Plotly. Install with `pip install plotly`."
            ) from exc

        self._px = px
        self.arr = np.asarray(coords)
        if self.arr.ndim != 2 or self.arr.shape[1] < 3:
            raise ValueError(
                f"Expected coords shape (n_samples, >=3), got {self.arr.shape}."
            )

        self.title = title
        self.figsize = figsize
        self.cmap = cmap
        self.point_size = point_size
        self.legend_title = legend_title
        self.envelope = envelope
        self.envelope_alpha = envelope_alpha
        self.envelope_linewidth = envelope_linewidth

        if self.envelope:
            raise NotImplementedError(
                "Plotly-native Embedding3DPlot does not implement `envelope` yet."
            )

        self.color_values = _resolve_point_colors(
            n_samples=self.arr.shape[0],
            labels=labels,
            properties=properties,
            color_by=color_by,
        )
        self.group_values = _resolve_point_groups(
            n_samples=self.arr.shape[0],
            labels=labels,
            properties=properties,
            group_by=envelope_by,
        )

        self._fig: Any | None = None

    @staticmethod
    def _camera_from_elev_azim(
        elev: float, azim: float, radius: float = 1.8
    ) -> dict[str, dict[str, float]]:
        """Convert Matplotlib-style (elev, azim) to Plotly camera eye."""
        elev_rad = np.deg2rad(elev)
        azim_rad = np.deg2rad(azim)
        return {
            "eye": {
                "x": float(radius * np.cos(elev_rad) * np.cos(azim_rad)),
                "y": float(radius * np.cos(elev_rad) * np.sin(azim_rad)),
                "z": float(radius * np.sin(elev_rad)),
            }
        }

    def _build(self, rebuild: bool = False) -> Any:
        if self._fig is not None and not rebuild:
            return self._fig

        px = self._px
        x = self.arr[:, 0]
        y = self.arr[:, 1]
        z = self.arr[:, 2]

        if self.color_values is None:
            fig = px.scatter_3d(
                x=x,
                y=y,
                z=z,
                title=self.title,
                labels={"x": "Dim 1", "y": "Dim 2", "z": "Dim 3"},
            )
        else:
            if np.issubdtype(self.color_values.dtype, np.number):
                fig = px.scatter_3d(
                    x=x,
                    y=y,
                    z=z,
                    color=self.color_values,
                    title=self.title,
                    labels={
                        "x": "Dim 1",
                        "y": "Dim 2",
                        "z": "Dim 3",
                        "color": self.legend_title or "Value",
                    },
                )
            else:
                fig = px.scatter_3d(
                    x=x,
                    y=y,
                    z=z,
                    color=self.color_values.astype(str),
                    title=self.title,
                    labels={
                        "x": "Dim 1",
                        "y": "Dim 2",
                        "z": "Dim 3",
                        "color": self.legend_title or "Category",
                    },
                )

        fig.update_traces(marker={"size": self.point_size})
        fig.update_layout(
            width=int(self.figsize[0] * 100),
            height=int(self.figsize[1] * 100),
            scene={
                "xaxis_title": "Dim 1",
                "yaxis_title": "Dim 2",
                "zaxis_title": "Dim 3",
            },
        )
        if self.legend_title:
            fig.update_layout(legend_title_text=self.legend_title)

        self._fig = fig
        return fig

    def get_plot(
        self,
        elev: float | None = None,
        azim: float | None = None,
        rebuild: bool = False,
    ) -> Any:
        """Build/update and return a Plotly Figure for automation workflows."""
        fig = self._build(rebuild=rebuild)
        if elev is not None or azim is not None:
            camera = self._camera_from_elev_azim(
                elev=25.0 if elev is None else elev,
                azim=35.0 if azim is None else azim,
            )
            fig.update_layout(scene_camera=camera)
        return fig

    def show(
        self,
        elev: float | None = None,
        azim: float | None = None,
        rebuild: bool = False,
        renderer: str | None = None,
    ) -> Any:
        """Display the interactive Plotly figure and return it.

        Contract:
        - This is interactive display only
        - Use `get_plot(...)` for automation-only usage without UI side effects
        """
        fig = self.get_plot(elev=elev, azim=azim, rebuild=rebuild)
        fig.show(renderer=renderer)
        return fig

    def export_html(
        self,
        path: str,
        elev: float | None = None,
        azim: float | None = None,
        rebuild: bool = False,
        include_plotlyjs: str | bool = "cdn",
    ) -> None:
        """Export an interactive HTML artifact for automation/reporting."""
        fig = self.get_plot(elev=elev, azim=azim, rebuild=rebuild)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs=include_plotlyjs)

    def export_rotation_gif(
        self,
        path: str,
        elev: float = 25.0,
        azim_start: float = 0.0,
        azim_end: float = 360.0,
        n_frames: int = 120,
        fps: int = 24,
        dpi: int = 120,
        rebuild: bool = True,
    ) -> None:
        """Not supported in Plotly-native mode; use `export_html` instead."""
        raise NotImplementedError(
            "Plotly-native Embedding3DPlot does not support GIF export. "
            "Use `export_html(...)` for interactive export."
        )

    def close(self) -> None:
        """Release cached figure reference."""
        self._fig = None


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
