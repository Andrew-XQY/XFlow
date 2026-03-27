import importlib
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image
except Exception:  # optional dependency
    Image = None

from .typing import ImageLike

PLOTLY_DEFAULT_COLORWAY = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)


def _distinct_spectrum_colors(n_colors: int) -> list[str]:
    """Build a distinct qualitative palette for categorical points."""
    if n_colors <= 0:
        return []

    colors = list(
        PLOTLY_DEFAULT_COLORWAY[: min(n_colors, len(PLOTLY_DEFAULT_COLORWAY))]
    )
    if len(colors) >= n_colors:
        return colors

    # Fill remaining colors by alternating spectrum extremes toward center.
    import matplotlib.colors as mcolors

    n_extra = n_colors - len(colors)
    cmap = plt.get_cmap("turbo")
    positions = np.linspace(0.0, 1.0, n_extra, endpoint=True)

    left = 0
    right = n_extra - 1
    ordered_indices: list[int] = []
    while left <= right:
        ordered_indices.append(left)
        left += 1
        if left <= right:
            ordered_indices.append(right)
            right -= 1

    for idx in ordered_indices:
        colors.append(mcolors.to_hex(cmap(float(positions[idx]))))

    return colors


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


def _resolve_point_vector(
    n_samples: int,
    values: Any | None,
    name: str,
) -> np.ndarray | None:
    """Validate a per-point metadata vector aligned to sample order."""
    if values is None:
        return None

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D `{name}` metadata, got shape {arr.shape}.")
    if arr.shape[0] != n_samples:
        raise ValueError(
            f"`{name}` length mismatch: expected {n_samples}, got {arr.shape[0]}."
        )
    return arr


def _ordered_unique_strings(values: np.ndarray) -> list[str]:
    """Return first-seen unique string values, preserving input order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values.astype(str):
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _is_color_like(value: Any) -> bool:
    """Return True if the value can be interpreted as a color spec."""
    import matplotlib.colors as mcolors

    try:
        mcolors.to_rgba(value)
        return True
    except Exception:
        return False


def _resolve_marker_colors(color_values: np.ndarray) -> np.ndarray:
    """Resolve per-point marker colors from numeric, color-like, or categorical input."""
    if np.issubdtype(color_values.dtype, np.number):
        return color_values.astype(np.float64, copy=False)

    as_str = color_values.astype(str)
    if all(_is_color_like(v) for v in as_str):
        return as_str.astype(object, copy=False)

    categories = _ordered_unique_strings(as_str)
    palette = _distinct_spectrum_colors(len(categories))
    cat_to_color = {cat: palette[idx] for idx, cat in enumerate(categories)}
    return np.array([cat_to_color[v] for v in as_str], dtype=object)


def _colors_from_labels(label_values: np.ndarray) -> np.ndarray:
    """Build deterministic per-point colors from label groups."""
    categories = _ordered_unique_strings(label_values)
    palette = _distinct_spectrum_colors(len(categories))
    cat_to_color = {cat: palette[idx] for idx, cat in enumerate(categories)}
    return np.array([cat_to_color[v] for v in label_values], dtype=object)


def _scatter_3d_with_color(
    fig: Any,
    ax: Any,
    arr: np.ndarray,
    label_values: np.ndarray | None,
    marker_colors: np.ndarray | None,
    cmap: str,
    point_size: float,
    legend_loc: str,
) -> None:
    if label_values is None:
        groups = [("samples", np.ones(arr.shape[0], dtype=bool))]
        show_legend = False
    else:
        groups = [
            (label, label_values == label)
            for label in _ordered_unique_strings(label_values)
        ]
        show_legend = True

    has_numeric_colors = marker_colors is not None and np.issubdtype(
        np.asarray(marker_colors).dtype, np.number
    )
    first_scatter = None
    if has_numeric_colors:
        vmin = float(np.min(marker_colors))
        vmax = float(np.max(marker_colors))

    for label, mask in groups:
        kwargs: dict[str, Any] = {"s": point_size}
        if marker_colors is not None:
            if has_numeric_colors:
                kwargs["c"] = marker_colors[mask]
                kwargs["cmap"] = cmap
                kwargs["vmin"] = vmin
                kwargs["vmax"] = vmax
            else:
                kwargs["c"] = marker_colors[mask]

        sc = ax.scatter(
            arr[mask, 0],
            arr[mask, 1],
            arr[mask, 2],
            label=label if show_legend else None,
            **kwargs,
        )
        if first_scatter is None:
            first_scatter = sc

    if has_numeric_colors and first_scatter is not None:
        fig.colorbar(first_scatter, ax=ax)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ordered = sorted(zip(labels, handles), key=lambda item: item[0].lower())
        sorted_labels = [label for label, _ in ordered]
        sorted_handles = [handle for _, handle in ordered]
        ax.legend(
            sorted_handles,
            sorted_labels,
            loc=legend_loc,
            markerscale=3,
        )


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


def _projection_walls(arr: np.ndarray, gap_ratio: float) -> tuple[float, float, float]:
    """Compute x/y/z wall positions used for projection overlays."""
    mins = arr[:, :3].min(axis=0)
    maxs = arr[:, :3].max(axis=0)
    spans = np.maximum(maxs - mins, 1e-12)
    wall_x = float(mins[0] - gap_ratio * spans[0])
    wall_y = float(mins[1] - gap_ratio * spans[1])
    wall_z = float(mins[2] - gap_ratio * spans[2])
    return wall_x, wall_y, wall_z


class Embedding3DPlot:
    """Stateful Plotly-native 3D embedding plot object.

    Contract:
    - `coords` is 2D with shape (n_samples, >=3)
    - Point metadata (`labels`, `color`) is index-aligned with `coords`
    - `labels` controls legend/group names only
    - `color` controls marker colors only (numeric, direct color specs, or categories)
    - Projection envelopes group by `labels` (or all points if labels are not provided)
    - `get_plot(...)` builds/returns a Plotly Figure for automation workflows
    - `get_plot_with_projections(...)` overlays wall projections on the Plotly Figure
    - `get_matplotlib_plot(...)` builds/returns a Matplotlib Figure/Axes snapshot
    - `get_matplotlib_frame(...)` returns one RGB frame in memory for a camera angle
    - `show(...)` is interactive display only (no backend switching)
    - This object owns the cached Plotly figure until `close()`
    """

    def __init__(
        self,
        coords: Any,
        labels: Any | None = None,
        color: Any | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (7, 6),
        cmap: str = "tab10",
        point_size: float = 12,
        legend_loc: str = "upper right",
        envelope: bool = False,
        envelope_alpha: float = 0.12,
        envelope_linewidth: float = 0.8,
        show_projections: bool = False,
        projection_alpha: float = 0.35,
        projection_size_scale: float = 0.65,
        projection_gap_ratio: float = 0.06,
        projection_envelope: bool = False,
        projection_envelope_alpha: float = 0.18,
    ) -> None:
        try:
            import plotly.graph_objects as go
        except Exception as exc:
            raise ImportError(
                "Embedding3DPlot requires Plotly. Install with `pip install plotly`."
            ) from exc

        self._go = go
        self.arr = np.asarray(coords)
        if self.arr.ndim != 2 or self.arr.shape[1] < 3:
            raise ValueError(
                f"Expected coords shape (n_samples, >=3), got {self.arr.shape}."
            )

        self.title = title
        self.figsize = figsize
        self.cmap = cmap
        self.point_size = point_size
        self.legend_loc = legend_loc
        self.envelope = envelope
        self.envelope_alpha = envelope_alpha
        self.envelope_linewidth = envelope_linewidth
        self.show_projections = show_projections
        self.projection_alpha = projection_alpha
        self.projection_size_scale = projection_size_scale
        self.projection_gap_ratio = projection_gap_ratio
        self.projection_envelope = projection_envelope
        self.projection_envelope_alpha = projection_envelope_alpha

        if self.envelope:
            raise NotImplementedError(
                "Plotly-native Embedding3DPlot does not implement `envelope` yet."
            )

        n_samples = self.arr.shape[0]
        self.labels = _resolve_point_vector(n_samples, labels, name="labels")
        self.color_values = _resolve_point_vector(n_samples, color, name="color")
        self.label_values = (
            None if self.labels is None else self.labels.astype(str, copy=False)
        )
        self.group_values = self.label_values

        if self.color_values is None:
            self.marker_colors = (
                None
                if self.label_values is None
                else _colors_from_labels(self.label_values)
            )
        else:
            self.marker_colors = _resolve_marker_colors(self.color_values)

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

        go = self._go
        x = self.arr[:, 0]
        y = self.arr[:, 1]
        z = self.arr[:, 2]

        if self.label_values is None:
            groups = [("samples", np.ones(self.arr.shape[0], dtype=bool))]
            show_legend = False
        else:
            groups = [
                (label, self.label_values == label)
                for label in _ordered_unique_strings(self.label_values)
            ]
            show_legend = True
            legend_rank = {
                label: idx
                for idx, label in enumerate(
                    sorted(_ordered_unique_strings(self.label_values), key=str.lower)
                )
            }

        marker_colors = self.marker_colors
        has_numeric_colors = marker_colors is not None and np.issubdtype(
            np.asarray(marker_colors).dtype, np.number
        )
        if has_numeric_colors:
            cmin = float(np.min(marker_colors))
            cmax = float(np.max(marker_colors))

        fig = go.Figure()
        for idx, (label, mask) in enumerate(groups):
            marker: dict[str, Any] = {"size": self.point_size}

            if marker_colors is not None:
                subset = marker_colors[mask]
                if has_numeric_colors:
                    marker["color"] = subset.astype(np.float64)
                    marker["cmin"] = cmin
                    marker["cmax"] = cmax
                    marker["showscale"] = idx == 0
                    marker["colorbar"] = {"title": ""}
                else:
                    marker["color"] = subset.tolist()

            fig.add_trace(
                go.Scatter3d(
                    x=x[mask],
                    y=y[mask],
                    z=z[mask],
                    mode="markers",
                    marker=marker,
                    name=label,
                    showlegend=show_legend,
                    legendgroup=label,
                    legendrank=legend_rank[label] if show_legend else None,
                )
            )

        fig.update_layout(
            title=self.title,
            width=int(self.figsize[0] * 100),
            height=int(self.figsize[1] * 100),
            legend_title_text="",
            legend={"itemsizing": "constant"},
            scene={
                "xaxis_title": "Dim 1",
                "yaxis_title": "Dim 2",
                "zaxis_title": "Dim 3",
            },
        )

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

    @staticmethod
    def _rgba_from_hex(hex_color: str, alpha: float) -> str:
        color = hex_color.lstrip("#")
        if len(color) != 6:
            return f"rgba(120,120,120,{alpha})"
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    def get_plot_with_projections(
        self,
        elev: float | None = None,
        azim: float | None = None,
        rebuild: bool = False,
        show_projections: bool | None = None,
        projection_alpha: float | None = None,
        projection_size_scale: float | None = None,
        projection_gap_ratio: float | None = None,
        projection_envelope: bool | None = None,
        projection_envelope_alpha: float | None = None,
    ) -> Any:
        """Return Plotly Figure with added wall projections for each PCA axis.

        Projections are added onto three walls:
        - X wall: x = min(x) - gap
        - Y wall: y = min(y) - gap
        - Z wall: z = min(z) - gap

        Colors are copied from the original sample traces.
        """
        try:
            import plotly.graph_objects as go
        except Exception as exc:
            raise ImportError(
                "Embedding3DPlot projection overlay requires Plotly graph_objects."
            ) from exc

        fig = self.get_plot(elev=elev, azim=azim, rebuild=rebuild)

        projection_alpha = (
            self.projection_alpha if projection_alpha is None else projection_alpha
        )
        show_projections = (
            self.show_projections if show_projections is None else show_projections
        )
        projection_size_scale = (
            self.projection_size_scale
            if projection_size_scale is None
            else projection_size_scale
        )
        projection_gap_ratio = (
            self.projection_gap_ratio
            if projection_gap_ratio is None
            else projection_gap_ratio
        )
        projection_envelope = (
            self.projection_envelope
            if projection_envelope is None
            else projection_envelope
        )
        projection_envelope_alpha = (
            self.projection_envelope_alpha
            if projection_envelope_alpha is None
            else projection_envelope_alpha
        )

        # Remove previously-added projection overlays so repeated calls are idempotent.
        base_traces = [
            trace
            for trace in fig.data
            if getattr(trace, "legendgroup", None) != "__xflow_projection__"
        ]
        fig.data = tuple(base_traces)

        wall_x, wall_y, wall_z = _projection_walls(self.arr, projection_gap_ratio)

        if show_projections:
            for trace in base_traces:
                x = np.asarray(trace.x, dtype=np.float64)
                y = np.asarray(trace.y, dtype=np.float64)
                z = np.asarray(trace.z, dtype=np.float64)
                if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
                    continue

                marker = {}
                if getattr(trace, "marker", None) is not None:
                    marker = trace.marker.to_plotly_json()

                size = marker.get("size", self.point_size)
                if np.isscalar(size):
                    marker["size"] = float(size) * projection_size_scale
                else:
                    marker["size"] = (
                        np.asarray(size, dtype=np.float64) * projection_size_scale
                    ).tolist()
                marker["opacity"] = projection_alpha
                marker["showscale"] = False

                fig.add_trace(
                    go.Scatter3d(
                        x=np.full_like(x, wall_x),
                        y=y,
                        z=z,
                        mode="markers",
                        marker=marker,
                        name=f"{getattr(trace, 'name', '')} (x-wall)",
                        showlegend=False,
                        legendgroup="__xflow_projection__",
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=np.full_like(y, wall_y),
                        z=z,
                        mode="markers",
                        marker=marker,
                        name=f"{getattr(trace, 'name', '')} (y-wall)",
                        showlegend=False,
                        legendgroup="__xflow_projection__",
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=np.full_like(z, wall_z),
                        mode="markers",
                        marker=marker,
                        name=f"{getattr(trace, 'name', '')} (z-wall)",
                        showlegend=False,
                        legendgroup="__xflow_projection__",
                        hoverinfo="skip",
                    )
                )

        if projection_envelope:
            try:
                from scipy.spatial import ConvexHull, QhullError
            except Exception:
                warnings.warn(
                    "Projection envelope requires scipy; skipping envelope overlay.",
                    stacklevel=2,
                )
                return fig

            groups = (
                self.group_values.astype(str)
                if self.group_values is not None
                else np.array(["all"] * self.arr.shape[0], dtype=str)
            )

            color_map: dict[str, str] = {}
            for trace in base_traces:
                name = getattr(trace, "name", None)
                if name is None:
                    continue
                marker = getattr(trace, "marker", None)
                color = getattr(marker, "color", None)
                if isinstance(color, str):
                    color_map[str(name)] = color
                elif isinstance(color, (list, tuple, np.ndarray)) and len(color) > 0:
                    first = color[0]
                    if isinstance(first, str) and _is_color_like(first):
                        color_map[str(name)] = first

            unique_groups = np.unique(groups)
            fallback_colors = _distinct_spectrum_colors(len(unique_groups))
            for idx, group in enumerate(unique_groups):
                mask = groups == group
                pts = self.arr[mask, :3]
                if pts.shape[0] < 3:
                    continue

                base_color = color_map.get(str(group), fallback_colors[idx])
                face_color = (
                    self._rgba_from_hex(base_color, projection_envelope_alpha)
                    if isinstance(base_color, str) and base_color.startswith("#")
                    else base_color
                )

                walls = (
                    (np.column_stack([pts[:, 1], pts[:, 2]]), "x", wall_x),
                    (np.column_stack([pts[:, 0], pts[:, 2]]), "y", wall_y),
                    (np.column_stack([pts[:, 0], pts[:, 1]]), "z", wall_z),
                )
                for pts2d, wall_axis, wall_val in walls:
                    uniq = np.unique(pts2d, axis=0)
                    if uniq.shape[0] < 3:
                        continue
                    try:
                        hull = ConvexHull(uniq)
                    except QhullError:
                        continue
                    hull_pts = uniq[hull.vertices]
                    if hull_pts.shape[0] < 3:
                        continue

                    if wall_axis == "x":
                        xs = np.full(hull_pts.shape[0], wall_val)
                        ys = hull_pts[:, 0]
                        zs = hull_pts[:, 1]
                    elif wall_axis == "y":
                        xs = hull_pts[:, 0]
                        ys = np.full(hull_pts.shape[0], wall_val)
                        zs = hull_pts[:, 1]
                    else:
                        xs = hull_pts[:, 0]
                        ys = hull_pts[:, 1]
                        zs = np.full(hull_pts.shape[0], wall_val)

                    i = [0] * (hull_pts.shape[0] - 2)
                    j = list(range(1, hull_pts.shape[0] - 1))
                    k = list(range(2, hull_pts.shape[0]))
                    fig.add_trace(
                        go.Mesh3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            i=i,
                            j=j,
                            k=k,
                            color=face_color,
                            opacity=projection_envelope_alpha,
                            showscale=False,
                            showlegend=False,
                            name=f"{group} ({wall_axis}-wall envelope)",
                            legendgroup="__xflow_projection__",
                            hoverinfo="skip",
                        )
                    )

        fig.update_layout(
            scene={
                "xaxis": {"backgroundcolor": "rgba(245,245,245,0.5)"},
                "yaxis": {"backgroundcolor": "rgba(245,245,245,0.5)"},
                "zaxis": {"backgroundcolor": "rgba(245,245,245,0.5)"},
            }
        )
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

    def get_matplotlib_plot(
        self,
        elev: float = 25.0,
        azim: float = 35.0,
        show_projections: bool | None = None,
        projection_alpha: float | None = None,
        projection_size_scale: float | None = None,
        projection_gap_ratio: float | None = None,
        projection_envelope: bool | None = None,
        projection_envelope_alpha: float | None = None,
    ) -> tuple[Any, Any]:
        """Build and return a Matplotlib (fig, ax) snapshot at a camera angle."""
        import matplotlib.colors as mcolors

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        show_projections = (
            self.show_projections if show_projections is None else show_projections
        )
        projection_alpha = (
            self.projection_alpha if projection_alpha is None else projection_alpha
        )
        projection_size_scale = (
            self.projection_size_scale
            if projection_size_scale is None
            else projection_size_scale
        )
        projection_gap_ratio = (
            self.projection_gap_ratio
            if projection_gap_ratio is None
            else projection_gap_ratio
        )
        projection_envelope = (
            self.projection_envelope
            if projection_envelope is None
            else projection_envelope
        )
        projection_envelope_alpha = (
            self.projection_envelope_alpha
            if projection_envelope_alpha is None
            else projection_envelope_alpha
        )

        _scatter_3d_with_color(
            fig=fig,
            ax=ax,
            arr=self.arr,
            label_values=self.label_values,
            marker_colors=self.marker_colors,
            cmap=self.cmap,
            point_size=self.point_size,
            legend_loc=self.legend_loc,
        )

        mins = self.arr[:, :3].min(axis=0)
        maxs = self.arr[:, :3].max(axis=0)
        center = (mins + maxs) / 2.0
        radius = float((maxs - mins).max() / 2.0 + 1e-12)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        # Anchor projection/shading planes to axis panes to avoid mplot3d depth artifacts.
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        eps_x = max((xlim[1] - xlim[0]) * 1e-4, 1e-12)
        eps_y = max((ylim[1] - ylim[0]) * 1e-4, 1e-12)
        eps_z = max((zlim[1] - zlim[0]) * 1e-4, 1e-12)
        wall_x = float(xlim[0] + eps_x)
        wall_y = float(ylim[0] + eps_y)
        wall_z = float(zlim[0] + eps_z)

        x = self.arr[:, 0]
        y = self.arr[:, 1]
        z = self.arr[:, 2]

        if show_projections:
            proj_size = float(self.point_size) * projection_size_scale

            if self.marker_colors is None:
                proj_kwargs = {"c": "C0", "alpha": projection_alpha}
            elif np.issubdtype(np.asarray(self.marker_colors).dtype, np.number):
                proj_kwargs = {
                    "c": self.marker_colors,
                    "cmap": self.cmap,
                    "alpha": projection_alpha,
                }
            else:
                proj_kwargs = {"c": self.marker_colors, "alpha": projection_alpha}

            ax.scatter(np.full_like(x, wall_x), y, z, s=proj_size, **proj_kwargs)
            ax.scatter(x, np.full_like(y, wall_y), z, s=proj_size, **proj_kwargs)
            ax.scatter(x, y, np.full_like(z, wall_z), s=proj_size, **proj_kwargs)

        if projection_envelope:
            try:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                from scipy.spatial import ConvexHull, QhullError
            except Exception:
                warnings.warn(
                    "Projection envelope requires scipy; skipping envelope overlay.",
                    stacklevel=2,
                )
            else:
                groups = (
                    self.group_values.astype(str)
                    if self.group_values is not None
                    else np.array(["all"] * self.arr.shape[0], dtype=str)
                )
                unique_groups = np.unique(groups)
                fallback_colors = _distinct_spectrum_colors(len(unique_groups))

                group_to_color: dict[str, str] = {}
                if self.marker_colors is not None and not np.issubdtype(
                    np.asarray(self.marker_colors).dtype, np.number
                ):
                    for group in unique_groups:
                        mask = groups == group
                        if np.any(mask):
                            group_to_color[str(group)] = str(
                                self.marker_colors[mask][0]
                            )

                for idx, group in enumerate(unique_groups):
                    mask = groups == group
                    pts = self.arr[mask, :3]
                    if pts.shape[0] < 3:
                        continue

                    base_color = group_to_color.get(str(group), fallback_colors[idx])
                    face_rgba = mcolors.to_rgba(
                        base_color, alpha=projection_envelope_alpha
                    )

                    walls = (
                        (np.column_stack([pts[:, 1], pts[:, 2]]), "x", wall_x),
                        (np.column_stack([pts[:, 0], pts[:, 2]]), "y", wall_y),
                        (np.column_stack([pts[:, 0], pts[:, 1]]), "z", wall_z),
                    )
                    for pts2d, wall_axis, wall_val in walls:
                        uniq = np.unique(pts2d, axis=0)
                        if uniq.shape[0] < 3:
                            continue
                        try:
                            hull = ConvexHull(uniq)
                        except QhullError:
                            continue
                        poly2d = uniq[hull.vertices]
                        if poly2d.shape[0] < 3:
                            continue

                        if wall_axis == "x":
                            poly3d = [
                                (float(wall_val), float(v0), float(v1))
                                for v0, v1 in poly2d
                            ]
                        elif wall_axis == "y":
                            poly3d = [
                                (float(v0), float(wall_val), float(v1))
                                for v0, v1 in poly2d
                            ]
                        else:
                            poly3d = [
                                (float(v0), float(v1), float(wall_val))
                                for v0, v1 in poly2d
                            ]

                        ax.add_collection3d(
                            Poly3DCollection(
                                [poly3d],
                                facecolor=face_rgba,
                                edgecolor=face_rgba,
                                linewidths=self.envelope_linewidth,
                                alpha=projection_envelope_alpha,
                                zsort="min",
                            )
                        )

        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        if self.title:
            ax.set_title(self.title)

        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        return fig, ax

    def get_matplotlib_frame(
        self,
        elev: float = 25.0,
        azim: float = 35.0,
        dpi: int = 120,
        show_projections: bool | None = None,
        projection_alpha: float | None = None,
        projection_size_scale: float | None = None,
        projection_gap_ratio: float | None = None,
        projection_envelope: bool | None = None,
        projection_envelope_alpha: float | None = None,
    ) -> np.ndarray:
        """Return one RGB frame in memory for the requested camera angle."""
        fig, _ = self.get_matplotlib_plot(
            elev=elev,
            azim=azim,
            show_projections=show_projections,
            projection_alpha=projection_alpha,
            projection_size_scale=projection_size_scale,
            projection_gap_ratio=projection_gap_ratio,
            projection_envelope=projection_envelope,
            projection_envelope_alpha=projection_envelope_alpha,
        )
        fig.set_dpi(dpi)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return frame

    def close(self) -> None:
        """Release cached figure reference."""
        self._fig = None


class DimReducer:
    """
    Minimal unified API for dimensionality reduction.

    Contract:
    - Input `X` is ndarray-like with shape (n_samples, n_features)
    - Output is ndarray with shape (n_samples, n_components)
    - You may use a built-in method (`pca`, `tsne`, `umap`) or inject a custom model
    - Custom model must provide either:
      1) `fit_transform(X)`; or
      2) both `fit(X)` and `transform(X)`
    """

    def __init__(
        self,
        method: str = "pca",
        n_components: int = 2,
        random_state: int | None = 42,
        model: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.method = "custom" if model is not None else self._normalize_method(method)
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = model if model is not None else self._build_model()
        self._fitted_model: Any | None = None

        if model is not None and kwargs:
            raise ValueError(
                "`kwargs` are only used for built-in methods. "
                "For custom models, configure the model before injecting it."
            )

    @staticmethod
    def _normalize_method(method: str) -> str:
        """Normalize reducer names like 't-SNE' -> 'tsne'."""
        return re.sub(r"[^a-z0-9]+", "", str(method).lower())

    def _build_model(self):
        if self.method == "pca":
            from sklearn.decomposition import PCA

            return PCA(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
        if self.method == "tsne":
            try:
                open_tsne = importlib.import_module("openTSNE")
            except Exception as exc:
                raise ImportError(
                    "openTSNE is not installed. Install with `pip install openTSNE`."
                ) from exc
            return open_tsne.TSNE(
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
        target = self._fitted_model if self._fitted_model is not None else self.model
        return callable(getattr(target, "transform", None))

    def _validate_output(self, Z: Any, n_samples: int) -> np.ndarray:
        arr = np.asarray(Z)
        if arr.ndim != 2:
            raise ValueError(
                f"Reducer output must be 2D (n_samples, n_components), got shape {arr.shape}."
            )
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"Reducer output sample mismatch: expected {n_samples}, got {arr.shape[0]}."
            )
        if arr.shape[1] != self.n_components:
            raise ValueError(
                "Reducer output component mismatch: "
                f"expected {self.n_components}, got {arr.shape[1]}."
            )
        return arr.astype(np.float64, copy=False)

    def fit_transform(self, X: Any) -> np.ndarray:
        X_arr = _to_2d_feature_array(X)

        # openTSNE returns a TSNEEmbedding from `fit`, which carries `transform`.
        if self.method == "tsne":
            fit = getattr(self.model, "fit", None)
            if callable(fit):
                embedding = fit(X_arr)
                self._fitted_model = embedding
                return self._validate_output(embedding, n_samples=X_arr.shape[0])

        fit_transform = getattr(self.model, "fit_transform", None)
        if callable(fit_transform):
            out = fit_transform(X_arr)
            self._fitted_model = self.model
            return self._validate_output(out, n_samples=X_arr.shape[0])

        fit = getattr(self.model, "fit", None)
        transform = getattr(self.model, "transform", None)
        if callable(fit) and callable(transform):
            fit(X_arr)
            self._fitted_model = self.model
            return self._validate_output(transform(X_arr), n_samples=X_arr.shape[0])

        raise TypeError(
            "Reducer model must implement either `fit_transform(X)` or both `fit(X)` "
            "and `transform(X)`."
        )

    def transform(self, X: Any) -> np.ndarray:
        target = self._fitted_model if self._fitted_model is not None else self.model
        if not callable(getattr(target, "transform", None)):
            raise NotImplementedError(f"{self.method} does not support transform().")
        X_arr = _to_2d_feature_array(X)
        return self._validate_output(
            target.transform(X_arr),
            n_samples=X_arr.shape[0],
        )


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
    scale: str = "linear",
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
        scale: Display scale, either "linear" (default) or "log"
    """
    arr = to_numpy_image(img)
    if cmap is None:
        cmap = "gray" if arr.ndim == 2 else None
    scale = str(scale).strip().lower()
    if scale not in {"linear", "log"}:
        raise ValueError("`scale` must be either 'linear' or 'log'.")

    if figsize:
        plt.figure(figsize=figsize)
    if scale == "linear":
        plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        from matplotlib.colors import LogNorm

        arr_float = np.asarray(arr, dtype=np.float64)
        positive_vals = arr_float[arr_float > 0]
        if positive_vals.size == 0:
            raise ValueError("Log scale requires at least one positive pixel value.")

        log_vmin = float(vmin) if vmin is not None else float(np.min(positive_vals))
        log_vmax = float(vmax) if vmax is not None else float(np.max(arr_float))
        if log_vmin <= 0:
            raise ValueError("Log scale requires `vmin` to be > 0.")
        if log_vmax <= log_vmin:
            raise ValueError("For log scale, `vmax` must be greater than `vmin`.")

        plt.imshow(arr_float, cmap=cmap, norm=LogNorm(vmin=log_vmin, vmax=log_vmax))
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


def stack_mip(images: Iterable[ImageLike]) -> np.ndarray:
    """
    Pixel-wise maximum over all images.
    Assumes all images are same shape and already usable.
    """
    try:
        from tqdm import tqdm

        it = tqdm(images)
    except Exception:
        it = images

    mip = None
    for img in it:
        arr = to_numpy_image(img).astype(np.float64, copy=False)
        if mip is None:
            mip = arr.copy()
        else:
            np.maximum(mip, arr, out=mip)

    if mip is None:
        raise ValueError("Empty images iterable.")
    return mip
