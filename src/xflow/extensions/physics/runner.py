"""Accelerator physics-specific evaluation hooks."""

from __future__ import annotations

import csv
import importlib
import os
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

from ...evaluation.runner import (
    BaseEvalHook,
    EvalBatch,
    EvalContext,
    _batch_size_from_inputs,
    slice_sample,
)
from ...utils.typing import TensorLike
from .beam import extract_beam_parameters

# Beam parameter CSV contract shared by all hook internals.
BEAM_PARAM_KEYS: Tuple[str, ...] = (
    "h_centroid",
    "v_centroid",
    "h_width",
    "v_width",
)
BEAM_PARAM_SOURCES: Tuple[str, ...] = ("label", "reconstructed")
BEAM_PARAM_INDEX_COLUMNS: Tuple[str, ...] = (
    "batch_index",
    "sample_index",
    "global_sample_index",
)


def _beam_parameter_columns_for_source(
    source: str, methods: Sequence[str]
) -> List[str]:
    columns: List[str] = []
    for method in methods:
        for key in BEAM_PARAM_KEYS:
            columns.append(f"{source}_{method}_{key}")
    return columns


def _beam_parameter_csv_columns(methods: Sequence[str]) -> List[str]:
    columns = list(BEAM_PARAM_INDEX_COLUMNS)
    for source in BEAM_PARAM_SOURCES:
        columns.extend(_beam_parameter_columns_for_source(source, methods))
    return columns


class BeamParamHook(BaseEvalHook):
    """
    Extract per-sample beam parameters from predictions and optional targets.

    extractor: Callable[[TensorLike], dict]
    """

    def __init__(self, extractor: Callable[[TensorLike], dict]) -> None:
        self.extractor = extractor
        self.rows: list = []

    def on_batch(self, ctx: EvalContext, batch: EvalBatch) -> None:
        batch_size = _batch_size_from_inputs(batch.inputs)
        for i in range(batch_size):
            row = {
                "batch_index": batch.index,
                "sample_index": i,
                "pred": self.extractor(slice_sample(batch.predictions, i)),
            }
            if batch.targets is not None:
                row["gt"] = self.extractor(slice_sample(batch.targets, i))
            self.rows.append(row)


class BeamParamCSVHook(BaseEvalHook):
    """
    Save per-sample transverse beam parameters to CSV.

    For each sample, this hook extracts four normalized transverse parameters
    (h_centroid, v_centroid, h_width, v_width) using both "moments" and
    "gaussian" methods from both:
    - reconstructed image (model output)
    - label image (ground truth)

    Total parameter columns per sample: 16.
    """

    PARAMETER_KEYS: Tuple[str, ...] = BEAM_PARAM_KEYS

    def __init__(
        self,
        csv_path: str,
        methods: Sequence[str] = ("moments", "gaussian"),
        append: bool = True,
        flush_every: int = 0,
    ) -> None:
        self.csv_path = csv_path
        self.methods = tuple(methods)
        self.append = append
        self.flush_every = int(flush_every)

        self.rows: List[Dict[str, object]] = []
        self.saved_rows = 0
        self._global_sample_index = 0
        self._has_written = False

        self.columns = _beam_parameter_csv_columns(self.methods)

    def _parameter_columns(self, source: str) -> List[str]:
        return _beam_parameter_columns_for_source(source, self.methods)

    def _read_existing_header(self) -> List[str]:
        if not os.path.exists(self.csv_path):
            return []

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            return next(reader, [])

    def _empty_parameter_values(self, source: str) -> Dict[str, float]:
        return {name: float("nan") for name in self._parameter_columns(source)}

    def _extract_parameter_values(
        self,
        image: TensorLike,
        source: str,
    ) -> Dict[str, float]:
        values = self._empty_parameter_values(source)

        for method in self.methods:
            params = extract_beam_parameters(
                image=image,
                method=method,
                as_array=False,
                normalize=True,
            )

            if params is None:
                continue

            for key in self.PARAMETER_KEYS:
                col = f"{source}_{method}_{key}"
                if key in params:
                    values[col] = float(params[key])

        return values

    def _flush_rows(self) -> None:
        if not self.rows:
            return

        pd = importlib.import_module("pandas")

        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)

        df = pd.DataFrame(self.rows)
        for col in self.columns:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[self.columns]

        if self._has_written:
            mode = "a"
            header = False
        else:
            if self.append and os.path.exists(self.csv_path):
                mode = "a"
                header = False
            else:
                mode = "w"
                header = True

        df.to_csv(self.csv_path, mode=mode, header=header, index=False)

        self.saved_rows += len(df)
        self.rows.clear()
        self._has_written = True

    def on_start(self, ctx: EvalContext) -> None:
        if not self.append and os.path.exists(self.csv_path):
            os.remove(self.csv_path)

        if self.append and os.path.exists(self.csv_path):
            existing_header = self._read_existing_header()
            if existing_header and existing_header != self.columns:
                raise ValueError(
                    "Existing CSV header does not match expected BeamParamCSVHook columns. "
                    "Use append=False to overwrite or provide a different csv_path."
                )

    def on_batch(self, ctx: EvalContext, batch: EvalBatch) -> None:
        if batch.targets is None:
            raise ValueError("BeamParamCSVHook requires batch.targets.")

        batch_size = _batch_size_from_inputs(batch.inputs)

        for i in range(batch_size):
            pred_sample = slice_sample(batch.predictions, i)
            target_sample = slice_sample(batch.targets, i)

            row: Dict[str, object] = {
                "batch_index": int(batch.index),
                "sample_index": int(i),
                "global_sample_index": int(self._global_sample_index),
            }
            row.update(self._extract_parameter_values(target_sample, source="label"))
            row.update(
                self._extract_parameter_values(pred_sample, source="reconstructed")
            )

            self.rows.append(row)
            self._global_sample_index += 1

        if self.flush_every > 0 and len(self.rows) >= self.flush_every:
            self._flush_rows()

    def on_end(self, ctx: EvalContext) -> None:
        self._flush_rows()
        print(f"saved: {self.saved_rows} beam-parameter rows to {self.csv_path}")


class TripletSaverHook(BaseEvalHook):
    """
    Save (input speckle | ground truth | reconstruction) triplet per sample.

    Handles both 3D (single-sample) and 4D (batched) tensors via slice_sample,
    matching the unsqueeze logic in the original script.
    """

    def __init__(self, save_dir: str, cmap: str = "viridis", dpi: int = 80):
        self.save_dir = save_dir
        self.cmap = cmap
        self.dpi = dpi
        self.saved = 0

    def _to_numpy_image(self, value):
        return value.squeeze().numpy()

    def _fixed_range(self, *images) -> tuple[float, float]:
        max_value = max(float(image.max()) for image in images)
        return (0.0, 255.0) if max_value > 1.0 else (0.0, 1.0)

    def _normalized_image(self, image):
        image_min = float(image.min())
        image_max = float(image.max())
        if image_max <= image_min:
            return image * 0.0
        return (image - image_min) / (image_max - image_min)

    def on_start(self, ctx):
        os.makedirs(self.save_dir, exist_ok=True)

    def on_batch(self, ctx, batch):
        # batch tensors are already detached+CPU. Shape: (B, C, H, W) or (C, H, W).
        x, y_true, y_pred = batch.inputs, batch.targets, batch.predictions

        if y_true is None:
            raise ValueError("TripletSaverHook requires batch.targets.")

        # Fall back to a single sample if the dataset yields unbatched tensors.
        n = x.shape[0] if x.ndim == 4 else 1
        single = x.ndim == 3

        for i in range(n):
            xi = x if single else slice_sample(x, i)
            yi = y_true if single else slice_sample(y_true, i)
            pi = y_pred if single else slice_sample(y_pred, i)

            x_img = self._to_numpy_image(xi)
            y_img = self._to_numpy_image(yi)
            p_img = self._to_numpy_image(pi)
            x_norm = self._normalized_image(x_img)
            y_norm = self._normalized_image(y_img)
            p_norm = self._normalized_image(p_img)
            fixed_min, fixed_max = self._fixed_range(x_img, y_img, p_img)

            fig = plt.figure(figsize=(12.5, 6), constrained_layout=True)
            grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.06])
            axes = [
                [fig.add_subplot(grid[0, col]) for col in range(3)],
                [fig.add_subplot(grid[1, col]) for col in range(3)],
            ]
            top_cbar_ax = fig.add_subplot(grid[0, 3])
            bottom_cbar_ax = fig.add_subplot(grid[1, 3])

            top_titles = [
                "input fiber speckle (min-max)",
                "ground truth (original image) (min-max)",
                "reconstructed image (min-max)",
            ]
            bottom_titles = [
                f"input fiber speckle ({fixed_min:.0f}-{fixed_max:.0f})",
                f"ground truth (original image) ({fixed_min:.0f}-{fixed_max:.0f})",
                f"reconstructed image ({fixed_min:.0f}-{fixed_max:.0f})",
            ]
            normalized_images = [x_norm, y_norm, p_norm]
            fixed_images = [x_img, y_img, p_img]

            top_mappable = None
            for ax, title, image in zip(axes[0], top_titles, normalized_images):
                top_mappable = ax.imshow(image, cmap=self.cmap, vmin=0.0, vmax=1.0)
                ax.set_title(title)

            bottom_mappable = None
            for ax, title, image in zip(axes[1], bottom_titles, fixed_images):
                bottom_mappable = ax.imshow(
                    image,
                    cmap=self.cmap,
                    vmin=fixed_min,
                    vmax=fixed_max,
                )
                ax.set_title(title)

            for row in axes:
                for ax in row:
                    ax.axis("off")

            if top_mappable is not None:
                top_colorbar = fig.colorbar(top_mappable, cax=top_cbar_ax)
                top_colorbar.set_label("normalized intensity")

            if bottom_mappable is not None:
                bottom_colorbar = fig.colorbar(bottom_mappable, cax=bottom_cbar_ax)
                bottom_colorbar.set_label("intensity")

            out_path = os.path.join(self.save_dir, f"inference_{self.saved:05d}.png")
            fig.savefig(out_path, dpi=self.dpi)
            plt.close(fig)
            self.saved += 1

    def on_end(self, ctx):
        print(f"saved: {self.saved} images to {self.save_dir}")
