"""Accelerator physics-specific evaluation hooks."""

from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt

from ...evaluation.runner import (
    BaseEvalHook,
    EvalBatch,
    EvalContext,
    _batch_size_from_inputs,
    slice_sample,
)
from ...utils.typing import TensorLike


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
