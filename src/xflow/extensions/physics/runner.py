"""Accelerator physics-specific evaluation hooks."""

from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import torch

from ...evaluation.runner import (
    BaseEvalHook,
    EvalBatch,
    EvalContext,
    _batch_size_from_inputs,
    slice_sample,
)


class BeamParamHook(BaseEvalHook):
    """
    Extract per-sample beam parameters from predictions and optional targets.

    extractor: Callable[[torch.Tensor], dict]
    """

    def __init__(self, extractor: Callable[[torch.Tensor], dict]) -> None:
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

    def on_start(self, ctx):
        os.makedirs(self.save_dir, exist_ok=True)

    def on_batch(self, ctx, batch):
        # batch tensors are already detached+CPU. Shape: (B, C, H, W) or (C, H, W).
        x, y_true, y_pred = batch.inputs, batch.targets, batch.predictions

        # Fall back to a single sample if the dataset yields unbatched tensors.
        n = x.shape[0] if x.ndim == 4 else 1
        single = x.ndim == 3

        for i in range(n):
            xi = x if single else slice_sample(x, i)
            yi = y_true if single else slice_sample(y_true, i)
            pi = y_pred if single else slice_sample(y_pred, i)

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(xi.squeeze().numpy(), cmap=self.cmap)
            axes[0].set_title("input fiber speckle")
            axes[1].imshow(yi.squeeze().numpy(), cmap=self.cmap)
            axes[1].set_title("ground truth (original image)")
            axes[2].imshow(pi.squeeze().numpy(), cmap=self.cmap)
            axes[2].set_title("reconstructed image")
            for ax in axes:
                ax.axis("off")

            out_path = os.path.join(self.save_dir, f"inference_{self.saved:05d}.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=self.dpi)
            plt.close(fig)
            self.saved += 1

    def on_end(self, ctx):
        print(f"saved: {self.saved} images to {self.save_dir}")
