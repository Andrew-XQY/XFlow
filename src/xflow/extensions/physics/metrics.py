"""Pixel-wise image distance metrics for reconstruction quality reporting."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ...utils.typing import TensorLike
from ...utils.visualization import to_numpy_image


def compute_image_distance_metrics(
    reconstructed: TensorLike,
    label: TensorLike,
    data_range: Optional[float] = None,
) -> Dict[str, float]:
    """Compute per-sample PSNR, SSIM, and MAE between reconstruction and label.

    The function converts framework tensors/arrays to 2D NumPy images, computes
    MAE and PSNR directly, and computes SSIM using ``skimage.metrics`` when
    available. The scikit-image import is local so the module remains lightweight.
    If scikit-image is unavailable, a global SSIM fallback is used.

    Args:
        reconstructed: Reconstructed/predicted image tensor.
        label: Ground-truth/label image tensor.
        data_range: Optional explicit intensity range for PSNR/SSIM. If omitted,
            range is inferred from both images.

    Returns:
        Dict with keys: ``psnr``, ``ssim``, ``mae``.
    """
    recon = _to_2d_float_image(reconstructed)
    target = _to_2d_float_image(label)

    if recon.shape != target.shape:
        raise ValueError(
            "reconstructed and label must have identical shapes, "
            f"got {recon.shape} vs {target.shape}"
        )

    if data_range is None:
        max_value = max(float(np.max(recon)), float(np.max(target)))
        min_value = min(float(np.min(recon)), float(np.min(target)))
        dynamic_range = max_value - min_value
        if dynamic_range <= 0.0:
            dynamic_range = 1.0
    else:
        dynamic_range = float(data_range)
        if dynamic_range <= 0.0:
            raise ValueError("data_range must be positive")

    diff = recon - target
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(np.square(diff)))
    psnr = (
        float("inf") if mse == 0.0 else float(10.0 * np.log10((dynamic_range**2) / mse))
    )
    ssim = float(_compute_ssim(recon, target, dynamic_range))

    return {"psnr": psnr, "ssim": ssim, "mae": mae}


def _to_2d_float_image(value: TensorLike) -> np.ndarray:
    """Convert supported tensor/array image input to a 2D float64 array."""
    arr = np.asarray(to_numpy_image(value), dtype=np.float64)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        # Keep the metric API grayscale-friendly when channel data appears.
        arr = np.mean(arr, axis=-1)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeeze, got shape {arr.shape}")

    return arr


def _compute_ssim(recon: np.ndarray, target: np.ndarray, data_range: float) -> float:
    """Compute SSIM, preferring skimage and falling back to a global formula."""
    try:
        from skimage.metrics import structural_similarity  # type: ignore

        return float(structural_similarity(target, recon, data_range=data_range))
    except Exception:
        # Fallback global SSIM approximation if skimage is unavailable.
        mu_x = float(np.mean(recon))
        mu_y = float(np.mean(target))
        sigma_x = float(np.var(recon))
        sigma_y = float(np.var(target))
        covariance = float(np.mean((recon - mu_x) * (target - mu_y)))

        c1 = float((0.01 * data_range) ** 2)
        c2 = float((0.03 * data_range) ** 2)

        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

        if denominator == 0.0:
            return 1.0 if np.allclose(recon, target) else 0.0

        return float(numerator / denominator)
