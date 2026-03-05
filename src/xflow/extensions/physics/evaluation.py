"""Core metrics for evaluation."""

from typing import Tuple

import numpy as np


def get_centroid(tensor, method: str = "first_moment") -> Tuple[float, float]:
    """Calculate the centroid (center point) of an image.

    Framework-agnostic implementation that works with PyTorch tensors, TensorFlow tensors,
    or NumPy arrays.

    Args:
        tensor: Input tensor/array (grayscale image recommended).
                Can be PyTorch tensor, TensorFlow tensor, or NumPy array.
        method: Either "first_moment" (weighted average) or "gaussian_mean" (fit Gaussian to projection)

    Returns:
        Tuple of (cx, cy) centroid coordinates in pixel indices

    Examples:
        >>> # Works with PyTorch
        >>> image = torch.randn(256, 256)
        >>> cx, cy = get_centroid(image, method="first_moment")

        >>> # Works with TensorFlow
        >>> image = tf.random.normal([256, 256])
        >>> cx, cy = get_centroid(image, method="gaussian_mean")

        >>> # Works with NumPy
        >>> image = np.random.randn(256, 256)
        >>> cx, cy = get_centroid(image)
    """
    # Convert to NumPy for framework-agnostic processing
    if hasattr(tensor, "numpy"):  # TensorFlow tensor or PyTorch CPU tensor
        try:
            arr = tensor.numpy()
        except (TypeError, RuntimeError):
            # PyTorch tensor on GPU - need to move to CPU first
            arr = tensor.cpu().numpy()
    elif hasattr(tensor, "__array__"):  # NumPy array or array-like
        arr = np.asarray(tensor)
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    # Ensure 2D array (squeeze if needed) for centroid calculation
    if arr.ndim > 2:
        # Take mean across channels if present
        while arr.ndim > 2:
            # Assume last dimension is channels for most formats
            if arr.shape[0] <= 4:  # Likely (C, H, W) format
                arr = arr.mean(axis=0)
            else:  # Likely (H, W, C) format
                arr = arr.mean(axis=-1)

    # Get projections (histograms) by summing along axes
    # axis=0 sums along height → width projection for x-coordinate
    # axis=1 sums along width → height projection for y-coordinate
    x_hist = arr.sum(axis=0)  # Width projection
    y_hist = arr.sum(axis=1)  # Height projection

    if method == "first_moment":
        # First moment centroid: weighted average
        x_idx = np.arange(arr.shape[1], dtype=np.float32)  # Width indices
        y_idx = np.arange(arr.shape[0], dtype=np.float32)  # Height indices

        x_sum = np.sum(x_hist)
        y_sum = np.sum(y_hist)

        cx = np.sum(x_hist * x_idx) / x_sum if x_sum > 0 else np.nan
        cy = np.sum(y_hist * y_idx) / y_sum if y_sum > 0 else np.nan

    elif method == "gaussian_mean":
        # Gaussian fitting: fit Gaussian to each projection histogram
        x_idx = np.arange(arr.shape[1], dtype=np.float32)  # Width indices
        y_idx = np.arange(arr.shape[0], dtype=np.float32)  # Height indices

        # Normalize projections
        x_sum = np.sum(x_hist)
        y_sum = np.sum(y_hist)

        if x_sum > 0 and y_sum > 0:
            x_hist_norm = x_hist / x_sum
            y_hist_norm = y_hist / y_sum

            # Moment matching: mean is the first moment
            cx = np.sum(x_hist_norm * x_idx)
            cy = np.sum(y_hist_norm * y_idx)
        else:
            cx = np.nan
            cy = np.nan

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'first_moment' or 'gaussian_mean'"
        )

    return float(cx), float(cy)
