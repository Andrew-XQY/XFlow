"""Accelerator physics-specific transform utilities for specialized data preprocessing."""

from typing import Dict, Optional, Sequence, Tuple

from ...data.transform import TransformRegistry
from ...evaluation.metrics import get_centroid
from ...utils.typing import TensorLike
from .beam import extract_beam_parameters

# Conditionally import TensorFlow version if available
try:
    from .beam import extract_beam_parameters_tf

    TF_AVAILABLE = True
except ImportError:
    extract_beam_parameters_tf = None
    TF_AVAILABLE = False


@TransformRegistry.register("split_width_with_analysis")
def split_width_with_analysis(
    image: TensorLike,
    swap: bool = False,
    return_all: bool = False,
    method: str = "moments",
) -> Optional[Tuple[TensorLike, Dict[str, float], TensorLike]]:
    """Split image at width midpoint and analyze left half for parameters.

    Args:
        image: Input image tensor
        swap: If True, swap left and right halves before processing
        return_all: If True, return (right_half, parameters, left_half), otherwise (right_half, parameters)
        method: Method to use for beam parameter extraction

    Returns:
        Tuple of (right_half_image, parameters_dict) or (right_half_image, parameters_dict, left_half_image)
        if return_all is True, or None if extraction fails or parameters are unreasonable
    """
    import numpy as np

    # Convert to numpy if needed
    if hasattr(image, "numpy"):
        image_np = image.numpy()
    else:
        image_np = np.asarray(image)

    # Split image at width midpoint
    width = image_np.shape[1]
    mid_point = width // 2
    left_half = image_np[:, :mid_point]
    right_half = image_np[:, mid_point:]

    if swap:
        left_half, right_half = right_half, left_half

    # Prepare left half for analysis: remove singleton channel dim if present
    left_for_analysis = left_half
    if left_for_analysis.ndim == 3 and left_for_analysis.shape[-1] == 1:
        left_for_analysis = np.squeeze(left_for_analysis, axis=-1)

    # Analyze left half to extract parameters
    parameters = extract_beam_parameters(left_for_analysis, method=method)

    # Check if parameter extraction failed
    if parameters is None:
        return None

    if return_all:
        return right_half, parameters, left_half  # input, label, callback plot
    else:
        return right_half, parameters  # input, label


@TransformRegistry.register("check_centroid")
def check_centroid(
    tensor: TensorLike,
    rect: Sequence[Tuple[int, int]],
    method: str = "first_moment",
    on_fail: str = "return",
) -> Optional[object]:
    """Check if the centroid of an image falls within a specified rectangular region.

    Args:
        tensor: Input tensor/array (grayscale image recommended).
                Can be PyTorch tensor, TensorFlow tensor, or NumPy array.
        rect: Rectangular region defined by two corner points as [(x1, y1), (x2, y2)]
              or ((x1, y1), (x2, y2)) where (x1, y1) is top-left and (x2, y2) is bottom-right.
              Can be any iterable of two points.
        method: One of:
            - "first_moment": compute centroid via intensity-weighted first moment
            - "gaussian_mean": fit Gaussian to projections and use the mean
            - "beam_params": use extract_beam_parameters() to get
              [h_centroid, v_centroid, h_width, v_width]
        on_fail: "return" to return None when centroid lies outside the region (default),
            "raise" to raise a ValueError instead of returning None.

    Returns:
        The original unchanged tensor if centroid is within the rectangle.
        When on_fail="return", None is returned if the centroid is outside the rectangle
        or cannot be computed. When on_fail="raise", a ValueError is raised instead.

    Examples:
        >>> # Check if centroid is within region (50, 50) to (200, 200)
        >>> image = torch.randn(256, 256)
        >>> result = check_centroid(image, [(50, 50), (200, 200)])
        >>> if result is not None:
        ...     print("Centroid is within region")

        >>> # Works with any tensor type
        >>> image = np.random.randn(256, 256)
        >>> result = check_centroid(image, ((100, 100), (150, 150)))
    """
    import numpy as np

    normalized_on_fail = on_fail.lower()
    if normalized_on_fail not in {"return", "raise"}:
        raise ValueError("on_fail must be either 'return' or 'raise'")

    # ------------------------------------------------------------------
    # Get centroid coordinates
    # ------------------------------------------------------------------
    if method == "beam_params":
        # Expected signature:
        #   extract_beam_parameters(tensor) -> [h_centroid, v_centroid, h_width, v_width]
        # or None if the image is considered invalid.
        params = extract_beam_parameters(tensor, normalize=False)

        # As requested: if the extractor returns None, immediately raise.
        if params is None:
            raise ValueError("extract_beam_parameters returned None (invalid image)")

        if len(params) != 4:
            raise ValueError(
                "extract_beam_parameters must return a sequence of four values: "
                "[h_centroid, v_centroid, h_width, v_width]"
            )

        h_centroid, v_centroid, h_width, v_width = params
        cx, cy = float(h_centroid), float(v_centroid)
    else:
        # Fallback to the original centroid logic.
        cx, cy = get_centroid(tensor, method=method)

    # Handle NaN case (empty or invalid centroid)
    if np.isnan(cx) or np.isnan(cy):
        if normalized_on_fail == "raise":
            raise ValueError("Centroid could not be computed (NaN values)")
        return None

    # ------------------------------------------------------------------
    # Rectangle bounds check (only uses h_centroid, v_centroid)
    # ------------------------------------------------------------------
    point1, point2 = rect
    x1, y1 = point1
    x2, y2 = point2

    # Ensure coordinates are in correct order (top-left to bottom-right)
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Check if centroid is within rectangle
    if x_min <= cx <= x_max and y_min <= cy <= y_max:
        return tensor
    else:
        if normalized_on_fail == "raise":
            raise ValueError(f"Centroid ({cx}, {cy}) outside rectangle bounds")
        return None


@TransformRegistry.register("torch_split_width_with_analysis")
def torch_split_width_with_analysis(
    tensor: TensorLike,
    swap: bool = False,
    width_dim: int = -1,
    return_all: bool = False,
    method: str = "moments",
):
    try:
        import torch

        width = tensor.shape[width_dim]
        mid_point = width // 2

        left_half = torch.split(tensor, mid_point, dim=width_dim)[0]
        right_half = torch.split(tensor, mid_point, dim=width_dim)[1]

        parameters = extract_beam_parameters(left_half, method=method)

        if parameters is None:
            return None
        if not return_all:
            return left_half, parameters
        if swap:
            return right_half, parameters, left_half
        return left_half, parameters, right_half

    except ImportError:
        raise RuntimeError("Split or beam parameter extraction failed")


@TransformRegistry.register("tf_split_width_with_analysis")
def tf_split_width_with_analysis(
    image: TensorLike,
    swap: bool = False,
    return_all: bool = False,
    method: str = "moments",
) -> Optional[Tuple[TensorLike, Dict[str, float], TensorLike]]:
    """Split image at width midpoint and analyze left half for parameters.

    Args:
        image: Input image tensor
        swap: If True, swap left and right halves before processing
        return_all: If True, return (right_half, parameters, left_half), otherwise (right_half, parameters)
        method: Method to use for beam parameter extraction

    Returns:
        Tuple of (right_half_image, parameters_dict) or (right_half_image, parameters_dict, left_half_image)
        if return_all is True, or None if extraction fails or parameters are unreasonable
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for tf_split_width_with_analysis but not available"
        )

    import tensorflow as tf

    # Split image at width midpoint
    width = tf.shape(image)[1]
    mid_point = width // 2
    left_half = image[:, :mid_point]
    right_half = image[:, mid_point:]

    if swap:
        left_half, right_half = right_half, left_half

    # Analyze left half to extract parameters
    try:
        tf_params = extract_beam_parameters_tf(left_half, method=method)
        # Convert TensorFlow tensor result to dictionary
        parameters = {
            "h_centroid": float(tf_params[0]),
            "h_width": float(tf_params[1]),
            "v_centroid": float(tf_params[2]),
            "v_width": float(tf_params[3]),
        }

    except Exception:
        parameters = None

    # Check if parameter extraction failed
    if parameters is None:
        return None

    if return_all:
        return right_half, parameters, left_half  # input, label, callback plot
    else:
        return right_half, parameters  # input, label
