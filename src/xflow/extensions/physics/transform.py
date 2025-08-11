"""Accelerator physics-specific transform utilities for specialized data preprocessing."""

from typing import Dict, Optional, Tuple

from ...data.transform import TransformRegistry
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
