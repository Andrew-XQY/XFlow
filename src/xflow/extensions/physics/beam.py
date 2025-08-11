"""Beam diagnostics utilities for transverse beam parameter extraction."""

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from ...utils.typing import TensorLike

# Try to import TensorFlow with fallback
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    if not TYPE_CHECKING:
        tf = None
    TF_AVAILABLE = False

# Type checking imports
if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)


# General NumPy implementation for beam parameter extraction
def extract_beam_parameters(
    image: TensorLike, method: str = "gaussian", as_array: bool = True
) -> Optional[Dict[str, float]]:
    """Extract normalized transverse beam parameters from beam distribution image.

    Args:
        image: 2D tensor representing transverse beam distribution
        method: Analysis method to use ("moments", "gaussian", etc.)
        as_array: If True, return parameters as a list [h_centroid, h_width, v_centroid, v_width] instead of dict.

    Returns:
        Dictionary or list of normalized beam parameters (0-1 range), or None if extraction fails.
    """
    try:
        import numpy as np

        # Convert to numpy if needed
        if hasattr(image, "numpy"):
            image_np = image.numpy()
        else:
            image_np = np.asarray(image)

        # Background subtraction - subtract minimum pixel value
        min_val = np.min(image_np)
        image_bg_sub = image_np - min_val

        # Calculate horizontal and vertical projections
        h_projection = np.sum(image_bg_sub, axis=0)  # Sum along vertical axis
        v_projection = np.sum(image_bg_sub, axis=1)  # Sum along horizontal axis

        analysis_func = _get_beam_analysis_function(method)
        h_centroid, h_width = analysis_func(h_projection)
        v_centroid, v_width = analysis_func(v_projection)

        raw_params = {
            "h_centroid": float(h_centroid),
            "h_width": float(h_width),
            "v_centroid": float(v_centroid),
            "v_width": float(v_width),
        }

        image_shape = image_np.shape
        normalized_params = normalize_beam_parameters(raw_params, image_shape)

        if normalized_params is None or not is_reasonable_beam_sample(
            normalized_params
        ):
            return None

        if as_array:
            keys = ["h_centroid", "v_centroid", "h_width", "v_width"]
            return [normalized_params[k] for k in keys]
        else:
            return normalized_params

    except Exception as e:
        import traceback

        logger.warning(f"Beam parameter extraction failed: {e}")
        logger.warning(
            f"Image type: {type(image)}, shape: {getattr(image, 'shape', 'unknown')}"
        )
        logger.warning(f"Full traceback:\n{traceback.format_exc()}")
        return None


# TensorFlow native implementation for beam parameter extraction
if TF_AVAILABLE:

    @tf.function
    def extract_beam_parameters_tf(
        image: TensorLike, method: str = "gaussian"
    ) -> tf.Tensor:
        """Extract normalized transverse beam parameters (TF native version).

        Args:
            image: 2D tensor representing transverse beam distribution
            method: Analysis method to use ("moments", "gaussian", etc.)

        Returns:
            tf.Tensor of shape [4] containing [h_centroid, h_width, v_centroid, v_width]
            All values normalized to 0-1 range.
        """
        # Background subtraction
        min_val = tf.reduce_min(image)
        image_bg_sub = image - min_val

        # Calculate projections
        h_projection = tf.reduce_sum(image_bg_sub, axis=0)
        v_projection = tf.reduce_sum(image_bg_sub, axis=1)

        # Get the analysis method and calculate beam moments from projections
        analysis_func = _get_beam_analysis_function_tf(method)
        h_centroid, h_width = analysis_func(h_projection)
        v_centroid, v_width = analysis_func(v_projection)

        # Normalize to 0-1 range using image dimensions
        height = tf.cast(tf.shape(image)[0], tf.float32)
        width = tf.cast(tf.shape(image)[1], tf.float32)

        h_centroid_norm = h_centroid / (width - 1.0)
        v_centroid_norm = v_centroid / (height - 1.0)
        h_width_norm = h_width / width
        v_width_norm = v_width / height

        return tf.stack([h_centroid_norm, h_width_norm, v_centroid_norm, v_width_norm])

    @tf.function
    def calculate_beam_moments_1d_tf(
        projection: TensorLike,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculate first and second moments of 1D beam distribution using TensorFlow.

        TensorFlow-native implementation for computing beam centroid and standard deviation
        from intensity distribution using statistical moment analysis.

        Args:
            projection: 1D tensor representing beam projection

        Returns:
            Tuple of (centroid, std_dev) - beam position and width as TF tensors
        """
        total_intensity = tf.reduce_sum(projection)
        weights = projection / total_intensity

        length = tf.cast(tf.shape(projection)[0], tf.float32)
        coords = tf.range(length, dtype=tf.float32)

        mean = tf.reduce_sum(coords * weights)
        variance = tf.reduce_sum(weights * tf.square(coords - mean))
        std = tf.sqrt(variance)

        return mean, std

    def _get_beam_analysis_function_tf(method: str):
        """Get the appropriate beam analysis function for TensorFlow."""
        if method == "moments":
            return calculate_beam_moments_1d_tf
        # elif method == "gaussian":
        #     return calculate_beam_gaussian_1d_tf  # Not implemented yet
        else:
            return calculate_beam_moments_1d_tf


# Helper function to get the analysis function based on method


def _get_beam_analysis_function(method: str):
    """Get the appropriate beam analysis function for NumPy."""
    if method == "moments":
        return calculate_beam_moments_1d
    elif method == "gaussian":
        return calculate_beam_gaussian_1d
    else:
        logger.warning(f"Unknown method '{method}', falling back to 'moments'")
        return calculate_beam_moments_1d


def calculate_beam_gaussian_1d(projection: TensorLike) -> Tuple[float, float]:
    """Calculate beam parameters using Gaussian curve fitting - accelerator physics standard."""
    import numpy as np
    from scipy.optimize import curve_fit

    # Convert to numpy if needed
    if hasattr(projection, "numpy"):
        proj_np = projection.numpy()
    else:
        proj_np = np.asarray(projection)

    # Check for zero intensity
    if np.sum(proj_np) == 0:
        return 0.0, 0.0

    # Create coordinate array
    length = len(proj_np)
    coords = np.arange(length, dtype=np.float32)

    # Define Gaussian function
    def gaussian(x, amplitude, mean, sigma, offset):
        return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset

    # Gold standard initial parameter estimation
    peak_idx = np.argmax(proj_np)
    amplitude_est = np.max(proj_np)
    mean_est = float(peak_idx)
    offset_est = np.min(proj_np)

    # Estimate sigma from FWHM (standard practice in accelerator physics)
    half_max = (amplitude_est + offset_est) / 2.0
    indices = np.where(proj_np >= half_max)[0]
    if len(indices) > 1:
        fwhm = indices[-1] - indices[0]
        sigma_est = fwhm / 2.355  # Convert FWHM to sigma
    else:
        fwhm = length / 5.0  # Conservative fallback estimate
        sigma_est = fwhm / 2.355

    # Ensure realistic sigma bounds
    sigma_est = max(0.5, min(sigma_est, length / 3.0))

    initial_guess = [amplitude_est - offset_est, mean_est, sigma_est, offset_est]

    # Adaptive bounds based on data characteristics
    min_sigma = max(0.2, length * 0.001)  # 0.2 pixels or 0.1% of detector
    max_sigma = min(length * 0.6, fwhm * 2)  # 60% detector or 2x FWHM estimate

    try:
        # Perform Gaussian curve fitting with physics-realistic bounds
        popt, _ = curve_fit(
            gaussian,
            coords,
            proj_np,
            p0=initial_guess,
            maxfev=1500,
            bounds=(
                [0, 0, min_sigma, 0],  # Lower bounds
                [np.inf, length - 1, max_sigma, np.inf],  # Upper bounds
            ),
        )

        amplitude, mean, sigma, offset = popt

        # Physics validation: reasonable parameters?
        if not (0.5 <= mean <= length - 1.5) or sigma <= min_sigma or sigma > max_sigma:
            raise ValueError("Unrealistic fit parameters")

        return float(mean), float(sigma)

    except Exception:
        # Fallback to moments - reuse existing implementation
        return calculate_beam_moments_1d(projection)


def calculate_beam_moments_1d(projection: TensorLike) -> Tuple[float, float]:
    """Calculate first and second moments of 1D beam distribution.

    Computes the first moment (centroid) and second central moment (standard deviation)
    of a beam intensity profile using statistical moment analysis.

    Args:
        projection: 1D tensor/array representing beam projection

    Returns:
        Tuple of (centroid, std_dev) - beam position and width in pixels
    """
    import numpy as np

    # Convert to numpy if needed
    if hasattr(projection, "numpy"):
        proj_np = projection.numpy()
    else:
        proj_np = np.asarray(projection)

    # Normalize projection to get probability distribution
    total_intensity = np.sum(proj_np)
    if total_intensity == 0:
        return 0.0, 0.0

    weights = proj_np / total_intensity

    # Create coordinate array
    length = len(proj_np)
    coords = np.arange(length, dtype=np.float32)

    # Calculate weighted mean (centroid)
    mean = np.sum(coords * weights)

    # Calculate weighted standard deviation (width)
    variance = np.sum(weights * np.square(coords - mean))
    std = np.sqrt(variance)

    return float(mean), float(std)


def normalize_to_range(
    value: float,
    min_val: float,
    max_val: float,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> float:
    """Normalize a value from one range to another.

    Args:
        value: Value to normalize
        min_val: Minimum of original range
        max_val: Maximum of original range
        target_min: Minimum of target range (default: 0.0)
        target_max: Maximum of target range (default: 1.0)

    Returns:
        Normalized value in target range
    """
    if max_val == min_val:
        return target_min

    # Normalize to 0-1, then scale to target range
    normalized = (value - min_val) / (max_val - min_val)
    return target_min + normalized * (target_max - target_min)


def normalize_beam_parameters(
    params: Dict[str, float], image_shape: Tuple[int, int]
) -> Optional[Dict[str, float]]:
    """Normalize beam parameters to 0-1 range based on image dimensions."""
    if not params:
        return None

    # Handle different image shapes (2D, 3D with channels, etc.)
    if len(image_shape) == 2:
        height, width = image_shape
    elif len(image_shape) == 3:
        height, width, _ = image_shape  # Ignore channel dimension
    elif len(image_shape) == 4:
        _, height, width, _ = image_shape  # Handle batch + channel dimensions
    else:
        logger.warning(f"Unexpected image shape: {image_shape}")
        return None

    normalized = {}

    # Normalize centroids by image dimensions
    if "h_centroid" in params:
        normalized["h_centroid"] = normalize_to_range(
            params["h_centroid"], 0, width - 1
        )

    if "v_centroid" in params:
        normalized["v_centroid"] = normalize_to_range(
            params["v_centroid"], 0, height - 1
        )

    # Normalize widths by image dimensions (width can be 0 to full dimension)
    if "h_width" in params:
        normalized["h_width"] = normalize_to_range(params["h_width"], 0, width)

    if "v_width" in params:
        normalized["v_width"] = normalize_to_range(params["v_width"], 0, height)

    return normalized


def is_reasonable_beam_sample(parameters: Dict[str, float]) -> bool:
    """Validate if beam parameters represent a reasonable sample.

    Args:
        parameters: Dictionary of normalized beam parameters (0-1 range)

    Returns:
        True if parameters are reasonable, False otherwise
    """
    if not parameters:
        return False

    required_keys = {"h_centroid", "h_width", "v_centroid", "v_width"}
    if not required_keys.issubset(parameters.keys()):
        return False

    # Check if all values are in valid range [0, 1]
    for key, value in parameters.items():
        if not (0.0 <= value <= 1.0):
            return False

    # Check if beam widths are reasonable (not too small, indicating noise)
    min_width_threshold = 0.01  # 1% of image dimension
    if (
        parameters["h_width"] < min_width_threshold
        or parameters["v_width"] < min_width_threshold
    ):
        return False

    # Check if centroids are not too close to edges (indicating clipping)
    edge_margin = 0.05  # 5% margin from edges
    if (
        parameters["h_centroid"] < edge_margin
        or parameters["h_centroid"] > (1.0 - edge_margin)
        or parameters["v_centroid"] < edge_margin
        or parameters["v_centroid"] > (1.0 - edge_margin)
    ):
        return False

    return True
