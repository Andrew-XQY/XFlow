"""Beam diagnostics utilities for transverse beam parameter extraction."""
import tensorflow as tf
from typing import Dict, Tuple
from ...utils.typing import TensorLike


# TensorFlow native implementation for beam parameter extraction
@tf.function
def extract_beam_parameters_tf(image: TensorLike) -> tf.Tensor:
    """Extract normalized transverse beam parameters (TF native version).
    
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
    
    # Fit Gaussian to projections
    h_centroid, h_width = fit_gaussian_1d_tf(h_projection)
    v_centroid, v_width = fit_gaussian_1d_tf(v_projection)
    
    # Normalize to 0-1 range using image dimensions
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    
    h_centroid_norm = h_centroid / (width - 1.0)
    v_centroid_norm = v_centroid / (height - 1.0)
    h_width_norm = h_width / width
    v_width_norm = v_width / height
    
    return tf.stack([h_centroid_norm, h_width_norm, v_centroid_norm, v_width_norm])

@tf.function
def fit_gaussian_1d_tf(projection: TensorLike) -> tuple[tf.Tensor, tf.Tensor]:
    """TF-native Gaussian fitting."""
    total_intensity = tf.reduce_sum(projection)
    weights = projection / total_intensity
    
    length = tf.cast(tf.shape(projection)[0], tf.float32)
    coords = tf.range(length, dtype=tf.float32)
    
    mean = tf.reduce_sum(coords * weights)
    variance = tf.reduce_sum(weights * tf.square(coords - mean))
    std = tf.sqrt(variance)
    
    return mean, std



# General NumPy implementation for beam parameter extraction

def extract_beam_parameters(image: TensorLike) -> Dict[str, float]:
    """Extract normalized transverse beam parameters from beam distribution image.
    
    Args:
        image: 2D tensor representing transverse beam distribution
        
    Returns:
        Dictionary containing normalized beam parameters (0-1 range), or empty dict if extraction fails:
        - "h_centroid": horizontal beam centroid (normalized)
        - "h_width": horizontal beam width (normalized)
        - "v_centroid": vertical beam centroid (normalized)
        - "v_width": vertical beam width (normalized)
    """
    try:
        # Background subtraction - subtract minimum pixel value
        min_val = tf.reduce_min(image)
        image_bg_sub = image - min_val
        
        # Calculate horizontal and vertical projections
        h_projection = tf.reduce_sum(image_bg_sub, axis=0)  # Sum along vertical axis
        v_projection = tf.reduce_sum(image_bg_sub, axis=1)  # Sum along horizontal axis
        
        # Fit Gaussian to projections
        h_centroid, h_width = fit_gaussian_1d(h_projection)
        v_centroid, v_width = fit_gaussian_1d(v_projection)
        
        # Raw parameters
        raw_params = {
            "h_centroid": float(h_centroid),
            "h_width": float(h_width),
            "v_centroid": float(v_centroid), 
            "v_width": float(v_width)
        }
        
        # Normalize to 0-1 range
        image_shape = (tf.shape(image)[0].numpy(), tf.shape(image)[1].numpy())
        normalized_params = normalize_beam_parameters(raw_params, image_shape)
        
        return normalized_params
        
    except Exception:
        return {}


def fit_gaussian_1d(projection: TensorLike) -> tuple[tf.Tensor, tf.Tensor]:
    """Fit Gaussian to 1D projection and return mean and std.
    
    Args:
        projection: 1D tensor representing beam projection
        
    Returns:
        Tuple of (mean, std) from Gaussian fit
    """
    # Normalize projection to get probability distribution
    total_intensity = tf.reduce_sum(projection)
    weights = projection / total_intensity
    
    # Create coordinate array
    length = tf.cast(tf.shape(projection)[0], tf.float32)
    coords = tf.range(length, dtype=tf.float32)
    
    # Calculate weighted mean (centroid)
    mean = tf.reduce_sum(coords * weights)
    
    # Calculate weighted standard deviation (width)
    variance = tf.reduce_sum(weights * tf.square(coords - mean))
    std = tf.sqrt(variance)
    
    return mean, std



def normalize_to_range(value: float, min_val: float, max_val: float, target_min: float = 0.0, target_max: float = 1.0) -> float:
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


def normalize_beam_parameters(params: Dict[str, float], image_shape: Tuple[int, int]) -> Dict[str, float]:
    """Normalize beam parameters to 0-1 range based on image dimensions.
    
    Args:
        params: Raw beam parameters dictionary
        image_shape: (height, width) of the original image
        
    Returns:
        Dictionary with normalized parameters (0-1 range)
    """
    if not params:
        return {}
    
    height, width = image_shape
    
    normalized = {}
    
    # Normalize centroids by image dimensions
    if "h_centroid" in params:
        normalized["h_centroid"] = normalize_to_range(params["h_centroid"], 0, width - 1)
    
    if "v_centroid" in params:
        normalized["v_centroid"] = normalize_to_range(params["v_centroid"], 0, height - 1)
    
    # Normalize widths by image dimensions (width can be 0 to full dimension)
    if "h_width" in params:
        normalized["h_width"] = normalize_to_range(params["h_width"], 0, width)
    
    if "v_width" in params:
        normalized["v_width"] = normalize_to_range(params["v_width"], 0, height)
    
    return normalized