"""Accelerator physics-specific transform utilities for specialized data preprocessing."""
from typing import Tuple, Dict, Any, Optional
from ...data.transform import TransformRegistry
from ...utils.typing import TensorLike
from .beam import extract_beam_parameters, extract_beam_parameters_tf


@TransformRegistry.register("split_width_with_analysis")
def split_width_with_analysis(image: TensorLike, swap: bool = False) -> Optional[Tuple[TensorLike, Dict[str, float]]]:
    """Split image at width midpoint and analyze left half for parameters.
    
    Args:
        image: Input image tensor
        swap: If True, swap left and right halves before processing
        
    Returns:
        Tuple of (right_half_image, parameters_dict) or None if extraction fails or parameters are unreasonable
    """
    import numpy as np
    
    # Convert to numpy if needed
    if hasattr(image, 'numpy'):
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
    
    # Analyze left half to extract parameters
    parameters = extract_beam_parameters(left_half)
    
    # Check if parameter extraction failed
    if parameters is None:
        return None
    
    return right_half, parameters


@TransformRegistry.register("tf_split_width_with_analysis")
def tf_split_width_with_analysis(image: TensorLike, swap: bool = False) -> Optional[Tuple[TensorLike, Dict[str, float]]]:
    """Split image at width midpoint and analyze left half for parameters.
    
    Args:
        image: Input image tensor
        swap: If True, swap left and right halves before processing
        
    Returns:
        Tuple of (right_half_image, parameters_dict) or None if extraction fails or parameters are unreasonable
    """
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
        tf_params = extract_beam_parameters_tf(left_half)
        # Convert TensorFlow tensor result to dictionary
        parameters = {
            "h_centroid": float(tf_params[0]),
            "h_width": float(tf_params[1]),
            "v_centroid": float(tf_params[2]),
            "v_width": float(tf_params[3])
        }
            
    except Exception:
        parameters = None
    
    # Check if parameter extraction failed
    if parameters is None:
        return None
    
    return right_half, parameters