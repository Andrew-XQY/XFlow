"""Accelerator physics-specific transform utilities for specialized data preprocessing."""
from typing import Tuple, Dict, Any
from ...data.transform import TransformRegistry
from ...utils.typing import TensorLike
from .beam import extract_beam_parameters


@TransformRegistry.register("split_width_with_analysis")
def split_width_with_analysis(image: TensorLike, swap: bool = False) -> Tuple[TensorLike, Dict[str, float]]:
    """Split image at width midpoint and analyze left half for parameters.
    
    Args:
        image: Input image tensor
        swap: If True, swap left and right halves before processing
        
    Returns:
        Tuple of (right_half_image, parameters_dict)
        where parameters_dict contains 4 float values
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
    parameters = extract_beam_parameters(left_half)
    
    return right_half, parameters