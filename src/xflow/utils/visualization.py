import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_image(img):
    """
    Plot an image from PIL, numpy, TensorFlow tensor, or PyTorch tensor.
    Automatically converts to 2D numpy array and displays with matplotlib.
    """
    # Convert to numpy
    if hasattr(img, 'numpy'):  # TensorFlow tensor
        arr = img.numpy()
    elif hasattr(img, 'detach'):  # PyTorch tensor  
        arr = img.detach().cpu().numpy()
    elif isinstance(img, Image.Image):  # PIL
        arr = np.array(img)
    elif isinstance(img, np.ndarray):  # Already numpy
        arr = img
    else:
        arr = np.array(img)  # Fallback
    
    # Squeeze all single dimensions to get [H, W]
    arr = np.squeeze(arr)
    
    # Take first image if batch remains
    if arr.ndim > 2:
        arr = arr[0] if arr.ndim == 3 else arr[0, 0]
    
    # Determine if grayscale or color
    cmap = 'gray' if arr.ndim == 2 else None
    
    # Plot
    plt.imshow(arr, cmap=cmap)
    plt.xlabel('X (pixel index)')
    plt.ylabel('Y (pixel index)')
    plt.colorbar(label='Pixel value')
    plt.tight_layout()
    plt.show()
