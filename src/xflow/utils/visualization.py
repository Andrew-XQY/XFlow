import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_image(img):
    """
    Plot an image from various formats (PIL Image, OpenCV numpy array, or raw numpy array).
    Displays the image with X,Y pixel indices on axes and a colorbar indicating pixel values.
    """
    # Convert PIL Image to numpy array
    if isinstance(img, Image.Image):
        arr = np.array(img)
    # Accept numpy arrays directly (OpenCV images are numpy arrays too)
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        # Fallback: try converting to array
        try:
            arr = np.array(img)
        except Exception:
            raise TypeError(f"Unsupported image type: {type(img)}")

    # Determine if grayscale or color
    cmap = 'gray' if arr.ndim == 2 else None

    # Display image
    plt.imshow(arr, cmap=cmap)
    plt.xlabel('X (pixel index)')
    plt.ylabel('Y (pixel index)')
    plt.colorbar(label='Pixel value')
    plt.tight_layout()
    plt.show()
