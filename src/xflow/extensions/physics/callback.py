"""Accelerator physics-specific callback utilities for beam diagnostics visualization."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tensorflow as tf
from ...trainers.callback import CallbackRegistry


@CallbackRegistry.register("centroid_ellipse_callback")
def make_centroid_ellipse_callback(sample_data):
    """Create a callback that visualizes beam centroid and width ellipses."""
    
    class CentroidEllipseCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            # Store one batch for visualization
            self.sample_batch = next(iter(sample_data.take(1) if hasattr(sample_data, 'take') else sample_data))
            
        def on_epoch_end(self, epoch, logs=None):
            try:
                # Unpack: (input, target, reference_image) or (input, target)
                if len(self.sample_batch) == 3:
                    A, y_true, B_img = self.sample_batch
                else:
                    A, y_true = self.sample_batch
                    B_img = A
                
                # Get predictions
                y_pred = self.model.predict(A, verbose=0)
                
                # Use first sample
                img = B_img[0].numpy() if hasattr(B_img[0], 'numpy') else B_img[0]
                true_params = y_true[0].numpy() if hasattr(y_true[0], 'numpy') else y_true[0]
                pred_params = y_pred[0]
                
                # Convert to dictionaries
                keys = ['h_centroid', 'v_centroid', 'h_width', 'v_width']
                true_dict = dict(zip(keys, true_params))
                pred_dict = dict(zip(keys, pred_params))
                
                # Plot
                fig, ax = plt.subplots(figsize=(6, 6))
                plot_centroid_ellipse(ax, img, true_dict, color='red', marker='x')
                plot_centroid_ellipse(ax, img, pred_dict, color='blue', marker='+')
                ax.set_title(f'Epoch {epoch + 1}')
                plt.tight_layout()
                plt.show()
                    
            except Exception as e:
                print(f"Callback error: {e}")
    
    return CentroidEllipseCallback()


def plot_centroid_ellipse(ax, image, params, color='red', marker='x'):
    """Plot beam centroid and ellipse on image."""
    img = np.array(image)
    if img.ndim == 3:
        img = img[:, :, 0] if img.shape[2] > 1 else img.squeeze()
    
    h, w = img.shape
    x = params['h_centroid'] * w
    y = params['v_centroid'] * h
    ew = 2 * params['h_width'] * w
    eh = 2 * params['v_width'] * h
    
    ax.imshow(img, cmap='gray')
    ax.plot(x, y, marker, color=color, markersize=8, markeredgewidth=2)
    ellipse = Ellipse((x, y), ew, eh, edgecolor=color, facecolor='none', linewidth=2)
    ax.add_patch(ellipse)
    ax.axis('off')
