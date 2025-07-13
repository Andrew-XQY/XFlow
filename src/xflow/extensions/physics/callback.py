"""Accelerator physics-specific callback utilities for beam diagnostics visualization."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tensorflow as tf
from ...trainers.callback import CallbackRegistry


@CallbackRegistry.register("centroid_ellipse_callback")
def make_centroid_ellipse_callback(test_ds, sample_size=10):
    """Create a callback that visualizes beam centroid and width ellipses using a fixed sample from test_ds."""
    class CentroidEllipseCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.test_ds = test_ds
            self.sample_size = sample_size
            # Prepare initial sample batch
            self._refresh_sample()

        def _refresh_sample(self):
            # Unbatch then re-batch to get exactly `sample_size` examples
            ds = self.test_ds.unbatch().batch(self.sample_size)
            self.sample_batch = next(iter(ds))

        def on_epoch_begin(self, epoch, logs=None):
            # Optionally refresh sample each epoch
            self._refresh_sample()

        def on_epoch_end(self, epoch, logs=None):
            try:
                # Unpack the batch; allow for (A, y_true, B_img) or (A, y_true)
                parts = list(self.sample_batch)
                A, y_true = parts[0], parts[1]
                B_img = parts[2] if len(parts) == 3 else A

                # Predict on the inputs
                y_pred = self.model.predict(A, verbose=0)
                
                # Use first example for visualization
                img = B_img[0].numpy() if hasattr(B_img[0], 'numpy') else B_img[0]
                true_params = y_true[0].numpy() if hasattr(y_true[0], 'numpy') else y_true[0]
                pred_params = y_pred[0]

                # Map to parameter names
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
