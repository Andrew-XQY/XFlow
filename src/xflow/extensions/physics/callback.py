"""Accelerator physics-specific callback utilities for beam diagnostics visualization."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tensorflow as tf
from ...trainers.callback import CallbackRegistry


@CallbackRegistry.register("centroid_ellipse_callback")
def make_centroid_ellipse_callback(dataset=None):
    """Callback that visualizes beam centroid and width ellipses using a fixed sample from dataset."""
    class CentroidEllipseCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.dataset = dataset
            self.sample_batch = None
            if self.dataset is not None:
                self._refresh_sample()

        def set_dataset(self, dataset):
            self.dataset = dataset
            self._refresh_sample()

        def _refresh_sample(self):
            if self.dataset is None:
                raise ValueError("Dataset must be set before using the callback.")
            # If user passed exactly (A, y_true, B_img), treat that as one batch:
            if isinstance(self.dataset, tuple) and len(self.dataset) in (2, 3):
                self.sample_batch = self.dataset
            else:
                # Otherwise assume it's an iterable of batches
                self.sample_batch = next(iter(self.dataset))

        def on_epoch_begin(self, epoch, logs=None):
            if self.dataset is not None:
                self._refresh_sample()

        def on_epoch_end(self, epoch, logs=None):
            if self.sample_batch is None:
                print("No dataset set for visualization.")
                return
            try:
                # Unpack batch tuple of length 2 or 3
                if len(self.sample_batch) == 3:
                    A, y_true, B_img = self.sample_batch
                else:
                    A, y_true = self.sample_batch
                    B_img = A

                # Predict and grab first example
                y_pred = self.model.predict(A, verbose=0)
                img = B_img[0].numpy() if hasattr(B_img[0], 'numpy') else B_img[0]
                true_params = y_true[0].numpy() if hasattr(y_true[0], 'numpy') else y_true[0]
                pred_params = y_pred[0]

                keys = ['h_centroid', 'v_centroid', 'h_width', 'v_width']
                true_dict = dict(zip(keys, true_params))
                pred_dict = dict(zip(keys, pred_params))

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