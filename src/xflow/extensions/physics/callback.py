"""Accelerator physics-specific callback utilities for beam diagnostics visualization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from ...trainers.callback import CallbackRegistry
from ...utils.visualization import to_numpy_image


@CallbackRegistry.register("centroid_ellipse_callback")
def make_centroid_ellipse_callback(dataset=None, save_dir=None):
    """Callback that visualizes beam centroid and width ellipses using a fixed sample from dataset."""
    import tensorflow as tf

    class CentroidEllipseCallback(tf.keras.callbacks.Callback):
        def __init__(self, save_dir=save_dir):
            super().__init__()
            self.dataset = dataset
            self.sample_batch = None
            self.save_dir = save_dir
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
                img = B_img[0].numpy() if hasattr(B_img[0], "numpy") else B_img[0]
                true_params = (
                    y_true[0].numpy() if hasattr(y_true[0], "numpy") else y_true[0]
                )
                pred_params = y_pred[0]

                keys = ["h_centroid", "v_centroid", "h_width", "v_width"]
                true_dict = dict(zip(keys, true_params))
                pred_dict = dict(zip(keys, pred_params))

                fig, ax = plt.subplots(figsize=(6, 6))
                # Construct save_path if save_dir is set
                save_path = None
                if self.save_dir is not None:
                    import os

                    abs_save_dir = os.path.abspath(self.save_dir)
                    os.makedirs(abs_save_dir, exist_ok=True)
                    save_path = os.path.join(abs_save_dir, f"epoch_{epoch+1}.png")
                result_fig = plot_centroid_ellipse(
                    ax, img, true_dict, pred_params=pred_dict, save_path=save_path
                )
                ax.set_title(f"Epoch {epoch + 1}")
                plt.tight_layout()
                try:
                    from IPython.display import clear_output, display

                    clear_output(wait=True)
                    display(fig)
                except ImportError:
                    plt.show()
                # If save_path is set, plot_centroid_ellipse already saves and closes the figure

            except Exception as e:
                print(f"Callback error: {e}")

    return CentroidEllipseCallback()


def plot_centroid_ellipse(
    ax,
    image,
    true_params,
    pred_params=None,
    true_label="True",
    pred_label="Predicted",
    save_path=None,
):
    """Plot beam centroid and ellipse on image, with legend, colorbar, normalized ticks, and param text."""
    img = np.array(image)
    if img.ndim == 3:
        img = img[:, :, 0] if img.shape[2] > 1 else img.squeeze()
    h, w = img.shape

    # Show image and colorbar
    im = ax.imshow(img, cmap="viridis")  # gray, viridis
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity")

    # Plot true ellipse
    x_true = true_params["h_centroid"] * w
    y_true = true_params["v_centroid"] * h
    ew_true = 2 * true_params["h_width"] * w
    eh_true = 2 * true_params["v_width"] * h
    ellipse_true = Ellipse(
        (x_true, y_true),
        ew_true,
        eh_true,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
        label=true_label,
    )
    ax.add_patch(ellipse_true)
    ax.plot(x_true, y_true, "x", color="red", markersize=6, markeredgewidth=2)

    # Plot predicted ellipse if provided
    handles = [ellipse_true]
    labels = [true_label]
    if pred_params is not None:
        x_pred = pred_params["h_centroid"] * w
        y_pred = pred_params["v_centroid"] * h
        ew_pred = 2 * pred_params["h_width"] * w
        eh_pred = 2 * pred_params["v_width"] * h
        ellipse_pred = Ellipse(
            (x_pred, y_pred),
            ew_pred,
            eh_pred,
            edgecolor="blue",
            facecolor="none",
            linewidth=2,
            label=pred_label,
        )
        ax.add_patch(ellipse_pred)
        ax.plot(x_pred, y_pred, "+", color="blue", markersize=8, markeredgewidth=2)
        handles.append(ellipse_pred)
        labels.append(pred_label)

    # Add legend
    ax.legend(handles, labels, loc="best")

    # Normalized ticks: 0.0, 0.1, ..., 1.0
    xticks = [int(round(i * (w - 1))) for i in np.linspace(0, 1, 11)]
    yticks = [int(round(i * (h - 1))) for i in np.linspace(0, 1, 11)]
    ticklabels = [f"{i/10:.1f}" for i in range(11)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ticklabels)

    # Print params below the image (centered under the axis), aligned by key
    keys = ["h_centroid", "v_centroid", "h_width", "v_width"]

    def fmt(val):
        return f"{val: .4f}" if isinstance(val, float) else str(val)

    if pred_params is not None:
        header = f"{'':12}  {'True':>12}  {'Pred':>12}"
        lines = [header]
        for k in keys:
            tval = fmt(true_params.get(k, ""))
            pval = fmt(pred_params.get(k, ""))
            lines.append(f"{k:12}  {tval:>12}  {pval:>12}")
        param_text = "\n".join(lines)
    else:
        header = f"{'':12}  {'True':>12}"
        lines = [header]
        for k in keys:
            tval = fmt(true_params.get(k, ""))
            lines.append(f"{k:12}  {tval:>12}")
        param_text = "\n".join(lines)

    fig = ax.figure
    # Place text below the axis, centered
    fig.subplots_adjust(bottom=0.22)  # Make space for text
    fig.text(
        0.5,
        0.08,
        param_text,
        ha="center",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    ax.axis("on")

    # Save or return
    fig = ax.figure
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        return fig


# --- New Callback for Image Reconstruction ---
from datetime import datetime


def _visualize_image_reconstruction(
    img_in, img_pred, img_true, epoch, save_dir=None, cmap="viridis"
):
    """Shared visualization logic for image reconstruction callbacks."""
    import os

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    images = [img_in, img_pred, img_true, img_in, img_pred, img_true]
    titles = [
        "Input",
        "Reconstructed",
        "Ground Truth",
        "Input (rescale)",
        "Reconstructed (rescale)",
        "Ground Truth (rescale)",
    ]

    for i, ax in enumerate(axs.flat):
        if i < 3:
            ax.imshow(images[i], cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(images[i], cmap=cmap)
        ax.set_title(titles[i])
        ax.axis("off")
    plt.tight_layout()

    if save_dir is not None:
        abs_save_dir = os.path.abspath(save_dir)
        os.makedirs(abs_save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(abs_save_dir, f"epoch_{epoch+1}_{timestamp}.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        try:
            from IPython.display import clear_output, display

            clear_output(wait=True)
            display(fig)
        except ImportError:
            plt.show()

    def get_max(arr):
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        return np.max(arr)

    print(
        f"input image max pixel: {get_max(img_in)}, "
        f"ground truth image max pixel: {get_max(img_true)}, "
        f"reconstructed image max pixel: {get_max(img_pred)}"
    )


@CallbackRegistry.register("image_reconstruction_callback")
def make_image_reconstruction_callback(dataset=None, save_dir=None, cmap="viridis"):
    """Callback that visualizes input, predicted, and ground truth images side by side, with and without min-max rescaling."""
    import tensorflow as tf

    class ImageReconstructionCallback(tf.keras.callbacks.Callback):
        def __init__(self, save_dir=save_dir, cmap=cmap):
            super().__init__()
            self.dataset = dataset
            self.sample_batch = None
            self.save_dir = save_dir
            self.cmap = cmap
            if self.dataset is not None:
                self._refresh_sample()

        def set_dataset(self, dataset):
            self.dataset = dataset
            self._refresh_sample()

        def _refresh_sample(self):
            if self.dataset is None:
                raise ValueError("Dataset must be set before using the callback.")
            # If user passed exactly (X, Y), treat that as one batch:
            if isinstance(self.dataset, tuple) and len(self.dataset) == 2:
                self.sample_batch = self.dataset
            else:
                # Otherwise assume it's an iterable of batches
                self.sample_batch = next(iter(self.dataset))

        def on_epoch_begin(self, epoch, logs=None):
            if self.dataset is not None:
                self._refresh_sample()
            if self.sample_batch is None:
                print("No dataset set for visualization.")
                return
            try:
                X, Y = self.sample_batch
                # Randomly choose one sample
                idx = np.random.randint(0, len(X))
                x = X[idx : idx + 1]  # keep batch dim
                y_true = Y[idx]
                y_pred = self.model.predict(x, verbose=0)

                img_in = to_numpy_image(x)
                img_pred = to_numpy_image(y_pred)
                img_true = to_numpy_image(y_true)

                _visualize_image_reconstruction(
                    img_in, img_pred, img_true, epoch, self.save_dir, self.cmap
                )

            except Exception as e:
                print(f"ImageReconstructionCallback error: {e}")

    return ImageReconstructionCallback()


@CallbackRegistry.register("torch_image_reconstruction_callback")
def make_torch_image_reconstruction_callback(
    dataset=None, save_dir=None, cmap="viridis"
):
    """Create PyTorch image reconstruction visualization callback."""
    import os
    from datetime import datetime

    from ...trainers.callback import PyTorchCallback

    class TorchImageReconstructionCallback(PyTorchCallback):
        def __init__(self, save_dir=save_dir, cmap=cmap):
            super().__init__()
            self.dataset = dataset
            self.sample_batch = None
            self.save_dir = save_dir
            self.cmap = cmap
            if self.dataset is not None:
                self._refresh_sample()

        def set_dataset(self, dataset):
            self.dataset = dataset
            self._refresh_sample()

        def _refresh_sample(self):
            if self.dataset is None:
                raise ValueError("Dataset must be set before using the callback.")
            if isinstance(self.dataset, tuple) and len(self.dataset) == 2:
                self.sample_batch = self.dataset
            else:
                self.sample_batch = next(iter(self.dataset))

        def on_epoch_begin(self, epoch, model=None, **kwargs):
            if self.dataset is not None:
                self._refresh_sample()
            if self.sample_batch is None or model is None:
                return
            try:
                import torch

                X, Y = self.sample_batch
                idx = np.random.randint(0, len(X))
                x = X[idx : idx + 1]
                y_true = Y[idx]

                model.eval()
                with torch.no_grad():
                    if hasattr(x, "to"):
                        x = x.to(next(model.parameters()).device)
                    y_pred = model(x)
                    if hasattr(y_pred, "cpu"):
                        y_pred = y_pred.cpu()

                img_in = to_numpy_image(x)
                img_pred = to_numpy_image(y_pred)
                img_true = to_numpy_image(y_true)

                _visualize_image_reconstruction(
                    img_in, img_pred, img_true, epoch, self.save_dir, self.cmap
                )

            except Exception as e:
                print(f"TorchImageReconstructionCallback error: {e}")

    return TorchImageReconstructionCallback()
