import matplotlib as mpl

# APS-style color palette (Physical Review Letters)
APS_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion/red
    "#CC79A7",  # reddish purple
    "#999999",  # medium gray
]


def set_aps_single_column(figsize=(3.25, 2.5), scale=1.0, legend_background=True):
    """
    Configure Matplotlib rcParams for APS single-column figures.

    This function sets parameters to ensure that plots are formatted
    appropriately for single-column figures in APS journals, such as
    Physical Review Letters (PRL). The settings adjust font sizes,
    tick marks, axis lines, and savefig options to maintain consistency
    and clarity in the final publication.

    Usage:
        set_aps_single_column()
    """
    mpl.rcParams.update(
        {
            # Font settings:
            "font.family": "sans-serif",  # APS standard font family
            "font.sans-serif": [
                "Arial",
                "Helvetica",
            ],  # Preferred fonts: Arial, Helvetica
            "font.size": 9.0 * scale,  # Base font size for tick labels and general text
            "axes.labelsize": 10.0 * scale,  # Axis label font size
            "axes.titlesize": 10.0 * scale,  # Title font size
            "legend.fontsize": 8.0
            * scale,  # Legend text font size (legend is optional but standard)
            "xtick.labelsize": 9.0 * scale,  # X-axis tick label size
            "ytick.labelsize": 9.0 * scale,  # Y-axis tick label size
            # Tick marks and axis lines:
            "xtick.direction": "in",  # Ticks inward
            "ytick.direction": "in",  # Ticks inward
            "xtick.major.size": 4.0 * scale,  # Length of major ticks
            "xtick.minor.size": 2.0 * scale,  # Length of minor ticks
            "ytick.major.size": 4.0 * scale,  # Length of major ticks
            "ytick.minor.size": 2.0 * scale,  # Length of minor ticks
            "xtick.major.width": 1.0,  # Width of major ticks
            "ytick.major.width": 1.0,  # Width of major ticks
            "axes.linewidth": 1.0,  # Width of the axis lines
            "lines.linewidth": 0.75 * scale,  # Line width for plots
            "lines.markersize": 6 * scale,  # Adjust as needed for clarity
            "image.cmap": "viridis",  # Default colormap for images
            # Legend settings based on argument
            "legend.frameon": legend_background,
            "legend.facecolor": "white" if legend_background else "none",
            "legend.edgecolor": "black" if legend_background else "none",
            "legend.framealpha": 0.8 if legend_background else 0.0,
            # Figure size:
            "figure.figsize": figsize,
            # Spacing
            "axes.labelpad": 6.0
            * scale,  # default is 4.0 :contentReference[oaicite:0]{index=0}
            "xtick.major.pad": 3.0
            * scale,  # via rc('xtick.major', pad=…) :contentReference[oaicite:1]{index=1}
            "ytick.major.pad": 3.0 * scale,
            # Savefig resolution and font embedding:
            "savefig.dpi": 600,  # Resolution for figure saving
            "pdf.fonttype": 42,  # Embed fonts as TrueType
            "ps.fonttype": 42,  # Embed fonts as TrueType
        }
    )


def set_aps_double_column(figsize=(7.0, 3.5), scale=1.0, legend_background=True):
    """
    Configure Matplotlib rcParams for APS double-column figures.

    This function sets parameters to ensure that plots are formatted
    appropriately for double-column figures in APS journals. The settings
    adjust font sizes, tick marks, axis lines, and savefig options to
    maintain consistency and clarity in the final publication.

    Usage:
        set_aps_double_column()
    """
    mpl.rcParams.update(
        {
            # Font settings:
            "font.family": "sans-serif",  # APS standard font family
            "font.sans-serif": [
                "Arial",
                "Helvetica",
            ],  # Preferred fonts: Arial, Helvetica
            "font.size": 8.0 * scale,  # Base font size for tick labels and general text
            "axes.labelsize": 9.0 * scale,  # Axis label font size
            "axes.titlesize": 9.0 * scale,  # Title font size
            "legend.fontsize": 8.0
            * scale,  # Legend text font size (legend is optional but standard)
            "xtick.labelsize": 8.0 * scale,  # X-axis tick label size
            "ytick.labelsize": 8.0 * scale,  # Y-axis tick label size
            # Tick marks and axis lines:
            "xtick.direction": "in",  # Ticks inward
            "ytick.direction": "in",  # Ticks inward
            "xtick.major.size": 4.0 * scale,  # Length of major ticks
            "xtick.minor.size": 2.0 * scale,  # Length of minor ticks
            "ytick.major.size": 4.0 * scale,  # Length of major ticks
            "ytick.minor.size": 2.0 * scale,  # Length of minor ticks
            "xtick.major.width": 1.0,  # Width of major ticks
            "ytick.major.width": 1.0,  # Width of major ticks
            "axes.linewidth": 1.0,  # Width of the axis lines
            "lines.linewidth": 0.75 * scale,  # Line width for plots
            "lines.markersize": 6 * scale,  # Adjust as needed for clarity
            "image.cmap": "viridis",  # Default colormap for images
            # Legend settings based on argument
            "legend.frameon": legend_background,
            "legend.facecolor": "white" if legend_background else "none",
            "legend.edgecolor": "black" if legend_background else "none",
            "legend.framealpha": 0.8 if legend_background else 0.0,
            # Figure size:
            "figure.figsize": figsize,
            # Spacing
            "axes.labelpad": 5.0
            * scale,  # default is 4.0 :contentReference[oaicite:0]{index=0}
            "xtick.major.pad": 3.0
            * scale,  # default is 3 :contentReference[oaicite:1]{index=1}
            "ytick.major.pad": 3.0 * scale,
            # Savefig resolution and font embedding:
            "savefig.dpi": 600,  # Resolution for figure saving
            "pdf.fonttype": 42,  # Embed fonts as TrueType
            "ps.fonttype": 42,  # Embed fonts as TrueType
        }
    )
