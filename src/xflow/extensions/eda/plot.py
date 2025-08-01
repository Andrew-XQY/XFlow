import ast
import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def is_simple_type(val):
    """
    Return True if val is a simple Python type (int, float, str, bool, None), False for list, dict, set, or other objects.
    """
    return isinstance(
        val, (int, float, str, bool, type(None), np.integer, np.floating, np.bool_)
    )


def convert_to_numeric(series):
    """
    Try to convert a pandas Series to numeric, otherwise return as is.
    Extendable for more conversion rules.
    """

    def try_convert(val):
        # Try numeric
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
        # Try literal eval (for lists, dicts, etc.)
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError, TypeError):
            pass
        return val  # fallback: return as is

    # Only try conversion for object/string columns
    if series.dtype == object:
        return series.apply(try_convert)
    return series


def calculate_histogram(iterable, bins=None):
    """
    Count occurrences for any iterable (numeric or not).
    Returns (values, counts) for plotting.
    """
    # If numeric and bins specified, use numpy histogram
    arr = np.array(iterable)
    if np.issubdtype(arr.dtype, np.number) and bins is not None:
        counts, bin_edges = np.histogram(arr, bins=bins)
        # Use bin centers for plotting
        values = (bin_edges[:-1] + bin_edges[1:]) / 2
        return values, counts
    # Otherwise, treat as categorical
    counter = Counter(iterable)
    values, counts = zip(*sorted(counter.items(), key=lambda x: str(x[0])))
    return values, counts


def df_distribution_plot(df, subplot_cols=3, bins=20):
    """
    Plot histograms for each column in df.
    """
    n_cols = len(df.columns)
    n_rows = math.ceil(n_cols / subplot_cols)
    fig, axes = plt.subplots(
        n_rows, subplot_cols, figsize=(5 * subplot_cols, 4 * n_rows)
    )
    axes = axes.flatten() if n_cols > 1 else [axes]

    print(f"Found {n_cols} columns. Plotting distributions:")
    for idx, col in enumerate(df.columns):
        col_values = df[col].dropna()
        ax = axes[idx]
        if not any(is_simple_type(v) for v in col_values):
            print(f"\nSkipping column '{col}' (no simple types detected)")
            ax.set_title(f"{col}\n(Skipped: not simple type)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        progress = f"[{'=' * (idx+1)}{' ' * (n_cols-idx-1)}]"
        print(f"{progress} ({idx+1}/{n_cols}) Processing column: {col}", end="\r")
        series = convert_to_numeric(df[col])
        values, counts = calculate_histogram(series, bins=bins)
        if np.issubdtype(np.array(values).dtype, np.number):
            ax.bar(
                values, counts, width=(values[1] - values[0]) if len(values) > 1 else 1
            )
        else:
            ax.bar(values, counts)
            ax.set_xticklabels(values, rotation=45, ha="right")
        ax.set_title(str(col))
    print()  # Newline after progress bar
    # Hide unused subplots
    for ax in axes[n_cols:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
