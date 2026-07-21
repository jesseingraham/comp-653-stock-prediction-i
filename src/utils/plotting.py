"""Plotting helpers for summarizing RNN training and hyperparameter tuning.

These functions take plain pandas/NumPy inputs (extracted from a Ray Tune
``ResultGrid`` or `trainer.evaluate_on_test`) and return Matplotlib figures, so
they are reusable across the modelling notebooks and testable without Ray.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Hyperparameters conventionally shown on a log scale.
_LOG_PARAMS = ("lr", "weight_decay", "config/lr", "config/weight_decay")


def plot_trial_performance(
    results_df: pd.DataFrame,
    metric: str = "val_rmse",
    mode: str = "min",
) -> Figure:
    """Bar chart of each trial's final `metric`, best trial highlighted.

    Args:
        results_df: Per-trial summary (e.g. ``ResultGrid.get_dataframe()``).
        metric: Column to rank trials by.
        mode: ``"min"`` or ``"max"`` -- which end is best.

    Returns:
        The Matplotlib figure.
    """
    values = results_df[metric].to_numpy(dtype=float)
    order = np.argsort(values)
    if mode == "max":
        order = order[::-1]
    ranked = values[order]
    best_pos = 0

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#c0c0c0"] * len(ranked)
    if len(colors):
        colors[best_pos] = "#d62728"
    ax.bar(range(len(ranked)), ranked, color=colors)
    ax.set_xlabel("Trial (ranked)")
    ax.set_ylabel(metric)
    ax.set_title(f"Trial performance by {metric} (best highlighted)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_learning_curves(
    curves: Sequence[pd.DataFrame],
    metric: str = "val_rmse",
    iteration_col: str = "training_iteration",
    best_index: int | None = None,
) -> Figure:
    """Overlay per-trial learning curves (metric vs. epoch).

    Trials that ASHA stopped early simply have shorter curves.

    Args:
        curves: One DataFrame per trial (e.g. each result's
            ``metrics_dataframe``), each with an iteration column and `metric`.
        metric: Metric column to plot.
        iteration_col: Column holding the epoch / training iteration.
        best_index: Index into `curves` to highlight, if known.

    Returns:
        The Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, curve in enumerate(curves):
        if metric not in curve or iteration_col not in curve:
            continue
        is_best = i == best_index
        ax.plot(
            curve[iteration_col],
            curve[metric],
            color="#d62728" if is_best else "#9ecae1",
            linewidth=2.0 if is_best else 0.9,
            zorder=3 if is_best else 1,
            label="best trial" if is_best else None,
        )
    ax.set_xlabel(iteration_col)
    ax.set_ylabel(metric)
    ax.set_title("Learning curves across tuning trials")
    ax.grid(True, alpha=0.3)
    if best_index is not None:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_hyperparameter_effects(
    results_df: pd.DataFrame,
    params: Sequence[str],
    metric: str = "val_rmse",
) -> Figure:
    """Small-multiple scatter plots of `metric` against each hyperparameter.

    Args:
        results_df: Per-trial summary (e.g. ``ResultGrid.get_dataframe()``).
        params: Column names of the hyperparameters to plot. Columns absent
            from `results_df` are skipped.
        metric: Metric column for the y-axis.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If none of `params` are present in `results_df`.
    """
    present = [p for p in params if p in results_df.columns]
    if not present:
        raise ValueError("None of the requested params are in results_df.")

    n_cols = min(3, len(present))
    n_rows = (len(present) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False
    )
    flat_axes = axes.flatten()
    y = results_df[metric].to_numpy(dtype=float)
    for ax, param in zip(flat_axes, present):
        ax.scatter(results_df[param], y, alpha=0.7, color="#1f77b4")
        if param in _LOG_PARAMS:
            ax.set_xscale("log")
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    for ax in flat_axes[len(present) :]:
        ax.set_visible(False)
    fig.suptitle(f"{metric} vs. hyperparameters")
    fig.tight_layout()
    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: Any | None = None,
    title: str = "Test set: predicted vs. actual",
) -> Figure:
    """Line plot of actual vs. predicted targets over the test period.

    Args:
        y_true: Actual targets, shape ``(n,)`` or ``(n, 1)``.
        y_pred: Predicted targets, same shape as `y_true`.
        index: Optional x-axis values (e.g. dates); defaults to ``0..n-1``.
        title: Plot title.

    Returns:
        The Matplotlib figure.
    """
    true = np.asarray(y_true).reshape(len(y_true), -1)[:, 0]
    pred = np.asarray(y_pred).reshape(len(y_pred), -1)[:, 0]
    x = np.arange(len(true)) if index is None else index

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, true, label="actual", color="#1f77b4", linewidth=1.0)
    ax.plot(x, pred, label="predicted", color="#d62728", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Date" if index is not None else "Test step")
    ax.set_ylabel("Target")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
