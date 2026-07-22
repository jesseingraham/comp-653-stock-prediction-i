"""Turn a feature table into supervised look-back windows for the RNNs.

Parameterized, pure transform: given a chronologically ordered feature
DataFrame it produces sliding look-back windows ``X`` of shape
``(n_windows, window_size, n_features)`` and multi-step return targets ``y`` of
shape ``(n_windows, horizon)``. The ``02_feature_engineering`` notebook chooses
the window size, horizon, and feature columns and persists the arrays to the
Shared Drive; this module makes no sizing decisions itself.

Design choices (see project discussion):
    * Targets are per-step **log returns** of the target price column by default
      (``y[:, k] = log(price[t+k+1] / price[t+k])``); simple returns optional.
    * Feature windows are emitted **as-is** -- no scaling. Per-fold scaling
      happens in the trainer so walk-forward cross-validation stays leakage-free.
    * The window covers rows through day ``t``; targets start strictly at
      ``t+1``, so no future information leaks into a window.
    * Chronological order is preserved (no shuffling) and windows containing any
      NaN feature or target are dropped (count reported).
    * Each window is tagged with its first target date, for plotting and the
      chronological train/test split downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DRIVE_DATA_PATH

# Conventional filenames for the two windowed feature sets on the Drive.
BASELINE_WINDOWS_FILENAME = "sp500_windows_baseline.npz"
AUGMENTED_WINDOWS_FILENAME = "sp500_windows_augmented.npz"


def _returns(price: np.ndarray, kind: str) -> np.ndarray:
    """Per-step returns of a price series (index 0 is NaN).

    Args:
        price: 1-D price array.
        kind: ``"log"`` for log returns or ``"simple"`` for percentage returns.

    Returns:
        Array the same length as `price`; ``out[t]`` is the return from ``t-1``
        to ``t`` and ``out[0]`` is NaN.

    Raises:
        ValueError: If `kind` is not ``"log"`` or ``"simple"``.
    """
    out = np.full(len(price), np.nan, dtype=np.float64)
    if kind == "log":
        out[1:] = np.log(price[1:] / price[:-1])
    elif kind == "simple":
        out[1:] = price[1:] / price[:-1] - 1.0
    else:
        raise ValueError(f"Unknown return kind {kind!r}; use 'log' or 'simple'.")
    return out


def make_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "Close",
    window_size: int = 30,
    horizon: int = 1,
    stride: int = 1,
    return_kind: str = "log",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build look-back windows and multi-step return targets from `df`.

    For a window ending at row ``t`` (features from ``t-window_size+1`` through
    ``t``), the target is the next `horizon` returns of `target_column`:
    ``[r(t+1), ..., r(t+horizon)]``. Windows with any NaN feature or target are
    dropped.

    Args:
        df: Chronologically ordered feature table, indexed by date.
        feature_columns: Columns to window into `X` (used as-is, unscaled).
        target_column: Price column the returns are computed from.
        window_size: Number of past timesteps per window.
        horizon: Number of future steps to predict.
        stride: Step between consecutive window start positions.
        return_kind: ``"log"`` or ``"simple"`` returns.

    Returns:
        ``(X, y, dates)`` where ``X`` is ``(n, window_size, n_features)``
        float32, ``y`` is ``(n, horizon)`` float32, and ``dates`` holds each
        window's first target date (``t+1``).

    Raises:
        ValueError: If required columns are missing or sizing is non-positive.
    """
    missing = [c for c in [*feature_columns, target_column] if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")
    if min(window_size, horizon, stride) < 1:
        raise ValueError("window_size, horizon, and stride must all be >= 1.")

    features = df[feature_columns].to_numpy(dtype=np.float64)
    returns = _returns(df[target_column].to_numpy(dtype=np.float64), return_kind)
    index = df.index.to_numpy()
    n_rows = len(df)

    x_list, y_list, date_list = [], [], []
    # Window ends at t; targets use returns[t+1 .. t+horizon], so the last valid
    # end index is n_rows - 1 - horizon.
    for end in range(window_size - 1, n_rows - horizon, stride):
        window = features[end - window_size + 1 : end + 1]
        target = returns[end + 1 : end + 1 + horizon]
        if np.isnan(window).any() or np.isnan(target).any():
            continue
        x_list.append(window)
        y_list.append(target)
        date_list.append(index[end + 1])

    x = np.asarray(x_list, dtype=np.float32).reshape(
        -1, window_size, len(feature_columns)
    )
    y = np.asarray(y_list, dtype=np.float32).reshape(-1, horizon)
    dates = np.asarray(date_list)

    dropped = (len(range(window_size - 1, n_rows - horizon, stride))) - len(x)
    print(
        f"Built {len(x):,} windows (size={window_size}, horizon={horizon}, "
        f"features={len(feature_columns)}); dropped {dropped} with NaNs."
    )
    return x, y, dates


def save_windows(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    filename: str,
    data_dir: str | Path = DRIVE_DATA_PATH,
) -> Path:
    """Persist windowed arrays to a compressed ``.npz`` under `data_dir`.

    Args:
        x: Windowed features, ``(n, window_size, n_features)``.
        y: Targets, ``(n, horizon)``.
        dates: Per-window first target dates, ``(n,)``.
        filename: Output file name (e.g. `BASELINE_WINDOWS_FILENAME`).
        data_dir: Destination directory (defaults to the Shared Drive path).

    Returns:
        The path the file was written to.
    """
    dest_dir = Path(data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    np.savez_compressed(path, X=x, y=y, dates=dates)
    return path


def main() -> None:
    """Smoke demo on synthetic data (shape check only)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-01", periods=300, freq="B")
    frame = pd.DataFrame(
        {
            "Open": rng.uniform(90, 110, 300),
            "High": rng.uniform(90, 110, 300),
            "Low": rng.uniform(90, 110, 300),
            "Close": rng.uniform(90, 110, 300),
            "Volume": rng.integers(1_000, 5_000, 300),
        },
        index=dates,
    )
    x, y, tags = make_windows(
        frame, ["Open", "High", "Low", "Close", "Volume"], window_size=30, horizon=1
    )
    print("X:", x.shape, "| y:", y.shape, "| dates:", tags.shape)


if __name__ == "__main__":
    main()
