"""Derive stationary model inputs from cleaned OHLCV data.

The forecasting targets are log returns, so the inputs must be stationary too:
raw price levels drift from ~1,270 (2006) to ~6,850 (2025), which puts test-era
values outside anything the model saw during training, and scaling cannot undo
that. This module converts the cleaned OHLCV table into scale-free features that
mean the same thing at any price level.

`make_base_features` returns the original columns (so `Close` remains available
as the windowing target) plus the derived feature columns named in
`BASE_FEATURE_COLUMNS`. It is the price-only feature set behind the baseline
model; the pattern columns from `patterns` are appended alongside these for the
augmented model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Stationary, scale-free features derived from OHLCV.
LOG_RETURN = "log_return"
HIGH_LOW_RANGE = "high_low_range"
OPEN_CLOSE_CHANGE = "open_close_change"
LOG_VOLUME_CHANGE = "log_volume_change"

# The price-only feature set used by the baseline model. Volatility columns are
# appended by `make_base_features` according to its `volatility_windows`.
BASE_FEATURE_COLUMNS = [
    LOG_RETURN,
    HIGH_LOW_RANGE,
    OPEN_CLOSE_CHANGE,
    LOG_VOLUME_CHANGE,
]


def volatility_column(window: int) -> str:
    """Return the column name for a rolling-volatility feature."""
    return f"volatility_{window}"


def make_base_features(
    df: pd.DataFrame,
    volatility_windows: tuple[int, ...] = (5, 20),
) -> tuple[pd.DataFrame, list[str]]:
    """Add stationary feature columns to a cleaned OHLCV frame.

    Derived features:
        * ``log_return`` -- log change in Close from the previous bar.
        * ``high_low_range`` -- ``(High - Low) / Close``, the bar's span.
        * ``open_close_change`` -- ``(Close - Open) / Open``, the bar's body.
        * ``log_volume_change`` -- log change in Volume. Non-positive volumes
          (e.g. holiday half-sessions) become NaN rather than infinities.
        * ``volatility_<w>`` -- rolling standard deviation of ``log_return``
          over the trailing ``w`` bars, for each `volatility_windows` entry.

    All rolling statistics look strictly backwards, and the leading rows that
    cannot be computed stay NaN; `windowing.make_windows` drops any window
    containing NaN.

    Args:
        df: Cleaned OHLCV frame indexed by date, ascending.
        volatility_windows: Trailing window lengths for volatility features.

    Returns:
        A ``(frame, feature_columns)`` pair: the input columns plus the derived
        ones, and the list of derived feature column names.

    Raises:
        KeyError: If any of Open/High/Low/Close/Volume is missing.
    """
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input is missing required columns: {missing}")

    out = df.copy()
    close = out["Close"]

    out[LOG_RETURN] = np.log(close / close.shift(1))
    out[HIGH_LOW_RANGE] = (out["High"] - out["Low"]) / close
    out[OPEN_CLOSE_CHANGE] = (out["Close"] - out["Open"]) / out["Open"]

    # Guard against zero/negative volume so the ratio cannot produce infinities.
    volume = out["Volume"].where(out["Volume"] > 0)
    out[LOG_VOLUME_CHANGE] = np.log(volume / volume.shift(1))

    feature_columns = list(BASE_FEATURE_COLUMNS)
    for window in volatility_windows:
        name = volatility_column(window)
        out[name] = out[LOG_RETURN].rolling(window).std()
        feature_columns.append(name)

    return out, feature_columns
