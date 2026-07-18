"""Clean and validate raw S&P 500 OHLCV data for downstream modelling.

Consumes the raw frame produced by `src.data.fetcher` and returns a validated,
gap-resilient OHLCV series that the feature-engineering and windowing stages can
rely on. The cleaner enforces structural invariants (sorted, unique, tz-naive
dates and numeric dtypes), drops the redundant ``Adj Close`` column, and repairs
missing or OHLC-inconsistent bars by forward-filling the most recent valid bar.
Real market moves and zero-volume sessions are preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Price columns retained in the cleaned output. "Adj Close" is intentionally
# excluded: for a price index it is identical to "Close".
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
CLEAN_COLUMNS = [*PRICE_COLUMNS, "Volume"]


def _invalid_row_mask(df: pd.DataFrame) -> pd.Series:
    """Flag rows that are missing or violate OHLC logic.

    A bar is invalid if any OHLCV field is NaN, any price is non-positive, or the
    High/Low bounds are inconsistent with each other or with Open/Close.

    Args:
        df: OHLCV frame with the columns in `CLEAN_COLUMNS`.

    Returns:
        A boolean Series (indexed like `df`) that is True for invalid rows.
    """
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    missing = df[CLEAN_COLUMNS].isna().any(axis=1)
    non_positive = (df[PRICE_COLUMNS] <= 0).any(axis=1)
    bad_bounds = (
        (high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)
    )
    return missing | non_positive | bad_bounds


def _assert_clean(df: pd.DataFrame) -> None:
    """Assert the post-conditions the downstream pipeline relies on.

    Args:
        df: The cleaned OHLCV frame.

    Raises:
        AssertionError: If any invariant (no NaNs, sorted unique index, valid
            OHLC bounds) does not hold.
    """
    assert not df.isna().any().any(), "Cleaned frame still contains NaNs."
    assert df.index.is_monotonic_increasing, "Index is not sorted ascending."
    assert not df.index.duplicated().any(), "Index still has duplicate dates."
    assert not _invalid_row_mask(df).any(), "OHLC integrity violations remain."


def clean_ohlcv(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Validate and clean a raw OHLCV frame for downstream modelling.

    Cleaning steps:
        1. Normalize the index to a tz-naive, ascending, duplicate-free
           ``Date`` index.
        2. Keep only numeric OHLCV columns (dropping the redundant
           ``Adj Close``).
        3. Replace missing or OHLC-inconsistent bars with the most recent valid
           bar (row-level forward fill); drop any leading bars with no
           predecessor to fill from.

    Extreme-but-real market moves are preserved. Zero-volume sessions are kept
    and reported rather than removed.

    Args:
        df: Raw OHLCV frame from `src.data.fetcher`.
        verbose: If True, print a one-line summary of what changed.

    Returns:
        A cleaned frame indexed by ``Date`` with columns `CLEAN_COLUMNS`.

    Raises:
        KeyError: If any required OHLCV column is missing from `df`.
    """
    missing_cols = [col for col in CLEAN_COLUMNS if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Input is missing required columns: {missing_cols}")

    out = df.copy()

    # 1. Normalize the date index: datetime, tz-naive, unique, sorted.
    index = pd.to_datetime(out.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    out.index = index
    out.index.name = "Date"
    n_dupes = int(out.index.duplicated().sum())
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # 2. Keep only OHLCV as numeric; this drops the redundant Adj Close column.
    #    Non-numeric values coerce to NaN so the fill in step 3 repairs them.
    out = out[CLEAN_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # 3. Repair missing/corrupt bars by forward-filling whole valid bars, so a
    #    repaired bar is a copy of a known-good one and stays self-consistent.
    invalid = _invalid_row_mask(out)
    n_invalid = int(invalid.sum())
    out.loc[invalid] = np.nan
    out = out.ffill()

    n_leading_dropped = int(out.isna().any(axis=1).sum())
    out = out.dropna()

    # Volume is a count; restore integer dtype now that NaNs are gone.
    out["Volume"] = out["Volume"].astype("int64")
    n_zero_vol = int((out["Volume"] == 0).sum())

    if verbose:
        print(
            f"Cleaned OHLCV: {len(df):,} -> {len(out):,} rows | "
            f"dupe dates dropped: {n_dupes} | "
            f"invalid bars filled: {n_invalid} | "
            f"leading rows dropped: {n_leading_dropped} | "
            f"zero-volume days kept: {n_zero_vol}"
        )

    _assert_clean(out)
    return out


def main() -> None:
    """Fetch raw data and run the cleaner, printing a summary report."""
    from src.data.fetcher import fetch_ohlcv

    raw = fetch_ohlcv()
    clean_ohlcv(raw)


if __name__ == "__main__":
    main()
