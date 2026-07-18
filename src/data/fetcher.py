"""Fetch raw daily OHLCV data for the S&P 500 index from Yahoo Finance.

This module is the data-acquisition entry point for the pipeline, driven by the
`01_data_collection` notebook. It downloads roughly 20 years of daily bars for
the index configured in `config.py` and persists them as Parquet to the project
Shared Drive (`DRIVE_DATA_PATH`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DRIVE_DATA_PATH, END_DATE, START_DATE, TICKER

# Field columns kept from Yahoo Finance, in canonical order.
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# Default filename for the persisted raw dataset.
RAW_FILENAME = "sp500_daily.parquet"


def fetch_ohlcv(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Download daily OHLCV bars for a single ticker from Yahoo Finance.

    Args:
        ticker: Yahoo Finance symbol (defaults to the configured index).
        start: Inclusive start date, "YYYY-MM-DD".
        end: Exclusive end date, "YYYY-MM-DD".

    Returns:
        A DataFrame indexed by a "Date" DatetimeIndex with the columns in
        `OHLCV_COLUMNS`.

    Raises:
        ValueError: If Yahoo Finance returns no rows for the request.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker!r} between {start} and {end}.")

    # For a single ticker yfinance still returns MultiIndex columns
    # (field, ticker); collapse them to just the field level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    df.index.name = "Date"

    return df[OHLCV_COLUMNS]


def save_raw(
    df: pd.DataFrame,
    filename: str = RAW_FILENAME,
    data_dir: str | Path = DRIVE_DATA_PATH,
) -> Path:
    """Write a raw OHLCV DataFrame to Parquet, creating `data_dir` if needed.

    Args:
        df: The OHLCV DataFrame to persist.
        filename: Output file name written inside `data_dir`.
        data_dir: Destination directory (defaults to the Shared Drive path).

    Returns:
        The path the file was written to.
    """
    dest_dir = Path(data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    df.to_parquet(path)
    return path


def get_sp500_data(
    save: bool = True,
    filename: str = RAW_FILENAME,
    data_dir: str | Path = DRIVE_DATA_PATH,
) -> pd.DataFrame:
    """Fetch S&P 500 daily OHLCV and optionally persist it to the Shared Drive.

    This is the public entry point used by the data-collection notebook.

    Args:
        save: If True, write the result to `data_dir` as Parquet.
        filename: Output file name when saving.
        data_dir: Destination directory when saving.

    Returns:
        The fetched OHLCV DataFrame.
    """
    df = fetch_ohlcv()
    if save:
        path = save_raw(df, filename=filename, data_dir=data_dir)
        print(
            f"Saved {len(df):,} rows ({df.index.min().date()} to "
            f"{df.index.max().date()}) to {path}"
        )
    return df


def main() -> None:
    """Fetch and persist the S&P 500 dataset when run as a script."""
    get_sp500_data()


if __name__ == "__main__":
    main()
