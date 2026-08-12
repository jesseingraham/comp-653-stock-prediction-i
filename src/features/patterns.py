"""Chart-pattern features from `zeta-zetra/chart_patterns`.

Wraps four of the library's detectors -- pivot points, double tops/bottoms,
head-and-shoulders, and flags -- into per-bar binary flag columns that line up
with `base_features.make_base_features` output. `add_pattern_features` appends
these columns and returns their names, mirroring that module's contract so
`02_feature_engineering` can do:

    features, base_columns = make_base_features(clean, ...)
    features, pattern_columns = add_pattern_features(features)
    # feature_columns for the augmented dataset = base_columns + pattern_columns

Vendoring note (why this isn't a normal pip dependency): the package on PyPI/
GitHub pins `pandas==1.3.5`, caps Python at `<3.11`, and pins the Windows-only
`kaleido==0.1.0.post1` -- any one of these breaks a `pip install -r
requirements.txt` that also needs `ray`/`optuna`, since pip resolves the whole
file before installing anything. Its modules also import
`chart_patterns.chart_patterns.*`, a nested layout its own flat install never
creates. The detector code itself has no problem with modern pandas, so the
setup is to clone the repo and put the clone's *parent* directory on
`sys.path` (done once, in the notebook's environment-setup cell):

    !git clone -q https://github.com/zeta-zetra/chart_patterns.git /content/vendor/chart_patterns
    sys.path.append('/content/vendor')

This module then imports `chart_patterns.chart_patterns.*` normally.

Detector quirks this module works around:
    * Every detector indexes by integer row *label*, not position (e.g.
      `ohlc.loc[idx, "low"]`), so the input needs a plain `0..n-1` RangeIndex --
      not `df`'s date index.
    * Column names must be lower-case (`open`/`high`/`low`/`close`).
    * `find_doubles_pattern`'s `double="both"` branch is unreachable (an
      `if ... or "both"` shadows the `elif` that would catch bottoms), so tops
      and bottoms are detected with two separate calls here instead.
    * Each detector's output only marks the bar at which a completed pattern
      is *recognized* (a lagging signal by construction, since e.g. a double
      top needs the second peak to have already formed) -- these are flags on
      the confirming bar, not the pattern's start.
"""

from __future__ import annotations

import pandas as pd

from chart_patterns.chart_patterns.doubles import find_doubles_pattern
from chart_patterns.chart_patterns.flag import find_flag_pattern
from chart_patterns.chart_patterns.head_and_shoulders import find_head_and_shoulders
from chart_patterns.chart_patterns.pivot_points import find_all_pivot_points

# Binary pattern-flag columns appended by `add_pattern_features`.
PIVOT_HIGH = "pattern_pivot_high"
PIVOT_LOW = "pattern_pivot_low"
DOUBLE_TOP = "pattern_double_top"
DOUBLE_BOTTOM = "pattern_double_bottom"
HEAD_AND_SHOULDERS = "pattern_head_and_shoulders"
FLAG = "pattern_flag"

PATTERN_FEATURE_COLUMNS = [
    PIVOT_HIGH,
    PIVOT_LOW,
    DOUBLE_TOP,
    DOUBLE_BOTTOM,
    HEAD_AND_SHOULDERS,
    FLAG,
]


def _ohlc_for_detectors(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-cased OHLC with a `0..n-1` RangeIndex, as the detectors require."""
    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input is missing required columns: {missing}")
    return df[required].rename(columns=str.lower).reset_index(drop=True)


def add_pattern_features(
    df: pd.DataFrame,
    pivot_left: int = 3,
    pivot_right: int = 3,
    doubles_lookback: int = 25,
    hs_lookback: int = 60,
    flag_lookback: int = 25,
) -> tuple[pd.DataFrame, list[str]]:
    """Append chart-pattern flag columns to a cleaned OHLCV frame.

    Runs four `chart_patterns` detectors and reduces each to a `0`/`1` column
    aligned back onto `df`'s original (date) index:
        * ``pattern_pivot_high`` / ``pattern_pivot_low`` -- local swing
          high/low, via `find_all_pivot_points`.
        * ``pattern_double_top`` / ``pattern_double_bottom`` -- via
          `find_doubles_pattern`, called once per side (see module docstring).
        * ``pattern_head_and_shoulders`` -- via `find_head_and_shoulders`.
        * ``pattern_flag`` -- via `find_flag_pattern`.

    All columns are dense `0`/`1` (no NaNs), so they never cause
    `windowing.make_windows` to drop a window on their account; only the base
    features' leading NaNs do that.

    Args:
        df: Cleaned OHLCV frame indexed by date, ascending (as returned by
            `base_features.make_base_features`, or the raw cleaned frame).
        pivot_left: Candles to the left considered when testing a pivot.
        pivot_right: Candles to the right considered when testing a pivot.
        doubles_lookback: Bars of history the double-top/bottom scan considers.
        hs_lookback: Bars of history the head-and-shoulders scan considers.
        flag_lookback: Bars of history the flag scan considers.

    Returns:
        A ``(frame, pattern_columns)`` pair: `df` plus the six pattern
        columns, and the list of their names (`PATTERN_FEATURE_COLUMNS`).

    Raises:
        KeyError: If any of Open/High/Low/Close is missing.
    """
    ohlc = _ohlc_for_detectors(df)
    n = len(ohlc)

    pivots = find_all_pivot_points(
        ohlc.copy(), left_count=pivot_left, right_count=pivot_right
    )
    pivot_high = (pivots["pivot"] == 2).to_numpy()
    pivot_low = (pivots["pivot"] == 1).to_numpy()

    tops = find_doubles_pattern(ohlc.copy(), lookback=doubles_lookback, double="tops")
    double_top = (
        (tops["chart_type"] == "double") & (tops["double_type"] == "tops")
    ).to_numpy()

    bottoms = find_doubles_pattern(
        ohlc.copy(), lookback=doubles_lookback, double="bottoms"
    )
    double_bottom = (
        (bottoms["chart_type"] == "double") & (bottoms["double_type"] == "bottoms")
    ).to_numpy()

    hs = find_head_and_shoulders(ohlc.copy(), lookback=hs_lookback)
    head_and_shoulders = (hs["chart_type"] == "hs").to_numpy()

    flags = find_flag_pattern(ohlc.copy(), lookback=flag_lookback)
    flag = (flags["chart_type"] == "flag").to_numpy()

    out = df.copy()
    out[PIVOT_HIGH] = pivot_high.astype("int8")
    out[PIVOT_LOW] = pivot_low.astype("int8")
    out[DOUBLE_TOP] = double_top.astype("int8")
    out[DOUBLE_BOTTOM] = double_bottom.astype("int8")
    out[HEAD_AND_SHOULDERS] = head_and_shoulders.astype("int8")
    out[FLAG] = flag.astype("int8")

    assert len(out) == n, "pattern flags must align 1:1 with the input rows"

    return out, list(PATTERN_FEATURE_COLUMNS)
