"""Chart-pattern features from `zeta-zetra/chart_patterns`.

Wraps three of the library's detectors -- pivot points, double tops/bottoms,
and flags -- into per-bar binary flag columns that line up with
`base_features.make_base_features` output. `add_pattern_features` appends these
columns and returns their names, mirroring that module's contract so
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
    * Every detector confirms a pattern at bar ``t`` by inspecting bars *after*
      ``t`` (a pivot high is only a pivot once the following bars fail to
      exceed it), yet writes its answer onto bar ``t``. Used as-is that is
      lookahead bias: a window ending at ``t`` carries a flag encoding bars
      beyond ``t``, and ``t+1`` is exactly what the model predicts.
      `add_pattern_features` therefore shifts every column forward to the bar
      where it first becomes knowable; see `pattern_lookahead_bars` for the
      per-column lag, which is NOT uniform.
    * The resulting features are lagging by construction -- a double top is
      only usable some bars after its second peak forms. That is the real
      constraint a trader operates under.

Head and shoulders is deliberately excluded. `find_head_and_shoulders` rejects a
candidate unless the neckline is near-flat, via ``upper_slmin`` -- an *absolute*
slope limit in price units per bar, defaulting to 0.0001. That is a sensible
"flat" threshold at forex price levels (~1.1) but is orders of magnitude too
strict for an index in the thousands, so it matched nothing at all on our data.
Relaxing it enough to match anything admits mostly ordinary trend bumps, and
even an appropriately price-relative limit yields only a couple of genuine
patterns across twenty years -- too few for a model to learn from, and it costs
two extra full pivot passes to compute. Recommended future work: scale
``upper_slmin`` to the price level, and use intraday bars or a cross-section of
tickers so the pattern occurs often enough to be informative.
"""

from __future__ import annotations

import pandas as pd

from chart_patterns.chart_patterns.doubles import find_doubles_pattern
from chart_patterns.chart_patterns.flag import find_flag_pattern
from chart_patterns.chart_patterns.pivot_points import find_all_pivot_points

# Binary pattern-flag columns appended by `add_pattern_features`.
PIVOT_HIGH = "pattern_pivot_high"
PIVOT_LOW = "pattern_pivot_low"
DOUBLE_TOP = "pattern_double_top"
DOUBLE_BOTTOM = "pattern_double_bottom"
FLAG = "pattern_flag"

PATTERN_FEATURE_COLUMNS = [
    PIVOT_HIGH,
    PIVOT_LOW,
    DOUBLE_TOP,
    DOUBLE_BOTTOM,
    FLAG,
]

# `doubles.py` and `flag.py` call `find_all_pivot_points(ohlc)` with no arguments,
# so their pivots always use the library's own default right_count. Our
# `pivot_right` does NOT reach them.
_LIBRARY_PIVOT_RIGHT = 3


def pattern_lookahead_bars(pivot_right: int = 3) -> dict[str, int]:
    """Return how many future bars each detector consults, per column.

    A detector confirms a pattern at bar ``t`` by inspecting bars after ``t``, so
    its raw output is only knowable ``lookahead`` bars later. `add_pattern_features`
    shifts each column by its own value here; they are not uniform.

    The ``*_lookback`` parameters contribute nothing: every detector loops
    ``range(lookback, len(ohlc))`` over ``ohlc.loc[candle_idx - lookback:candle_idx]``,
    which is strictly backwards. Only the pivot confirmation looks forward.

    Args:
        pivot_right: Right-hand candles used for the standalone pivot columns.

    Returns:
        A mapping from pattern column name to its lookahead in bars.
    """
    return {
        PIVOT_HIGH: pivot_right,
        PIVOT_LOW: pivot_right,
        # Doubles and flags depend on the pivot at the confirming bar itself,
        # computed with the library's hardcoded right_count.
        DOUBLE_TOP: _LIBRARY_PIVOT_RIGHT,
        DOUBLE_BOTTOM: _LIBRARY_PIVOT_RIGHT,
        FLAG: _LIBRARY_PIVOT_RIGHT,
    }


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
    flag_lookback: int = 25,
) -> tuple[pd.DataFrame, list[str]]:
    """Append chart-pattern flag columns to a cleaned OHLCV frame.

    Runs three `chart_patterns` detectors and reduces each to a `0`/`1` column
    aligned back onto `df`'s original (date) index:
        * ``pattern_pivot_high`` / ``pattern_pivot_low`` -- local swing
          high/low, via `find_all_pivot_points`.
        * ``pattern_double_top`` / ``pattern_double_bottom`` -- via
          `find_doubles_pattern`, called once per side (see module docstring).
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
        flag_lookback: Bars of history the flag scan considers.

    Returns:
        A ``(frame, pattern_columns)`` pair: `df` plus the five pattern
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

    flags = find_flag_pattern(ohlc.copy(), lookback=flag_lookback)
    flag = (flags["chart_type"] == "flag").to_numpy()

    raw = {
        PIVOT_HIGH: pivot_high,
        PIVOT_LOW: pivot_low,
        DOUBLE_TOP: double_top,
        DOUBLE_BOTTOM: double_bottom,
        FLAG: flag,
    }
    lookahead = pattern_lookahead_bars(pivot_right=pivot_right)

    out = df.copy()
    for column in PATTERN_FEATURE_COLUMNS:
        # Move each flag to the bar where it first becomes knowable. Each
        # detector consults a different number of future bars, so the shifts are
        # per column -- a uniform shift would leave the slower ones leaking.
        series = pd.Series(raw[column].astype("int8"), index=df.index)
        out[column] = series.shift(lookahead[column]).fillna(0).astype("int8")

    assert len(out) == n, "pattern flags must align 1:1 with the input rows"

    return out, list(PATTERN_FEATURE_COLUMNS)
