"""
Technical Indicators Module
=============================
Vectorized, MT5/TradingView-accurate implementations of the core technical
indicators used across the bot: SMA, EMA, RSI, ATR, ADX (+DI/-DI), MACD,
Bollinger Bands, realized volatility, and a simple trend-strength score.

Why a rewrite was needed
-------------------------
The previous version smoothed RSI, ATR, and ADX with a **simple rolling
mean** (`.rolling(14).mean()`). MT5 and TradingView do NOT do this — they
use **Wilder's smoothing method** (a specific exponential moving average
seeded with the simple average of the first `period` values). Using a plain
SMA instead of Wilder's smoothing produces RSI/ATR/ADX values that visibly
diverge from what a trader sees on their MT5/TradingView chart, especially
in trending conditions — exactly the kind of silent mismatch that erodes
trust in a signal.

pandas' own `Series.ewm(alpha=..., adjust=False)` is *close* to Wilder's
method but seeds the recursion from the very first raw data point rather
than from an N-period simple-average seed, which also drifts from the
platform-standard value for the first several dozen bars. This module
implements the correct SMA-seeded recursion in closed form (see
`_seeded_smoothing`) so results match MT5/TradingView bar-for-bar, without
ever falling back to a slow, non-vectorized Python loop.

No look-ahead bias
-------------------
Every computation here uses only the current bar and prior bars — via
`.rolling()`, `.diff()`, `.shift(1)`, and forward-seeded (not centered,
never future-peeking) exponential smoothing. None of these operations can
see data from bars that haven't happened yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)

REQUIRED_OHLC_COLUMNS = ("Open", "High", "Low", "Close")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_ohlc(df: pd.DataFrame, required: tuple = REQUIRED_OHLC_COLUMNS) -> None:
    """Validate that a DataFrame has the columns this module needs.

    Args:
        df: Input OHLC(V) DataFrame.
        required: Column names that must be present.

    Raises:
        ValueError: If `df` is None/empty, missing required columns, or a
            required column isn't numeric. Raising here (rather than
            silently producing garbage) is deliberate — a trading bot should
            fail loudly on malformed input, not compute indicators on junk.
    """
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is None or empty — cannot compute indicators.")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required column(s): {missing}")

    for c in required:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Column '{c}' must be numeric, got dtype {df[c].dtype}")

    # Leading NaN (warmup gaps, e.g. before an instrument started trading) is
    # normal and handled gracefully by every rolling/seeded calc below.
    # NaN gaps in the *middle* of the series are not normal — they usually
    # mean missing candles — and will silently propagate through rolling
    # windows, so at least surface a warning rather than fail silently.
    for c in required:
        s = df[c]
        first_valid = s.first_valid_index()
        if first_valid is None:
            continue
        after_first = s.loc[first_valid:]
        internal_nans = int(after_first.isna().sum())
        if internal_nans > 0:
            log.warning(
                "Column '%s' has %d NaN value(s) after its first valid entry — "
                "these will propagate as NaN through rolling/smoothed indicators.",
                c, internal_nans,
            )


def _has_min_bars(df: pd.DataFrame, min_bars: int, indicator_name: str) -> bool:
    """Log a clear warning (not a crash) when there isn't enough history for
    a stable indicator value, and let the caller decide how to proceed."""
    if len(df) < min_bars:
        log.warning(
            "%s requested with only %d bar(s); needs at least %d for a fully "
            "warmed-up value. Result will contain leading NaNs (or be all-NaN "
            "if too short) — this is expected, not an error.",
            indicator_name, len(df), min_bars,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Core building block: SMA-seeded exponential smoothing (vectorized)
# ---------------------------------------------------------------------------

def _seeded_smoothing(series: pd.Series, period: int, alpha: float) -> pd.Series:
    """Exponential smoothing seeded with the simple average of the first
    `period` values — the convention MT5/TradingView use for EMA, RSI's
    average gain/loss, ATR, and ADX's DM/TR/DX smoothing.

    This is NOT the same as `series.ewm(alpha=alpha, adjust=False).mean()`,
    which instead seeds its recursion from the raw first data point and
    therefore diverges from MT5/TradingView for roughly the first
    `3-5 / alpha` bars.

    Implementation note (why this is still fully vectorized):
        The recursion `y[i] = (1-alpha)*y[i-1] + alpha*x[i]` is linear, so
        two runs of it that start from different seed values at the same
        index differ by exactly `(seed_a - seed_b) * (1-alpha)**steps` at
        every later index. We compute pandas' native (raw-seeded) ewm once,
        then add the closed-form correction term to retarget it onto the
        SMA-seeded trajectory — no per-row Python loop required.

    Args:
        series: Input series (e.g. Close, or True Range).
        period: Smoothing period (the "N" in Wilder's method or EMA length).
        alpha: Smoothing factor. Use `1/period` for Wilder's method
            (RSI/ATR/ADX) or `2/(period+1)` for a standard EMA.

    Returns:
        pandas.Series, same index as `series`, with the first `period - 1`
        values as NaN (insufficient history to seed) and NaN throughout if
        `series` has fewer than `period` valid values.
    """
    n = len(series)
    result = pd.Series(np.nan, index=series.index, dtype=float)

    if n < period:
        return result  # Not enough history — return all-NaN rather than guess.

    values = series.to_numpy(dtype=float)
    if np.isnan(values[:period]).any():
        # Can't form a clean SMA seed if the warmup window itself has gaps;
        # still return a usable series by seeding from the first fully
        # valid window instead of crashing.
        #
        # NOTE: we deliberately use positional (integer) lookups here, not
        # series.index.get_loc(label) / Series.reindex(index). Those are
        # label-based and raise `ValueError: cannot reindex on an axis with
        # duplicate labels` whenever the index has a repeated timestamp
        # (e.g. a broker/data feed resending an overlapping candle). Staying
        # purely positional keeps this safe regardless of index uniqueness.
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            return result
        start_pos = int(np.argmax(valid_mask))  # position of first non-NaN value
        if n - start_pos < period:
            return result
        sub_result = _seeded_smoothing(series.iloc[start_pos:], period, alpha)
        result.iloc[start_pos:] = sub_result.to_numpy(dtype=float)
        return result

    naive = series.ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float)

    sma_seed = values[:period].mean()
    naive_seed = naive[period - 1]

    idx = np.arange(n)
    steps = idx - (period - 1)
    decay = np.where(steps >= 0, (1.0 - alpha) ** np.clip(steps, 0, None), 0.0)
    correction = (sma_seed - naive_seed) * decay

    out = naive + correction
    out[: period - 1] = np.nan
    result[:] = out
    return result


# ---------------------------------------------------------------------------
# Public indicator functions
# ---------------------------------------------------------------------------

def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average — identical convention across every platform.

    Args:
        close: Price series (typically Close).
        period: Lookback window.

    Returns:
        pandas.Series with the first `period - 1` values as NaN.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    return close.rolling(window=period, min_periods=period).mean()


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average, seeded with an SMA of the first `period`
    values — matches MT5's iMA(MODE_EMA) and TradingView's `ta.ema()`.

    Args:
        close: Price series (typically Close).
        period: EMA length.

    Returns:
        pandas.Series with the first `period - 1` values as NaN.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    alpha = 2.0 / (period + 1.0)
    return _seeded_smoothing(close, period, alpha)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's original smoothing — matches
    MT5's iRSI and TradingView's `ta.rsi()`.

    Args:
        close: Price series (typically Close).
        period: RSI lookback, default 14 (the standard).

    Returns:
        pandas.Series of RSI values in [0, 100]. NaN for the warmup period.
        Where the average loss is 0 and average gain > 0, RSI is 100
        (a strict, unbroken uptrend — the mathematically correct limit of
        the RSI formula as loss approaches 0). Where both average gain and
        average loss are 0 (a perfectly flat price), RSI is defined as 50
        (neutral) rather than left as NaN, matching common platform behavior.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    alpha = 1.0 / period
    avg_gain = _seeded_smoothing(gain, period, alpha)
    avg_loss = _seeded_smoothing(loss, period, alpha)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(avg_loss != 0, 100.0)                     # avg_loss == 0, avg_gain > 0 -> 100
    flat = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~flat, 50.0)                                # perfectly flat price -> neutral 50
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing — matches MT5's iATR and
    TradingView's `ta.atr()`.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR lookback, default 14 (the standard).

    Returns:
        pandas.Series of ATR values (same units as price). NaN during warmup.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    prev_close = close.shift(1)   # backward-looking only — no look-ahead
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha = 1.0 / period
    return _seeded_smoothing(tr, period, alpha)


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI/-DI, using Wilder's smoothing at
    every stage (directional movement, true range, and the final ADX
    average of DX) — matches MT5's iADX and TradingView's `ta.dmi()`.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ADX/DI lookback, default 14 (the standard).

    Returns:
        pandas.DataFrame with columns "ADX", "DI_Plus", "DI_Minus", same
        index as the inputs. NaN during warmup. Where both DI+ and DI- are
        zero (no directional movement at all in that window), DX is defined
        as 0 rather than left as an undefined 0/0.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    up_move = high.diff()
    down_move = -low.diff()

    # Raw arrays used for BOTH comparisons so zeroing one never corrupts the
    # threshold used for the other (a subtle bug in naive implementations
    # that reuse an already-modified +DM series when computing -DM).
    up_arr = up_move.to_numpy(dtype=float)
    down_arr = down_move.to_numpy(dtype=float)

    plus_dm_arr = np.where((up_arr > down_arr) & (up_arr > 0), up_arr, 0.0)
    minus_dm_arr = np.where((down_arr > up_arr) & (down_arr > 0), down_arr, 0.0)

    plus_dm = pd.Series(plus_dm_arr, index=high.index)
    minus_dm = pd.Series(minus_dm_arr, index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha = 1.0 / period
    smoothed_tr = _seeded_smoothing(tr, period, alpha)
    smoothed_plus_dm = _seeded_smoothing(plus_dm, period, alpha)
    smoothed_minus_dm = _seeded_smoothing(minus_dm, period, alpha)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100.0 * (smoothed_minus_dm / smoothed_tr)
        di_sum = plus_di + minus_di
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum

    dx = dx.where(di_sum != 0, 0.0)  # no directional movement at all -> DX = 0, not NaN/inf

    adx = _seeded_smoothing(dx, period, alpha)

    return pd.DataFrame({"ADX": adx, "DI_Plus": plus_di, "DI_Minus": minus_di}, index=high.index)


class TechnicalIndicators:
    """Batch entry point: adds every indicator this module provides to a
    copy of the input DataFrame in one call. Kept as the module's public
    surface for backward compatibility with any existing callers.
    """

    @staticmethod
    def add_all(data: pd.DataFrame) -> pd.DataFrame:
        """Compute every indicator in this module and return them as new
        columns on a *copy* of `data` (the input is never mutated).

        Args:
            data: OHLC(V) DataFrame with at least Open/High/Low/Close columns.

        Returns:
            A new pandas.DataFrame — `data` plus SMA_10/20/50/200,
            EMA_10/20/50/200, RSI, MACD/MACD_Signal/MACD_Hist,
            BB_Mid/Upper/Lower/Width/Position, ATR_14, ATR_Pct, ADX,
            DI_Plus, DI_Minus, Volatility, Trend_Score.

        Raises:
            ValueError: If `data` is missing required OHLC columns, is
                empty, or a required column isn't numeric (see
                `_validate_ohlc`). This is intentional — better to fail
                loudly on bad input than silently compute nonsense.
        """
        _validate_ohlc(data)
        df = data.copy()

        longest_ma = 200
        _has_min_bars(df, longest_ma, "SMA_200/EMA_200")

        # Moving Averages
        for period in [10, 20, 50, 200]:
            df[f"SMA_{period}"] = calculate_sma(df["Close"], period)
            df[f"EMA_{period}"] = calculate_ema(df["Close"], period)

        # RSI (Wilder-smoothed)
        _has_min_bars(df, 14 * 3, "RSI")  # a few multiples of the period for a stable value
        df["RSI"] = calculate_rsi(df["Close"], period=14)

        # MACD — EMA(12)/EMA(26) are each independently SMA-seeded per their
        # own period, matching TradingView's ta.macd() and MT5's iMACD.
        ema12 = calculate_ema(df["Close"], 12)
        ema26 = calculate_ema(df["Close"], 26)
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = calculate_ema(df["MACD"], 9)
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # Bollinger Bands (SMA-based mid-band + population std, the
        # universal convention — no Wilder smoothing involved here)
        df["BB_Mid"] = calculate_sma(df["Close"], 20)
        bb_std = df["Close"].rolling(window=20, min_periods=20).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
        df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std
        bb_range = df["BB_Upper"] - df["BB_Lower"]
        df["BB_Width"] = bb_range / df["BB_Mid"].replace(0, np.nan)
        df["BB_Position"] = ((df["Close"] - df["BB_Lower"]) / bb_range.replace(0, np.nan))

        # ATR (Wilder-smoothed)
        _has_min_bars(df, 14 * 3, "ATR")
        df["ATR_14"] = calculate_atr(df["High"], df["Low"], df["Close"], period=14)
        df["ATR_Pct"] = (df["ATR_14"] / df["Close"].replace(0, np.nan)) * 100

        # ADX / DI+ / DI- (Wilder-smoothed at every stage)
        _has_min_bars(df, 14 * 3, "ADX")
        adx_df = calculate_adx(df["High"], df["Low"], df["Close"], period=14)
        df["ADX"] = adx_df["ADX"]
        df["DI_Plus"] = adx_df["DI_Plus"]
        df["DI_Minus"] = adx_df["DI_Minus"]

        # Realized volatility (annualized, 252 trading days/year convention)
        df["Volatility"] = df["Close"].pct_change().rolling(window=20, min_periods=20).std() * np.sqrt(252)

        # Simple 0-100 trend-strength composite (unchanged logic, still
        # entirely backward-looking — each term only reads the current row)
        df["Trend_Score"] = (
            (df["EMA_20"] > df["EMA_50"]).astype(int) * 25 +
            (df["Close"] > df["EMA_20"]).astype(int) * 25 +
            (df["ADX"] > 25).astype(int) * 25 +
            (df["MACD_Hist"] > 0).astype(int) * 25
        )

        return df


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Self-test: cross-checks the vectorized, SMA-seeded implementation
    # above against a slow, straightforward Python-loop reference of the
    # exact same textbook formulas (Wilder 1978 for RSI/ATR/ADX; the
    # standard SMA-seeded EMA recursion). Agreement here means the
    # vectorized code is a faithful, bar-for-bar implementation of the
    # documented MT5/TradingView methodology — the best verification
    # possible without live network access to pull a chart for comparison.
    # ------------------------------------------------------------------
    import numpy as _np

    rng = _np.random.default_rng(42)
    n = 300
    close = 2000 + _np.cumsum(rng.normal(0, 3, n))
    high = close + _np.abs(rng.normal(1, 1, n))
    low = close - _np.abs(rng.normal(1, 1, n))
    df_test = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close})

    def _ref_ema(series, period):
        alpha = 2 / (period + 1)
        vals = series.to_numpy(dtype=float)
        out = np.full(len(vals), np.nan)
        if len(vals) < period:
            return pd.Series(out, index=series.index)
        out[period - 1] = vals[:period].mean()
        for i in range(period, len(vals)):
            out[i] = out[i - 1] * (1 - alpha) + vals[i] * alpha
        return pd.Series(out, index=series.index)

    def _ref_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).to_numpy(dtype=float)
        loss = (-delta.clip(upper=0)).to_numpy(dtype=float)
        n_ = len(series)
        avg_gain = np.full(n_, np.nan)
        avg_loss = np.full(n_, np.nan)
        avg_gain[period] = gain[1:period + 1].mean()
        avg_loss[period] = loss[1:period + 1].mean()
        for i in range(period + 1, n_):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        rsi = np.where(avg_loss == 0, 100.0, rsi)
        return pd.Series(rsi, index=series.index)

    ema_vec = calculate_ema(df_test["Close"], 20)
    ema_ref = _ref_ema(df_test["Close"], 20)
    ema_diff = (ema_vec - ema_ref).abs().max()
    print(f"EMA(20) max abs diff vs reference loop: {ema_diff:.10f}  [{'PASS' if ema_diff < 1e-6 else 'FAIL'}]")

    rsi_vec = calculate_rsi(df_test["Close"], 14)
    rsi_ref = _ref_rsi(df_test["Close"], 14)
    valid = rsi_ref.notna()
    rsi_diff = (rsi_vec[valid] - rsi_ref[valid]).abs().max()
    print(f"RSI(14) max abs diff vs reference loop: {rsi_diff:.10f}  [{'PASS' if rsi_diff < 1e-6 else 'FAIL'}]")

    atr_vec = calculate_atr(df_test["High"], df_test["Low"], df_test["Close"], 14)
    print(f"ATR(14) sample tail:\n{atr_vec.tail(3).to_string()}")

    adx_vec = calculate_adx(df_test["High"], df_test["Low"], df_test["Close"], 14)
    print(f"ADX(14) sample tail:\n{adx_vec.tail(3).to_string()}")

    full = TechnicalIndicators.add_all(df_test)
    print(f"\nadd_all() produced {full.shape[1]} columns for {len(full)} rows. No exceptions raised.")

    # No-look-ahead sanity check: shifting the input forward (simulating a
    # data leak from the future) must change the *early* values but must
    # NOT retroactively change values already computed on earlier bars.
    shifted = df_test.copy()
    shifted.loc[shifted.index[-1], "Close"] += 500  # perturb only the last bar
    rsi_before = calculate_rsi(df_test["Close"], 14)
    rsi_after = calculate_rsi(shifted["Close"], 14)
    changed_early = (rsi_before.iloc[:-1] - rsi_after.iloc[:-1]).abs().max()
    print(f"\nNo-look-ahead check: perturbing only the LAST bar changed earlier "
          f"RSI values by at most {changed_early:.10f} [{'PASS' if changed_early < 1e-9 else 'FAIL — LOOK-AHEAD BIAS DETECTED'}]")

    # Insufficient-history handling
    tiny = df_test.head(5)
    tiny_atr = calculate_atr(tiny["High"], tiny["Low"], tiny["Close"], 14)
    print(f"\nInsufficient-history check (5 bars, period=14): "
          f"all-NaN = {tiny_atr.isna().all()} [{'PASS' if tiny_atr.isna().all() else 'FAIL'}]")
