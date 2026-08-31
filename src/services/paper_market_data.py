"""Deterministic, time-anchored offline market data for paper mode.

Paper mode must never depend on Yahoo Finance or another external market-data
provider. This service generates a synthetic OHLCV stream suitable for
UI/demo/paper analysis and paper backtests.

BUG FIX #1 (audit): the original implementation seeded a single RNG per
(symbol, timeframe) and drew `bars` values every call, always keeping the
*most recent* draw as the "current" price regardless of the real wall-clock
`end` timestamp passed in. That made `paper_price()` return the exact same
bid/ask forever. Fixed by keying every bar's price to its absolute bar index
since a fixed epoch, computed from the real timestamp — deterministic for a
given (symbol, timeframe, timestamp), but genuinely time-varying.

BUG FIX #2 (audit): the first fix used `hashlib.sha256(...)` +
`np.random.default_rng(seed)` PER BAR in a Python loop. That's fine for a
few hundred bars but becomes extremely slow for large multi-year backtest
ranges (tens/hundreds of thousands of bars) — it could hang for minutes.
Replaced with a fully vectorized numpy integer hash (SplitMix64-style),
which is deterministic and reproducible exactly like before, but computes
100k+ bars in milliseconds instead of minutes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import numpy as np
import pandas as pd


_BASE_PRICES = {
    "XAUUSD": 2450.0, "XAUUSDm": 2450.0,
    "EURUSD": 1.0850, "EURUSDm": 1.0850,
    "GBPUSD": 1.2700, "GBPUSDm": 1.2700,
    "USDJPY": 155.0, "USDJPYm": 155.0,
    "AUDUSD": 0.6600, "AUDUSDm": 0.6600,
}
_STEP = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)

_U64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _seed64(text: str) -> np.uint64:
    raw = hashlib.sha256(text.encode()).hexdigest()[:16]
    return np.uint64(int(raw, 16))


def _vec_hash01(seed_text: str, idx: np.ndarray) -> np.ndarray:
    """Vectorized, deterministic pseudo-random values in [0, 1) for each
    integer in `idx`, keyed by `seed_text`. SplitMix64-style integer mixer —
    fast for large arrays, unlike per-element hashlib+RNG calls."""
    base = _seed64(seed_text)
    x = (idx.astype(np.uint64) + base + np.uint64(0x9E3779B97F4A7C15)) & _U64_MASK
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _U64_MASK
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _U64_MASK
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _vec_normal(seed_text: str, idx: np.ndarray, std: float = 1.0) -> np.ndarray:
    """Vectorized approx-normal noise via Box-Muller on two hash streams."""
    u1 = np.clip(_vec_hash01(seed_text + ":u1", idx), 1e-12, 1.0)
    u2 = _vec_hash01(seed_text + ":u2", idx)
    return std * np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def _bar_index(ts: pd.Timestamp, minutes: int) -> int:
    delta = ts.to_pydatetime() - _EPOCH
    return int(delta.total_seconds() // (minutes * 60))


def _symbol_harmonics(symbol: str):
    seed = int(_seed64(symbol + ":harmonics"))
    rng = np.random.default_rng(seed % (2**32 - 1))
    phases = rng.uniform(0, 2 * np.pi, 3)
    freqs = rng.uniform(0.015, 0.06, 3)
    weights = np.array([1.0, 0.6, 0.35])
    return phases, freqs, weights


def _level(symbol: str, base: float, idx) -> np.ndarray:
    """Deterministic, continuous-looking price level for absolute bar index/indices."""
    phases, freqs, weights = _symbol_harmonics(symbol)
    scale = max(base * 0.006, 0.05)
    idx = np.asarray(idx, dtype=np.float64)
    trend = sum(w * np.sin(idx * f + p) for f, p, w in zip(freqs, phases, weights))
    trend = trend / weights.sum() * scale
    noise = _vec_normal(symbol + ":level", np.asarray(idx, dtype=np.int64), scale * 0.25)
    return base + trend + noise


def generate_paper_ohlcv(symbol: str = "XAUUSD", timeframe: str = "M15",
                         bars: int = 500, end: datetime | None = None) -> pd.DataFrame:
    """Return synthetic OHLCV data anchored to real elapsed time."""
    bars = max(10, min(int(bars), 120000))
    minutes = _STEP.get(timeframe.upper())
    if minutes is None:
        raise ValueError(f"Unsupported paper timeframe: {timeframe}")
    if end is None:
        end = datetime.now(timezone.utc)
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    end_ts = end_ts.floor(f"{minutes}min")
    idx_dates = pd.date_range(end=end_ts, periods=bars, freq=f"{minutes}min", tz="UTC")

    base = float(_BASE_PRICES.get(symbol, 100.0))
    end_idx = _bar_index(end_ts, minutes)
    bar_indices = np.arange(end_idx - bars + 1, end_idx + 1, dtype=np.int64)

    close = _level(symbol, base, bar_indices)
    close = np.maximum(close, base * 0.2)
    open_prev = _level(symbol, base, bar_indices - 1)
    open_ = np.maximum(open_prev, base * 0.2)

    scale = max(base * 0.0008, 0.02)
    spread_u = _vec_hash01(symbol + ":" + timeframe + ":spread", bar_indices)
    spread = (0.15 + 0.75 * spread_u) * scale + scale * 0.1
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol_u = _vec_hash01(symbol + ":" + timeframe + ":vol", bar_indices)
    volume = (100 + vol_u * 1900).astype(np.int64)

    return pd.DataFrame({
        "Date": idx_dates, "Open": open_, "High": high,
        "Low": low, "Close": close, "Volume": volume
    }).reset_index(drop=True)


def paper_price(symbol: str = "XAUUSD", timeframe: str = "M15") -> dict[str, float]:
    """Live-ish synthetic bid/ask that ticks with real wall-clock time.

    Interpolates between the previous and current bar's level using the
    fractional position within the current bar (based on actual elapsed
    seconds), plus a small per-second deterministic jitter, so consecutive
    calls a few seconds apart return visibly different, non-static values.
    """
    minutes = _STEP.get(timeframe.upper(), 15)
    now = datetime.now(timezone.utc)
    base = float(_BASE_PRICES.get(symbol, 100.0))
    now_ts = pd.Timestamp(now)
    bar_start = now_ts.floor(f"{minutes}min")
    idx = _bar_index(bar_start, minutes)
    frac = (now_ts - bar_start).total_seconds() / (minutes * 60.0)
    level_prev = float(_level(symbol, base, np.array([idx], dtype=np.int64))[0])
    level_next = float(_level(symbol, base, np.array([idx + 1], dtype=np.int64))[0])
    mid = level_prev + (level_next - level_prev) * frac

    scale = max(base * 0.0008, 0.02)
    second_bucket = int(now.timestamp())
    jitter = float(_vec_normal(symbol + ":" + timeframe + ":tick",
                                np.array([second_bucket], dtype=np.int64), scale * 0.2)[0])
    close = max(mid + jitter, base * 0.2)
    spread = max(close * 0.0002, 0.0002)
    return {"bid": close, "ask": close + spread}
