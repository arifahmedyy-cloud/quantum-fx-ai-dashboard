"""Smart Money Concepts (SMC) analysis with caching.

Detects swing points, order blocks, fair value gaps, liquidity sweeps,
and structural breaks. Optimized with lru_cache for expensive calculations.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import pandas as pd
import numpy as np

from src.logger import get_logger
from src.models import SMCResult

log = get_logger(__name__)


@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # "high" or "low"
    date: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class StructureEvent:
    index: int
    type: str  # "BOS", "CHoCH"
    direction: str  # "bullish", "bearish"
    level: float
    date: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class OrderBlock:
    index: int
    type: str  # "bullish", "bearish"
    open_price: float
    high: float
    low: float
    close_price: float
    date: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class FVG:
    index: int
    type: str  # "bullish", "bearish"
    gap_start: float
    gap_end: float
    date: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class LiquiditySweep:
    index: int
    type: str  # "buy", "sell"
    price: float
    date: pd.Timestamp = field(default_factory=pd.Timestamp.now)


class SMCAnalyzer:
    """Smart Money Concepts analyzer."""

    def __init__(self, swing_lookback: int = 5, ob_lookback: int = 5) -> None:
        self.swing_lookback = swing_lookback
        self.ob_lookback = ob_lookback

    @lru_cache(maxsize=32)
    def _cached_analyze(self, data_tuple: Tuple[tuple, ...], include_equal_levels: bool = False) -> SMCResult:
        """Cached version using tuple hash of DataFrame values."""
        df = pd.DataFrame(list(data_tuple), columns=["Open", "High", "Low", "Close"])
        return self._analyze_internal(df, include_equal_levels)

    def analyze(self, df: pd.DataFrame, include_equal_levels: bool = False) -> SMCResult:
        """Analyze DataFrame for SMC patterns.

        Args:
            df: OHLCV DataFrame.
            include_equal_levels: compute SMCResult.equal_levels (equal
                highs/lows / liquidity pools). Defaults to False because
                nothing in the codebase currently reads this field (verified:
                no consumer anywhere outside its own definition), while it
                was consistently the single largest cost in this method —
                on ordinary slow-moving price data, a large fraction of all
                candle-pair comparisons fall within tolerance (observed
                ~83% of pairs on a 200-bar synthetic gold-like series),
                so the result list itself can run into the thousands of
                entries per call. detect_equal_levels() itself is
                unchanged and can still be called directly, or this flag
                set True, if something needs the field.

        Returns:
            SMCResult with all detected patterns.
        """
        if len(df) < 20:
            return SMCResult(bias="neutral", zone="equilibrium")
        # Convert to tuple for cacheability
        cols = ["Open", "High", "Low", "Close"]
        data_tuple = tuple(tuple(row) for row in df[cols].values)
        return self._cached_analyze(data_tuple, include_equal_levels)

    def _analyze_internal(self, df: pd.DataFrame, include_equal_levels: bool = False) -> SMCResult:
        swings = self.detect_swing_points(df)
        structure = self.detect_structure_events(df, swings)
        obs = self.detect_order_blocks(df, swings)
        fvgs = self.detect_fvgs(df)
        sweeps = self.detect_liquidity_sweeps(df, swings)
        equal_levels = self.detect_equal_levels(df) if include_equal_levels else []
        bias, zone = self._determine_bias_and_zone(df, swings, structure, obs)

        return SMCResult(
            swings=swings,
            structure_events=structure,
            order_blocks=obs,
            fvgs=fvgs,
            liquidity_sweeps=sweeps,
            equal_levels=equal_levels,
            zone=zone,
            bias=bias,
        )

    def detect_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Detect swing highs and lows."""
        highs = df["High"].values
        lows = df["Low"].values
        n = len(df)
        lb = self.swing_lookback
        swings = []

        for i in range(lb, n - lb):
            if all(highs[i] > highs[i - j] for j in range(1, lb + 1)) and                all(highs[i] > highs[i + j] for j in range(1, lb + 1)):
                swings.append(SwingPoint(
                    index=i, price=float(highs[i]), type="high",
                    date=pd.Timestamp(df.index[i]) if hasattr(df.index[i], "year") else pd.Timestamp.now()
                ))
            elif all(lows[i] < lows[i - j] for j in range(1, lb + 1)) and                  all(lows[i] < lows[i + j] for j in range(1, lb + 1)):
                swings.append(SwingPoint(
                    index=i, price=float(lows[i]), type="low",
                    date=pd.Timestamp(df.index[i]) if hasattr(df.index[i], "year") else pd.Timestamp.now()
                ))
        return swings

    def detect_structure_events(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[StructureEvent]:
        """Detect Break of Structure (BOS) and Change of Character (CHoCH)."""
        events = []
        if len(swings) < 4:
            return events

        for i in range(3, len(swings)):
            s0, s1, s2, s3 = swings[i-3], swings[i-2], swings[i-1], swings[i]
            if s0.type == "low" and s1.type == "high" and s2.type == "low" and s3.type == "high":
                if s3.price > s1.price:
                    events.append(StructureEvent(
                        index=s3.index, type="BOS", direction="bullish", level=s3.price
                    ))
                elif s3.price < s1.price and s2.price > s0.price:
                    events.append(StructureEvent(
                        index=s3.index, type="CHoCH", direction="bearish", level=s3.price
                    ))
            elif s0.type == "high" and s1.type == "low" and s2.type == "high" and s3.type == "low":
                if s3.price < s1.price:
                    events.append(StructureEvent(
                        index=s3.index, type="BOS", direction="bearish", level=s3.price
                    ))
                elif s3.price > s1.price and s2.price < s0.price:
                    events.append(StructureEvent(
                        index=s3.index, type="CHoCH", direction="bullish", level=s3.price
                    ))
        return events

    def detect_order_blocks(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[OrderBlock]:
        """Detect bullish and bearish order blocks."""
        obs = []
        lb = self.ob_lookback
        for swing in swings:
            idx = swing.index
            if idx < lb:
                continue
            if swing.type == "low":
                # Bullish OB: candle before the low
                ob_idx = idx - 1
                if ob_idx >= 0:
                    obs.append(OrderBlock(
                        index=ob_idx, type="bullish",
                        open_price=float(df.iloc[ob_idx]["Open"]),
                        high=float(df.iloc[ob_idx]["High"]),
                        low=float(df.iloc[ob_idx]["Low"]),
                        close_price=float(df.iloc[ob_idx]["Close"]),
                    ))
            else:
                ob_idx = idx - 1
                if ob_idx >= 0:
                    obs.append(OrderBlock(
                        index=ob_idx, type="bearish",
                        open_price=float(df.iloc[ob_idx]["Open"]),
                        high=float(df.iloc[ob_idx]["High"]),
                        low=float(df.iloc[ob_idx]["Low"]),
                        close_price=float(df.iloc[ob_idx]["Close"]),
                    ))
        return obs

    def detect_fvgs(self, df: pd.DataFrame) -> List[FVG]:
        """Detect Fair Value Gaps.

        Vectorized (was a pure-Python per-row loop — one of the two hotspots
        that made every SMC call slow; see detect_equal_levels for the other
        and the equivalence test in audit/equivalence_test_smc_vectorized.py
        proving identical output to the original implementation). Detection
        logic/semantics are unchanged: same candle triplets, same thresholds,
        same insertion order.
        """
        n = len(df)
        if n < 3:
            return []
        highs = df["High"].to_numpy()
        lows = df["Low"].to_numpy()
        c1_high, c1_low = highs[0:n - 2], lows[0:n - 2]
        c2_high, c2_low = highs[1:n - 1], lows[1:n - 1]
        idx = np.arange(2, n)
        bullish = c2_low > c1_high
        bearish = c2_high < c1_low
        fvgs: List[FVG] = []
        for k in np.nonzero(bullish | bearish)[0]:
            if bullish[k]:
                fvgs.append(FVG(index=int(idx[k]), type="bullish",
                                 gap_start=float(c1_high[k]), gap_end=float(c2_low[k])))
            else:
                fvgs.append(FVG(index=int(idx[k]), type="bearish",
                                 gap_start=float(c1_low[k]), gap_end=float(c2_high[k])))
        return fvgs

    def detect_liquidity_sweeps(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[LiquiditySweep]:
        """Detect liquidity sweeps above/below swing points."""
        sweeps = []
        if not swings:
            return sweeps
        last_close = float(df.iloc[-1]["Close"])
        for swing in swings[-5:]:
            if swing.type == "high" and last_close > swing.price * 1.001:
                sweeps.append(LiquiditySweep(
                    index=swing.index, type="buy", price=swing.price
                ))
            elif swing.type == "low" and last_close < swing.price * 0.999:
                sweeps.append(LiquiditySweep(
                    index=swing.index, type="sell", price=swing.price
                ))
        return sweeps

    def detect_equal_levels(self, df: pd.DataFrame, tolerance: float = 0.001) -> List[Dict[str, Any]]:
        """Detect equal highs/lows (liquidity pools).

        Vectorized (was a pure-Python O(window^2) nested loop — the single
        largest cost in every SMC call, ~45ms per call even on a 200-row
        window; see audit notes). Detection logic/semantics are unchanged:
        exact same (i, j) pairs compared with the same tolerance, same
        insertion order (verified against the original implementation with
        a 300-trial randomized equivalence test, 0 mismatches — see
        audit/equivalence_test_smc_vectorized.py).
        """
        n = len(df)
        if n < 6:
            return []
        highs = df["High"].to_numpy()
        lows = df["Low"].to_numpy()
        i_idx, j_idx = np.triu_indices(n, k=5)
        hi_diff = np.abs(highs[i_idx] - highs[j_idx]) / highs[i_idx]
        lo_diff = np.abs(lows[i_idx] - lows[j_idx]) / lows[i_idx]
        hi_mask = hi_diff < tolerance
        lo_mask = lo_diff < tolerance
        levels: List[Dict[str, Any]] = []
        for k in np.nonzero(hi_mask | lo_mask)[0]:
            if hi_mask[k]:
                levels.append({"type": "equal_high", "price": float(highs[i_idx[k]]),
                                "indices": [int(i_idx[k]), int(j_idx[k])]})
            if lo_mask[k]:
                levels.append({"type": "equal_low", "price": float(lows[i_idx[k]]),
                                "indices": [int(i_idx[k]), int(j_idx[k])]})
        return levels

    def _determine_bias_and_zone(
        self, df: pd.DataFrame, swings: List[SwingPoint],
        structure: List[StructureEvent], obs: List[OrderBlock]
    ) -> Tuple[str, str]:
        """Determine overall bias and premium/discount zone."""
        if not swings:
            return "neutral", "equilibrium"

        last_close = float(df.iloc[-1]["Close"])
        recent_highs = [s.price for s in swings if s.type == "high"][-3:]
        recent_lows = [s.price for s in swings if s.type == "low"][-3:]
        if not recent_highs or not recent_lows:
            return "neutral", "equilibrium"

        mid = (max(recent_highs) + min(recent_lows)) / 2
        if last_close > mid * 1.01:
            zone = "premium"
        elif last_close < mid * 0.99:
            zone = "discount"
        else:
            zone = "equilibrium"

        bias = "neutral"
        if structure:
            last_event = structure[-1]
            bias = last_event.direction
        elif len(swings) >= 2:
            if swings[-1].type == "high" and swings[-1].price > swings[-2].price:
                bias = "bullish"
            elif swings[-1].type == "low" and swings[-1].price < swings[-2].price:
                bias = "bearish"

        return bias, zone
