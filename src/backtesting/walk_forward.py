"""Walk-forward validation.

Splits historical data into sequential windows and runs the SAME
BacktestEngine used for a single backtest on each out-of-sample window in
turn — this is what tells you whether a strategy's edge holds up across
different time periods, rather than being a fluke of whichever single
period you happened to backtest.

Deliberately built as a thin orchestration layer over `BacktestEngine`
rather than a second, parallel implementation of signal generation — the
project only has one place that decides BUY/SELL/NO_TRADE, and this reuses
it exactly as live trading would.

Usage:
    from src.backtesting.walk_forward import WalkForwardValidator
    from src.config import RiskConfig

    wf = WalkForwardValidator(risk_config=RiskConfig(), window_bars=1000, step_bars=250)
    report = wf.run(df)  # df: OHLCV with a datetime index or 'Date' column

    report.is_robust          -> bool, see is_robust() docstring
    report.folds               -> per-window BacktestSummary + date range
    report.aggregate_profit_factor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import statistics

import pandas as pd

from src.logger import get_logger
from src.models import BacktestSummary
from src.config import RiskConfig
from src.backtesting.backtest_engine import BacktestEngine

log = get_logger(__name__)


@dataclass
class WalkForwardFold:
    """One out-of-sample window's result."""
    fold_index: int
    start_index: int
    end_index: int
    start_date: Optional[str]
    end_date: Optional[str]
    summary: BacktestSummary


@dataclass
class WalkForwardReport:
    folds: List[WalkForwardFold] = field(default_factory=list)
    window_bars: int = 0
    step_bars: int = 0
    aggregate_profit_factor: float = 0.0
    aggregate_win_rate: float = 0.0
    aggregate_trades: int = 0
    worst_fold_max_dd: float = 0.0
    profit_factor_std: float = 0.0
    profitable_fold_pct: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def is_robust(
        self,
        min_profitable_fold_pct: float = 60.0,
        max_pf_std: float = 1.0,
        min_folds: int = 4,
    ) -> bool:
        """A strategy that only looks good in an aggregate/single backtest
        but falls apart fold-to-fold is overfit to that one period. This is
        a blunt, intentionally conservative pass/fail:
            - at least `min_folds` folds were actually run (a "walk-forward"
              of 1-2 folds proves nothing),
            - at least `min_profitable_fold_pct`% of folds were profitable
              (not just the aggregate), and
            - fold-to-fold profit factor doesn't swing wildly
              (std <= max_pf_std).
        This does NOT mean "will be profitable live" — it means "did not
        fail the most basic overfitting sanity check." Treat it as a gate
        that must pass, not a claim of future profitability.
        """
        if len(self.folds) < min_folds:
            return False
        return self.profitable_fold_pct >= min_profitable_fold_pct and self.profit_factor_std <= max_pf_std


class WalkForwardValidator:
    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        window_bars: int = 1000,
        step_bars: int = 250,
        initial_balance: float = 10000.0,
        spread: float = 0.5,
    ) -> None:
        """
        Args:
            risk_config: Risk settings applied identically in every fold
                (a real walk-forward would re-optimize per fold; this
                project doesn't currently expose tunable strategy
                parameters to search over, so this validates robustness of
                the FIXED current strategy/config across time periods,
                which is still a meaningful and much cheaper check than
                none at all).
            window_bars: Size of each out-of-sample test window.
            step_bars: How far to slide the window forward between folds.
                step_bars < window_bars means folds overlap; step_bars ==
                window_bars means folds are back-to-back with no overlap
                (the more standard walk-forward setup).
            initial_balance: Starting balance for EVERY fold independently
                (folds are evaluated in isolation on purpose, so a bad fold
                doesn't drag down every subsequent fold's starting capital
                and make results order-dependent).
        """
        self.risk_config = risk_config or RiskConfig()
        self.window_bars = window_bars
        self.step_bars = step_bars
        self.initial_balance = initial_balance
        self.spread = spread

    def run(self, df: pd.DataFrame) -> WalkForwardReport:
        """Run walk-forward validation over the full historical dataset.

        Args:
            df: Full OHLCV history. Must have at least `window_bars` rows,
                ideally several multiples of it (more folds = a more
                meaningful robustness check).

        Returns:
            WalkForwardReport. If there isn't enough data for even one full
            window, returns an empty report with a warning rather than
            raising — callers should check `len(report.folds) == 0`.
        """
        report = WalkForwardReport(window_bars=self.window_bars, step_bars=self.step_bars)

        if df is None or len(df) < self.window_bars:
            msg = (f"Not enough data for walk-forward: {0 if df is None else len(df)} bars available, "
                   f"need at least {self.window_bars} for a single fold.")
            log.warning(msg)
            report.warnings.append(msg)
            return report

        df = df.reset_index(drop=True)
        n = len(df)
        fold_index = 0
        start = 0

        while start + self.window_bars <= n:
            end = start + self.window_bars
            window_df = df.iloc[start:end].copy()

            engine = BacktestEngine(
                initial_balance=self.initial_balance,
                risk_config=self.risk_config,
                spread=self.spread,
            )
            try:
                summary = engine.run(window_df)
            except Exception as exc:
                log.error("Fold %d (bars %d-%d) raised during backtest: %s", fold_index, start, end, exc)
                report.warnings.append(f"Fold {fold_index} failed: {exc}")
                start += self.step_bars
                fold_index += 1
                continue

            start_date = self._safe_date(window_df, 0)
            end_date = self._safe_date(window_df, -1)

            report.folds.append(WalkForwardFold(
                fold_index=fold_index, start_index=start, end_index=end,
                start_date=start_date, end_date=end_date, summary=summary,
            ))
            log.info("Fold %d [%s -> %s]: %d trades, PF=%.2f, win_rate=%.1f%%, max_dd=%.1f%%",
                      fold_index, start_date, end_date, summary.trades,
                      summary.profit_factor, summary.win_rate, summary.max_dd)

            start += self.step_bars
            fold_index += 1

        self._aggregate(report)
        return report

    @staticmethod
    def _safe_date(window_df: pd.DataFrame, pos: int) -> Optional[str]:
        if "Date" in window_df.columns:
            try:
                return str(window_df["Date"].iloc[pos])
            except Exception:
                return None
        return None

    def _aggregate(self, report: WalkForwardReport) -> None:
        if not report.folds:
            report.warnings.append("No folds completed — cannot compute aggregate statistics.")
            return

        pfs = [f.summary.profit_factor for f in report.folds if f.summary.trades > 0]
        win_rates = [f.summary.win_rate for f in report.folds if f.summary.trades > 0]
        total_trades = sum(f.summary.trades for f in report.folds)
        max_dds = [f.summary.max_dd for f in report.folds]
        profitable = sum(1 for f in report.folds if f.summary.trades > 0 and f.summary.profit_factor > 1.0)
        folds_with_trades = sum(1 for f in report.folds if f.summary.trades > 0)

        # BUG FIX (audit): a fold with trades but zero LOSING trades has
        # profit_factor = inf (total_profit / 0). That's a genuinely good
        # outcome, not an error condition, but statistics.pstdev()/mean()
        # crash outright on an infinite value (AttributeError on Python
        # 3.12's Fraction-based implementation) — so a single perfect fold
        # took down the entire report. Excluded from the finite mean/std
        # calculation (which those statistics can't meaningfully represent
        # anyway) but still counted as a profitable fold, and called out
        # explicitly in the warnings so it's visible rather than silently
        # dropped.
        finite_pfs = [pf for pf in pfs if pf not in (float("inf"), float("-inf"))]
        inf_pf_count = len(pfs) - len(finite_pfs)

        report.aggregate_trades = total_trades
        if finite_pfs:
            report.aggregate_profit_factor = round(statistics.mean(finite_pfs), 3)
        elif pfs:
            report.aggregate_profit_factor = float("inf")
        else:
            report.aggregate_profit_factor = 0.0
        report.aggregate_win_rate = round(statistics.mean(win_rates), 2) if win_rates else 0.0
        report.worst_fold_max_dd = round(max(max_dds), 2) if max_dds else 0.0
        report.profit_factor_std = round(statistics.pstdev(finite_pfs), 3) if len(finite_pfs) > 1 else 0.0
        report.profitable_fold_pct = round(profitable / folds_with_trades * 100, 1) if folds_with_trades else 0.0

        if inf_pf_count:
            report.warnings.append(
                f"{inf_pf_count} fold(s) had trades but zero losing trades (profit_factor=inf) — "
                f"excluded from the profit-factor mean/std above, but still counted as profitable."
            )
        if folds_with_trades < len(report.folds):
            report.warnings.append(
                f"{len(report.folds) - folds_with_trades} of {len(report.folds)} folds had zero trades "
                f"— excluded from profit-factor aggregation, but still count against robustness."
            )
