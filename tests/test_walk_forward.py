"""Tests for src/backtesting/walk_forward.py.

Covers a confirmed audit bug: a fold with trades but zero losing trades has
profit_factor = inf (division by zero losses), which crashed
statistics.pstdev()/mean() outright (AttributeError on Python 3.12's
Fraction-based implementation) — so a single "perfect" fold (a GOOD
outcome) took down the entire walk-forward report. Fixed by excluding
infinite values from the finite mean/std calculation while still counting
them as profitable folds and surfacing them in report.warnings.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src.backtesting.walk_forward import WalkForwardValidator, WalkForwardReport
from src.config import RiskConfig
from src.services.paper_market_data import generate_paper_ohlcv


class TestWalkForwardAggregateCrash:
    def test_run_does_not_crash_with_all_winning_fold(self):
        """Regression test for the exact crash: a fold with 100% win rate
        (profit_factor=inf) must not raise."""
        df = generate_paper_ohlcv("XAUUSD", "H1", 3000, end=datetime(2026, 8, 1, tzinfo=timezone.utc))
        wf = WalkForwardValidator(risk_config=RiskConfig(), window_bars=500, step_bars=250)
        report = wf.run(df)  # must not raise
        assert isinstance(report, WalkForwardReport)
        assert len(report.folds) > 0

    def test_aggregate_handles_infinite_profit_factor_directly(self):
        """Unit-level test of the aggregation math itself, independent of
        whether a real backtest happens to produce an inf-PF fold."""
        from src.backtesting.walk_forward import WalkForwardFold
        from src.models import BacktestSummary

        def _summary(trades, pf, win_rate, max_dd):
            return BacktestSummary(
                final_balance=10000.0, return_pct=0.0, trades=trades, wins=int(trades * win_rate / 100),
                win_rate=win_rate, profit_factor=pf, max_dd=max_dd,
                equity_curve=[10000.0], regime_breakdown={},
            )

        report = WalkForwardReport(window_bars=500, step_bars=250)
        report.folds = [
            WalkForwardFold(0, 0, 500, "2026-01-01", "2026-01-05", _summary(5, 1.5, 60.0, 3.0)),
            WalkForwardFold(1, 250, 750, "2026-01-03", "2026-01-08", _summary(2, float("inf"), 100.0, 0.5)),
            WalkForwardFold(2, 500, 1000, "2026-01-06", "2026-01-11", _summary(0, 0.0, 0.0, 0.0)),
        ]
        wf = WalkForwardValidator()
        wf._aggregate(report)  # must not raise

        assert report.aggregate_profit_factor == 1.5  # only the finite PF counted
        assert report.profit_factor_std == 0.0  # only 1 finite value -> no std
        assert any("profit_factor=inf" in w for w in report.warnings)
        # both traded folds were profitable (1.5 > 1.0, inf > 1.0)
        assert report.profitable_fold_pct == 100.0

    def test_aggregate_all_finite_still_works_normally(self):
        """Make sure the fix didn't change behavior for the normal
        (no-infinity) case."""
        from src.backtesting.walk_forward import WalkForwardFold
        from src.models import BacktestSummary

        def _summary(trades, pf, win_rate, max_dd):
            return BacktestSummary(
                final_balance=10000.0, return_pct=0.0, trades=trades, wins=int(trades * win_rate / 100),
                win_rate=win_rate, profit_factor=pf, max_dd=max_dd,
                equity_curve=[10000.0], regime_breakdown={},
            )

        report = WalkForwardReport(window_bars=500, step_bars=250)
        report.folds = [
            WalkForwardFold(0, 0, 500, "d1", "d2", _summary(5, 1.5, 60.0, 3.0)),
            WalkForwardFold(1, 250, 750, "d3", "d4", _summary(5, 0.9, 40.0, 5.0)),
        ]
        wf = WalkForwardValidator()
        wf._aggregate(report)

        assert report.aggregate_profit_factor == pytest.approx(1.2, abs=0.01)
        assert not any("profit_factor=inf" in w for w in report.warnings)


class TestWalkForwardBasic:
    def test_not_enough_data_returns_empty_report_with_warning(self):
        df = generate_paper_ohlcv("XAUUSD", "H1", 100)
        wf = WalkForwardValidator(window_bars=500, step_bars=250)
        report = wf.run(df)
        assert report.folds == []
        assert len(report.warnings) == 1

    def test_is_robust_requires_minimum_folds(self):
        report = WalkForwardReport()
        assert report.is_robust(min_folds=4) is False
