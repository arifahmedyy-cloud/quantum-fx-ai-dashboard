"""Deterministic regression tests for backtest execution correctness."""
from types import SimpleNamespace

import pandas as pd

from src.backtesting.backtest_engine import BacktestEngine
from src.config import BacktestConfig, RiskConfig
from src.models import SignalOutput


def bars(n=206):
    dates = pd.date_range("2025-01-01", periods=n, freq="15min")
    rows = []
    for i, dt in enumerate(dates):
        price = 100.0 + (i * 0.01)
        rows.append({"Date": dt, "Open": price, "High": price + 0.5, "Low": price - 0.5, "Close": price + 0.1, "Volume": 100})
    return pd.DataFrame(rows)


def signal_fn(window, i):
    close = float(window.iloc[-1]["Close"])
    return SignalOutput("BUY", 90, "TRENDING", "test", 2.0, 5.0, 2.0, 90,
                        close - 1.0, close + 2.0, close, 0.01, "test", {})


def install_noop_decision(engine):
    def decide(signal, smc, **kwargs):
        return SimpleNamespace(action=signal.action, ai_score=90, sl=signal.sl, tp=signal.tp,
                               entry=signal.entry, lot_size=signal.lot_size)
    engine.decision.decide = decide
    engine.smc.analyze = lambda window: SimpleNamespace(bias="neutral", zone="equilibrium")


def test_analysis_window_is_bounded_not_full_history():
    """Regression test for the O(n^2) scaling bug: the window handed to
    strategy_fn/regime/SMC must be capped at BacktestConfig.analysis_window_bars
    bars, not the full df.iloc[:i+1] history-to-date. Before the fix, a
    growing unbounded window (combined with SMC's O(window) internal scans)
    made backtests effectively unusable beyond a few thousand candles."""
    df = bars(n=260)
    seen_lengths = []

    def probing_signal_fn(window, i):
        seen_lengths.append(len(window))
        close = float(window.iloc[-1]["Close"])
        return SignalOutput("BUY", 90, "TRENDING", "test", 2.0, 5.0, 2.0, 90,
                             close - 1.0, close + 2.0, close, 0.01, "test", {})

    engine = BacktestEngine(
        initial_balance=10000,
        risk_config=RiskConfig(max_open_trades=1, enable_ml_filter=False, enable_ai_reviewer=False),
        backtest_config=BacktestConfig(spread=0.0, slippage=0.0, warmup_bars=200, analysis_window_bars=50),
    )
    install_noop_decision(engine)
    engine.run(df, strategy_fn=probing_signal_fn)
    assert seen_lengths, "signal_fn should have been called at least once"
    assert max(seen_lengths) <= 50


def test_entry_is_next_bar_open_not_signal_close():
    df = bars()
    # Prevent an exit on the first execution candle.
    df.loc[201, ["High", "Low"]] = [102.5, 101.5]
    engine = BacktestEngine(
        initial_balance=10000,
        risk_config=RiskConfig(max_open_trades=1, enable_ml_filter=False, enable_ai_reviewer=False),
        backtest_config=BacktestConfig(spread=0.0, slippage=0.0, warmup_bars=200),
    )
    install_noop_decision(engine)
    engine.run(df, strategy_fn=signal_fn)
    assert engine.trades
    assert engine.trades[0].signal_time < engine.trades[0].entry_time
    assert engine.trades[0].entry_time == df.iloc[201]["Date"].to_pydatetime()
    assert engine.trades[0].entry_price == float(df.iloc[201]["Open"])


def test_sl_uses_low_not_close_and_actual_sl_price():
    df = bars()
    # Signal at 200; execution at 201; candle 201 touches SL while close remains above SL.
    df.loc[201, "Low"] = 99.0
    df.loc[201, "Close"] = 101.0
    engine = BacktestEngine(
        initial_balance=10000,
        risk_config=RiskConfig(max_open_trades=1, enable_ml_filter=False, enable_ai_reviewer=False),
        backtest_config=BacktestConfig(spread=0.0, slippage=0.0, warmup_bars=200),
    )
    install_noop_decision(engine)
    engine.run(df, strategy_fn=signal_fn)
    assert engine.trades
    trade = engine.trades[0]
    assert trade.exit_reason == "SL"
    assert trade.exit_price == trade.sl
    assert trade.profit_loss < 0


def test_same_candle_policy_sl_first():
    df = bars()
    df.loc[201, "Low"] = 99.0
    df.loc[201, "High"] = 103.0
    engine = BacktestEngine(
        initial_balance=10000,
        risk_config=RiskConfig(max_open_trades=1, enable_ml_filter=False, enable_ai_reviewer=False),
        backtest_config=BacktestConfig(spread=0.0, slippage=0.0, intrabar_policy="SL_FIRST", warmup_bars=200),
    )
    install_noop_decision(engine)
    engine.run(df, strategy_fn=signal_fn)
    assert engine.trades[0].exit_reason == "SL"


def test_max_open_trades_allows_multiple_independent_positions():
    df = bars()
    # Keep all candles away from SL/TP so positions coexist until end-of-data.
    engine = BacktestEngine(
        initial_balance=10000,
        risk_config=RiskConfig(max_open_trades=2, enable_ml_filter=False, enable_ai_reviewer=False),
        backtest_config=BacktestConfig(spread=0.0, slippage=0.0, warmup_bars=200),
    )
    install_noop_decision(engine)
    engine.run(df, strategy_fn=signal_fn)
    assert len(engine.trades) >= 2
    # At least two trades must have overlapping lifetimes when max_open_trades=2.
    assert engine.trades[0].entry_time < engine.trades[1].entry_time
    assert engine.trades[0].exit_time >= engine.trades[1].entry_time


def test_gap_through_stop_does_not_manufacture_sl_fill():
    df = bars()
    # Signal at 200 has SL below its close; next open gaps below that SL.
    df.loc[201, "Open"] = 99.0
    df.loc[201, "High"] = 99.5
    df.loc[201, "Low"] = 98.5
    df.loc[201, "Close"] = 99.2
    engine = BacktestEngine(
        initial_balance=10000,
        risk_config=RiskConfig(max_open_trades=1, enable_ml_filter=False, enable_ai_reviewer=False),
        backtest_config=BacktestConfig(spread=0.0, slippage=0.0, warmup_bars=200),
    )
    install_noop_decision(engine)
    engine.run(df, strategy_fn=signal_fn)
    assert engine.trades
    trade = engine.trades[0]
    assert trade.exit_reason == "SL"
    assert trade.exit_price == trade.entry_price


def test_backtest_default_uses_regime_detector_and_logs_decisions():
    df = bars()
    engine = BacktestEngine(
        initial_balance=10000,
        risk_config=RiskConfig(max_open_trades=1, enable_ml_filter=False, enable_ai_reviewer=False),
        backtest_config=BacktestConfig(spread=0.0, slippage=0.0, warmup_bars=200),
    )
    engine.decision.decide = lambda signal, smc, **kwargs: SimpleNamespace(
        action="NO_TRADE", ai_score=signal.confidence, sl=signal.sl, tp=signal.tp,
        entry=signal.entry, lot_size=signal.lot_size, explanation="test rejection"
    )
    engine.smc.analyze = lambda window: SimpleNamespace(bias="neutral", zone="equilibrium")
    engine.run(df)
    assert engine.validation["regime_detector_active"] is True
    assert engine.decision_log
    assert all("reason" in row and "regime" in row for row in engine.decision_log)
