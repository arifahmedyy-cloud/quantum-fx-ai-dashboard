"""Regression tests for the circuit-breaker and break-even integration gaps
found during audit.

Both RiskManager (consecutive-loss cooldown / "circuit breaker") and
TrailingStopManager (break-even + trailing stop) were ALREADY fully
implemented and unit-tested in isolation. The bugs were purely at the
live-loop integration layer:

  1. RiskManager.record_trade_result() — which increments/resets the
     consecutive-loss counter that daily_guard()'s circuit-breaker check
     depends on — was never called anywhere in app.py. The counter simply
     never moved in live trading, so the circuit breaker could never
     trigger no matter how many trades lost in a row. Fixed by calling it
     from _reconcile_closed_trades() (the same place journal.record_trade()
     is finalized).
  2. daily_guard()'s bounded-cooldown path only activates when
     current_time is not None, but the live loop's call site omitted it
     entirely — silently disabling consecutive_loss_cooldown_hours even
     when configured. Fixed by passing current_time=datetime.now().

Break-Even/Trailing (_apply_trailing_stops -> TrailingStopManager) was
already correctly wired; these tests just confirm the full path (position
open -> record_position_risk -> price moves -> _apply_trailing_stops ->
broker.modify_position_sl_tp) actually moves the SL through app.py's real
function, not just TrailingStopManager in isolation.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.trading.broker_connector import PaperBroker
from src.trading.trailing_stop_manager import TrailingStopManager
from src.trading.risk_manager import RiskManager
from src.models import TradeSignal
from src.services.journal_service import JournalService
from src.services.notification_service import NotificationService
from src.config import NotificationConfig, RiskConfig

import app


@pytest.fixture
def journal(tmp_path):
    return JournalService(db_path=str(tmp_path / "test_journal.db"))


@pytest.fixture
def broker():
    b = PaperBroker(initial_balance=10000, leverage=100)
    b.connect()
    return b


@pytest.fixture
def notifier():
    return NotificationService(NotificationConfig(enabled=False))


def _open_and_lose(broker, journal, risk):
    signal = TradeSignal(symbol="XAUUSD", direction="BUY", entry_price=2450.0,
                          sl=2440.0, tp=2470.0, lot_size=0.1, strategy="test",
                          ai_score=80, regime="trending")
    result = broker.send_order(signal)
    ticket = result["ticket"]
    journal.record_order_open(
        ticket=ticket, entry_time=datetime.now(), symbol="XAUUSD", direction="BUY",
        entry_price=2450.0, stop_loss=2440.0, take_profit=2470.0, lot_size=0.1,
        risk_pct=1.0, ai_score=80, regime="trending", smc_bias="bullish",
        confluence_notes="test", strategy="test")
    broker.update_price("XAUUSD", bid=2439.0, ask=2439.5)
    broker.check_sl_tp("XAUUSD", bid=2439.0, ask=2439.5)
    app._reconcile_closed_trades(
        broker, journal, NotificationService(NotificationConfig(enabled=False)), health=None,
        risk=risk, account_balance=broker.get_account_info().balance,
    )


class TestCircuitBreakerIntegration:
    def test_consecutive_losses_counter_updates_via_live_reconciliation(self, broker, journal):
        risk = RiskManager(RiskConfig(max_consecutive_losses=3))
        assert risk._consecutive_losses == 0
        _open_and_lose(broker, journal, risk)
        assert risk._consecutive_losses == 1, (
            "record_trade_result was not called from _reconcile_closed_trades — "
            "the circuit breaker counter is dead again."
        )

    def test_circuit_breaker_blocks_after_n_consecutive_losses(self, broker, journal):
        risk = RiskManager(RiskConfig(max_consecutive_losses=3))
        for _ in range(3):
            _open_and_lose(broker, journal, risk)
        guard = risk.daily_guard(broker.get_account_info().balance, 10000, current_time=datetime.now())
        assert guard.should_block_new_trades is True
        assert "consecutive losses" in guard.reason.lower()

    def test_circuit_breaker_releases_after_configured_cooldown(self, broker, journal):
        risk = RiskManager(RiskConfig(max_consecutive_losses=2, consecutive_loss_cooldown_hours=1.0))
        for _ in range(2):
            _open_and_lose(broker, journal, risk)
        now = datetime.now()
        blocked = risk.daily_guard(broker.get_account_info().balance, 10000, current_time=now)
        assert blocked.should_block_new_trades is True

        later = now + timedelta(hours=2)  # > 1.0h cooldown
        released = risk.daily_guard(broker.get_account_info().balance, 10000, current_time=later)
        assert released.should_block_new_trades is False

    def test_trading_loop_daily_guard_call_passes_current_time(self):
        """Static check that the live loop's daily_guard() call site
        passes current_time — omitting it silently disables the bounded
        cooldown path (see risk_manager.py's daily_guard docstring)."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path) as f:
            source = f.read()
        start = source.index("def trading_loop(")
        end = source.index("\ndef ", start + 1)
        loop_body = source[start:end]
        guard_call_idx = loop_body.index("risk.daily_guard(")
        # current_time= must appear within the same statement (before the
        # closing paren reasonably close by).
        snippet = loop_body[guard_call_idx:guard_call_idx + 200]
        assert "current_time=" in snippet, (
            "trading_loop's daily_guard() call no longer passes current_time — "
            "the consecutive-loss cooldown will silently stop working."
        )


class TestBreakEvenIntegrationViaAppFunction:
    def test_apply_trailing_stops_moves_sl_to_breakeven(self, broker, journal):
        risk_config = RiskConfig(use_trailing_stop=True, breakeven_trigger_r=0.5,
                                  breakeven_buffer_pct=0.05, trailing_trigger_r=1.5,
                                  trailing_distance_r=0.5)
        trailing_mgr = TrailingStopManager(risk_config)
        config = type("C", (), {"risk": risk_config})()

        signal = TradeSignal(symbol="XAUUSD", direction="BUY", entry_price=2450.0,
                              sl=2440.0, tp=2470.0, lot_size=0.1, strategy="test",
                              ai_score=80, regime="trending")
        result = broker.send_order(signal)
        ticket = result["ticket"]
        journal.record_order_open(
            ticket=ticket, entry_time=datetime.now(), symbol="XAUUSD", direction="BUY",
            entry_price=2450.0, stop_loss=2440.0, take_profit=2470.0, lot_size=0.1,
            risk_pct=1.0, ai_score=80, regime="trending", smc_bias="bullish",
            confluence_notes="test", strategy="test")
        journal.record_position_risk(ticket=ticket, symbol="XAUUSD", direction="BUY",
                                      entry_price=2450.0, initial_sl=2440.0)

        assert broker.get_positions()[0].sl == 2440.0

        broker.update_price("XAUUSD", bid=2454.5, ask=2455.0)  # 0.5R profit
        app._apply_trailing_stops(config, broker, journal, trailing_mgr)

        new_sl = broker.get_positions()[0].sl
        assert new_sl > 2440.0, "Break-even did not move the SL via the real app.py function"
        assert new_sl < 2455.0

    def test_apply_trailing_stops_advances_further_at_trailing_trigger(self, broker, journal):
        risk_config = RiskConfig(use_trailing_stop=True, breakeven_trigger_r=0.5,
                                  breakeven_buffer_pct=0.05, trailing_trigger_r=1.5,
                                  trailing_distance_r=0.5)
        trailing_mgr = TrailingStopManager(risk_config)
        config = type("C", (), {"risk": risk_config})()

        signal = TradeSignal(symbol="XAUUSD", direction="BUY", entry_price=2450.0,
                              sl=2440.0, tp=2470.0, lot_size=0.1, strategy="test",
                              ai_score=80, regime="trending")
        result = broker.send_order(signal)
        ticket = result["ticket"]
        journal.record_order_open(
            ticket=ticket, entry_time=datetime.now(), symbol="XAUUSD", direction="BUY",
            entry_price=2450.0, stop_loss=2440.0, take_profit=2470.0, lot_size=0.1,
            risk_pct=1.0, ai_score=80, regime="trending", smc_bias="bullish",
            confluence_notes="test", strategy="test")
        journal.record_position_risk(ticket=ticket, symbol="XAUUSD", direction="BUY",
                                      entry_price=2450.0, initial_sl=2440.0)

        broker.update_price("XAUUSD", bid=2454.5, ask=2455.0)
        app._apply_trailing_stops(config, broker, journal, trailing_mgr)
        breakeven_sl = broker.get_positions()[0].sl

        broker.update_price("XAUUSD", bid=2464.5, ask=2465.0)  # 1.5R profit
        app._apply_trailing_stops(config, broker, journal, trailing_mgr)
        trailing_sl = broker.get_positions()[0].sl

        assert trailing_sl > breakeven_sl, "Trailing stop did not advance beyond break-even"

    def test_disabled_trailing_stop_never_moves_sl(self, broker, journal):
        risk_config = RiskConfig(use_trailing_stop=False)
        trailing_mgr = TrailingStopManager(risk_config)
        config = type("C", (), {"risk": risk_config})()

        signal = TradeSignal(symbol="XAUUSD", direction="BUY", entry_price=2450.0,
                              sl=2440.0, tp=2470.0, lot_size=0.1, strategy="test",
                              ai_score=80, regime="trending")
        result = broker.send_order(signal)
        ticket = result["ticket"]
        journal.record_order_open(
            ticket=ticket, entry_time=datetime.now(), symbol="XAUUSD", direction="BUY",
            entry_price=2450.0, stop_loss=2440.0, take_profit=2470.0, lot_size=0.1,
            risk_pct=1.0, ai_score=80, regime="trending", smc_bias="bullish",
            confluence_notes="test", strategy="test")
        journal.record_position_risk(ticket=ticket, symbol="XAUUSD", direction="BUY",
                                      entry_price=2450.0, initial_sl=2440.0)

        broker.update_price("XAUUSD", bid=2464.5, ask=2465.0)
        app._apply_trailing_stops(config, broker, journal, trailing_mgr)
        assert broker.get_positions()[0].sl == 2440.0
