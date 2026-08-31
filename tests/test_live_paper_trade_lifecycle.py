"""Regression tests for bugs found while auditing the live paper-trading
path end-to-end (opening a real position through send_order, moving price
to hit SL, and confirming it actually closes and gets recorded correctly).

Three confirmed, related bugs:
  1. PaperBroker.check_sl_tp() existed but was never called anywhere in
     app.py's live trading_loop, so a paper position never automatically
     closed when price hit its SL or TP — it just sat open forever with
     floating P&L updating, an inaccurate simulation of real risk
     discipline. Fixed by calling it once per symbol per loop iteration.
  2. JournalService.record_trade() — which finalizes a completed trade
     into the persistent `trades` table — was never called anywhere in
     the app. Orders opened via record_order_open() were stored in
     pending_orders but never finalized on close, so the Journal tab,
     CSV/JSON export, and win-rate calculations were always empty
     regardless of how many trades actually closed. Fixed by adding
     app._reconcile_closed_trades(), called once per loop iteration,
     which diffs pending orders against the broker's trade history
     (works uniformly for PaperBroker and MT5Broker) and finalizes
     anything that has closed.
  3. record_trade()'s r_multiple calculation ignored contract_size
     entirely (risk_amount = abs(entry-sl) * lot_size only), so for any
     instrument with contract_size != 1 (e.g. XAUUSD at 100 oz/lot) the
     stored R-multiple was wrong by exactly that factor — a real ~1.15R
     loss was recorded as -115R. Fixed by adding a contract_size
     parameter that _reconcile_closed_trades resolves and passes through.
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.trading.broker_connector import PaperBroker
from src.models import TradeSignal
from src.services.journal_service import JournalService
from src.services.notification_service import NotificationService
from src.config import NotificationConfig

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


def _open_position(broker, journal, symbol="XAUUSD", direction="BUY",
                    entry=2450.0, sl=2440.0, tp=2470.0, lot=0.1):
    signal = TradeSignal(symbol=symbol, direction=direction, entry_price=entry,
                          sl=sl, tp=tp, lot_size=lot, strategy="Regime (default)",
                          ai_score=78, regime="trending")
    result = broker.send_order(signal)
    ticket = result["ticket"]
    journal.record_order_open(
        ticket=ticket, entry_time=datetime.now(), symbol=symbol, direction=direction,
        entry_price=entry, stop_loss=sl, take_profit=tp, lot_size=lot,
        risk_pct=1.0, ai_score=78, regime="trending", smc_bias="bullish",
        confluence_notes="regime+SMC agree", strategy="Regime (default)")
    return ticket


class TestPaperSLTPAutoClose:
    def test_position_does_not_close_on_price_move_alone(self, broker, journal):
        """update_price alone (no check_sl_tp) must NOT close the position —
        confirms the baseline behavior this bug fix builds on."""
        _open_position(broker, journal)
        broker.update_price("XAUUSD", bid=2439.0, ask=2439.5)  # past SL
        assert len(broker.get_positions()) == 1

    def test_check_sl_tp_closes_position_on_sl_hit(self, broker, journal):
        _open_position(broker, journal)
        broker.update_price("XAUUSD", bid=2439.0, ask=2439.5)
        broker.check_sl_tp("XAUUSD", bid=2439.0, ask=2439.5)
        assert len(broker.get_positions()) == 0, "Position did not auto-close on SL hit"

    def test_check_sl_tp_closes_position_on_tp_hit(self, broker, journal):
        _open_position(broker, journal)
        broker.update_price("XAUUSD", bid=2471.0, ask=2471.5)  # past TP
        broker.check_sl_tp("XAUUSD", bid=2471.0, ask=2471.5)
        assert len(broker.get_positions()) == 0, "Position did not auto-close on TP hit"

    def test_check_sl_tp_leaves_position_open_if_not_hit(self, broker, journal):
        _open_position(broker, journal)
        broker.update_price("XAUUSD", bid=2452.0, ask=2452.5)  # between SL and TP
        broker.check_sl_tp("XAUUSD", bid=2452.0, ask=2452.5)
        assert len(broker.get_positions()) == 1


class TestJournalReconciliation:
    def test_closed_trade_gets_recorded_in_journal(self, broker, journal, notifier):
        _open_position(broker, journal)
        assert journal.trades == []  # nothing finalized yet

        broker.update_price("XAUUSD", bid=2439.0, ask=2439.5)
        broker.check_sl_tp("XAUUSD", bid=2439.0, ask=2439.5)
        app._reconcile_closed_trades(broker, journal, notifier, health=None)

        trades = journal.trades
        assert len(trades) == 1, "record_trade was not called — journal still empty"
        assert trades[0]["result"] == "LOSS"
        assert trades[0]["profit_loss"] < 0
        assert journal.open_orders == [], "pending order was not consumed"

    def test_open_position_is_not_reconciled_early(self, broker, journal, notifier):
        _open_position(broker, journal)
        broker.update_price("XAUUSD", bid=2452.0, ask=2452.5)  # no SL/TP hit
        app._reconcile_closed_trades(broker, journal, notifier, health=None)
        assert journal.trades == []
        assert len(journal.open_orders) == 1

    def test_winning_trade_recorded_as_win(self, broker, journal, notifier):
        _open_position(broker, journal)
        broker.update_price("XAUUSD", bid=2471.0, ask=2471.5)
        broker.check_sl_tp("XAUUSD", bid=2471.0, ask=2471.5)
        app._reconcile_closed_trades(broker, journal, notifier, health=None)
        trades = journal.trades
        assert len(trades) == 1
        assert trades[0]["result"] == "WIN"
        assert trades[0]["profit_loss"] > 0


class TestRMultipleContractSize:
    """Regression test for the contract-size bug in r_multiple."""

    def test_r_multiple_accounts_for_gold_contract_size(self, journal):
        # XAUUSD: 10-point risk, 0.1 lot, contract_size=100 -> $100 risk.
        # A $115 loss should be ~ -1.15R, not -115R.
        journal.record_trade(
            entry_time=datetime.now(), exit_time=datetime.now(),
            symbol="XAUUSD", direction="BUY", entry_price=2450.0, exit_price=2439.0,
            stop_loss=2440.0, take_profit=2470.0, lot_size=0.1, profit_loss=-115.0,
            risk_pct=1.0, contract_size=100.0,
        )
        trade = journal.trades[0]
        assert -2.0 < trade["r_multiple"] < -1.0, (
            f"r_multiple is wrong: {trade['r_multiple']} (contract_size not applied)"
        )

    def test_r_multiple_defaults_to_1x_when_contract_size_omitted(self, journal):
        # Backward-compat: omitting contract_size should behave like before
        # (contract_size=1.0) — risk_amount = |entry-sl| * lot_size exactly.
        # Using round numbers here (distance=10, lot=1.0) so the expected
        # risk_amount is exactly 10 and profit_loss=-10 gives r_multiple=-1.0.
        journal.record_trade(
            entry_time=datetime.now(), exit_time=datetime.now(),
            symbol="TESTSYM", direction="BUY", entry_price=100.0, exit_price=90.0,
            stop_loss=90.0, take_profit=120.0, lot_size=1.0, profit_loss=-10.0,
            risk_pct=1.0,
        )
        trade = journal.trades[0]
        assert trade["r_multiple"] == pytest.approx(-1.0, abs=0.01)
