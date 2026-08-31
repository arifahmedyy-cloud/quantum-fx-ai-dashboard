"""Tests for broker_connector.py."""

import pytest
from src.trading.broker_connector import PaperBroker, BrokerConnector
from src.models import TradeSignal


class TestPaperBroker:
    def test_connect(self):
        broker = PaperBroker(initial_balance=10000)
        assert broker.connect()
        assert broker.is_connected()

    def test_disconnect(self):
        broker = PaperBroker()
        broker.connect()
        assert broker.disconnect()
        assert not broker.is_connected()

    def test_get_account_info(self):
        broker = PaperBroker(initial_balance=5000, leverage=50)
        broker.connect()
        info = broker.get_account_info()
        assert info.balance == 5000
        assert info.leverage == 50
        assert info.margin_level == 100.0

    def test_send_order_buy(self):
        broker = PaperBroker()
        broker.connect()
        signal = TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1)
        result = broker.send_order(signal)
        assert result["success"]
        assert result["ticket"] > 0
        assert len(broker.get_positions()) == 1

    def test_send_order_sell(self):
        broker = PaperBroker()
        broker.connect()
        signal = TradeSignal(symbol="XAUUSD", direction="SELL",
            entry_price=2450, sl=2455, tp=2440, lot_size=0.1)
        result = broker.send_order(signal)
        assert result["success"]
        assert len(broker.get_positions()) == 1

    def test_send_order_not_connected(self):
        broker = PaperBroker()
        signal = TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1)
        result = broker.send_order(signal)
        assert not result["success"]
        assert "Not connected" in result["error"]

    def test_send_order_insufficient_margin(self):
        broker = PaperBroker(initial_balance=100, leverage=10)
        broker.connect()
        signal = TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=10.0)
        result = broker.send_order(signal)
        assert not result["success"]
        assert "Insufficient margin" in result["error"]

    def test_close_position(self):
        broker = PaperBroker()
        broker.connect()
        signal = TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1)
        result = broker.send_order(signal)
        assert broker.close_position(result["ticket"])
        assert len(broker.get_positions()) == 0

    def test_close_position_not_found(self):
        broker = PaperBroker()
        broker.connect()
        assert not broker.close_position(99999)

    def test_close_all_positions(self):
        broker = PaperBroker()
        broker.connect()
        for _ in range(3):
            broker.send_order(TradeSignal(symbol="XAUUSD", direction="BUY",
                entry_price=2450, sl=2445, tp=2460, lot_size=0.1))
        assert len(broker.get_positions()) == 3
        assert broker.close_all_positions()
        assert len(broker.get_positions()) == 0

    def test_modify_sl_tp(self):
        broker = PaperBroker()
        broker.connect()
        result = broker.send_order(TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1))
        ticket = result["ticket"]
        assert broker.modify_position_sl_tp(ticket, sl=2440, tp=2470)
        pos = broker.get_positions()[0]
        assert pos.sl == 2440
        assert pos.tp == 2470

    def test_partial_close(self):
        broker = PaperBroker()
        broker.connect()
        result = broker.send_order(TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=1.0))
        ticket = result["ticket"]
        result = broker.close_position_partial(ticket, 0.5)
        assert result["success"]
        assert result["closed_volume"] == 0.5
        assert result["remaining_volume"] == 0.5

    def test_margin_calculation(self):
        broker = PaperBroker(initial_balance=10000, leverage=100)
        broker.connect()
        margin = broker._margin_required(1.0, 2450.0)
        assert margin == 2450.0

    def test_notional_calculation(self):
        broker = PaperBroker()
        notional = broker._notional(1.0, 2450.0)
        assert notional == 2450.0 * 100.0

    def test_check_sl_tp(self):
        broker = PaperBroker()
        broker.connect()
        broker.send_order(TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1))
        broker.update_price("XAUUSD", 2444.0, 2444.5)
        broker.check_sl_tp("XAUUSD", 2444.0, 2444.5)
        assert len(broker.get_positions()) == 0

    def test_health_check(self):
        broker = PaperBroker()
        broker.connect()
        health = broker.health_check()
        assert health["connected"]
        assert health["terminal_ok"]


class TestPaperBrokerMultiSymbolContractSize:
    """Regression tests for the multi-symbol contract-size bug found in
    audit: gold uses 100 oz/lot, standard forex pairs use 100,000 units/lot.
    Before the fix, every symbol used gold's contract size, producing
    wildly wrong P&L/margin for forex trades."""

    def test_forex_price_is_not_gold_fallback(self):
        broker = PaperBroker()
        broker.connect()
        price = broker.get_price("EURUSD")
        assert price["bid"] < 10  # a real EURUSD price, not gold's ~2450

    def test_forex_margin_uses_100k_contract_size(self):
        broker = PaperBroker(leverage=100.0)
        broker.connect()
        signal = TradeSignal(symbol="EURUSD", direction="BUY",
            entry_price=1.0850, sl=1.0800, tp=1.0900, lot_size=0.1)
        result = broker.send_order(signal)
        assert result["success"]
        account = broker.get_account_info()
        # notional = 0.1 lot * 1.0852 * 100,000 = ~10,852; margin = notional/100 = ~108.5
        expected_margin = 0.1 * 1.0852 * 100_000 / 100.0
        assert abs(account.margin - expected_margin) < 1.0

    def test_gold_still_uses_100oz_contract_size(self):
        broker = PaperBroker(leverage=100.0)
        broker.connect()
        signal = TradeSignal(symbol="XAUUSD", direction="BUY",
            entry_price=2450, sl=2445, tp=2460, lot_size=0.1)
        broker.send_order(signal)
        account = broker.get_account_info()
        expected_margin = 0.1 * 2450.5 * 100 / 100.0
        assert abs(account.margin - expected_margin) < 1.0

    def test_forex_pl_on_close_is_realistic(self):
        broker = PaperBroker(leverage=100.0)
        broker.connect()
        signal = TradeSignal(symbol="EURUSD", direction="BUY",
            entry_price=1.0850, sl=1.0800, tp=1.0900, lot_size=1.0)
        result = broker.send_order(signal)
        ticket = result["ticket"]
        broker.update_price("EURUSD", bid=1.0950, ask=1.0952)
        broker.close_position(ticket)
        # 0.0100 move * 1.0 lot * 100,000 = ~$1000, not ~$1 (which the old
        # gold-sized 100x multiplier would have produced)
        assert broker._balance > 10000 + 500


class TestBrokerConnectorABC:
    def test_abstract_class(self):
        with pytest.raises(TypeError):
            BrokerConnector()


class _FakeSymbolInfo:
    def __init__(self, visible=True):
        self.visible = visible
        self.trade_contract_size = 100.0
        self.trade_tick_size = 0.01
        self.trade_tick_value = 1.0
        self.volume_min = 0.01
        self.volume_max = 100.0
        self.volume_step = 0.01
        self.digits = 2
        self.point = 0.01


class _FakeTick:
    def __init__(self, bid, ask, time=0):
        self.bid = bid
        self.ask = ask
        self.time = time


class _FakeMT5Module:
    """Mimics the pieces of the MetaTrader5 package MT5Broker touches,
    for a broker whose native symbol has a suffix (e.g. Exness' 'XAUUSDm')
    that differs from the generic name a user might configure ('XAUUSD')."""

    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440

    def __init__(self, native_symbol="XAUUSDm"):
        self.native_symbol = native_symbol
        self.select_calls = []

    def terminal_info(self):
        return object()  # any non-None value signals "terminal is up"

    def symbol_info(self, name):
        return _FakeSymbolInfo() if name == self.native_symbol else None

    def symbol_select(self, name, visible):
        self.select_calls.append((name, visible))
        return True

    def symbol_info_tick(self, name):
        return _FakeTick(2450.0, 2450.5) if name == self.native_symbol else None

    def copy_rates_from_pos(self, name, timeframe, start, count):
        if name != self.native_symbol:
            return None
        import numpy as np
        dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"),
                 ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")]
        rows = [(1700000000 + i * 60, 2450.0, 2451.0, 2449.0, 2450.5, 100) for i in range(count)]
        return np.array(rows, dtype=dtype)


class TestMT5BrokerSymbolSuffixResolution:
    """Regression tests for a confirmed audit bug: a live-trading user's
    configured symbol (e.g. "XAUUSD") often doesn't match the broker's
    actual suffixed symbol name (e.g. Exness' "XAUUSDm"). load_historical_data
    already resolved this correctly via resolve_symbol() for backtests, but
    get_ohlcv/get_symbol_specs/get_price (used by the LIVE trading loop) did
    not — they used the raw requested symbol as-is, so MT5 calls silently
    failed forever with "No OHLCV data for XAUUSD M5" even though the
    broker-native "XAUUSDm" worked fine and MT5 was connected."""

    def _connected_broker(self, native_symbol="XAUUSDm"):
        from src.trading.broker_connector import MT5Broker
        broker = MT5Broker(login=0, password="", server="",
                            symbol_candidates=["XAUUSD", native_symbol])
        broker._mt5 = _FakeMT5Module(native_symbol=native_symbol)
        broker._connected = True
        broker.resolved_symbol = broker.resolve_symbol()
        return broker

    def test_get_ohlcv_resolves_suffixed_symbol(self):
        broker = self._connected_broker()
        df = broker.get_ohlcv(symbol="XAUUSD", timeframe="M5", bars=10)
        assert not df.empty, "get_ohlcv failed to resolve XAUUSD -> XAUUSDm"
        assert len(df) == 10

    def test_get_price_resolves_suffixed_symbol(self):
        broker = self._connected_broker()
        price = broker.get_price(symbol="XAUUSD")
        assert price["bid"] == 2450.0
        assert price["ask"] == 2450.5

    def test_get_symbol_specs_resolves_suffixed_symbol(self):
        broker = self._connected_broker()
        specs = broker.get_symbol_specs(symbol="XAUUSD")
        assert specs.get("symbol") == "XAUUSDm"
        assert specs.get("contract_size") == 100.0

    def test_get_ohlcv_still_works_when_symbol_already_matches(self):
        broker = self._connected_broker()
        df = broker.get_ohlcv(symbol="XAUUSDm", timeframe="M5", bars=5)
        assert not df.empty
