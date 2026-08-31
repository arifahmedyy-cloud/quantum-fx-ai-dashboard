"""Tests for risk_manager.py."""

import pytest
from src.trading.risk_manager import RiskManager
from src.config import RiskConfig


class TestRiskManager:
    def test_drawdown_adjustment_no_drawdown(self, risk_config):
        rm = RiskManager(risk_config)
        result = rm.compute_drawdown_adjusted_risk(10000, 10000, 1.0)
        assert result == 1.0

    def test_drawdown_adjustment_with_drawdown(self, risk_config):
        rm = RiskManager(risk_config)
        result = rm.compute_drawdown_adjusted_risk(9000, 10000, 1.0)
        expected = 1.0 * (1 - 0.1 * 0.5)
        assert abs(result - expected) < 0.01

    def test_drawdown_adjustment_max_reduction(self, risk_config):
        rm = RiskManager(risk_config)
        result = rm.compute_drawdown_adjusted_risk(1000, 10000, 1.0)
        assert result == 0.1

    def test_kelly_insufficient_history(self, risk_config):
        rm = RiskManager(risk_config)
        suggested, kelly = rm.kelly_position_size(10000, 2450, 2445, 1.0)
        assert suggested == 1.0
        assert kelly == 0.0

    def test_kelly_with_history(self, risk_config):
        rm = RiskManager(risk_config)
        for _ in range(15): rm.record_trade_result(100.0)
        for _ in range(5): rm.record_trade_result(-50.0)
        suggested, kelly = rm.kelly_position_size(10000, 2450, 2445, 1.0)
        assert suggested > 0
        assert kelly > 0
        assert suggested <= risk_config.max_risk_pct

    def test_lot_size_calculation(self, risk_config):
        rm = RiskManager(risk_config)
        lot = rm.calculate_lot_size(10000, 2450, 2445, 1.0)
        assert lot > 0
        assert lot <= 2.0

    def test_lot_size_zero_distance(self, risk_config):
        rm = RiskManager(risk_config)
        lot = rm.calculate_lot_size(10000, 2450, 2450, 1.0)
        assert lot == 0.01

    def test_daily_guard_no_breach(self, risk_config):
        rm = RiskManager(risk_config)
        guard = rm.daily_guard(10000, 10000)
        assert not guard.should_block_new_trades
        assert not guard.should_close_all

    def test_daily_guard_loss_limit(self, risk_config):
        rm = RiskManager(risk_config)
        for _ in range(10): rm.record_trade_result(-600.0)
        guard = rm.daily_guard(4000, 10000)
        assert guard.should_block_new_trades
        assert guard.should_close_all
        assert "Daily loss limit" in guard.reason

    def test_daily_guard_consecutive_losses(self, risk_config):
        rm = RiskManager(risk_config)
        for _ in range(3): rm.record_trade_result(-100.0)
        guard = rm.daily_guard(10000, 10000)
        assert guard.should_block_new_trades
        assert "consecutive losses" in guard.reason
        assert guard.guard_name == "max_consecutive_losses"

    def test_consecutive_losses_reset_on_new_trading_day(self):
        """Regression test for the permanent-block bug: the consecutive-loss
        counter must NOT survive a day rollover. Before the fix, it only
        reset on a winning trade — which can never happen while the guard
        itself is blocking new trades — so a single 3-loss streak silently
        blocked the account for the rest of the backtest/session."""
        from datetime import datetime as dt
        config = RiskConfig(max_consecutive_losses=3, daily_loss_limit_pct=99.0)
        rm = RiskManager(config)
        for _ in range(3):
            rm.record_trade_result(-100.0, current_time=dt(2025, 1, 2, 10), balance_before_trade=10000.0)
        blocked = rm.daily_guard(9700, 10000, current_time=dt(2025, 1, 2, 11))
        assert blocked.should_block_new_trades
        assert blocked.guard_name == "max_consecutive_losses"

        # New trading day: guard must clear even though no winning trade
        # ever occurred (none could, while blocked).
        after_rollover = rm.daily_guard(9700, 10000, current_time=dt(2025, 1, 3, 0, 1))
        assert not after_rollover.should_block_new_trades

    def test_consecutive_losses_permanent_block_does_not_recur_across_many_days(self):
        """Without the fix, this would stay permanently blocked for all 10
        simulated days after the first 3-loss streak on day 1."""
        from datetime import datetime as dt, timedelta
        config = RiskConfig(max_consecutive_losses=3, daily_loss_limit_pct=99.0)
        rm = RiskManager(config)
        for _ in range(3):
            rm.record_trade_result(-100.0, current_time=dt(2025, 1, 1, 10), balance_before_trade=10000.0)
        unblocked_days = 0
        for day_offset in range(1, 11):
            g = rm.daily_guard(9700, 10000, current_time=dt(2025, 1, 1) + timedelta(days=day_offset))
            if not g.should_block_new_trades:
                unblocked_days += 1
        assert unblocked_days == 10  # every subsequent day should be tradable again

    def test_consecutive_loss_cooldown_hours_resets_before_next_day(self):
        """The bounded-cooldown option resets on a fixed timer rather than
        waiting for the calendar day to roll over."""
        from datetime import datetime as dt
        config = RiskConfig(max_consecutive_losses=3, daily_loss_limit_pct=99.0,
                             consecutive_loss_cooldown_hours=6.0)
        rm = RiskManager(config)
        for _ in range(3):
            rm.record_trade_result(-100.0, current_time=dt(2025, 1, 2, 10), balance_before_trade=10000.0)
        still_blocked = rm.daily_guard(9700, 10000, current_time=dt(2025, 1, 2, 14))  # +4h
        assert still_blocked.should_block_new_trades
        resumed = rm.daily_guard(9700, 10000, current_time=dt(2025, 1, 2, 16, 1))  # +6h1m
        assert not resumed.should_block_new_trades

    def test_validate_signal_hard_max_risk_cap(self):
        config = RiskConfig(base_risk_pct=1.0, max_risk_pct=2.0, enable_ml_filter=False)
        rm = RiskManager(config)
        # $1,000 risk on $10,000 = 10%, must be rejected despite any base-risk sanity logic.
        valid, reason = rm.validate_signal(2450, 2440, 2470, 2.0, 10000, 70, contract_size=100)
        assert not valid
        assert "max risk" in reason.lower()

    def test_daily_loss_uses_day_start_balance(self):
        config = RiskConfig(daily_loss_limit_pct=5.0, max_consecutive_losses=99)
        rm = RiskManager(config)
        # First trade establishes the historical day's starting balance at 10,000.
        rm.record_trade_result(-400.0, current_time=__import__("datetime").datetime(2025, 1, 2), balance_before_trade=10000.0)
        guard = rm.daily_guard(9600.0, 12000.0, current_time=__import__("datetime").datetime(2025, 1, 2, 12))
        assert not guard.should_block_new_trades
        # A 5% loss is measured against 10,000, not the lifetime peak of 12,000.
        rm.record_trade_result(-100.0, current_time=__import__("datetime").datetime(2025, 1, 2, 13), balance_before_trade=9600.0)
        guard = rm.daily_guard(9500.0, 12000.0, current_time=__import__("datetime").datetime(2025, 1, 2, 14))
        assert guard.should_block_new_trades

    def test_validate_signal_valid(self, risk_config):
        rm = RiskManager(risk_config)
        valid, reason = rm.validate_signal(2450, 2445, 2460, 0.1, 10000, 70)
        assert valid
        assert reason == ""

    def test_validate_signal_low_ai_score(self, risk_config):
        rm = RiskManager(risk_config)
        valid, reason = rm.validate_signal(2450, 2445, 2460, 0.1, 10000, 40)
        assert not valid
        assert "AI score" in reason

    def test_validate_signal_poor_rr(self, risk_config):
        rm = RiskManager(risk_config)
        valid, reason = rm.validate_signal(2450, 2445, 2446, 0.1, 10000, 70)
        assert not valid
        assert "Reward" in reason

    def test_validate_signal_zero_sl(self, risk_config):
        rm = RiskManager(risk_config)
        valid, reason = rm.validate_signal(2450, 0, 2460, 0.1, 10000, 70)
        assert not valid
        assert "SL" in reason

    def test_risk_summary(self, risk_config):
        rm = RiskManager(risk_config)
        rm.record_trade_result(100.0)
        rm.record_trade_result(-50.0)
        summary = rm.get_risk_summary()
        assert summary["total_trades"] == 2
        assert summary["consecutive_losses"] == 1
