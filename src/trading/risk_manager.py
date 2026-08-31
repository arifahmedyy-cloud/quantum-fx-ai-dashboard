"""Risk management with Kelly sizing, drawdown adjustment, and daily guards.

All position sizing and risk validation logic centralized here.
"""

from __future__ import annotations

import math
from datetime import datetime, date
from typing import Optional, Dict, Any, List

from src.logger import get_logger
from src.models import DailyGuardStatus
from src.config import RiskConfig
from src.exceptions import RiskError

log = get_logger(__name__)


class RiskManager:
    """Centralized risk management for all trades."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_reset_date: Optional[date] = None
        self._trade_history: List[Dict[str, Any]] = []
        self._max_daily_pnl = 0.0
        self._guard_logged_reason: Optional[str] = None  # dedupes repeated-bar log spam
        self._daily_start_balance: Optional[float] = None
        # Timestamp at which _consecutive_losses first reached the configured
        # threshold; used only by the bounded-cooldown path (see daily_guard).
        self._consecutive_loss_block_start: Optional[datetime] = None

    def _reset_daily_if_needed(self, as_of: Optional[date] = None, starting_balance: Optional[float] = None) -> None:
        """Reset daily counters when the day rolls over.

        Args:
            as_of: The date to check against. Defaults to the real
                wall-clock date (correct for live trading). CRITICAL for
                backtesting: without passing the simulated bar's date here,
                this always compares against `date.today()` — the date the
                backtest process is actually running, not the historical
                date being simulated. Since a backtest of months of data
                runs in seconds of wall-clock time, `date.today()` never
                changes mid-run, so daily counters would never reset and
                the "daily" guard would silently become a guard over the
                ENTIRE backtest's cumulative P&L instead of each simulated
                day — making every fold after an early bad stretch
                artificially blocked for the rest of the run.
        """
        today = as_of if as_of is not None else date.today()
        if self._last_reset_date != today:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._max_daily_pnl = 0.0
            self._last_reset_date = today
            self._daily_start_balance = float(starting_balance) if starting_balance is not None and starting_balance > 0 else self._daily_start_balance
            self._guard_logged_reason = None
            # BUG FIX: the consecutive-loss counter used to only reset on a
            # winning trade. Since a win can never occur while this same
            # guard is blocking new trades, that made "N consecutive losses"
            # a PERMANENT block for the remainder of the backtest/session
            # once triggered once, despite being logged and named as a DAILY
            # guard. Reset it here so it behaves as a same-day guard, unless
            # a bounded cooldown is configured, in which case that mechanism
            # (see daily_guard()) owns the reset instead.
            if self.config.consecutive_loss_cooldown_hours is None:
                self._consecutive_losses = 0
                self._consecutive_loss_block_start = None
            log.info("Daily risk counters reset (day=%s, start_balance=%s)", today, self._daily_start_balance)
        elif self._daily_start_balance is None and starting_balance is not None and starting_balance > 0:
            self._daily_start_balance = float(starting_balance)

    def compute_drawdown_adjusted_risk(
        self,
        balance: float,
        peak_balance: float,
        base_risk_pct: float,
    ) -> float:
        """Reduce risk proportionally to current drawdown.

        Args:
            balance: Current account balance.
            peak_balance: Highest balance achieved.
            base_risk_pct: Base risk percentage.

        Returns:
            Adjusted risk percentage.
        """
        if peak_balance <= 0 or balance <= 0:
            return base_risk_pct
        drawdown_pct = (peak_balance - balance) / peak_balance
        if drawdown_pct <= 0:
            return base_risk_pct
        if drawdown_pct >= 0.5:
            # Severe drawdown circuit breaker: cap risk to the floor.
            return max(0.1, base_risk_pct * 0.1)
        reduction = 1 - (drawdown_pct * self.config.drawdown_reduction_factor)
        return max(0.1, base_risk_pct * max(reduction, 0.1))

    def kelly_position_size(
        self,
        balance: float,
        entry: float,
        sl: float,
        base_risk_pct: float,
    ) -> tuple[float, float]:
        """Calculate position size using Kelly Criterion with fractional Kelly.

        Args:
            balance: Account balance.
            entry: Entry price.
            sl: Stop loss price.
            base_risk_pct: Base risk percentage.

        Returns:
            Tuple of (suggested_risk_pct, kelly_pct).
        """
        if len(self._trade_history) < self.config.kelly_min_trades:
            log.info("Kelly: insufficient history (%d < %d), using base risk",
                     len(self._trade_history), self.config.kelly_min_trades)
            return base_risk_pct, 0.0

        wins = [t for t in self._trade_history if t.get("profit_loss", 0) > 0]
        losses = [t for t in self._trade_history if t.get("profit_loss", 0) < 0]
        if not wins or not losses:
            return base_risk_pct, 0.0

        win_rate = len(wins) / len(self._trade_history)
        avg_win = sum(t["profit_loss"] for t in wins) / len(wins)
        avg_loss = abs(sum(t["profit_loss"] for t in losses) / len(losses))
        if avg_loss == 0:
            return base_risk_pct, 0.0

        kelly_pct = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        kelly_pct = max(0, min(kelly_pct, 0.25))
        suggested = round(
            min(max(kelly_pct * self.config.kelly_fraction * 100, 0.25), self.config.max_risk_pct),
            2,
        )
        log.info("Kelly: win_rate=%.2f kelly=%.4f suggested=%.2f%%", win_rate, kelly_pct, suggested)
        return suggested, kelly_pct

    def calculate_lot_size(
        self,
        balance: float,
        entry: float,
        sl: float,
        risk_pct: float,
        leverage: float = 100.0,
        contract_size: float = 100.0,
        quote_currency: str = "USD",
        tick_size: Optional[float] = None,
        tick_value: Optional[float] = None,
        volume_min: float = 0.01,
        volume_max: float = 100.0,
        volume_step: float = 0.01,
    ) -> float:
        """Calculate lot size for the risk_pct being risked on this trade.

        Generalized from a gold-only formula (which is why the defaults
        below still match gold's 100 oz/lot — existing XAUUSD-only callers
        get identical behavior to before). For other instruments, pass the
        symbol's real `contract_size` and `quote_currency` from
        `SymbolManager.profile(symbol)`:
            - Forex majors (contract_size=100000): a 1.0-unit price move on
              a USD-quoted pair (EURUSD, GBPUSD, AUDUSD) is worth
              contract_size dollars per lot directly.
            - USD-base pairs quoted in a non-USD currency (USDJPY,
              quote_currency="JPY"): converted to USD using the entry price
              itself as the JPY/USD proxy rate. This is the standard retail
              approximation, not an exact live-rate conversion — acceptable
              for position sizing, not for accounting.
            - Gold (contract_size=100): $1 move = $100 per lot.
            - Crypto (contract_size=1, e.g. BTCUSD at 1 lot = 1 BTC):
              $1 move = $1 per lot. Verify your broker's actual contract
              size before relying on this — it varies by broker for crypto.

        Args:
            balance: Account balance.
            entry: Entry price.
            sl: Stop loss.
            risk_pct: Risk percentage.
            leverage: Account leverage.
            contract_size: Units of the base instrument per 1.0 lot.
            quote_currency: The currency the symbol is priced in. Only
                "USD" is treated as needing no conversion; anything else
                approximates a conversion via the entry price.

        Returns:
            Lot size (rounded to 2 decimals), minimum 0.01.
        """
        risk_amount = balance * (risk_pct / 100.0)
        price_distance = abs(entry - sl)
        if price_distance <= 0:
            log.warning("Zero price distance, defaulting to 0.01 lot")
            return 0.01

        if tick_size and tick_value and tick_size > 0 and tick_value > 0:
            value_per_price_unit_per_lot = tick_value / tick_size
        elif quote_currency == "USD":
            value_per_price_unit_per_lot = contract_size
        else:
            if entry <= 0:
                log.warning("Non-positive entry price for a non-USD-quoted symbol, defaulting to minimum lot")
                return max(volume_min, 0.01)
            value_per_price_unit_per_lot = contract_size / entry

        raw_lots = risk_amount / (price_distance * value_per_price_unit_per_lot)

        # Margin check. Notional value of the position in USD:
        #   - USD-quoted symbols (gold, EURUSD, GBPUSD, AUDUSD, BTCUSD):
        #     contract_size units of the base instrument, priced in USD.
        #   - USD-BASE symbols quoted in another currency (USDJPY):
        #     contract_size IS already the USD amount (100,000 units of the
        #     USD base currency) — multiplying by the JPY price would be wrong.
        if quote_currency == "USD":
            notional_usd = raw_lots * contract_size * entry
        else:
            notional_usd = raw_lots * contract_size
        margin = notional_usd / leverage
        if margin > balance * 0.5:
            max_notional = balance * 0.5 * leverage
            raw_lots = max_notional / (contract_size * entry) if quote_currency == "USD" else max_notional / contract_size
            log.warning("Margin limit hit, reduced to %.2f lots", raw_lots)
        raw_lots = min(float(volume_max), max(float(volume_min), raw_lots))
        step = max(float(volume_step), 1e-9)
        raw_lots = math.floor((raw_lots + 1e-12) / step) * step
        raw_lots = min(float(volume_max), max(float(volume_min), raw_lots))
        return float(round(raw_lots, 8))

    def daily_guard(
        self,
        balance: float,
        peak_balance: float,
        current_time: Optional[datetime] = None,
    ) -> DailyGuardStatus:
        """Check if daily limits are breached.

        Args:
            balance: Current balance.
            peak_balance: Peak balance.
            current_time: The current moment to evaluate against. In live
                trading, omit this (defaults to real time). In a backtest,
                ALWAYS pass the simulated bar's timestamp here — see
                `_reset_daily_if_needed`'s docstring for why this matters.

        Returns:
            DailyGuardStatus with block/close recommendations.
        """
        as_of_date = current_time.date() if current_time is not None else None
        self._reset_daily_if_needed(as_of_date, starting_balance=balance)

        # Bounded-cooldown path: if configured, the consecutive-loss guard
        # resets a fixed number of hours after it was triggered instead of
        # waiting for the calendar day to roll over (see config field docs).
        if (
            self.config.consecutive_loss_cooldown_hours is not None
            and self._consecutive_losses >= self.config.max_consecutive_losses
            and self._consecutive_loss_block_start is not None
            and current_time is not None
        ):
            elapsed_hours = (current_time - self._consecutive_loss_block_start).total_seconds() / 3600.0
            if elapsed_hours >= self.config.consecutive_loss_cooldown_hours:
                log.info(
                    "Consecutive-loss cooldown elapsed (%.1fh >= %.1fh); resuming trading",
                    elapsed_hours, self.config.consecutive_loss_cooldown_hours,
                )
                self._consecutive_losses = 0
                self._consecutive_loss_block_start = None

        should_block = False
        should_close = False
        reason = ""
        guard_name = ""

        baseline = self._daily_start_balance if self._daily_start_balance and self._daily_start_balance > 0 else balance
        daily_pnl_pct = (self._daily_pnl / baseline * 100) if baseline > 0 else 0

        if daily_pnl_pct <= -self.config.daily_loss_limit_pct:
            should_block = True
            should_close = True
            reason = f"Daily loss limit hit: {daily_pnl_pct:.2f}%"
            guard_name = "daily_loss_limit"
        elif self._consecutive_losses >= self.config.max_consecutive_losses:
            should_block = True
            reason = f"Max consecutive losses ({self.config.max_consecutive_losses}) reached"
            guard_name = "max_consecutive_losses"
        elif self.config.daily_profit_lock_pct and daily_pnl_pct >= self.config.daily_profit_lock_pct:
            should_block = True
            reason = f"Daily profit target reached: {daily_pnl_pct:.2f}%"
            guard_name = "daily_profit_lock"

        # Log only on the transition into a triggered state, not on every
        # single call (a backtest calls this once per bar — logging every
        # time a guard stays triggered floods the log with thousands of
        # identical lines and buries anything actually useful).
        if reason and reason != self._guard_logged_reason:
            log.warning("DAILY GUARD: %s", reason)
            self._guard_logged_reason = reason
        elif not reason:
            self._guard_logged_reason = None

        return DailyGuardStatus(
            should_block_new_trades=should_block,
            should_close_all=should_close,
            daily_pnl_pct=daily_pnl_pct,
            reason=reason,
            guard_name=guard_name,
        )

    def record_trade_result(self, profit_loss: float, current_time: Optional[datetime] = None,
                            balance_before_trade: Optional[float] = None) -> None:
        """Record trade result for consecutive loss tracking.

        Args:
            profit_loss: The realized P/L of the closed trade.
            current_time: The moment the trade closed. In a backtest, ALWAYS
                pass the simulated bar's timestamp — this must use the same
                notion of "now" as `daily_guard()`, or the two will reset
                against different clocks (this one against wall-clock, that
                one against simulated time) and thrash each other's daily
                counters. See `_reset_daily_if_needed`'s docstring.
        """
        as_of_date = current_time.date() if current_time is not None else None
        self._reset_daily_if_needed(as_of_date, starting_balance=balance_before_trade)
        record_time = current_time if current_time is not None else datetime.now()
        self._trade_history.append({"profit_loss": profit_loss, "time": record_time})
        self._daily_pnl += profit_loss
        self._daily_trades += 1
        if profit_loss < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses == self.config.max_consecutive_losses:
                # Mark the moment the block started, for the bounded-cooldown
                # path in daily_guard(). Harmless no-op when cooldown is off.
                self._consecutive_loss_block_start = record_time
        else:
            self._consecutive_losses = 0
            self._consecutive_loss_block_start = None
        if self._daily_pnl > self._max_daily_pnl:
            self._max_daily_pnl = self._daily_pnl
        log.info("Risk record: P/L=$%.2f daily_pnl=$%.2f consecutive_losses=%d",
                 profit_loss, self._daily_pnl, self._consecutive_losses)

    def validate_signal(
        self,
        entry: float,
        sl: float,
        tp: float,
        lot_size: float,
        balance: float,
        ai_score: int,
        ml_confidence: Optional[float] = None,
        contract_size: float = 100.0,
        tick_size: Optional[float] = None,
        tick_value: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Validate a trade signal against all risk rules.

        Args:
            entry: Entry price.
            sl: Stop loss.
            tp: Take profit.
            lot_size: Calculated lot size.
            balance: Account balance.
            ai_score: AI confidence score.
            ml_confidence: ML model confidence.

        Returns:
            Tuple of (is_valid, reason).
        """
        if ai_score < self.config.min_ai_score:
            return False, f"AI score {ai_score} below minimum {self.config.min_ai_score}"
        if self.config.enable_ml_filter and ml_confidence is not None:
            if ml_confidence < self.config.min_ml_confidence:
                return False, f"ML confidence {ml_confidence:.1f} below minimum {self.config.min_ml_confidence}"
        if sl <= 0 or tp <= 0:
            return False, "SL and TP must be positive"
        if entry == sl:
            return False, "Entry equals SL (zero risk distance)"
        risk_distance = abs(entry - sl)
        reward_distance = abs(tp - entry)
        if reward_distance < risk_distance:
            return False, f"Reward ({reward_distance:.2f}) < Risk ({risk_distance:.2f})"
        max_risk_amount = balance * (self.config.max_risk_pct / 100.0)
        risk_amount = balance * (self.config.base_risk_pct / 100.0)
        if tick_size and tick_value and tick_size > 0 and tick_value > 0:
            calculated_risk = (risk_distance / tick_size) * tick_value * lot_size
        else:
            calculated_risk = risk_distance * lot_size * contract_size
        if calculated_risk > max_risk_amount + 1e-9:
            return False, f"Calculated risk ${calculated_risk:.2f} exceeds max risk ${max_risk_amount:.2f}"
        return True, ""

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk status summary."""
        self._reset_daily_if_needed()
        return {
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "total_trades": len(self._trade_history),
            "max_daily_pnl": self._max_daily_pnl,
            "win_rate": len([t for t in self._trade_history if t.get("profit_loss", 0) > 0]) / max(1, len(self._trade_history)),
        }
