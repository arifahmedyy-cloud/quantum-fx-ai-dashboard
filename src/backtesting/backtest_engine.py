"""Auditable historical backtest engine.

Execution model:
- signals are generated from a completed candle close;
- orders execute on the following candle open;
- SL/TP are tested against candle High/Low (bid/ask adjusted for spread);
- ambiguous same-candle SL/TP is resolved by an explicit policy;
- historical timestamps are never replaced with wall-clock time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.config import BacktestConfig, RiskConfig
from src.logger import get_logger
from src.models import BacktestSummary, SignalOutput
from src.trading.decision_engine import DecisionEngine
from src.trading.indicators import TechnicalIndicators
from src.trading.regime_detector import RegimeDetector
from src.trading.risk_manager import RiskManager
from src.trading.smc import SMCAnalyzer
from src.trading.strategies import StrategySignal, strategy_signal_to_signal_output

log = get_logger(__name__)


def _bar_time(row: pd.Series) -> Optional[datetime]:
    """Return the historical timestamp from a Date column or DatetimeIndex."""
    if "Date" in row.index and pd.notna(row["Date"]):
        return pd.Timestamp(row["Date"]).to_pydatetime()
    if isinstance(row.name, (pd.Timestamp, datetime)):
        return pd.Timestamp(row.name).to_pydatetime()
    return None


def _normalize_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.index.name or "index": "Date"})
        elif "datetime" in out.columns:
            out = out.rename(columns={"datetime": "Date"})
        elif "time" in out.columns:
            out = out.rename(columns={"time": "Date"})
    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Backtest data missing columns: {missing}")
    out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    for c in ["Open", "High", "Low", "Close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Volume" not in out.columns:
        if "tick_volume" in out.columns:
            out["Volume"] = pd.to_numeric(out["tick_volume"], errors="coerce")
        else:
            out["Volume"] = 0.0
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[(out["High"] >= out[["Open", "Close"]].max(axis=1)) &
              (out["Low"] <= out[["Open", "Close"]].min(axis=1)) &
              (out["High"] >= out["Low"]) & (out["Close"] > 0)]
    out["Date"] = out["Date"].dt.tz_convert("UTC").dt.tz_localize(None)
    return out.reset_index(drop=True)


@dataclass
class SimulatedTrade:
    entry_time: datetime
    exit_time: Optional[datetime] = None
    signal_time: Optional[datetime] = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lot_size: float = 0.0
    profit_loss: float = 0.0
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    result: str = ""
    regime: str = ""
    r_multiple: float = 0.0
    duration_hours: float = 0.0
    exit_reason: str = ""


class BacktestEngine:
    """Chronological, multi-position backtest simulator."""

    def __init__(self, initial_balance: float = 10000.0,
                 risk_config: Optional[RiskConfig] = None,
                 spread: float = 0.5,
                 backtest_config: Optional[BacktestConfig] = None,
                 symbol_specs: Optional[Dict[str, float]] = None) -> None:
        self.initial_balance = float(initial_balance)
        self.risk_config = risk_config or RiskConfig()
        self.config = backtest_config or BacktestConfig(spread=spread)
        self.config.validate()
        self.symbol_specs = symbol_specs or {}
        self.spread = float(self.config.spread)
        self.equity_curve: List[float] = []
        self.trades: List[SimulatedTrade] = []
        self.validation: Dict[str, Any] = {}
        self.decision_log: List[Dict[str, Any]] = []
        self.guard_events: List[Dict[str, Any]] = []
        self.guard_diagnostics: Dict[str, Any] = {}
        self.regime_detector = RegimeDetector()
        self.smc = SMCAnalyzer()
        self.decision = DecisionEngine(self.risk_config)
        self.risk = RiskManager(self.risk_config)

    def _normalize_signal(self, sig: Any, current: pd.Series, strategy_fn: Callable) -> SignalOutput:
        if isinstance(sig, StrategySignal):
            return strategy_signal_to_signal_output(sig, getattr(strategy_fn, "__name__", "Custom"))
        if isinstance(sig, SignalOutput):
            return sig
        if isinstance(sig, dict):
            close = float(current["Close"])
            return SignalOutput(
                action=sig.get("action", "NO_TRADE"), confidence=int(sig.get("confidence", 50)),
                regime=sig.get("regime", "custom"), strategy=sig.get("strategy", "Custom"),
                expected_pf=float(sig.get("expected_pf", 1.5)), expected_max_dd=float(sig.get("expected_max_dd", 5.0)),
                expected_avg_rr=float(sig.get("expected_avg_rr", 2.0)), consistency_score=int(sig.get("confidence", 50)),
                sl=float(sig.get("sl", close)), tp=float(sig.get("tp", close)), entry=float(sig.get("entry", close)),
                lot_size=float(sig.get("lot_size", 0.01)), explanation=str(sig.get("explanation", "Custom strategy")), metrics=sig.get("metrics", {}),
            )
        raise TypeError(f"Unsupported strategy output type: {type(sig).__name__}")

    def _costed_entry(self, direction: str, open_price: float) -> float:
        slip = self.config.slippage
        if direction == "BUY":
            return open_price + self.spread / 2 + slip
        return open_price - self.spread / 2 - slip

    def _exit_market_price(self, direction: str, raw_price: float) -> float:
        slip = self.config.slippage
        if direction == "BUY":
            return raw_price - self.spread / 2 - slip
        return raw_price + self.spread / 2 + slip

    def _pnl(self, trade: SimulatedTrade, exit_price: float) -> float:
        # contract_size is the monetary value of a 1.0 price move per lot
        # unless real tick_size/tick_value are supplied.
        tick_size = self.symbol_specs.get("tick_size")
        tick_value = self.symbol_specs.get("tick_value")
        if tick_size and tick_value and tick_size > 0 and tick_value > 0:
            value = ((exit_price - trade.entry_price) / tick_size) * tick_value * trade.lot_size
        else:
            value = (exit_price - trade.entry_price) * trade.lot_size * float(self.symbol_specs.get("contract_size", 100.0))
        return value if trade.direction == "BUY" else -value

    def _check_exit(self, trade: SimulatedTrade, bar: pd.Series,
                    intrabar: Optional[pd.DataFrame] = None) -> Optional[tuple[float, str, datetime]]:
        """Return actual exit price/reason/time using OHLC and explicit ambiguity policy."""
        def hits(row: pd.Series) -> tuple[bool, bool]:
            high = float(row["High"]); low = float(row["Low"])
            if trade.direction == "BUY":
                bid_high, bid_low = high, low
                return bid_low <= trade.sl, bid_high >= trade.tp
            ask_high, ask_low = high + self.spread, low + self.spread
            return ask_high >= trade.sl, ask_low <= trade.tp

        # Gap-through protection: if the market opens beyond a protective
        # stop/target, the trade cannot be filled at the historical level that
        # is already behind the market. It exits at the executable entry/open
        # price instead of manufacturing a better fill.
        bar_open = float(bar["Open"])
        if trade.direction == "BUY":
            if bar_open <= trade.sl:
                exit_time = _bar_time(bar)
                if exit_time is None:
                    raise ValueError("Historical bar has no timestamp")
                return trade.entry_price, "SL", exit_time
            if bar_open >= trade.tp:
                exit_time = _bar_time(bar)
                if exit_time is None:
                    raise ValueError("Historical bar has no timestamp")
                return trade.entry_price, "TP", exit_time
        else:
            if bar_open >= trade.sl:
                exit_time = _bar_time(bar)
                if exit_time is None:
                    raise ValueError("Historical bar has no timestamp")
                return trade.entry_price, "SL", exit_time
            if bar_open <= trade.tp:
                exit_time = _bar_time(bar)
                if exit_time is None:
                    raise ValueError("Historical bar has no timestamp")
                return trade.entry_price, "TP", exit_time

        sl_hit, tp_hit = hits(bar)
        exit_time = _bar_time(bar)
        if exit_time is None:
            raise ValueError("Historical bar has no timestamp")

        if sl_hit and tp_hit and self.config.intrabar_policy == "LOWER_TIMEFRAME" and intrabar is not None and not intrabar.empty:
            for _, ib in intrabar.sort_values("Date").iterrows():
                ish, ith = hits(ib)
                if ish:
                    return self._exit_market_price(trade.direction, trade.sl), "SL", _bar_time(ib)  # type: ignore[arg-type]
                if ith:
                    return self._exit_market_price(trade.direction, trade.tp), "TP", _bar_time(ib)  # type: ignore[arg-type]
        if sl_hit and (not tp_hit or self.config.intrabar_policy == "SL_FIRST"):
            return self._exit_market_price(trade.direction, trade.sl), "SL", exit_time
        if tp_hit:
            return self._exit_market_price(trade.direction, trade.tp), "TP", exit_time
        return None

    def _close_trade(self, trade: SimulatedTrade, exit_price: float, exit_time: datetime,
                     reason: str, balance: float) -> float:
        trade.exit_time = exit_time
        trade.exit_price = float(exit_price)
        trade.exit_reason = reason
        trade.gross_pnl = self._pnl(trade, trade.exit_price)
        trade.commission = self.config.commission_per_lot * trade.lot_size
        trade.slippage_cost = self.config.slippage * trade.lot_size * float(self.symbol_specs.get("contract_size", 100.0)) * 2
        trade.profit_loss = trade.gross_pnl - trade.commission
        risk_distance = abs(trade.entry_price - trade.sl)
        trade.r_multiple = (trade.profit_loss / (risk_distance * trade.lot_size * float(self.symbol_specs.get("contract_size", 100.0)))) if risk_distance > 0 else 0.0
        trade.result = "WIN" if trade.profit_loss > 0 else "LOSS" if trade.profit_loss < 0 else "BREAKEVEN"
        trade.duration_hours = max(0.0, (exit_time - trade.entry_time).total_seconds() / 3600.0)
        self.risk.record_trade_result(
            trade.profit_loss,
            current_time=exit_time,
            balance_before_trade=balance,
        )
        self.trades.append(trade)
        return balance + trade.profit_loss

    def run(self, df: pd.DataFrame, strategy_fn: Optional[Callable[[pd.DataFrame, int], Any]] = None,
            intrabar_df: Optional[pd.DataFrame] = None,
            track_guard_diagnostics: bool = False) -> BacktestSummary:
        """Run a strictly chronological backtest with next-bar execution.

        Args:
            df: OHLCV candles.
            strategy_fn: Optional custom strategy callable.
            intrabar_df: Optional lower-timeframe data for intrabar SL/TP resolution.
            track_guard_diagnostics: When False (default), behavior is
                identical to before this flag existed — a guard-blocked bar
                skips signal generation entirely. When True, the engine
                additionally generates (but never acts on) a signal on
                guard-blocked bars purely to measure how many would-be
                signals each guard rejected. This is opt-in because it does
                extra SMC/decision-engine work per blocked bar; it never
                changes which trades are taken.
        """
        df = _normalize_utc(df)
        if len(df) <= self.config.warmup_bars + 1:
            raise ValueError(f"Not enough candles: need > {self.config.warmup_bars + 1}, got {len(df)}")
        df = TechnicalIndicators.add_all(df)
        if intrabar_df is not None:
            intrabar_df = _normalize_utc(intrabar_df)
        self.risk = RiskManager(self.risk_config)
        self.trades = []
        self.decision_log = []
        self.equity_curve = [self.initial_balance]
        balance = self.initial_balance
        peak_equity = balance
        pending: List[Dict[str, Any]] = []
        open_trades: List[SimulatedTrade] = []

        # --- Guard diagnostics state (see task 5/6 in the risk-manager audit) ---
        self.guard_events = []
        guard_candles_blocked: Dict[str, int] = {}
        guard_signals_blocked: Dict[str, int] = {}
        active_guard_block: Optional[Dict[str, Any]] = None

        for i in range(self.config.warmup_bars, len(df)):
            current = df.iloc[i]
            bar_time = _bar_time(current)
            if bar_time is None:
                raise ValueError("Historical bar has no timestamp")

            # 1) Execute signals generated on the previous completed candle.
            if pending:
                for order in pending:
                    if len(open_trades) >= self.risk_config.max_open_trades:
                        continue
                    entry_price = self._costed_entry(order["direction"], float(current["Open"]))
                    sl, tp = order["sl"], order["tp"]
                    risk_pct = self.risk_config.base_risk_pct
                    if self.risk_config.use_auto_drawdown_risk:
                        risk_pct = self.risk.compute_drawdown_adjusted_risk(balance, peak_equity, risk_pct)
                    specs = self.symbol_specs
                    lot = self.risk.calculate_lot_size(
                        balance, entry_price, sl, risk_pct,
                        contract_size=float(specs.get("contract_size", 100.0)),
                        quote_currency=str(specs.get("quote_currency", "USD")),
                        tick_size=specs.get("tick_size"), tick_value=specs.get("tick_value"),
                        volume_min=float(specs.get("volume_min", 0.01)), volume_max=float(specs.get("volume_max", 100.0)),
                        volume_step=float(specs.get("volume_step", 0.01)),
                    )
                    valid, _ = self.risk.validate_signal(
                        entry_price, sl, tp, lot, balance, int(order["ai_score"]),
                        contract_size=float(specs.get("contract_size", 100.0)),
                        tick_size=specs.get("tick_size"), tick_value=specs.get("tick_value"),
                    )
                    if not valid:
                        continue
                    trade = SimulatedTrade(signal_time=order["signal_time"], entry_time=bar_time,
                                           direction=order["direction"], entry_price=entry_price,
                                           sl=sl, tp=tp, lot_size=lot, regime=order["regime"])
                    open_trades.append(trade)
                pending = []

            # 2) Apply exits to all open trades using current candle range.
            for trade in list(open_trades):
                ib = None
                if intrabar_df is not None and self.config.use_m1_intrabar and i + 1 < len(df):
                    start, end = pd.Timestamp(bar_time), pd.Timestamp(_bar_time(df.iloc[i + 1]))
                    ib = intrabar_df[(intrabar_df["Date"] >= start) & (intrabar_df["Date"] < end)]
                hit = self._check_exit(trade, current, ib)
                if hit:
                    exit_price, reason, exit_time = hit
                    balance = self._close_trade(trade, exit_price, exit_time, reason, balance)
                    open_trades.remove(trade)

            # 3) Historical daily guard. It may close all positions and blocks new ones.
            guard = self.risk.daily_guard(balance, peak_equity, current_time=bar_time)
            if guard.should_close_all and open_trades:
                for trade in list(open_trades):
                    exit_price = self._exit_market_price(trade.direction, float(current["Close"]))
                    balance = self._close_trade(trade, exit_price, bar_time, "Daily guard", balance)
                    open_trades.remove(trade)
            pending = [] if guard.should_block_new_trades else pending

            # 3b) Guard diagnostics bookkeeping (candle-level; always cheap —
            # no extra strategy/decision calls happen here). Tracks how long
            # each guard stays triggered and when it opens/closes, so the
            # audit can show whether a block resets within the intended
            # period or persists indefinitely.
            if guard.should_block_new_trades:
                gname = guard.guard_name or "unknown"
                guard_candles_blocked[gname] = guard_candles_blocked.get(gname, 0) + 1
                if active_guard_block is None or active_guard_block["guard_name"] != gname:
                    if active_guard_block is not None:
                        active_guard_block["block_end"] = bar_time
                        self.guard_events.append(active_guard_block)
                    active_guard_block = {
                        "guard_name": gname, "reason": guard.reason,
                        "block_start": bar_time, "block_end": None,
                        "candles_blocked": 0, "signals_blocked": 0,
                    }
                active_guard_block["candles_blocked"] += 1
            elif active_guard_block is not None:
                active_guard_block["block_end"] = bar_time
                self.guard_events.append(active_guard_block)
                active_guard_block = None

            # 4) Mark-to-market equity at completed candle close.
            equity = balance
            for trade in open_trades:
                mark = self._exit_market_price(trade.direction, float(current["Close"]))
                equity += self._pnl(trade, mark)
            peak_equity = max(peak_equity, equity)
            self.equity_curve.append(equity)

            # 5) Generate a signal only after candle i is complete; queue for i+1.
            #    Default behavior (track_guard_diagnostics=False) is unchanged:
            #    a guard-blocked bar skips signal generation entirely. Opt-in
            #    diagnostics mode additionally generates (but never acts on) a
            #    signal on blocked bars, purely to count rejected signals.
            if i >= len(df) - 1:
                continue
            if guard.should_block_new_trades and not track_guard_diagnostics:
                continue
            # PERFORMANCE/PARITY FIX: bound the analysis window to the same
            # rolling lookback the live bot actually uses (see
            # BacktestConfig.analysis_window_bars). The previous
            # `df.iloc[:i + 1]` grew by one row every bar with no upper
            # bound, so SMC's swing/structure/OB/FVG/sweep detection (each
            # an O(window length) scan) and the per-bar `.copy()` itself
            # made the whole backtest O(n^2) in candle count — impractically
            # slow for any realistic backtest size — and meant the backtest
            # analyzed far more history per bar than live trading ever does.
            window_start = max(0, i + 1 - self.config.analysis_window_bars)
            window = df.iloc[window_start:i + 1].copy()
            if strategy_fn:
                signal = self._normalize_signal(strategy_fn(window, i), current, strategy_fn)
            else:
                signal = self.regime_detector.generate_signal(window)
            smc_result = self.smc.analyze(window)
            decision = self.decision.decide(signal, smc_result, current_price=float(current["Close"]))
            would_signal = decision.action in ("BUY", "SELL")
            self.decision_log.append({
                "time": bar_time.isoformat(),
                "regime": signal.regime,
                "regime_action": signal.action,
                "regime_confidence": signal.confidence,
                "strategy": signal.strategy,
                "smc_bias": smc_result.bias,
                "smc_zone": smc_result.zone,
                "decision": decision.action,
                "decision_score": decision.ai_score,
                "reason": getattr(decision, "explanation", ""),
                "accepted": would_signal,
                "guard_blocked": bool(guard.should_block_new_trades),
                "guard_name": guard.guard_name,
            })
            if guard.should_block_new_trades:
                if would_signal:
                    gname = guard.guard_name or "unknown"
                    guard_signals_blocked[gname] = guard_signals_blocked.get(gname, 0) + 1
                    if active_guard_block is not None:
                        active_guard_block["signals_blocked"] += 1
                continue
            if would_signal and len(open_trades) + len(pending) < self.risk_config.max_open_trades:
                pending.append({
                    "direction": decision.action, "sl": float(decision.sl), "tp": float(decision.tp),
                    "ai_score": decision.ai_score, "signal_time": bar_time, "regime": signal.regime,
                })

        # Pending orders at the final candle have no next-open execution and are discarded.
        if open_trades:
            final = df.iloc[-1]
            final_time = _bar_time(final)
            if final_time is None:
                raise ValueError("Final historical candle has no timestamp")
            if self.config.end_of_data_policy in {"CLOSE_AT_LAST_CLOSE", "MARK_TO_MARKET"}:
                for trade in list(open_trades):
                    exit_price = self._exit_market_price(trade.direction, float(final["Close"]))
                    balance = self._close_trade(trade, exit_price, final_time, "End of data", balance)
                    open_trades.remove(trade)
                self.equity_curve.append(balance)

        wins = sum(t.profit_loss > 0 for t in self.trades)
        losses = sum(t.profit_loss < 0 for t in self.trades)
        gross_profit = sum(t.profit_loss for t in self.trades if t.profit_loss > 0)
        gross_loss = abs(sum(t.profit_loss for t in self.trades if t.profit_loss < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        max_dd = 0.0; peak = self.equity_curve[0] if self.equity_curve else self.initial_balance
        for eq in self.equity_curve:
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak * 100 if peak else 0.0)
        regime_breakdown: Dict[str, Dict[str, Any]] = {}
        for t in self.trades:
            g = regime_breakdown.setdefault(t.regime, {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})
            g["trades"] += 1; g["wins"] += int(t.profit_loss > 0); g["losses"] += int(t.profit_loss < 0); g["pnl"] += t.profit_loss
        self.validation = {
            "timezone": "UTC internal",
            "signal_on": self.config.signal_on,
            "entry_on": self.config.entry_on,
            "intrabar_policy": self.config.intrabar_policy,
            "end_of_data_policy": self.config.end_of_data_policy,
            "spread": self.config.spread,
            "slippage": self.config.slippage,
            "commission_per_lot": self.config.commission_per_lot,
            "warmup_bars": self.config.warmup_bars,
            "lookahead_safe_execution": True,
            "max_open_trades": self.risk_config.max_open_trades,
            "regime_detector_active": strategy_fn is None,
            "ai_reviewer": "DISABLED_FOR_BACKTEST" if self.risk_config.enable_ai_reviewer else "DISABLED",
            "decision_log_entries": len(self.decision_log),
            "strict_smc_confluence": self.risk_config.strict_smc_confluence,
        }

        # Close out a guard block still active when the data ends (block_end
        # stays None to flag "never reset within the backtest window").
        if active_guard_block is not None:
            self.guard_events.append(active_guard_block)
            active_guard_block = None

        total_bars = max(0, len(df) - self.config.warmup_bars)
        total_blocked = sum(guard_candles_blocked.values())
        self.guard_diagnostics = {
            "track_guard_diagnostics": track_guard_diagnostics,
            "total_bars_evaluated": total_bars,
            "candles_blocked_by_guard": dict(guard_candles_blocked),
            "pct_bars_blocked_by_guard": {
                g: round(c / total_bars * 100, 2) if total_bars else 0.0
                for g, c in guard_candles_blocked.items()
            },
            "total_candles_blocked": total_blocked,
            "pct_bars_blocked_total": round(total_blocked / total_bars * 100, 2) if total_bars else 0.0,
            "signals_blocked_by_guard": dict(guard_signals_blocked),
            "guard_events": self.guard_events,
        }

        return BacktestSummary(
            final_balance=balance,
            return_pct=(balance - self.initial_balance) / self.initial_balance * 100 if self.initial_balance else 0.0,
            trades=len(self.trades), wins=wins,
            win_rate=wins / len(self.trades) * 100 if self.trades else 0.0,
            profit_factor=pf, max_dd=max_dd,
            equity_curve=self.equity_curve, regime_breakdown=regime_breakdown,
            guard_diagnostics=self.guard_diagnostics,
        )
