"""XAU/USD AI Trading Bot — Production Dashboard."""

from __future__ import annotations

import os, sys, time, threading
from datetime import datetime, timedelta, timezone
from typing import Any, List, Dict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# BUG FIX (audit): `threading.Thread(target=trading_loop, ...)` spawns a
# plain Python thread with NO Streamlit ScriptRunContext attached. Streamlit
# binds `st.session_state` to the ScriptRunContext of whichever thread is
# running it — a background thread without that context cannot see the
# main session's state at all, so `st.session_state.broker` (and every
# other key) raises AttributeError/KeyError inside the thread ("st.session_state
# has no attribute 'broker'"), even though it was set correctly on the main
# thread. This is a well-known Streamlit threading gotcha; the documented
# fix is to explicitly copy the calling thread's ScriptRunContext onto the
# new thread with add_script_run_ctx before starting it.
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:  # pragma: no cover - older Streamlit versions
    try:
        from streamlit.script_run_context import add_script_run_ctx, get_script_run_ctx
    except ImportError:
        add_script_run_ctx = None
        get_script_run_ctx = None


def _start_background_thread(target, args=()) -> threading.Thread:
    """Start a daemon thread that can safely access st.session_state.

    See the BUG FIX note above import add_script_run_ctx for why this is
    necessary instead of a bare `threading.Thread(...).start()`.
    """
    thread = threading.Thread(target=target, args=args, daemon=True)
    if add_script_run_ctx is not None and get_script_run_ctx is not None:
        ctx = get_script_run_ctx()
        if ctx is not None:
            add_script_run_ctx(thread, ctx)
    thread.start()
    return thread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger import configure_logging, get_logger
from src.config import load_config, BotConfig
from src.models import TradeSignal
from src.trading.broker_connector import PaperBroker, MT5Broker, MT5BridgeBroker, _contract_size
from src.trading.regime_detector import RegimeDetector
from src.trading.smc import SMCAnalyzer
from src.trading.decision_engine import DecisionEngine
from src.trading.risk_manager import RiskManager
from src.trading.symbol_manager import SymbolManager
from src.trading.session_guard import SessionGuard
from src.trading.economic_calendar_guard import EconomicCalendarGuard
from src.trading.correlation_guard import CorrelationGuard, OpenExposure
from src.trading.indicators import TechnicalIndicators
from src.trading.strategies import get_strategy, STRATEGIES, strategy_signal_to_signal_output
from src.trading.trailing_stop_manager import TrailingStopManager
from src.services.paper_market_data import generate_paper_ohlcv, paper_price
from src.services.journal_service import JournalService
from src.services.notification_service import NotificationService
from src.services.health_service import HealthService
from src.services.news_service import NewsService
from src.services.ai_service import AIService
from src.services import credential_store
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.monte_carlo import MonteCarloSimulator
from src.backtesting.performance_report import PerformanceReporter
from src.backtesting.mt5_data_cache import MT5DataCache
from src.ui.components import (
    render_header, render_sidebar_config, render_account_card,
    render_positions_table, render_trade_history, render_signal_card,
    render_health_status, render_backtest_results, render_monte_carlo,
    render_spread_widget, render_session_widget, render_daily_summary_widget,
)

log = get_logger(__name__)


def init_session_state() -> None:
    defaults = {
        "bot_running": False, "last_update": None, "broker": None,
        "data_service": None, "journal": None, "notifier": None,
        "health": None, "regime_detector": None, "smc": None,
        "decision_engine": None, "risk_manager": None,
        "current_signal": None, "backtest_summary": None,
        "monte_carlo_report": None, "peak_balance": 10000.0,
        "error_count": 0, "last_error": None, "broker_signature": None,
        "news_service": None, "ai_service": None,
        "current_news": None, "current_ai": None,
        "symbol_manager": None, "current_signals": {},
        "live_strategy_name": "Regime (default)", "symbol_signal_cache": {},
        "session_guard": None, "calendar_guard": None, "ml_model": None,
        "ml_model_status": "not loaded",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.get("ml_model") is None:
        _try_load_ml_model()


_DEFAULT_ML_MODEL_PATH = os.path.join("models", "ml_signal_model.json")


def _try_load_ml_model(path: str = _DEFAULT_ML_MODEL_PATH) -> None:
    """Load a previously-trained ML signal model if one exists on disk.

    Non-fatal by design: a missing/unreadable model just means the ML
    filter behaves as if disabled (ml_confidence stays None), exactly the
    same as before this feature was completed. Train one with
    `python tools/train_ml_signal.py`.
    """
    from src.trading.ml_signal import MLSignalModel
    try:
        st.session_state.ml_model = MLSignalModel.load(path)
        st.session_state.ml_model_status = f"loaded ({path})"
        log.info("ML signal model loaded from %s", path)
    except Exception as exc:
        st.session_state.ml_model = None
        st.session_state.ml_model_status = f"not available ({exc})"
        log.info("No ML signal model loaded: %s", exc)


def create_broker(config: BotConfig) -> Any:
    if config.broker == "paper":
        return PaperBroker(initial_balance=10000.0, leverage=config.mt5.leverage, symbol="XAUUSD")
    elif config.broker == "mt5":
        return MT5Broker(
            login=config.mt5.login, password=config.mt5.password,
            server=config.mt5.server, leverage=config.mt5.leverage,
            reconnect_attempts=config.mt5.reconnect_attempts,
            reconnect_backoff_seconds=config.mt5.reconnect_backoff_seconds,
            reconnect_backoff_multiplier=config.mt5.reconnect_backoff_multiplier,
        )
    elif config.broker == "mt5_bridge":
        # App runs on Linux/Docker; MT5 itself is reached over HTTP via a
        # small Windows-side bridge service (see mt5_bridge/). This is the
        # broker to use in the Docker deployment described in the README.
        return MT5BridgeBroker(
            base_url=config.mt5_bridge.base_url, token=config.mt5_bridge.token,
            symbol_candidates=config.mt5_bridge.symbol_candidates,
            reconnect_attempts=config.mt5_bridge.reconnect_attempts,
            reconnect_backoff_seconds=config.mt5_bridge.reconnect_backoff_seconds,
            reconnect_backoff_multiplier=config.mt5_bridge.reconnect_backoff_multiplier,
            request_timeout_seconds=config.mt5_bridge.request_timeout_seconds,
            leverage=config.mt5_bridge.leverage,
        )
    raise ValueError(f"Unknown broker: {config.broker}")


def _fetch_live_ohlcv(
    broker: Any, config: BotConfig, symbol: str, timeframe: str = "H1",
    bars: int = 200, yf_period: str = "5d", yf_interval: str = "15m"
) -> pd.DataFrame:
    """Fetch market data from the selected execution mode only.

    PAPER mode uses deterministic local synthetic data.
    MT5 / MT5 bridge uses broker-native OHLCV only.
    There is intentionally NO Yahoo Finance fallback in the trading path.
    """
    if config.broker == "paper":
        paper_symbol = "XAUUSD" if symbol.lower().startswith("xauusd") else symbol
        df = generate_paper_ohlcv(paper_symbol, timeframe, bars)
        px = paper_price(paper_symbol, timeframe)
        if broker is not None and hasattr(broker, "update_price"):
            broker.update_price(paper_symbol, px["bid"], px["ask"])
        return df

    if config.broker in ("mt5", "mt5_bridge") and broker is not None and broker.is_connected():
        df = broker.get_ohlcv(symbol=symbol, timeframe=timeframe, bars=bars)
        if df is None or df.empty:
            raise RuntimeError(
                f"No broker-native MT5 data for {symbol} {timeframe}; "
                "Yahoo fallback is disabled."
            )
        return df

    raise RuntimeError(
        f"Broker {config.broker!r} is not connected; no external/Yahoo fallback is permitted."
    )


def _run_symbol_cycle(
    symbol: str, config: BotConfig, broker: Any, journal: Any, notifier: Any, health: Any,
    regime: Any, smc: Any, decision: Any, risk: Any, news_service: Any, ai_service: Any,
    symbol_manager: Any, correlation_guard: Any, account: Any, live_strategy_name: str,
    signal_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one full analysis+trade cycle for a single symbol. Returns the
    signal dict for display. Never raises — caller's try/except wraps this,
    but errors here are also caught locally so one bad symbol doesn't stop
    the others in the same cycle.

    Indicators/regime-or-strategy/SMC are only recomputed when a new candle
    is detected (candle-based caching) — `signal_cache` persists this across
    calls (owned by trading_loop, one entry per symbol) so repeated 3-second
    auto-refresh cycles reuse the same analysis until the market actually
    produces a new bar.
    """
    try:
        df = _fetch_live_ohlcv(broker, config, symbol, timeframe="H1", bars=200)
        if df is None or df.empty:
            log.warning("No data received for %s", symbol)
            return {}

        health.record_market_data_update()

        # BUG FIX (audit): df.index[-1] is a plain positional RangeIndex
        # (0..bars-1) for BOTH paper data (generate_paper_ohlcv) and MT5
        # data (MT5Broker._rates_to_dataframe) — neither sets a DatetimeIndex.
        # Since `bars` is always the same fixed count, df.index[-1] was
        # ALWAYS the same integer (e.g. 199) on every loop iteration,
        # regardless of real elapsed time. That made the cache_key below
        # constant forever, so the signal cache never invalidated on a new
        # candle: df_ind/regime_signal/smc_result/current_price were computed
        # once on the very first cycle and then reused unchanged for the
        # entire session — the bot would look "stuck"/frozen after its first
        # analysis. Use the actual bar timestamp from the "Date" column
        # instead, which genuinely changes as new candles form.
        candle_time = df["Date"].iloc[-1] if "Date" in df.columns else df.index[-1]
        cache_key = (candle_time, live_strategy_name)
        cached = signal_cache.get(symbol)

        if cached is not None and cached.get("cache_key") == cache_key:
            df_ind = cached["df_ind"]
            regime_signal = cached["regime_signal"]
            smc_result = cached["smc_result"]
            current_price = cached["current_price"]
        else:
            df_ind = TechnicalIndicators.add_all(df)
            current_price = float(df_ind.iloc[-1]["Close"])
            smc_result = smc.analyze(df_ind)

            if live_strategy_name and live_strategy_name != "Regime (default)":
                strategy_signal = get_strategy(live_strategy_name).generate(df_ind)
                regime_signal = strategy_signal_to_signal_output(strategy_signal, live_strategy_name)
            else:
                regime_signal = regime.generate_signal(df_ind)

            signal_cache[symbol] = {
                "cache_key": cache_key, "df_ind": df_ind, "regime_signal": regime_signal,
                "smc_result": smc_result, "current_price": current_price,
            }

        strategy_label = live_strategy_name if live_strategy_name and live_strategy_name != "Regime (default)" else "Regime+SMC"

        profile = symbol_manager.profile(symbol)
        try:
            price_info = broker.get_price(symbol)
            spread = (price_info.get("ask", 0) - price_info.get("bid", 0)) / profile.pip_size
            if spread > profile.max_spread_pips:
                log.info("%s spread too wide (%.1f pips), skipping this cycle", symbol, spread)
                return {"action": "NO_TRADE", "entry": current_price, "sl": 0, "tp": 0,
                        "ai_score": 0, "regime": regime_signal.regime, "smc_bias": smc_result.bias,
                        "explanation": f"Spread too wide ({spread:.1f} pips)"}
        except Exception:
            pass  # spread check is best-effort; don't block trading on it

        news_sentiment = None
        if news_service is not None:
            news_sentiment = news_service.fetch_news_sentiment(symbol)

        ml_confidence = None
        # BUG FIX/FEATURE COMPLETION (audit): this was previously hardcoded
        # to None unconditionally, so the "ML filter" risk setting did
        # nothing at all — MLSignalModel was fully implemented but never
        # wired into the live decision path (and had no save/load, so a
        # trained model couldn't even survive between app restarts).
        # ml_confidence must be 0-100 and DIRECTIONAL (does the model agree
        # with the specific action regime_signal is proposing?), not a
        # raw magnitude — otherwise a model confidently predicting the
        # OPPOSITE direction would still get averaged in as if it agreed.
        ml_model = st.session_state.get("ml_model")
        if ml_model is not None and config.risk.enable_ml_filter and regime_signal.action in ("BUY", "SELL"):
            try:
                ml_result = ml_model.predict_latest(df_ind)
                if ml_result is not None:
                    directional_prob = ml_result.prob_up if regime_signal.action == "BUY" else (1.0 - ml_result.prob_up)
                    ml_confidence = directional_prob * 100.0
            except Exception as exc:
                log.warning("ML model prediction failed for %s: %s", symbol, exc)

        ai_reviewer_confidence = None
        ai_result = None
        if ai_service is not None and ai_service.is_available:
            ai_result = ai_service.analyze(regime_signal, smc_result, news_sentiment, current_price)
            if ai_result.available:
                ai_reviewer_confidence = ai_result.confidence

        final_decision = decision.decide(
            regime_signal, smc_result, ml_confidence=ml_confidence,
            ai_reviewer_confidence=ai_reviewer_confidence, current_price=current_price)

        signal_dict = {
            "symbol": symbol, "action": final_decision.action, "entry": final_decision.entry,
            "sl": final_decision.sl, "tp": final_decision.tp,
            "ai_score": final_decision.ai_score, "regime": regime_signal.regime,
            "smc_bias": smc_result.bias, "explanation": final_decision.explanation,
        }

        if final_decision.action in ("BUY", "SELL"):
            positions = broker.get_positions()
            if len(positions) >= config.risk.max_open_trades:
                log.info("Max open trades reached (%d)", len(positions))
                return signal_dict

            open_exposure = [OpenExposure(symbol=p.symbol, direction=p.position_type) for p in positions]
            corr_result = correlation_guard.check(open_exposure, symbol, final_decision.action)
            if not corr_result.allowed:
                log.info("Correlation guard blocked %s %s: %s", final_decision.action, symbol, corr_result.reason)
                signal_dict["explanation"] = corr_result.reason
                signal_dict["action"] = "NO_TRADE"
                return signal_dict

            risk_pct = config.risk.base_risk_pct
            if config.risk.use_auto_drawdown_risk:
                risk_pct = risk.compute_drawdown_adjusted_risk(
                    account.balance, st.session_state.peak_balance, risk_pct)
            if config.risk.use_kelly_sizing:
                kelly_pct, _ = risk.kelly_position_size(
                    account.balance, final_decision.entry, final_decision.sl, risk_pct)
                risk_pct = kelly_pct

            lot_size = risk.calculate_lot_size(
                account.balance, final_decision.entry, final_decision.sl,
                risk_pct, leverage=account.leverage,
                contract_size=profile.contract_size, quote_currency=profile.quote_currency)

            is_valid, reason = risk.validate_signal(
                final_decision.entry, final_decision.sl, final_decision.tp,
                lot_size, account.balance, final_decision.ai_score)
            if not is_valid:
                log.warning("Signal rejected for %s: %s", symbol, reason)
                return signal_dict

            signal = TradeSignal(
                symbol=symbol, direction=final_decision.action,
                entry_price=final_decision.entry, sl=final_decision.sl,
                tp=final_decision.tp, lot_size=lot_size,
                strategy=strategy_label, ai_score=final_decision.ai_score,
                regime=regime_signal.regime)
            result = broker.send_order(signal)
            if result["success"]:
                journal.record_order_open(
                    ticket=result["ticket"], entry_time=datetime.now(),
                    symbol=symbol, direction=final_decision.action,
                    entry_price=final_decision.entry, stop_loss=final_decision.sl,
                    take_profit=final_decision.tp, lot_size=lot_size,
                    risk_pct=risk_pct, ai_score=final_decision.ai_score,
                    regime=regime_signal.regime, smc_bias=smc_result.bias,
                    confluence_notes=" | ".join(final_decision.confluence_notes),
                    strategy=strategy_label)
                journal.record_position_risk(
                    ticket=result["ticket"], symbol=symbol, direction=final_decision.action,
                    entry_price=final_decision.entry, initial_sl=final_decision.sl)
                health.record_order(action=final_decision.action, symbol=symbol,
                                    success=True, ticket=result["ticket"])
                if notifier:
                    notifier.notify_trade(final_decision.action, final_decision.entry,
                                          final_decision.sl, final_decision.tp, lot_size)
            else:
                health.record_order(action=final_decision.action, symbol=symbol,
                                    success=False, reason=result.get("error"))

        return signal_dict
    except Exception as exc:
        log.error("Cycle error for %s: %s", symbol, exc, exc_info=True)
        return {}


def _reconcile_closed_trades(broker: Any, journal: Any, notifier: Any, health: Any,
                              risk: Any = None, account_balance: Optional[float] = None) -> None:
    """Detect positions that have closed (SL/TP hit, manual close, daily
    guard force-close, or broker-side liquidation) and finalize their
    journal entry.

    BUG FIX (audit): JournalService.record_trade() — the method that
    writes a completed trade into the `trades` table — was never called
    anywhere in the app. Positions opened via record_order_open() were
    stored in `pending_orders` but never finalized on close, so the
    Journal tab, CSV/JSON export, and any win-rate/history calculation
    that reads journal.trades were always empty regardless of how many
    trades actually closed. This reconciles pending orders against the
    broker's trade history (works for both PaperBroker and MT5Broker,
    which expose the same get_trade_history() dict shape) once per loop
    iteration and finalizes anything that has closed.
    """
    try:
        pending = journal.open_orders
        if not pending:
            return
        open_tickets = {p.ticket for p in broker.get_positions()}
        history = broker.get_trade_history() if hasattr(broker, "get_trade_history") else []
        history_by_ticket = {h["ticket"]: h for h in history if h.get("ticket") is not None}

        for order in pending:
            ticket = order["ticket"]
            if ticket in open_tickets:
                continue  # still open
            closed = history_by_ticket.get(ticket)
            if closed is None:
                continue  # closed but not yet reflected in trade history; retry next loop
            exit_price = closed.get("close_price") or 0.0
            profit_loss = closed.get("pl", 0.0) or 0.0
            exit_time = closed.get("time") or datetime.now()
            symbol = order.get("symbol", "")
            # BUG FIX (audit): r_multiple was silently wrong by the
            # contract-size factor (see journal_service.py note) — resolve
            # the instrument's real contract size (broker-reported for MT5,
            # the same lookup PaperBroker's own P&L math uses otherwise)
            # so the R-multiple stored here is meaningful.
            contract_size = 1.0
            try:
                if hasattr(broker, "get_symbol_specs"):
                    specs = broker.get_symbol_specs(symbol) or {}
                    contract_size = float(specs.get("contract_size") or 0.0) or _contract_size(symbol)
                else:
                    contract_size = _contract_size(symbol)
            except Exception:
                contract_size = _contract_size(symbol)
            journal.record_trade(
                entry_time=order.get("entry_time"), exit_time=exit_time,
                symbol=symbol, direction=order.get("direction", ""),
                entry_price=order.get("entry_price") or 0.0, exit_price=exit_price,
                stop_loss=order.get("stop_loss") or 0.0, take_profit=order.get("take_profit") or 0.0,
                lot_size=order.get("lot_size") or 0.0, profit_loss=profit_loss,
                risk_pct=order.get("risk_pct") or 0.0, ai_score=order.get("ai_score"),
                regime=order.get("regime", ""), smc_bias=order.get("smc_bias", ""),
                confluence_notes=order.get("confluence_notes"), strategy=order.get("strategy", ""),
                contract_size=contract_size,
            )
            journal.pop_pending_order(ticket)
            journal.clear_position_risk(ticket)
            if health is not None:
                health.record_trade(profit_loss=profit_loss)
            # BUG FIX (audit): RiskManager.record_trade_result() — the
            # method that updates the consecutive-loss counter the
            # circuit-breaker / cooldown guard in daily_guard() depends on
            # — was defined but never called anywhere in the live loop.
            # That made the entire consecutive-loss guard (and the bounded
            # cooldown, consecutive_loss_cooldown_hours) permanently dead
            # in live trading: the counter simply never moved, so it could
            # never reach max_consecutive_losses no matter how many trades
            # actually lost in a row. This is the exact same "recorder
            # method exists but is never invoked" pattern as the journal
            # and health bugs fixed earlier in this audit.
            if risk is not None:
                risk.record_trade_result(
                    profit_loss=profit_loss, current_time=exit_time,
                    balance_before_trade=account_balance,
                )
            if notifier is not None:
                emoji = "✅" if profit_loss > 0 else ("❌" if profit_loss < 0 else "➖")
                notifier.send(f"{emoji} *Trade Closed* #{ticket}\n"
                              f"{order.get('direction')} {order.get('symbol')} | P/L: ${profit_loss:.2f}")
            log.info("Journal finalized: #%s %s %s P/L=$%.2f", ticket,
                      order.get("direction"), order.get("symbol"), profit_loss)
    except Exception as exc:
        log.error("Trade reconciliation failed: %s", exc, exc_info=True)


def _apply_trailing_stops(config: BotConfig, broker: Any, journal: Any, trailing_mgr: Any) -> None:
    """Check every open position for a trailing-stop/break-even update.

    Runs once per trading-loop iteration (not per symbol) since it needs to
    see all open positions together. Never raises — a failure here should
    not stop the trading loop.
    """
    if not config.risk.use_trailing_stop:
        return
    try:
        positions = broker.get_positions()
        open_tickets = {p.ticket for p in positions}

        # Garbage-collect risk-reference rows for positions that are no
        # longer open (closed by SL/TP or manually) so the table doesn't
        # grow unbounded.
        for ticket in journal.all_position_risk_tickets():
            if ticket not in open_tickets:
                journal.clear_position_risk(ticket)

        for pos in positions:
            risk_ref = journal.get_position_risk(pos.ticket)
            if not risk_ref:
                continue  # opened before this feature existed, or by another tool — skip safely
            try:
                price_info = broker.get_price(pos.symbol)
                current_price = price_info["bid"] if pos.position_type == "SELL" else price_info["ask"]
            except Exception:
                continue

            result = trailing_mgr.compute_new_sl(
                direction=pos.position_type, entry_price=risk_ref["entry_price"],
                initial_sl=risk_ref["initial_sl"], current_price=current_price, current_sl=pos.sl,
            )
            if result.new_sl is not None:
                ok = broker.modify_position_sl_tp(pos.ticket, sl=result.new_sl)
                if ok:
                    log.info("Trailing stop updated #%d %s: SL -> %.5f (%s)",
                             pos.ticket, pos.symbol, result.new_sl, result.reason)
    except Exception as exc:
        log.error("Trailing stop check failed: %s", exc, exc_info=True)


def trading_loop(config: BotConfig) -> None:
    broker = st.session_state.broker
    if broker is None:
        log.error("Broker not initialized")
        return
    journal = st.session_state.journal
    notifier = st.session_state.notifier
    health = st.session_state.health
    regime = st.session_state.regime_detector
    smc = st.session_state.smc
    decision = st.session_state.decision_engine
    risk = st.session_state.risk_manager
    news_service = st.session_state.news_service
    ai_service = st.session_state.ai_service
    symbol_manager = st.session_state.symbol_manager
    correlation_guard = CorrelationGuard(symbol_manager, max_net_usd_exposure=config.risk.max_net_usd_exposure)
    trailing_mgr = TrailingStopManager(config.risk)
    _last_notified_guard_reason = None

    while st.session_state.bot_running:
        try:
            account = broker.get_account_info()
            if account.balance > st.session_state.peak_balance:
                st.session_state.peak_balance = account.balance

            # BUG FIX (audit): daily_guard() must receive current_time
            # explicitly — the bounded consecutive-loss cooldown path
            # inside it only activates when current_time is not None (see
            # risk_manager.py). Omitting it silently disabled
            # consecutive_loss_cooldown_hours even when configured.
            guard = risk.daily_guard(account.balance, st.session_state.peak_balance,
                                      current_time=datetime.now())
            if guard.should_block_new_trades:
                if guard.should_close_all:
                    broker.close_all_positions()
                    health.record_error(guard.reason)
                # BUG FIX (audit): notify_drawdown() existed on
                # NotificationService but was never called — a daily
                # loss-limit / consecutive-loss / profit-lock guard could
                # trip (and even force-close all positions) without ever
                # alerting a configured Telegram/Discord channel. Notify
                # once per NEW trigger (not every 10s loop) using the
                # guard's own reason as the de-dupe key.
                if notifier is not None and guard.reason != _last_notified_guard_reason:
                    notifier.notify_drawdown(guard.daily_pnl_pct, config.risk.daily_loss_limit_pct)
                    notifier.send(f"🛑 *Risk Guard Triggered*\n{guard.reason}"
                                  + ("\nAll open positions were closed." if guard.should_close_all else
                                     "\nNew trades are blocked; existing positions continue."))
                    _last_notified_guard_reason = guard.reason
                time.sleep(10)
                continue
            _last_notified_guard_reason = None

            session_guard = st.session_state.session_guard
            calendar_guard = st.session_state.calendar_guard
            protection_blocked = False
            if session_guard is not None:
                sc = session_guard.check()
                if not sc.allowed:
                    log.info("Session guard: %s", sc.reason)
                    protection_blocked = True
            if not protection_blocked and calendar_guard is not None:
                cc = calendar_guard.check()
                if not cc.allowed:
                    log.info("Calendar guard: %s", cc.reason)
                    protection_blocked = True
            if protection_blocked:
                # Session/news protection blocks NEW trades only — existing
                # positions keep running under their own SL/TP/trailing
                # stop, same as the daily profit-lock behavior. It does not
                # force-close anything (unlike the daily LOSS limit above,
                # which does), since a scheduled news window isn't evidence
                # something has already gone wrong with an open position.
                _apply_trailing_stops(config, broker, journal, trailing_mgr)
                time.sleep(10)
                continue

            _apply_trailing_stops(config, broker, journal, trailing_mgr)

            signals_this_cycle = {}
            for symbol in symbol_manager.active_symbols:
                signal_dict = _run_symbol_cycle(
                    symbol, config, broker, journal, notifier, health, regime, smc,
                    decision, risk, news_service, ai_service, symbol_manager,
                    correlation_guard, account, st.session_state.live_strategy_name,
                    st.session_state.symbol_signal_cache,
                )
                if signal_dict:
                    signals_this_cycle[symbol] = signal_dict

                # BUG FIX (audit): PaperBroker.check_sl_tp() existed but was
                # never called anywhere in the live trading loop, so a paper
                # position would NEVER automatically close when price hit
                # its SL or TP — it would just sit open indefinitely with
                # floating P&L updating forever. This makes paper mode an
                # inaccurate simulation of the bot's actual risk discipline.
                # MT5Broker has no check_sl_tp method (the real broker
                # server enforces SL/TP natively), hence the hasattr guard —
                # same pattern already used for update_price above.
                if hasattr(broker, "check_sl_tp"):
                    try:
                        price = broker.get_price(symbol)
                        broker.check_sl_tp(symbol, price["bid"], price["ask"])
                    except Exception as exc:
                        log.warning("check_sl_tp failed for %s: %s", symbol, exc)

            _reconcile_closed_trades(broker, journal, notifier, health, risk=risk, account_balance=account.balance)

            if signals_this_cycle:
                st.session_state.current_signals = signals_this_cycle
                primary = symbol_manager.active_symbols[0]
                if primary in signals_this_cycle:
                    st.session_state.current_signal = signals_this_cycle[primary]

            st.session_state.last_update = datetime.now()
            st.session_state.error_count = 0
            time.sleep(3)
        except Exception as exc:
            log.error("Trading loop error: %s", exc, exc_info=True)
            st.session_state.error_count += 1
            st.session_state.last_error = str(exc)
            health.record_error(str(exc))
            # BUG FIX (audit): notify_error() existed on NotificationService
            # but was never actually called anywhere in the codebase, so a
            # configured Telegram/Discord alert would never fire for a real
            # trading-loop exception — only for trade execution, broker
            # disconnects, and daily risk-guard blocks.
            if notifier is not None:
                notifier.notify_error(str(exc))
            if st.session_state.error_count > 5:
                log.critical("Too many errors, stopping bot")
                if notifier is not None:
                    notifier.send(
                        "🛑 *Bot Stopped Automatically*\n"
                        f"Too many consecutive errors ({st.session_state.error_count}). "
                        f"Last error: {str(exc)[:300]}\n"
                        "The bot has stopped itself. Open the dashboard to investigate "
                        "before restarting."
                    )
                st.session_state.bot_running = False
            time.sleep(5)




def preflight_check(config: BotConfig, broker: Any) -> List[Dict[str, Any]]:
    """Run once before allowing 'Start Bot'. Returns a list of check results.

    Each item: {"label": str, "ok": bool, "critical": bool, "detail": str}
    Critical checks must all pass before the bot is allowed to start.
    Non-critical checks (AI/news) only show a warning — the bot can still
    run without them, just with reduced analysis.
    """
    checks = []

    connected = False
    try:
        connected = broker.is_connected()
    except Exception:
        pass
    checks.append({"label": "Broker connection", "ok": connected, "critical": True,
                    "detail": f"{config.broker.upper()} " + ("connected" if connected else "not connected")})

    valid_risk = 0 < config.risk.base_risk_pct <= config.risk.max_risk_pct
    checks.append({"label": "Risk settings", "ok": valid_risk, "critical": True,
                    "detail": "base risk % must be > 0 and ≤ max risk %"})

    data_ok = False
    # BUG FIX (audit): this hardcoded "XAUUSD" regardless of which symbol(s)
    # the user actually configured in "Symbols to trade". Combined with the
    # get_ohlcv suffix-resolution bug (now fixed), this made "Cannot start"
    # fire even when the broker connection and the user's real symbols were
    # completely fine. Check against the first active symbol if one is
    # configured, falling back to "XAUUSD" only if none is set yet.
    symbol_manager = st.session_state.get("symbol_manager")
    preflight_symbol = "XAUUSD"
    if symbol_manager is not None and getattr(symbol_manager, "active_symbols", None):
        preflight_symbol = symbol_manager.active_symbols[0]
    try:
        df = _fetch_live_ohlcv(broker, config, preflight_symbol, timeframe="M5",
                                bars=50, yf_period="1d", yf_interval="5m")
        data_ok = df is not None and not df.empty
    except Exception:
        data_ok = False
    checks.append({"label": "Market data", "ok": data_ok, "critical": True,
                    "detail": (f"price data reachable ({preflight_symbol})" if data_ok
                               else f"could not fetch price data for {preflight_symbol}")})

    ai_ok = st.session_state.ai_service is not None and st.session_state.ai_service.is_available
    checks.append({"label": "AI reviewer", "ok": ai_ok, "critical": False,
                    "detail": "active" if ai_ok else "not configured — bot will trade on rules only"})

    news_ok = config.news.enabled and bool(config.news.alpha_vantage_api_key)
    checks.append({"label": "News sentiment", "ok": news_ok, "critical": False,
                    "detail": "active" if news_ok else "not configured"})

    return checks


def main() -> None:
    render_header()
    init_session_state()
    ui_config = render_sidebar_config()

    try:
        config = load_config(overrides={
            "broker": ui_config["broker"],
            "mt5": {"login": ui_config["mt5_login"], "password": ui_config["mt5_password"],
                    "server": ui_config["mt5_server"], "leverage": ui_config["mt5_leverage"]},
            "mt5_bridge": {"base_url": ui_config["bridge_url"], "token": ui_config["bridge_token"]},
            "risk": {"base_risk_pct": ui_config["base_risk"], "max_risk_pct": ui_config["max_risk"],
                     "daily_loss_limit_pct": ui_config["daily_loss"], "max_open_trades": ui_config["max_trades"],
                     "use_kelly_sizing": ui_config["use_kelly"], "use_auto_drawdown_risk": ui_config["use_drawdown"]},
            "notifications": {"telegram_bot_token": ui_config["telegram_token"],
                              "telegram_chat_id": ui_config["telegram_chat"],
                              "discord_webhook_url": ui_config["discord_webhook"],
                              "enabled": bool(ui_config["telegram_token"] or ui_config["discord_webhook"])},
            **({"ai": {"provider": ui_config["ai_provider"],
                       "anthropic_api_key": ui_config["anthropic_key"],
                       "google_api_key": ui_config["google_key"],
                       "openai_api_key": ui_config["openai_key"],
                       "enabled": True}}
               if (ui_config["anthropic_key"] or ui_config["google_key"] or ui_config["openai_key"])
               else {}),
            **({"news": {"alpha_vantage_api_key": ui_config["alpha_vantage_key"], "enabled": True}}
               if ui_config["alpha_vantage_key"] else {}),
        }, validate=False)
    except Exception as exc:
        st.error(f"Config error: {exc}")
        return

    broker_signature = (config.broker, config.mt5.login, config.mt5.password,
                        config.mt5.server, config.mt5.leverage,
                        config.mt5_bridge.base_url, config.mt5_bridge.token,
                        config.ai.provider, config.ai.anthropic_api_key,
                        config.ai.google_api_key, config.ai.openai_api_key, config.ai.enabled,
                        config.news.alpha_vantage_api_key, config.news.enabled,
                        tuple(ui_config.get("active_symbols") or ["XAUUSD"]))

    if st.session_state.broker is None or st.session_state.broker_signature != broker_signature:
        if st.session_state.broker is not None:
            try:
                st.session_state.broker.disconnect()
            except Exception:
                pass
            if st.session_state.health is not None:
                st.session_state.health.stop()

        try:
            broker = create_broker(config)
            connected = broker.connect()
            if not connected:
                st.error(f"❌ {config.broker.upper()} connection FAILED — check login/password/server and try again.")
                st.session_state.broker = None
                st.session_state.broker_signature = None
                return
            st.session_state.broker = broker
            st.session_state.broker_signature = broker_signature
            st.session_state.journal = JournalService()
            st.session_state.notifier = NotificationService(config.notifications)
            st.session_state.health = HealthService(broker, interval_seconds=30, notifier=st.session_state.notifier)
            st.session_state.regime_detector = RegimeDetector()
            st.session_state.smc = SMCAnalyzer()
            st.session_state.decision_engine = DecisionEngine(config.risk)
            st.session_state.risk_manager = RiskManager(config.risk)
            st.session_state.news_service = NewsService(config.news)
            st.session_state.ai_service = AIService(config.ai)
            st.session_state.symbol_manager = SymbolManager(ui_config.get("active_symbols") or ["XAUUSD"])
            st.session_state.session_guard = SessionGuard(block_between_sessions=True)
            st.session_state.calendar_guard = EconomicCalendarGuard(
                blackout_minutes_before=30, blackout_minutes_after=30, min_impact="high")
            st.success(f"✅ {config.broker.upper()} connected successfully")
            if config.broker == "mt5":
                if ui_config.get("remember_login"):
                    credential_store.save_credentials(
                        "mt5", str(config.mt5.login), config.mt5.password, config.mt5.server)
                else:
                    credential_store.clear_credentials("mt5")
            if config.broker == "mt5_bridge":
                if ui_config.get("remember_bridge"):
                    credential_store.save_credentials(
                        "mt5_bridge", "", config.mt5_bridge.token, config.mt5_bridge.base_url)
                else:
                    credential_store.clear_credentials("mt5_bridge")
        except Exception as exc:
            st.error(f"Initialization failed: {exc}")
            log.error("Init error: %s", exc, exc_info=True)
            st.session_state.broker = None
            st.session_state.broker_signature = None
            return

    broker = st.session_state.broker
    journal = st.session_state.journal
    health = st.session_state.health

    st.caption(f"🔌 Broker: **{config.broker.upper()}** | Connection status: "
               f"{'🟢 Connected' if broker.is_connected() else '🔴 Disconnected'}")

    ai_ready = st.session_state.ai_service is not None and st.session_state.ai_service.is_available
    news_ready = config.news.enabled and bool(config.news.alpha_vantage_api_key)
    risk_ready = 0 < config.risk.base_risk_pct <= config.risk.max_risk_pct
    st.caption(
        f"{'🟢' if broker.is_connected() else '🔴'} Broker &nbsp; "
        f"{'🟢' if risk_ready else '🔴'} Risk config &nbsp; "
        f"{'🟢' if ai_ready else '⚪'} AI reviewer &nbsp; "
        f"{'🟢' if news_ready else '⚪'} News sentiment &nbsp; "
        f"{'🟢' if not st.session_state.bot_running or st.session_state.error_count == 0 else '🟡'} No recent errors",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["📊 Dashboard", "📈 Backtest", "🎲 Monte Carlo", "📋 Journal", "⚙️ Settings"])

    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Account Overview")
            try:
                account = broker.get_account_info()
                render_account_card({"balance": account.balance, "equity": account.equity,
                                     "margin": account.margin, "free_margin": account.free_margin})
            except Exception as exc:
                st.error(f"Account info error: {exc}")
            st.subheader("Open Positions")
            try:
                positions = broker.get_positions()
                render_positions_table([
                    {"ticket": p.ticket, "symbol": p.symbol, "type": p.position_type,
                     "volume": p.volume, "open_price": p.open_price, "sl": p.sl, "tp": p.tp, "profit": p.profit}
                    for p in positions])
            except Exception as exc:
                st.error(f"Positions error: {exc}")
            st.subheader("Current Signal")
            render_signal_card(st.session_state.current_signal)

            st.subheader("📰 News Sentiment & 🤖 AI Reviewer")
            ai_col1, ai_col2 = st.columns(2)
            with ai_col1:
                news = st.session_state.current_news
                if news:
                    label_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(news["label"], "⚪")
                    st.markdown(f"**News:** {label_emoji} {news['label'].title()} "
                                f"(score: {news['score']:.2f}, {news['article_count']} articles)")
                    if news["source"] == "none":
                        st.caption("Alpha Vantage key not set — news sentiment disabled")
                    for h in news["headlines"][:3]:
                        st.caption(f"• {h}")
                else:
                    st.caption("No news data yet — starts once the bot runs a cycle")
            with ai_col2:
                ai = st.session_state.current_ai
                if ai:
                    if ai["available"]:
                        st.markdown(f"**AI Reviewer confidence** ({st.session_state.ai_service.active_provider_name if st.session_state.ai_service else 'n/a'}): {ai['confidence']:.0f}/100")
                        st.progress(min(max(ai["confidence"] / 100, 0.0), 1.0))
                        st.caption(ai["reasoning"])
                    else:
                        st.caption(f"AI reviewer unavailable: {ai.get('error', 'not configured')}")
                else:
                    st.caption("Add an Anthropic API key in the sidebar to enable the AI reviewer")

            if st.button("🔍 Run Analysis Now", help="Fetch data and run one news + AI review cycle without starting the bot"):
                with st.spinner("Analyzing..."):
                    try:
                        df_now = _fetch_live_ohlcv(broker, config, "XAUUSD", timeframe="H1",
                                                    bars=200, yf_period="5d", yf_interval="15m")
                        if df_now is None or df_now.empty:
                            st.warning("No market data available")
                        else:
                            df_ind_now = TechnicalIndicators.add_all(df_now)
                            regime_now = st.session_state.regime_detector.generate_signal(df_ind_now)
                            smc_now = st.session_state.smc.analyze(df_ind_now)
                            price_now = float(df_ind_now.iloc[-1]["Close"])

                            news_now = None
                            if st.session_state.news_service is not None:
                                news_now = st.session_state.news_service.fetch_news_sentiment("XAUUSD")
                                st.session_state.current_news = {
                                    "score": news_now.score, "label": news_now.label,
                                    "article_count": news_now.article_count,
                                    "headlines": news_now.headlines, "source": news_now.source,
                                }
                            if st.session_state.ai_service is not None and st.session_state.ai_service.is_available:
                                ai_now = st.session_state.ai_service.analyze(regime_now, smc_now, news_now, price_now)
                                st.session_state.current_ai = {
                                    "confidence": ai_now.confidence, "reasoning": ai_now.reasoning,
                                    "available": ai_now.available, "error": ai_now.error,
                                }
                            else:
                                st.session_state.current_ai = {
                                    "confidence": None, "reasoning": "", "available": False,
                                    "error": "AI reviewer not configured",
                                }
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")

            st.subheader("Price Chart")

            @st.fragment(run_every="5s")
            def _live_chart():
                try:
                    df = _fetch_live_ohlcv(broker, config, "XAUUSD", timeframe="M5",
                                            bars=288, yf_period="1d", yf_interval="5m")
                    if df.empty:
                        st.warning("No chart data available")
                        return
                    fig = go.Figure(data=[go.Candlestick(
                        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                        name="XAU/USD")])

                    try:
                        chart_start = df.index.min()
                        closed_trades = [t for t in journal.trades
                                          if t.get("entry_time") and pd.Timestamp(t["entry_time"]) >= chart_start]
                        open_trades = [t for t in journal.open_orders
                                       if t.get("entry_time") and pd.Timestamp(t["entry_time"]) >= chart_start]
                        for label, trades, symbol_shape in (
                            ("Closed entry", closed_trades, None), ("Open entry", open_trades, None)
                        ):
                            buys = [t for t in trades if t.get("direction") == "BUY"]
                            sells = [t for t in trades if t.get("direction") == "SELL"]
                            if buys:
                                fig.add_trace(go.Scatter(
                                    x=[pd.Timestamp(t["entry_time"]) for t in buys],
                                    y=[t["entry_price"] for t in buys],
                                    mode="markers", name=f"{label} BUY",
                                    marker=dict(symbol="triangle-up", size=12, color="lime")))
                            if sells:
                                fig.add_trace(go.Scatter(
                                    x=[pd.Timestamp(t["entry_time"]) for t in sells],
                                    y=[t["entry_price"] for t in sells],
                                    mode="markers", name=f"{label} SELL",
                                    marker=dict(symbol="triangle-down", size=12, color="red")))
                    except Exception as marker_exc:
                        log.debug("Trade marker overlay skipped: %s", marker_exc)

                    fig.update_layout(template="plotly_dark", height=500,
                                       margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, width="stretch")
                    st.caption(f"🔄 Auto-refreshes every 5s | Last update: {datetime.now().strftime('%H:%M:%S')}")
                except Exception as exc:
                    st.warning(f"Chart unavailable: {exc}")

            _live_chart()
        with col2:
            st.subheader("Bot Control")
            live_strategy_options = ["Regime (default)"] + STRATEGIES
            selected_strategy = st.selectbox(
                "Live Strategy", live_strategy_options,
                index=live_strategy_options.index(st.session_state.live_strategy_name)
                if st.session_state.live_strategy_name in live_strategy_options else 0,
                disabled=st.session_state.bot_running,
                help="'Regime (default)' uses the built-in RegimeDetector (recommended). "
                     "Choosing a specific strategy replaces it for live trading — SMC, risk, "
                     "AI review, and correlation checks still run the same either way.")
            if selected_strategy != st.session_state.live_strategy_name:
                st.session_state.live_strategy_name = selected_strategy
                st.session_state.symbol_signal_cache = {}  # invalidate cache on strategy switch
            if st.button("▶️ Start Bot", disabled=st.session_state.bot_running):
                with st.spinner("Running pre-flight checks..."):
                    checks = preflight_check(config, broker)
                critical_failed = [c for c in checks if c["critical"] and not c["ok"]]
                for c in checks:
                    icon = "🟢" if c["ok"] else ("🔴" if c["critical"] else "🟡")
                    st.caption(f"{icon} {c['label']}: {c['detail']}")
                if critical_failed:
                    st.error("❌ Cannot start — fix the issue(s) marked 🔴 above first.")
                else:
                    st.session_state.bot_running = True
                    health.start()
                    _start_background_thread(trading_loop, args=(config,))
                    st.success("Bot started")
                    st.rerun()
            if st.button("⏹️ Stop Bot", disabled=not st.session_state.bot_running):
                st.session_state.bot_running = False
                health.stop()
                st.warning("Bot stopped")
                st.rerun()

            st.markdown("---")
            st.subheader("Market Context")
            try:
                price = broker.get_price("XAUUSD")
                render_spread_widget(price.get("bid", 0.0), price.get("ask", 0.0))
            except Exception as exc:
                st.caption(f"Spread unavailable: {exc}")
            render_session_widget()

            st.markdown("---")
            st.subheader("Today's Summary")
            try:
                today_str = datetime.now().strftime("%Y-%m-%d")
                trades_today = [t for t in journal.trades
                                 if t.get("entry_time") and str(t["entry_time"]).startswith(today_str)]
                render_daily_summary_widget(trades_today)
            except Exception as exc:
                st.caption(f"Summary unavailable: {exc}")

            st.markdown("---")
            st.subheader("Health Status")
            try:
                status = health.latest_status()
                render_health_status({"connected": status.connected, "terminal_ok": status.terminal_ok,
                                      "account_ok": status.account_ok, "uptime_seconds": status.uptime_seconds})
            except Exception as exc:
                st.error(f"Health error: {exc}")
            st.markdown("---")
            st.subheader("Risk Status")
            risk = st.session_state.risk_manager
            if risk:
                summary = risk.get_risk_summary()
                st.write(f"Daily P&L: ${summary['daily_pnl']:.2f}")
                st.write(f"Consecutive Losses: {summary['consecutive_losses']}")
                st.write(f"Total Trades: {summary['total_trades']}")
                st.write(f"Win Rate: {summary['win_rate']*100:.1f}%")
            if st.session_state.last_error:
                st.error(f"Last Error: {st.session_state.last_error[:100]}")

    with tabs[1]:
        st.subheader("Strategy Backtest")
        strategy_name = st.selectbox("Strategy", ["Regime (default)"] + STRATEGIES, index=0)
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
        timeframe = st.selectbox("MT5 Timeframe", ["M5", "M15", "M30", "H1"], index=1)
        symbol = st.selectbox("MT5 Symbol", ui_config.get("active_symbols") or ["XAUUSD"])
        initial_balance = st.number_input("Initial Balance", value=10000.0, step=1000.0)
        force_refresh = st.checkbox("Force refresh selected range from MT5", value=False, help="Normally the database is reused and only missing/new candles are downloaded. Enable this only when the broker history has been corrected.")
        st.caption("MT5 historical candles are stored in a local SQLite database. First run downloads the requested range; later runs reuse the database and fetch only missing/new candles.")
        cache = MT5DataCache()
        db_status = cache.status(symbol, timeframe)
        if db_status["rows"]:
            st.info(f"Database: {db_status['database']} | {symbol} {timeframe} | {db_status['rows']:,} candles | {db_status['first']} → {db_status['last']}")
        else:
            st.warning("No cached MT5 data for this symbol/timeframe yet. The first sync will download it from MT5.")

        col_sync, col_run = st.columns(2)
        sync_clicked = col_sync.button("⬇️ Sync / Update MT5 Data")
        run_clicked = col_run.button("▶ Run Backtest")

        if sync_clicked or run_clicked:
            with st.spinner("Synchronizing MT5 historical data..." if sync_clicked and not run_clicked else "Synchronizing MT5 data and running backtest..."):
                try:
                    months = {"1mo": 1, "3mo": 3, "6mo": 6, "1y": 12}[period]
                    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                    # Exclude the currently forming candle. All historical range
                    # calculations are UTC; local timezone is display-only.
                    tf_minutes = {"M5": 5, "M15": 15, "M30": 30, "H1": 60}[timeframe]
                    end = now - timedelta(minutes=(now.minute % tf_minutes), seconds=1)
                    start = end - timedelta(days=30 * months)
                    if config.broker != "mt5":
                        raise RuntimeError("Broker must be set to MT5 for broker-native backtesting. Do not silently use Yahoo GC=F.")
                    first_download = False
                    df = pd.DataFrame()
                    cache_action = "database"
                    if not force_refresh and not sync_clicked:
                        df = cache.load(symbol, timeframe, start, end)
                    if sync_clicked or force_refresh or df.empty:
                        if config.broker != "mt5":
                            raise RuntimeError("MT5 is required when the requested range is not already cached. No Yahoo/GC=F fallback is used.")
                        df, first_download = cache.update(broker, symbol, timeframe, start, end, force_refresh=force_refresh)
                        cache_action = "initial download" if first_download else ("forced refresh" if force_refresh else "incremental update")
                    if df.empty:
                        raise RuntimeError(f"No cached/MT5 historical data available for {symbol} {timeframe}. Sync from MT5 first.")
                    status = cache.status(symbol, timeframe)
                    st.success(f"MT5 database: {status['rows']:,} total candles | {status['first']} → {status['last']} | Source used: {'MT5 sync' if cache_action != 'database' else 'local database'}")
                    if sync_clicked and not run_clicked:
                        st.info("Data is saved locally. Future backtests reuse the SQLite database and only sync missing/new candles when you request an update.")
                    if not run_clicked:
                        st.stop()
                    st.info(f"Backtest data: MT5 database | Symbol: {symbol} | Timeframe: {timeframe} | Candles: {len(df)} | Cache action: {'initial download' if first_download else ('forced refresh' if force_refresh else 'incremental update')}")
                    strategy = None if strategy_name == "Regime (default)" else get_strategy(strategy_name)
                    
                    specs = broker.get_symbol_specs(symbol) if hasattr(broker, "get_symbol_specs") else {}
                    bt_cfg = config.backtest
                    bt_cfg.spread = float(st.number_input("Backtest Spread (price units)", value=float(bt_cfg.spread), min_value=0.0, step=0.01, key="bt_spread"))
                    bt_cfg.slippage = float(st.number_input("Slippage (price units)", value=float(bt_cfg.slippage), min_value=0.0, step=0.01, key="bt_slippage"))
                    bt_cfg.intrabar_policy = st.selectbox("Same-candle SL/TP policy", ["SL_FIRST", "TP_FIRST", "LOWER_TIMEFRAME"], index=["SL_FIRST", "TP_FIRST", "LOWER_TIMEFRAME"].index(bt_cfg.intrabar_policy), key="bt_intrabar")
                    engine = BacktestEngine(initial_balance=initial_balance, risk_config=config.risk, backtest_config=bt_cfg, symbol_specs=specs)
                    summary = engine.run(df, strategy_fn=(strategy.generate if strategy is not None else None))
                    st.info(
                        "Regime Detector: ACTIVE" if strategy is None else f"Strategy: {strategy_name} | Regime Detector: not primary"
                    )
                    if config.risk.enable_ai_reviewer:
                        st.warning("AI Reviewer: DISABLED FOR HISTORICAL BACKTEST to prevent non-causal/external-model leakage.")
                    # Build a complete dashboard payload from the actual simulated
                    # trades. This keeps the UI auditable instead of showing only
                    # five headline numbers.
                    trade_records = []
                    for trade in getattr(engine, "trades", []):
                        if hasattr(trade, "__dataclass_fields__"):
                            trade_records.append({name: getattr(trade, name) for name in trade.__dataclass_fields__})
                        else:
                            trade_records.append(dict(trade))
                    report = PerformanceReporter(trade_records).calculate_metrics(
                        starting_balance=initial_balance
                    )
                    metrics = {
                        "total_trades": report.total_trades,
                        "win_rate": report.win_rate,
                        "profit_factor": report.profit_factor,
                        "avg_win": report.avg_win,
                        "avg_loss": report.avg_loss,
                        "sharpe_ratio": report.sharpe_ratio,
                        "sortino_ratio": report.sortino_ratio,
                        "max_drawdown_pct": report.max_drawdown_pct,
                        "expectancy": report.expectancy,
                        "avg_trade_duration_hours": report.avg_trade_duration_hours,
                        "avg_r_multiple": report.avg_r_multiple,
                        "monthly_returns": report.monthly_returns,
                    }
                    st.session_state.backtest_summary = {
                        "initial_balance": initial_balance,
                        "final_balance": summary.final_balance, "return_pct": summary.return_pct,
                        "trades": summary.trades, "wins": summary.wins, "win_rate": summary.win_rate,
                        "profit_factor": summary.profit_factor, "max_dd": summary.max_dd,
                        "equity_curve": summary.equity_curve,
                        "regime_breakdown": summary.regime_breakdown,
                        "trade_records": trade_records,
                        "metrics": metrics,
                        "context": {
                            "source": "MT5", "symbol": symbol, "timeframe": timeframe,
                            "candles": len(df),
                            "cache": cache_action,
                            "assumptions": [
                                f"signal={engine.config.signal_on}",
                                f"entry={engine.config.entry_on}",
                                f"intrabar={engine.config.intrabar_policy}",
                                f"spread={engine.config.spread}",
                                f"slippage={engine.config.slippage}",
                                "UTC internal timestamps",
                                f"base_risk={config.risk.base_risk_pct}%",
                                f"max_risk={config.risk.max_risk_pct}%",
                                f"daily_loss_limit={config.risk.daily_loss_limit_pct}%",
                                f"strict_smc_confluence={config.risk.strict_smc_confluence}",
                            ],
                        },
                        "validation": {
                            "data_source": "MT5",
                            "rows": len(df),
                            "start": str(df["Date"].iloc[0]) if "Date" in df.columns and len(df) else None,
                            "end": str(df["Date"].iloc[-1]) if "Date" in df.columns and len(df) else None,
                            "duplicate_timestamps": int(df["Date"].duplicated().sum()) if "Date" in df.columns else None,
                            **getattr(engine, "validation", {}),
                            "symbol_specs": specs,
                        },
                    }
                    if engine.decision_log:
                        with st.expander("Decision Log — every accepted/rejected signal"):
                            st.dataframe(pd.DataFrame(engine.decision_log), width="stretch")
                    render_backtest_results(st.session_state.backtest_summary)
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
        elif st.session_state.backtest_summary:
            render_backtest_results(st.session_state.backtest_summary)

    with tabs[2]:
        st.subheader("Monte Carlo Simulation")
        n_sims = st.slider("Simulations", 100, 5000, 1000, 100)
        ruin_threshold = st.slider("Ruin Threshold %", 10, 90, 50, 5)
        if st.button("Run Monte Carlo"):
            with st.spinner("Running simulation..."):
                try:
                    trades = journal.trades
                    returns = [t.get("profit_loss", 0) for t in trades if t.get("profit_loss") is not None]
                    if len(returns) < 10:
                        st.warning("Need at least 10 trades for meaningful simulation")
                    else:
                        report = MonteCarloSimulator(n_simulations=n_sims).run(returns, ruin_threshold_pct=ruin_threshold)
                        st.session_state.monte_carlo_report = {
                            "n_simulations": report.n_simulations, "risk_of_ruin_pct": report.risk_of_ruin_pct,
                            "probability_of_loss_pct": report.probability_of_loss_pct,
                            "return_stats": report.return_stats, "worst_drawdown_pct": report.worst_drawdown_pct,
                            "average_drawdown_pct": report.average_drawdown_pct,
                            "confidence_intervals": report.confidence_intervals}
                        render_monte_carlo(st.session_state.monte_carlo_report)
                except Exception as exc:
                    st.error(f"Monte Carlo failed: {exc}")
        elif st.session_state.monte_carlo_report:
            render_monte_carlo(st.session_state.monte_carlo_report)

    with tabs[3]:
        st.subheader("Trade Journal")
        trades = journal.trades
        render_trade_history(trades, limit=100)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export CSV"):
                st.download_button("Download CSV", journal.to_csv(trades), "trades.csv", "text/csv")
        with col2:
            if st.button("Export JSON"):
                st.download_button("Download JSON", journal.to_json(trades), "trades.json", "application/json")
        st.subheader("Performance Analytics")
        if trades:
            st.json(PerformanceReporter(trades).to_dict())
        else:
            st.info("No trades recorded yet")

    with tabs[4]:
        st.subheader("Configuration")
        st.json(config.to_safe_dict())
        st.subheader("Logs")
        log_dir = config.logging.log_dir
        if os.path.exists(log_dir):
            for f in sorted(os.listdir(log_dir))[-5:]:
                st.text(f)


if __name__ == "__main__":
    configure_logging(level="INFO", console=True)
    main()
