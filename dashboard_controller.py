"""Process adapter for the React dashboard.

Keeps the existing trading engine untouched. It initializes the same runtime
objects used by app.py and calls app.trading_loop() in a separate process.
Stop is requested through a small file flag, which the controller checks.
"""
from __future__ import annotations
import json, os, threading, time
from pathlib import Path
from datetime import datetime

STATE = Path(os.getenv("QFX_DASHBOARD_STATE", ".qfx_dashboard_state.json"))
STOP = Path(os.getenv("QFX_DASHBOARD_STOP", ".qfx_dashboard_stop"))


def write_state(**values):
    STATE.write_text(json.dumps({"updated_at": datetime.now().isoformat(), **values}, default=str), encoding="utf-8")


def main():
    # The trading loop historically lives in app.py and references
    # st.session_state. We provide only that state container here; the
    # Streamlit UI itself is never imported or launched.
    import types, sys
    st = types.ModuleType("streamlit")
    class SessionState(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        def __contains__(self, key): return dict.__contains__(self, key)
    st.session_state = SessionState()
    sys.modules["streamlit"] = st

    import app
    from src.config import load_config
    from src.services.journal_service import JournalService
    from src.services.notification_service import NotificationService
    from src.services.health_service import HealthService
    from src.services.news_service import NewsService
    from src.services.ai_service import AIService
    from src.trading.regime_detector import RegimeDetector
    from src.trading.smc import SMCAnalyzer
    from src.trading.decision_engine import DecisionEngine
    from src.trading.risk_manager import RiskManager
    from src.trading.symbol_manager import SymbolManager
    from src.trading.session_guard import SessionGuard
    from src.trading.economic_calendar_guard import EconomicCalendarGuard

    config = load_config(validate=False)
    app.init_session_state()
    broker = app.create_broker(config)
    if not broker.connect():
        write_state(running=False, ok=False, error=f"{config.broker} connection failed")
        raise RuntimeError(f"{config.broker} connection failed")

    st.session_state.broker = broker
    st.session_state.journal = JournalService()
    st.session_state.notifier = NotificationService(config.notifications)
    st.session_state.health = HealthService(broker, interval_seconds=30, notifier=st.session_state.notifier)
    st.session_state.regime_detector = RegimeDetector()
    st.session_state.smc = SMCAnalyzer()
    st.session_state.decision_engine = DecisionEngine(config.risk)
    st.session_state.risk_manager = RiskManager(config.risk)
    st.session_state.news_service = NewsService(config.news)
    st.session_state.ai_service = AIService(config.ai)
    st.session_state.symbol_manager = SymbolManager(["XAUUSD"])
    st.session_state.session_guard = SessionGuard(block_between_sessions=True)
    st.session_state.calendar_guard = EconomicCalendarGuard(blackout_minutes_before=30, blackout_minutes_after=30, min_impact="high")
    st.session_state.bot_running = True
    st.session_state.peak_balance = float(broker.get_account_info().balance)
    STOP.unlink(missing_ok=True)

    def watcher():
        while st.session_state.bot_running:
            if STOP.exists():
                st.session_state.bot_running = False
                write_state(running=False, ok=True, reason="stop_requested")
                return
            time.sleep(0.5)

    threading.Thread(target=watcher, daemon=True).start()
    write_state(running=True, ok=True, broker=config.broker)
    try:
        app.trading_loop(config)
    finally:
        try:
            broker.disconnect()
        except Exception:
            pass
        st.session_state.bot_running = False
        write_state(running=False, ok=True, broker=config.broker)


if __name__ == "__main__":
    main()
