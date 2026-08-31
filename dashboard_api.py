"""React dashboard adapter. Core trading modules are not modified."""
from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import load_config
from src.trading.broker_connector import MT5Broker, MT5BridgeBroker, PaperBroker
from src.trading.indicators import TechnicalIndicators
from src.trading.regime_detector import RegimeDetector
from src.trading.smc import SMCAnalyzer
from src.trading.decision_engine import DecisionEngine
from src.services.ai_service import AIService
from src.services.news_service import NewsService
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_report import PerformanceReporter
from src.backtesting.mt5_data_cache import MT5DataCache
from src.trading.strategies import get_strategy
from src.services.paper_market_data import generate_paper_ohlcv, paper_price

app = FastAPI(title="Quantum FX AI Dashboard API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"])

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".qfx_dashboard_state.json"
STOP = ROOT / ".qfx_dashboard_stop"
_controller: subprocess.Popen | None = None
_broker: Any = None
_paper_broker: Any = None
_config: Any = None
_last_error: str | None = None


def _build_broker(cfg):
    if cfg.broker == "paper": return PaperBroker(initial_balance=10000.0, leverage=cfg.mt5.leverage, symbol="XAUUSD")
    if cfg.broker == "mt5": return MT5Broker(login=cfg.mt5.login, password=cfg.mt5.password, server=cfg.mt5.server, leverage=cfg.mt5.leverage, reconnect_attempts=cfg.mt5.reconnect_attempts, reconnect_backoff_seconds=cfg.mt5.reconnect_backoff_seconds, reconnect_backoff_multiplier=cfg.mt5.reconnect_backoff_multiplier)
    if cfg.broker == "mt5_bridge": return MT5BridgeBroker(base_url=cfg.mt5_bridge.base_url, token=cfg.mt5_bridge.token, symbol_candidates=cfg.mt5_bridge.symbol_candidates, reconnect_attempts=cfg.mt5_bridge.reconnect_attempts, reconnect_backoff_seconds=cfg.mt5_bridge.reconnect_backoff_seconds, reconnect_backoff_multiplier=cfg.mt5_bridge.reconnect_backoff_multiplier, request_timeout_seconds=cfg.mt5_bridge.request_timeout_seconds, leverage=cfg.mt5_bridge.leverage)
    raise RuntimeError(f"Unsupported broker: {cfg.broker}")


def _ensure_paper_broker():
    global _paper_broker
    if _paper_broker is None:
        _paper_broker = PaperBroker(initial_balance=10000.0, leverage=100.0, symbol="XAUUSD")
        _paper_broker.connect()
    return _paper_broker


def _paper_market(symbol: str, timeframe: str, bars: int):
    paper_symbol = "XAUUSD" if symbol.lower().startswith("xauusd") else symbol
    df = generate_paper_ohlcv(paper_symbol, timeframe, bars)
    px = paper_price(paper_symbol, timeframe)
    broker = _ensure_paper_broker()
    broker.update_price(paper_symbol, px["bid"], px["ask"])
    records = [
        {"time": str(row.Date), "open": float(row.Open), "high": float(row.High),
         "low": float(row.Low), "close": float(row.Close), "volume": float(row.Volume)}
        for row in df.itertuples(index=False)
    ]
    return {
        "source": "paper_synthetic", "mode": "paper",
        "symbol": paper_symbol, "requested_symbol": symbol, "timeframe": timeframe,
        "bid": px["bid"], "ask": px["ask"], "candles": records,
        "updated_at": datetime.now().isoformat()
    }


def _risk_summary_for_broker(broker, cfg):
    account = broker.get_account_info()
    from src.trading.risk_manager import RiskManager
    rm = RiskManager(cfg.risk)
    summary = rm.get_risk_summary()
    guard = rm.daily_guard(account.balance, account.balance)
    summary["guard_active"] = bool(guard.should_block_new_trades)
    summary["guard_reason"] = guard.reason
    return {"summary": summary}

def _start_bot_for_mode(mode: str):
    global _controller
    mode = mode.lower()
    if mode not in ("paper", "mt5"):
        raise HTTPException(400, "Mode must be paper or mt5")
    st = _read_state()
    if _controller and _controller.poll() is None or st.get("running"):
        raise HTTPException(409, f"Bot already running in {st.get('broker', 'unknown')} mode")
    cfg, _ = _ensure_broker()
    if mode == "mt5" and cfg.broker != "mt5":
        # Credentials remain in the normal config; only the broker selector is overridden
        pass
    STOP.unlink(missing_ok=True)
    env = os.environ.copy()
    env["BROKER"] = mode
    _controller = subprocess.Popen(
        [sys.executable, str(ROOT / "dashboard_controller.py")],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )
    return {"ok": True, "running": True, "mode": mode, "pid": _controller.pid}


def _ensure_broker():
    global _broker, _config, _last_error
    if _broker is None:
        _config = load_config(validate=False); _broker = _build_broker(_config)
    try:
        if not _broker.is_connected(): _broker.connect()
        _last_error = None
    except Exception as exc: _last_error = str(exc)
    return _config, _broker


_mt5_broker: Any = None


def _ensure_mt5_broker(cfg):
    """Always return a genuine MT5-backed broker for /api/mt5/* endpoints.

    BUG FIX: previously these endpoints called `_build_broker(cfg)`, which
    branches on `cfg.broker` (the globally configured broker mode, e.g.
    "paper"). When the app was running in paper mode, that silently
    returned a PaperBroker mislabeled as MT5 data — violating the
    requirement that MT5 endpoints only ever serve real MT5 data. This
    helper always constructs/reuses a real MT5Broker regardless of the
    active global broker mode.
    """
    global _mt5_broker
    if cfg.broker == "mt5":
        # The global broker IS already a real MT5Broker; reuse it.
        return _ensure_broker()[1]
    if _mt5_broker is None:
        _mt5_broker = MT5Broker(
            login=cfg.mt5.login, password=cfg.mt5.password, server=cfg.mt5.server,
            leverage=cfg.mt5.leverage, reconnect_attempts=cfg.mt5.reconnect_attempts,
            reconnect_backoff_seconds=cfg.mt5.reconnect_backoff_seconds,
            reconnect_backoff_multiplier=cfg.mt5.reconnect_backoff_multiplier,
        )
    if not _mt5_broker.is_connected():
        _mt5_broker.connect()
    return _mt5_broker


def _account_dict(a):
    return {"balance":float(a.balance),"equity":float(a.equity),"margin":float(a.margin),"free_margin":float(a.free_margin),"margin_level":float(a.margin_level),"currency":a.currency,"leverage":float(a.leverage)}

def _position_dict(p):
    return {"ticket":int(p.ticket),"symbol":p.symbol,"side":p.position_type,"volume":float(p.volume),"entry":float(p.open_price),"sl":float(p.sl or 0),"tp":float(p.tp or 0),"profit":float(p.profit),"open_time":p.open_time.isoformat() if isinstance(p.open_time,datetime) else str(p.open_time)}

def _read_state():
    try: return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception: return {"running":False,"ok":True}

@app.get("/api/health")
def health():
    cfg, broker = _ensure_broker()
    try: status=broker.health_check()
    except Exception as exc: return {"ok":False,"broker":cfg.broker,"connected":False,"terminal_ok":False,"account_ok":False,"last_error":str(exc)}
    st=_read_state()
    return {"ok":bool(status.get("connected")),"broker":cfg.broker,"connected":bool(status.get("connected")),"terminal_ok":bool(status.get("terminal_ok")),"account_ok":bool(status.get("account_ok")),"resolved_symbol":status.get("symbol") or getattr(broker,"resolved_symbol",None),"bot_running":bool(st.get("running")),"last_error":_last_error,"checked_at":datetime.now().isoformat()}

@app.get("/api/status")
def status(): return health()

@app.get("/api/account")
def account():
    cfg,b=_ensure_broker();
    if not b.is_connected(): raise HTTPException(503,"Broker is not connected")
    return {"source":cfg.broker,"account":_account_dict(b.get_account_info())}

@app.get("/api/positions")
def positions():
    cfg,b=_ensure_broker();
    if not b.is_connected(): raise HTTPException(503,"Broker is not connected")
    return {"source":cfg.broker,"positions":[_position_dict(p) for p in b.get_positions()]}

@app.get("/api/market")
def market(symbol:str=Query("XAUUSDm",min_length=2),timeframe:str=Query("M15"),bars:int=Query(120,ge=10,le=1000)):
    cfg,b=_ensure_broker();
    # BUG FIX: PaperBroker has no get_ohlcv (paper OHLCV comes from the
    # synthetic generator, not the broker object) — this generic route
    # previously 502'd whenever the active broker was "paper".
    if cfg.broker == "paper":
        return _paper_market(symbol, timeframe, bars)
    if not b.is_connected(): raise HTTPException(503,"Broker is not connected")
    try: df=b.get_ohlcv(symbol=symbol,timeframe=timeframe,bars=bars)
    except Exception as exc: raise HTTPException(502,str(exc))
    if df is None or df.empty: raise HTTPException(404,f"No broker data for {symbol} {timeframe}")
    records=[]
    for row in df.tail(bars).itertuples(index=False):
        dt=getattr(row,"Date",None)
        records.append({"time":dt.isoformat() if hasattr(dt,"isoformat") else str(dt),"open":float(row.Open),"high":float(row.High),"low":float(row.Low),"close":float(row.Close),"volume":float(getattr(row,"Volume",0))})
    price=b.get_price(symbol)
    return {"source":cfg.broker,"symbol":symbol,"timeframe":timeframe,"bid":float(price.get("bid",0) or 0),"ask":float(price.get("ask",0) or 0),"candles":records,"updated_at":datetime.now().isoformat()}

@app.get("/api/risk")
def risk():
    cfg, b = _ensure_broker()
    if not b.is_connected():
        raise HTTPException(503, "Broker is not connected")
    return _risk_summary_for_broker(b, cfg)

@app.get("/api/paper/risk")
def paper_risk():
    cfg, _ = _ensure_broker()
    return _risk_summary_for_broker(_ensure_paper_broker(), cfg)

@app.get("/api/mt5/risk")
def mt5_risk():
    cfg, b = _ensure_broker()
    if cfg.broker != "mt5":
        b = _ensure_mt5_broker(cfg)
    if not b.is_connected():
        raise HTTPException(503, "MT5 is not connected")
    return _risk_summary_for_broker(b, cfg)

@app.get("/api/settings")
def settings():
    from src.trading.symbol_manager import SymbolManager
    cfg,_=_ensure_broker(); return {"broker":cfg.broker,"symbols":SymbolManager.all_known_symbols(),"risk_per_trade":cfg.risk.base_risk_pct,"max_risk":cfg.risk.max_risk_pct,"daily_loss_limit":cfg.risk.daily_loss_limit_pct,"max_open_trades":cfg.risk.max_open_trades,"ai_provider":cfg.ai.provider,"ai_enabled":cfg.ai.enabled}

@app.get("/api/history")
def history(limit:int=Query(50,ge=1,le=500)):
    from src.services.journal_service import JournalService
    try: return {"trades":JournalService().trades[:limit]}
    except Exception as exc: return {"trades":[],"error":str(exc)}

@app.post("/api/bot/start")
def bot_start(mode: str = Query("mt5")):
    return _start_bot_for_mode(mode)

@app.post("/api/bot/start/paper")
def bot_start_paper():
    return _start_bot_for_mode("paper")

@app.post("/api/bot/start/mt5")
def bot_start_mt5():
    return _start_bot_for_mode("mt5")

@app.post("/api/bot/stop")
def bot_stop():
    global _controller
    STOP.write_text("stop",encoding="utf-8")
    return {"ok":True,"stopping":True}

@app.get("/api/paper/health")
def paper_health():
    b = _ensure_paper_broker()
    st = _read_state()
    running = bool(st.get("running") and st.get("broker") == "paper")
    return {"ok": True, "broker": "paper", "mode": "paper", "connected": b.is_connected(),
            "terminal_ok": True, "account_ok": True, "resolved_symbol": "XAUUSD",
            "bot_running": running, "last_error": st.get("error")}

@app.get("/api/paper/account")
def paper_account():
    b = _ensure_paper_broker()
    return {"source": "paper", "mode": "paper", "account": _account_dict(b.get_account_info())}

@app.get("/api/paper/positions")
def paper_positions():
    b = _ensure_paper_broker()
    return {"source": "paper", "mode": "paper", "positions": [_position_dict(p) for p in b.get_positions()]}

@app.get("/api/paper/market")
def paper_market(symbol: str = Query("XAUUSD"), timeframe: str = Query("M15"), bars: int = Query(120, ge=10, le=1000)):
    return _paper_market(symbol, timeframe, bars)

@app.get("/api/mt5/health")
def mt5_health():
    cfg, b = _ensure_broker()
    if cfg.broker != "mt5":
        b = _ensure_mt5_broker(cfg)
    try:
        status = b.health_check()
    except Exception as exc:
        return {"ok": False, "broker": "mt5", "mode": "mt5", "connected": False,
                "terminal_ok": False, "account_ok": False, "last_error": str(exc)}
    st = _read_state()
    running = bool(st.get("running") and st.get("broker") == "mt5")
    return {"ok": bool(status.get("connected")), "broker": "mt5", "mode": "mt5",
            "connected": bool(status.get("connected")), "terminal_ok": bool(status.get("terminal_ok")),
            "account_ok": bool(status.get("account_ok")),
            "resolved_symbol": status.get("symbol") or getattr(b, "resolved_symbol", None),
            "bot_running": running, "last_error": st.get("error")}

@app.get("/api/mt5/account")
def mt5_account():
    cfg, b = _ensure_broker()
    if cfg.broker != "mt5":
        b = _ensure_mt5_broker(cfg)
    if not b.is_connected():
        raise HTTPException(503, "MT5 is not connected")
    return {"source": "mt5", "mode": "mt5", "account": _account_dict(b.get_account_info())}

@app.get("/api/mt5/positions")
def mt5_positions():
    cfg, b = _ensure_broker()
    if cfg.broker != "mt5":
        b = _ensure_mt5_broker(cfg)
    if not b.is_connected():
        raise HTTPException(503, "MT5 is not connected")
    return {"source": "mt5", "mode": "mt5", "positions": [_position_dict(p) for p in b.get_positions()]}

@app.get("/api/mt5/market")
def mt5_market(symbol: str = Query("XAUUSDm"), timeframe: str = Query("M15"), bars: int = Query(120, ge=10, le=1000)):
    cfg, b = _ensure_broker()
    if cfg.broker != "mt5":
        b = _ensure_mt5_broker(cfg)
    if not b.is_connected():
        raise HTTPException(503, "MT5 is not connected")
    df = b.get_ohlcv(symbol=symbol, timeframe=timeframe, bars=bars)
    if df is None or df.empty:
        raise HTTPException(404, f"No MT5 data for {symbol} {timeframe}")
    records = []
    for row in df.tail(bars).itertuples(index=False):
        dtv = getattr(row, "Date", None)
        records.append({"time": dtv.isoformat() if hasattr(dtv, "isoformat") else str(dtv),
                        "open": float(row.Open), "high": float(row.High),
                        "low": float(row.Low), "close": float(row.Close),
                        "volume": float(getattr(row, "Volume", 0))})
    price = b.get_price(symbol)
    return {"source": "mt5", "mode": "mt5", "symbol": symbol, "timeframe": timeframe,
            "bid": float(price.get("bid", 0) or 0), "ask": float(price.get("ask", 0) or 0),
            "candles": records, "updated_at": datetime.now().isoformat()}

def _analysis_for_broker(cfg, broker, symbol: str, timeframe: str):
    if not broker.is_connected():
        raise HTTPException(503, "Broker is not connected")
    df = broker.get_ohlcv(symbol=symbol, timeframe=timeframe, bars=200)
    if df is None or df.empty:
        raise HTTPException(404, "No market data")
    df = TechnicalIndicators.add_all(df)
    current = float(df.iloc[-1]["Close"])
    smc = SMCAnalyzer().analyze(df)
    regime = RegimeDetector().generate_signal(df)
    decision = DecisionEngine(cfg.risk).decide(regime, smc, current_price=current)
    return {"symbol": symbol, "timeframe": timeframe, "price": current,
            "action": decision.action, "entry": decision.entry, "sl": decision.sl,
            "tp": decision.tp, "ai_score": decision.ai_score, "regime": regime.regime,
            "smc_bias": smc.bias, "explanation": decision.explanation}

@app.post("/api/analysis")
def analysis(symbol: str = Query("XAUUSDm"), timeframe: str = Query("M15")):
    cfg, b = _ensure_broker()
    # BUG FIX: same PaperBroker.get_ohlcv gap as /api/market — delegate to
    # the paper-synthetic analysis path when running in paper mode.
    if cfg.broker == "paper":
        return analysis_paper(symbol=symbol if symbol != "XAUUSDm" else "XAUUSD", timeframe=timeframe)
    return _analysis_for_broker(cfg, b, symbol, timeframe)

@app.post("/api/analysis/paper")
def analysis_paper(symbol: str = Query("XAUUSD"), timeframe: str = Query("M15")):
    cfg, _ = _ensure_broker()
    b = _ensure_paper_broker()
    # Paper data is generated locally so analysis is completely offline.
    df = generate_paper_ohlcv(symbol, timeframe, 200)
    if df is None or df.empty:
        raise HTTPException(404, "No paper market data")
    df = TechnicalIndicators.add_all(df)
    current = float(df.iloc[-1]["Close"])
    smc = SMCAnalyzer().analyze(df)
    regime = RegimeDetector().generate_signal(df)
    decision = DecisionEngine(cfg.risk).decide(regime, smc, current_price=current)
    return {"source": "paper_synthetic", "symbol": symbol, "timeframe": timeframe,
            "price": current, "action": decision.action, "entry": decision.entry,
            "sl": decision.sl, "tp": decision.tp, "ai_score": decision.ai_score,
            "regime": regime.regime, "smc_bias": smc.bias,
            "explanation": decision.explanation}

@app.post("/api/analysis/mt5")
def analysis_mt5(symbol: str = Query("XAUUSDm"), timeframe: str = Query("M15")):
    cfg, b = _ensure_broker()
    if cfg.broker != "mt5":
        b = _ensure_mt5_broker(cfg)
    return _analysis_for_broker(cfg, b, symbol, timeframe)


def _run_backtest_df(cfg, broker, df, symbol: str, timeframe: str, source: str):
    if df is None or df.empty:
        raise HTTPException(404, f"No {source} historical data available")
    engine = BacktestEngine(
        initial_balance=10000.0,
        risk_config=cfg.risk,
        backtest_config=cfg.backtest,
        symbol_specs=broker.get_symbol_specs(symbol) if hasattr(broker, "get_symbol_specs") else {},
    )
    summary = engine.run(df, strategy_fn=None)
    records = [
        {n: getattr(t, n) for n in t.__dataclass_fields__}
        if hasattr(t, "__dataclass_fields__") else dict(t)
        for t in getattr(engine, "trades", [])
    ]
    report = PerformanceReporter(records).calculate_metrics(starting_balance=10000.0)
    return {
        "ok": True, "source": source, "symbol": symbol, "timeframe": timeframe,
        "summary": {
            "initial_balance": 10000.0, "final_balance": summary.final_balance,
            "return_pct": summary.return_pct, "trades": summary.trades,
            "wins": summary.wins, "win_rate": summary.win_rate,
            "profit_factor": summary.profit_factor, "max_dd": summary.max_dd,
        },
        "metrics": {"sharpe_ratio": report.sharpe_ratio,
                    "sortino_ratio": report.sortino_ratio, "expectancy": report.expectancy},
        "rows": len(df),
    }


@app.post("/api/backtest/paper")
def backtest_paper(symbol: str = Query("XAUUSD"), timeframe: str = Query("M15"),
                   months: int = Query(1, ge=1, le=24)):
    cfg, _ = _ensure_broker()
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    mins = {"M5":5, "M15":15, "M30":30, "H1":60, "H4":240}.get(timeframe)
    if not mins:
        raise HTTPException(400, "Unsupported timeframe")
    bars = min(100000, int((30 * months * 24 * 60) / mins) + 250)
    df = generate_paper_ohlcv(symbol, timeframe, bars, end=end)
    paper = _ensure_paper_broker()
    return _run_backtest_df(cfg, paper, df, symbol, timeframe, "paper_synthetic")


@app.post("/api/backtest/mt5")
def backtest_mt5(symbol: str = Query("XAUUSDm"), timeframe: str = Query("M15"),
                 months: int = Query(1, ge=1, le=24)):
    cfg, b = _ensure_broker()
    if cfg.broker != "mt5":
        b = _ensure_mt5_broker(cfg)
    if not b.is_connected():
        raise HTTPException(503, "MT5 is not connected")
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    mins = {"M5":5, "M15":15, "M30":30, "H1":60, "H4":240}.get(timeframe)
    if not mins:
        raise HTTPException(400, "Unsupported timeframe")
    end = now - timedelta(minutes=now.minute % mins, seconds=1)
    start = end - timedelta(days=30 * months)
    cache = MT5DataCache()
    df = cache.load(symbol, timeframe, start, end)
    if df.empty:
        df, _ = cache.update(b, symbol, timeframe, start, end, force_refresh=False)
    if df.empty:
        raise HTTPException(404, "No MT5 historical data available; sync MT5 data first")
    return _run_backtest_df(cfg, b, df, symbol, timeframe, "mt5")


@app.post("/api/backtest")
def backtest(symbol: str = Query("XAUUSDm"), timeframe: str = Query("M15"),
             months: int = Query(1, ge=1, le=24), mode: str = Query("mt5")):
    if mode.lower() == "paper":
        return backtest_paper(symbol="XAUUSD", timeframe=timeframe, months=months)
    return backtest_mt5(symbol=symbol, timeframe=timeframe, months=months)


# --- Risk Manager settings: previously read-only (no switches at all). ---

class RiskSettingsUpdate(BaseModel):
    base_risk_pct: float | None = None
    max_risk_pct: float | None = None
    daily_loss_limit_pct: float | None = None
    max_open_trades: int | None = None
    use_auto_drawdown_risk: bool | None = None
    use_trailing_stop: bool | None = None
    use_kelly_sizing: bool | None = None
    enable_ml_filter: bool | None = None
    strict_smc_confluence: bool | None = None
    # Circuit breaker (consecutive-loss guard) and Break-Even/Trailing
    # tuning — these already existed in RiskManager/TrailingStopManager
    # but had no dashboard-editable settings; added together with the
    # fix that made the circuit breaker actually fire in live trading
    # (see app.py's _reconcile_closed_trades / daily_guard call sites).
    max_consecutive_losses: int | None = None
    consecutive_loss_cooldown_hours: float | None = None
    breakeven_trigger_r: float | None = None
    breakeven_buffer_pct: float | None = None
    trailing_trigger_r: float | None = None
    trailing_distance_r: float | None = None


_RISK_ENV_KEY_MAP = {
    "base_risk_pct": "RISK_BASE_PCT",
    "max_risk_pct": "RISK_MAX_PCT",
    "daily_loss_limit_pct": "RISK_DAILY_LOSS_LIMIT_PCT",
    "max_open_trades": "RISK_MAX_OPEN_TRADES",
    "use_auto_drawdown_risk": "RISK_USE_AUTO_DRAWDOWN",
    "use_trailing_stop": "RISK_USE_TRAILING_STOP",
    "use_kelly_sizing": "RISK_USE_KELLY_SIZING",
    "enable_ml_filter": "RISK_ENABLE_ML_FILTER",
    "strict_smc_confluence": "RISK_STRICT_SMC_CONFLUENCE",
    "max_consecutive_losses": "RISK_MAX_CONSECUTIVE_LOSSES",
    "consecutive_loss_cooldown_hours": "RISK_CONSECUTIVE_LOSS_COOLDOWN_HOURS",
    "breakeven_trigger_r": "RISK_BREAKEVEN_TRIGGER_R",
    "breakeven_buffer_pct": "RISK_BREAKEVEN_BUFFER_PCT",
    "trailing_trigger_r": "RISK_TRAILING_TRIGGER_R",
    "trailing_distance_r": "RISK_TRAILING_DISTANCE_R",
}


@app.get("/api/settings/risk")
def get_risk_settings():
    cfg, _ = _ensure_broker()
    r = cfg.risk
    return {
        "base_risk_pct": r.base_risk_pct, "max_risk_pct": r.max_risk_pct,
        "daily_loss_limit_pct": r.daily_loss_limit_pct, "max_open_trades": r.max_open_trades,
        "use_auto_drawdown_risk": r.use_auto_drawdown_risk, "use_trailing_stop": r.use_trailing_stop,
        "use_kelly_sizing": r.use_kelly_sizing, "enable_ml_filter": r.enable_ml_filter,
        "strict_smc_confluence": r.strict_smc_confluence,
        "max_consecutive_losses": r.max_consecutive_losses,
        "consecutive_loss_cooldown_hours": r.consecutive_loss_cooldown_hours,
        "breakeven_trigger_r": r.breakeven_trigger_r,
        "breakeven_buffer_pct": r.breakeven_buffer_pct,
        "trailing_trigger_r": r.trailing_trigger_r,
        "trailing_distance_r": r.trailing_distance_r,
    }


def _write_env_values(values: dict) -> None:
    env_path = Path(".env")
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    existing = {}
    for i, line in enumerate(lines):
        if "=" in line and not line.strip().startswith("#"):
            existing[line.split("=", 1)[0].strip()] = i
    for k, v in values.items():
        entry = f"{k}={v}"
        if k in existing:
            lines[existing[k]] = entry
        else:
            lines.append(entry)
    env_path.write_text("\n".join(lines) + ("\n" if lines else ""))


@app.post("/api/settings/risk")
def update_risk_settings(body: RiskSettingsUpdate):
    """Update risk-manager parameters. BUG FIX (audit): the Risk Manager
    panel was previously read-only — there was no way to change risk
    per trade, daily loss limit, max open trades, or any of the risk
    toggles from the dashboard at all. This endpoint applies changes
    immediately to the dashboard's own in-memory config (so subsequent
    Analysis/Backtest calls use the new values) and persists them to
    .env so they survive a restart.
    """
    cfg, _ = _ensure_broker()
    updates = body.dict(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields provided")
    if "base_risk_pct" in updates and not (0 < updates["base_risk_pct"] <= 100):
        raise HTTPException(400, "base_risk_pct must be between 0 and 100")
    if "max_risk_pct" in updates and not (0 < updates["max_risk_pct"] <= 100):
        raise HTTPException(400, "max_risk_pct must be between 0 and 100")
    if "daily_loss_limit_pct" in updates and not (0 < updates["daily_loss_limit_pct"] <= 100):
        raise HTTPException(400, "daily_loss_limit_pct must be between 0 and 100")
    if "max_open_trades" in updates and not (1 <= updates["max_open_trades"] <= 50):
        raise HTTPException(400, "max_open_trades must be between 1 and 50")
    if "max_consecutive_losses" in updates and not (1 <= updates["max_consecutive_losses"] <= 20):
        raise HTTPException(400, "max_consecutive_losses must be between 1 and 20")
    if "consecutive_loss_cooldown_hours" in updates and not (0 < updates["consecutive_loss_cooldown_hours"] <= 168):
        raise HTTPException(400, "consecutive_loss_cooldown_hours must be between 0 and 168 (1 week)")
    if "breakeven_trigger_r" in updates and not (0 < updates["breakeven_trigger_r"] <= 10):
        raise HTTPException(400, "breakeven_trigger_r must be between 0 and 10")
    if "trailing_trigger_r" in updates and not (0 < updates["trailing_trigger_r"] <= 10):
        raise HTTPException(400, "trailing_trigger_r must be between 0 and 10")
    if ("breakeven_trigger_r" in updates and "trailing_trigger_r" in updates
            and updates["breakeven_trigger_r"] >= updates["trailing_trigger_r"]):
        raise HTTPException(400, "breakeven_trigger_r must be smaller than trailing_trigger_r")

    env_updates = {}
    for field, value in updates.items():
        setattr(cfg.risk, field, value)
        env_key = _RISK_ENV_KEY_MAP.get(field)
        if env_key:
            env_updates[env_key] = str(value)
    try:
        _write_env_values(env_updates)
        persisted = True
    except Exception:
        persisted = False

    return {
        "ok": True,
        "applied": updates,
        "persisted_to_env": persisted,
        "note": ("Applied immediately to this dashboard session (Analysis/Backtest "
                 "will use the new values right away). A bot that is already RUNNING "
                 "will not pick up these changes until it is stopped and restarted, "
                 "since it loaded its own config in a separate process at start time."),
    }
