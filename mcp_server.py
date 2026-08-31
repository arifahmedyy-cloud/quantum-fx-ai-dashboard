"""MCP server for Quantum FX AI — exposes the dashboard_api.py backend to
Claude Desktop (or any MCP-compatible client) as a set of tools.

DESIGN / SAFETY NOTE (read this before adding more tools):
This server intentionally exposes READ-ONLY and ANALYSIS tools only
(account info, positions, market data, running an AI analysis pass,
running a backtest, reading/updating risk *settings*). It does NOT expose
a tool that places a live order or starts/stops the bot's live trading
loop. That is a deliberate choice, not an oversight:

  - LLMs can misread numbers, hallucinate, or be manipulated by prompt
    injection (e.g. text pulled from a news headline). A tool that lets
    an LLM push a real market order removes the human-in-the-loop check
    that the DecisionEngine + RiskManager pipeline is built around.
  - Your existing architecture already puts the LLM (AIService) in an
    advisory role — it contributes an "AI score" to the decision, but
    `DecisionEngine`/`RiskManager` make the final call. This MCP server
    preserves that same boundary for Claude Desktop: Claude can look at
    the bot's data and reasoning, and suggest things, but a human decides
    whether to click Start/Stop or change risk settings.

If you deliberately want to add an order-placing or bot-start/stop tool
later, do it as a clearly separate, explicitly-confirmed tool — do not
fold it into `get_account` or similar "safe" tools.

Setup:
    pip install mcp
    (dashboard_api.py must be running, e.g. `uvicorn dashboard_api:app --port 8808`)

Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "quantum-fx-ai": {
          "command": "python",
          "args": ["C:\\path\\to\\qfx\\mcp_server.py"],
          "env": {"QFX_API_URL": "http://127.0.0.1:8808"}
        }
      }
    }
"""
from __future__ import annotations

import os
from typing import Any

import requests
from mcp.server.mcpserver import MCPServer

API_BASE = os.environ.get("QFX_API_URL", "http://127.0.0.1:8808")
TIMEOUT = 30

server = MCPServer(
    name="quantum-fx-ai",
    instructions=(
        "Tools for reading and analyzing the Quantum FX AI trading bot's "
        "state (account, positions, market data, signals, backtests, risk "
        "settings). This server does not place trades or start/stop the "
        "bot — those actions must be done by a human in the dashboard UI."
    ),
)


def _get(path: str, params: dict | None = None) -> Any:
    r = requests.get(f"{API_BASE}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(path: str, params: dict | None = None, json_body: dict | None = None) -> Any:
    r = requests.post(f"{API_BASE}{path}", params=params, json=json_body, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _api_error(exc: Exception) -> dict:
    return {
        "error": str(exc),
        "hint": (f"Is the dashboard API running at {API_BASE}? Start it with "
                 f"'python -m uvicorn dashboard_api:app --host 127.0.0.1 --port 8808'."),
    }


@server.tool()
def get_health() -> dict:
    """Check whether the dashboard API and broker connection are up, and
    which broker mode (paper/mt5) is currently active."""
    try:
        return _get("/api/health")
    except Exception as exc:
        return _api_error(exc)


@server.tool()
def get_account(mode: str = "paper") -> dict:
    """Get account balance, equity, margin, and free margin.

    Args:
        mode: "paper" or "mt5" — which broker's account to read.
    """
    try:
        return _get(f"/api/{mode}/account")
    except Exception as exc:
        return _api_error(exc)


@server.tool()
def get_positions(mode: str = "paper") -> dict:
    """Get currently open positions (empty list if none).

    Args:
        mode: "paper" or "mt5" — which broker's positions to read.
    """
    try:
        return _get(f"/api/{mode}/positions")
    except Exception as exc:
        return _api_error(exc)


@server.tool()
def get_market(symbol: str = "XAUUSD", timeframe: str = "M15", bars: int = 80,
               mode: str = "paper") -> dict:
    """Get recent OHLCV candle data and the current bid/ask for a symbol.

    Args:
        symbol: e.g. "XAUUSD", "EURUSD" (paper mode always uses XAUUSD).
        timeframe: one of M5, M15, M30, H1, H4.
        bars: number of recent candles to return (10-1000).
        mode: "paper" (synthetic data) or "mt5" (real broker data).
    """
    try:
        return _get(f"/api/{mode}/market", params={"symbol": symbol, "timeframe": timeframe, "bars": bars})
    except Exception as exc:
        return _api_error(exc)


@server.tool()
def run_analysis(symbol: str = "XAUUSD", timeframe: str = "M15", mode: str = "paper") -> dict:
    """Run one real analysis pass (SMC + regime detection + AI score +
    DecisionEngine) and return the resulting signal. This does NOT place
    any order — it only reports what the bot's analysis currently says.

    Args:
        symbol: e.g. "XAUUSD", "EURUSD".
        timeframe: one of M5, M15, M30, H1, H4.
        mode: "paper" or "mt5".
    """
    try:
        return _post(f"/api/analysis/{mode}", params={"symbol": symbol, "timeframe": timeframe})
    except Exception as exc:
        return _api_error(exc)


@server.tool()
def run_backtest(symbol: str = "XAUUSD", timeframe: str = "M15", months: int = 1,
                  mode: str = "paper") -> dict:
    """Run a historical backtest and return performance metrics (win rate,
    Sharpe ratio, max drawdown, trade count, etc.). No real money or orders
    are involved — this replays historical/synthetic data.

    Args:
        symbol: e.g. "XAUUSD", "EURUSD".
        timeframe: one of M5, M15, M30, H1, H4.
        months: backtest duration, 1-24. Longer ranges on fine timeframes
            (M5/M15) take longer to compute — prefer H1/H4 for months > 6.
        mode: "paper" (synthetic data) or "mt5" (real broker history).
    """
    try:
        return _post(f"/api/backtest/{mode}", params={"symbol": symbol, "timeframe": timeframe, "months": months})
    except Exception as exc:
        return _api_error(exc)


@server.tool()
def get_history(limit: int = 20) -> dict:
    """Get recent trade journal history (closed trades)."""
    try:
        return _get("/api/history", params={"limit": limit})
    except Exception as exc:
        return _api_error(exc)


@server.tool()
def get_risk_settings() -> dict:
    """Get the current risk-manager configuration (risk per trade, max
    risk, daily loss limit, max open trades, and feature toggles)."""
    try:
        return _get("/api/settings/risk")
    except Exception as exc:
        return _api_error(exc)


@server.tool()
def suggest_risk_settings_change(base_risk_pct: float | None = None,
                                  max_risk_pct: float | None = None,
                                  daily_loss_limit_pct: float | None = None,
                                  max_open_trades: int | None = None) -> dict:
    """Propose a change to risk settings for the user to review — this does
    NOT apply the change. It returns the current settings, the proposed
    new settings, and a plain-English summary of what would change, so a
    human can decide whether to apply it via the dashboard's Risk Manager
    panel (or explicitly ask to actually apply it through apply_risk_settings).
    """
    try:
        current = _get("/api/settings/risk")
    except Exception as exc:
        return _api_error(exc)
    proposed = dict(current)
    changes = {}
    for key, val in [("base_risk_pct", base_risk_pct), ("max_risk_pct", max_risk_pct),
                      ("daily_loss_limit_pct", daily_loss_limit_pct),
                      ("max_open_trades", max_open_trades)]:
        if val is not None and val != current.get(key):
            changes[key] = {"from": current.get(key), "to": val}
            proposed[key] = val
    return {
        "current_settings": current,
        "proposed_settings": proposed,
        "changes": changes,
        "note": ("This is a PROPOSAL only — nothing has been changed. Review "
                 "the changes and, if you agree, ask to apply them via "
                 "apply_risk_settings, or change them yourself in the "
                 "dashboard's Risk Manager panel."),
    }


@server.tool()
def apply_risk_settings(base_risk_pct: float | None = None,
                         max_risk_pct: float | None = None,
                         daily_loss_limit_pct: float | None = None,
                         max_open_trades: int | None = None,
                         confirm: bool = False) -> dict:
    """Actually apply a risk-settings change. Requires confirm=True to take
    effect — this is a real change to how much the bot risks per trade, so
    it should only be called after the user has explicitly agreed to the
    specific numbers (e.g. after reviewing suggest_risk_settings_change).

    Note: a bot that is already RUNNING will not pick up these changes
    until it is stopped and restarted from the dashboard.
    """
    if not confirm:
        return {"applied": False, "reason": "confirm=True is required to apply a risk-settings change."}
    body = {k: v for k, v in {
        "base_risk_pct": base_risk_pct, "max_risk_pct": max_risk_pct,
        "daily_loss_limit_pct": daily_loss_limit_pct, "max_open_trades": max_open_trades,
    }.items() if v is not None}
    if not body:
        return {"applied": False, "reason": "No settings provided to change."}
    try:
        return _post("/api/settings/risk", json_body=body)
    except Exception as exc:
        return _api_error(exc)


@server.tool()
def get_settings() -> dict:
    """Get general bot settings: broker mode, tracked symbols, risk
    parameters, and AI provider configuration."""
    try:
        return _get("/api/settings")
    except Exception as exc:
        return _api_error(exc)


if __name__ == "__main__":
    server.run(transport="stdio")
