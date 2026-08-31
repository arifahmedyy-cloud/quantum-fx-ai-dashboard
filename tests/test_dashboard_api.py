"""Regression tests for dashboard_api.py (React/FastAPI adapter).

These cover the bugs found during the audit:
  1. /api/settings crashed (cfg.symbols didn't exist).
  2. /api/mt5/* routes could silently serve PaperBroker data mislabeled
     as MT5 when the global broker mode was "paper".
  3. Paper synthetic price data was frozen (identical on every call).
  4. /api/market and /api/analysis (generic, unprefixed) crashed for
     PaperBroker, which has no get_ohlcv().

All tests run with BROKER=paper so no real MT5 connection or orders are
ever involved.
"""
import os
import sys
import time
import importlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["BROKER"] = "paper"

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import dashboard_api as api
    importlib.reload(api)
    state = api.STATE
    stop = api.STOP
    state.unlink(missing_ok=True)
    stop.unlink(missing_ok=True)
    yield TestClient(api.app)
    state.unlink(missing_ok=True)
    stop.unlink(missing_ok=True)


def test_settings_endpoint_does_not_crash(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert "XAUUSD" in body["symbols"]
    assert body["broker"] == "paper"


def test_mt5_health_never_fakes_connection_when_unavailable(client):
    # No MetaTrader5 terminal/package in this environment -> must honestly
    # report disconnected, never silently fall back to PaperBroker data.
    r = client.get("/api/mt5/health")
    assert r.status_code == 200
    body = r.json()
    assert body["broker"] == "mt5"
    assert body["connected"] is False


def test_mt5_account_503_not_fake_paper_data(client):
    r = client.get("/api/mt5/account")
    assert r.status_code == 503


def test_mt5_market_503_not_fake_paper_data(client):
    r = client.get("/api/mt5/market")
    assert r.status_code == 503


def test_paper_market_is_synthetic_and_labeled(client):
    r = client.get("/api/paper/market?symbol=XAUUSD&timeframe=M15&bars=50")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "paper_synthetic"
    assert len(body["candles"]) == 50


def test_generic_market_endpoint_works_in_paper_mode(client):
    r = client.get("/api/market?symbol=XAUUSD&timeframe=M15&bars=30")
    assert r.status_code == 200
    assert r.json()["source"] == "paper_synthetic"


def test_generic_analysis_endpoint_works_in_paper_mode(client):
    r = client.post("/api/analysis?symbol=XAUUSD&timeframe=M15")
    assert r.status_code == 200
    assert r.json()["source"] == "paper_synthetic"


def test_paper_backtest_runs_and_returns_metrics(client):
    r = client.post("/api/backtest/paper?symbol=XAUUSD&timeframe=M15&months=1")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["source"] == "paper_synthetic"
    assert "sharpe_ratio" in body["metrics"]


def test_bot_start_stop_lifecycle_paper_mode_only(client):
    r = client.post("/api/bot/start/paper")
    assert r.status_code == 200
    assert r.json()["mode"] == "paper"
    time.sleep(1.5)
    r2 = client.post("/api/bot/stop")
    assert r2.status_code == 200
    time.sleep(1.5)


def test_paper_price_not_static_over_time():
    from src.services.paper_market_data import paper_price
    p1 = paper_price("XAUUSD", "M15")
    time.sleep(1.1)
    p2 = paper_price("XAUUSD", "M15")
    assert p1 != p2, "paper price must not be frozen across real time"
