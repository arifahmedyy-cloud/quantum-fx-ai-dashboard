# Quantum FX AI — New React Dashboard

The legacy Streamlit dashboard is disabled as a launcher. The supported UI is React + FastAPI.

## Data modes
- PAPER: deterministic local synthetic market data.
- MT5: broker-native MetaTrader 5 data only.
- Yahoo Finance / `GC=F`: removed from the production data path.

## Controls
- Start Paper / Start MT5
- Stop Bot
- Run Analysis
- Paper Backtest
- MT5 Backtest
- Refresh
- Positions / History
- Risk guard status

## Run
1. Run `Setup.bat` once.
2. Run `Run Dashboard.bat`.
3. The dashboard opens at `http://127.0.0.1:5173`.

The trading engine, SMC, RiskManager, DecisionEngine, BacktestEngine and MT5 broker implementations are kept as the existing core. The dashboard uses `dashboard_api.py` as the adapter.

API is mode-separated: `/api/paper/*` for paper and `/api/mt5/*` for live MT5. No Yahoo fallback exists.
