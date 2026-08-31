# React Dashboard Integration Audit

## Scope
The React dashboard was connected through a new, separate read-only FastAPI adapter. The existing trading/backtest/MT5/SMC/RiskManager code was not modified.

## Main-code verification
SHA-256 comparisons against `v2_backtest_production_ready.zip` show these core files are unchanged:
- `app.py`
- `src/ui/components.py`
- `src/backtesting/backtest_engine.py`
- `src/config.py`
- `src/trading/broker_connector.py`
- `src/trading/decision_engine.py`
- `src/trading/risk_manager.py`
- `src/trading/smc.py`
- `src/trading/strategies.py`
- `src/models.py`
- `src/services/ai_service.py`

## New integration layer
- `dashboard_api.py` — read-only API bridge
- `dashboard_api_requirements.txt` — FastAPI/Uvicorn dependencies
- `frontend/src/main.jsx` — keeps the existing visual design but replaces demo values with API-backed data
- `frontend/.env.example` — API URL configuration
- `Run Dashboard API.bat`
- `Run Full React Dashboard.bat`
- `Setup React Dashboard.bat`

## Read-only endpoints
- `GET /api/health`
- `GET /api/status`
- `GET /api/account`
- `GET /api/positions`
- `GET /api/market?symbol=XAUUSDm&timeframe=M15&bars=80`

No order/signal-execution endpoint was added. The dashboard cannot place or close trades through this bridge.

## Runtime validation
Static validation passed for the API module (`python -m py_compile dashboard_api.py`). Node.js/npm are present, but frontend dependencies were not installed in the audit sandbox, so a production Vite build could not be executed there. Run `Setup React Dashboard.bat` on the Windows PC, then use `Run Full React Dashboard.bat`.

## Data behavior
The dashboard now reports real broker/API state. It does not manufacture a CONNECTED/LIVE state when the API/broker is unavailable. Broker data is sourced from the configured `paper`, `mt5`, or `mt5_bridge` connector.
