# Integration boundary

The dashboard is intentionally isolated under `frontend/` so the existing main code remains unchanged.

Recommended future endpoints:
- `GET /api/status`
- `GET /api/account`
- `GET /api/market?symbol=XAUUSDm&timeframe=M15`
- `GET /api/positions`
- `GET /api/signals`
- `GET /api/health`
- `POST /api/bot/start`
- `POST /api/bot/stop`
- `POST /api/backtest`
- WebSocket `/ws/live`
