# Quantum FX AI React Dashboard

This frontend is a real mode-separated dashboard, not a static demo.

## Modes

- **PAPER**: deterministic local synthetic OHLCV. No Yahoo Finance/network market-data dependency.
- **MT5 LIVE**: broker-native MetaTrader 5 data only. No Yahoo Finance fallback.
- **Paper Backtest**: runs the existing BacktestEngine on deterministic paper data.
- **MT5 Backtest**: runs the existing BacktestEngine on cached/real MT5 historical data.

## Run

1. Run `Setup React Dashboard.bat`.
2. Run `Run Full React Dashboard.bat`.
3. Open the dashboard at `http://127.0.0.1:5173`.

The React UI talks to `dashboard_api.py`. Trading/strategy/SMC/Risk/Backtest/MT5 core modules are not rewritten; the adapter selects the data source and execution mode.
# Quantum FX AI React Dashboard

This frontend is a real mode-separated dashboard, not a static demo.

## Modes

- **PAPER**: deterministic local synthetic OHLCV. No Yahoo Finance/network market-data dependency.
- **MT5 LIVE**: broker-native MetaTrader 5 data only. No Yahoo Finance fallback.
- **Paper Backtest**: runs the existing BacktestEngine on deterministic paper data.
- **MT5 Backtest**: runs the existing BacktestEngine on cached/real MT5 historical data.

## Run

1. Run `Setup React Dashboard.bat`.
2. Run `Run Full React Dashboard.bat`.
3. Open the dashboard at `http://127.0.0.1:5173`.

The React UI talks to `dashboard_api.py`. Trading/strategy/SMC/Risk/Backtest/MT5 core modules are not rewritten; the adapter selects the data source and execution mode.
