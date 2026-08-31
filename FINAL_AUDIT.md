# Final A-Z UI / Data / Backtest Audit

## What was changed

- The supported launcher is now `Run Dashboard.bat`.
- The old Streamlit launcher `Run Gold Bot.bat` is disabled; it no longer starts the old UI.
- React + FastAPI are the supported dashboard path.
- Paper and MT5 modes use separate API routes.
- Yahoo Finance / `GC=F` has been removed from the production market-data dependency/path.
- Paper mode uses deterministic local synthetic OHLCV.
- MT5 mode uses broker-native MT5 OHLCV/account/positions only.
- Added mode-specific risk and analysis endpoints.
- Fixed mode-specific bot status reporting.
- Removed the fake/static chart fallback; the chart is empty until real paper/MT5 candles arrive.
- Positions/History tab now reads actual history.
- Risk guard status is read from the adapter's real RiskManager state.
- Start/Stop runs through the existing trading loop via `dashboard_controller.py`.
- Paper Backtest and MT5 Backtest remain separate and use the existing `BacktestEngine`.

## Core integrity

The following trading/backtest/MT5 core modules were SHA-256 compared against the uploaded base ZIP and are unchanged:

- Strategy
- SMC
- RiskManager
- DecisionEngine
- MT5 broker connector
- BacktestEngine
- MT5 data cache
- Regime detector
- Indicators
- Correlation/session/calendar guards
- Trailing stop
- Symbol manager

## Runtime verification performed here

- Python compilation: PASS
- Full pytest suite: 141 PASS, 1 FAIL
- The single failure is `GeminiProvider` because this sandbox does not have `google.genai` installed. The project requirements still include `google-genai`; this needs the normal Windows setup to install it.
- Paper API health: PASS
- Paper market data: PASS
- Paper risk endpoint: PASS
- Paper analysis: PASS
- Paper backtest: PASS
- Actual Paper Start -> running state -> Stop flow: PASS
- After Stop, controller state returned `bot_running=False`: PASS
- No Python source imports `yfinance`: PASS
- Core integrity comparison: PASS
- React package installation/build: NOT VERIFIED here because npm registry access timed out in this environment.

## Manual Windows/MT5 validation still required

1. Run `Setup.bat`.
2. Open MT5 and confirm the target symbol, e.g. `XAUUSDm`.
3. Run `Run Dashboard.bat`.
4. Confirm the browser opens at `127.0.0.1:5173`.
5. Select MT5 LIVE and verify CONNECTED + real balance/equity/candles.
6. Run MT5 Analysis.
7. Run MT5 Backtest.
8. Test Start MT5, then Stop Bot.
9. Select PAPER and verify that it does not depend on Yahoo/network market data.
10. Run Paper Backtest and Start Paper/Stop.

## Important

This package intentionally does not add a direct Buy/Sell order endpoint to the React dashboard. Existing bot execution remains inside the existing trading engine/RiskManager/MT5 path.
