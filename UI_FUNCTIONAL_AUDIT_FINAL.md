# Quantum FX AI — UI Functional Integration Audit

## Scope
The existing trading core was intentionally not edited. Changes are limited to the dashboard adapter/controller and React UI wiring.

## Wired actions
- Refresh: GET health/account/positions/market
- Start Bot: POST `/api/bot/start`, launches `dashboard_controller.py`, which initializes the same objects and calls the existing `app.trading_loop()` unchanged.
- Stop Bot: POST `/api/bot/stop`, creates a stop flag consumed by the controller.
- Run Analysis: POST `/api/analysis`, uses broker OHLCV + TechnicalIndicators + SMC + RegimeDetector + DecisionEngine; it does not call `send_order`.
- Run Backtest: POST `/api/backtest`, uses the existing MT5 cache + BacktestEngine + PerformanceReporter. It follows the existing policy requiring broker=mt5.
- Risk: adapter exposes `guard_active` without changing RiskManager.
- Account/positions/market: broker-backed, not hard-coded demo values.

## Core integrity
The following core files were not edited by this integration:
- app.py
- src/trading/decision_engine.py
- src/trading/smc.py
- src/trading/risk_manager.py
- src/backtesting/backtest_engine.py
- src/trading/broker_connector.py

## Validation performed
- Python AST/compile: PASS for dashboard_api.py and dashboard_controller.py.
- Vite config added with React plugin registration.
- `npm install` could not complete in the sandbox because the package registry request timed out; therefore a production React build could not be executed here.
- Real MT5 terminal execution still requires the user's Windows PC/MT5 environment.

## Important behavior
The dashboard does not auto-start the bot. The Start Bot button is an explicit action. Stop is explicit and uses a controller stop signal.

## Remaining manual validation on Windows
1. Run Setup / install dependencies.
2. Start the dashboard.
3. Confirm `/api/health` shows the configured broker.
4. Click Start Bot and confirm the controller process appears and MT5 receives normal bot cycles.
5. Click Stop Bot and confirm the controller exits and no new orders are generated.
6. Run Analysis and confirm a signal payload appears without creating an order.
7. Run Backtest with MT5 historical data and confirm metrics are returned.
