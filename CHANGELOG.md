# Changelog — Backtest Production Hardening

## 2026-08-13
- Added `Regime (default)` as the primary backtest strategy selection.
- Wired the default backtest path to `RegimeDetector` instead of silently selecting a standalone strategy.
- Added strict, configurable Regime/SMC confluence gating (`RISK_STRICT_SMC_CONFLUENCE`, default `true`).
- Added an explicit `max_risk_pct` hard ceiling in `RiskManager.validate_signal()`.
- Corrected daily-loss percentage baseline to use the historical day's starting balance rather than lifetime peak balance.
- Added historical backtest decision logging for accepted and rejected signals.
- Exposed backtest decision logs and execution/risk assumptions in the dashboard.
- Made AI Reviewer status explicit for historical backtests; external/live AI review is not invoked per historical candle.
- Replaced deprecated Streamlit `use_container_width` calls with `width="stretch"`.
- Added regression tests covering default Regime execution, decision logging, max-risk ceiling, and daily-loss baseline.
