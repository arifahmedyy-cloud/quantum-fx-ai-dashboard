# Test Report — Backtest Production Hardening

## Executed in this environment

### Focused critical suite
`37 passed`

Covered:
- Backtest execution correctness
- Risk Manager
- Decision Engine
- Regime Detector
- New max-risk hard cap
- New daily-loss baseline
- Default Regime backtest path
- Decision log generation

### Broader suite excluding unavailable third-party/UI dependencies
`123 passed`

Excluded:
- `tests/test_candle_caching.py` — imports Streamlit; Streamlit is not installed in this sandbox.
- `tests/test_ai_providers.py` — Gemini provider test requires `google-genai`; that package is not installed in this sandbox.

## Compile check
`python -m compileall -q src app.py` — PASS

## Synthetic end-to-end backtest
PASS:
- 260 synthetic M15 candles
- default Regime Detector path
- chronological execution
- decision logging
- equity/P&L generation
- no runtime exception

## Environment-blocked validation

Not performed here:
- Real MetaTrader 5 terminal connectivity
- Real broker symbol specifications fetched from a live MT5 terminal
- Streamlit UI runtime
- Gemini SDK provider runtime
- User's real XAUUSDm historical dataset

These must be validated on the Windows MT5 machine before real-money deployment.

## Important limitation

This package is code/test hardened for backtesting, but it cannot honestly be labelled fully production/live-ready until real MT5 forward/demo validation has been completed on the user's Windows terminal and broker account.
