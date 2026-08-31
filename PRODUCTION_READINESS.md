# Production Readiness Gate

## Current status
**BACKTEST READY / DEMO VALIDATION REQUIRED**

## Passed
- Regime Detector is active in the default backtest path.
- Strategy and SMC feed Decision Engine.
- Risk Manager remains the final risk gate.
- Max risk has an explicit hard ceiling.
- Daily loss uses the historical day's starting balance.
- Next-candle entry model is tested.
- OHLC SL/TP behavior is tested.
- Same-candle policy is explicit.
- Historical timestamps are causal/UTC-normalized.
- Decision logs are generated.
- SQLite incremental data architecture is preserved.
- Critical Python tests pass in the available environment.

## Not yet proven in this environment
- Live MT5 connectivity/reconnect behavior.
- Real XAUUSDm broker specification retrieval.
- Full Streamlit runtime.
- Gemini SDK runtime.
- Real 903 MB MT5 dataset end-to-end performance.
- Demo/forward trading behavior under live spread/slippage/latency.

## Required before real money
1. Install all requirements on Windows.
2. Connect the MT5 terminal to the broker.
3. Sync XAUUSDm data into SQLite.
4. Run the M15 backtest using the real cached data.
5. Run walk-forward/out-of-sample validation.
6. Run MT5 demo/forward testing.
7. Verify order rejection, reconnect, spread, slippage, SL/TP and emergency shutdown.
8. Only then consider a small real-money pilot.

No profitability guarantee is implied by this readiness status.
