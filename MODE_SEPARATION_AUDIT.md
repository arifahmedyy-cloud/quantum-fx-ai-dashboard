# Mode Separation Audit

## Data routing
- PAPER: `src/services/paper_market_data.py` only.
- MT5: `MT5Broker.get_ohlcv()` only.
- Yahoo/GC=F is no longer used by `app._fetch_live_ohlcv()` and is not a fallback for MT5.
- Existing `DataService` remains only for legacy tooling/tests; it is not in the live trading data path.

## UI
- PAPER and MT5 LIVE are separate modes.
- Paper Backtest and MT5 Backtest are separate buttons/endpoints.
- MT5 buttons never fall back to Yahoo.
- Main strategy, SMC, RiskManager, DecisionEngine, BacktestEngine, and MT5 connector implementations were not rewritten.

## Manual Windows validation still required
- Confirm MT5 terminal is open/login is valid.
- Confirm `XAUUSDm` is the resolved broker symbol.
- Click MT5 LIVE and verify candle timestamps/prices match MT5.
- Click MT5 Backtest after historical MT5 cache is available.
