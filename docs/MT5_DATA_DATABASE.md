# MT5 Historical Data Database

The backtest UI uses a persistent SQLite database at `data/mt5_database/market_data.db`.

## Behavior

1. First backtest/sync for a symbol + timeframe downloads the requested closed-candle range from the connected MT5 terminal.
2. Candles are normalized, validated, deduplicated and stored with a unique `(symbol, timeframe, timestamp)` key.
3. Later backtests read the database and only ask MT5 for an older extension, a newer tail, or a short internal data gap.
4. The currently forming candle is excluded by the UI before synchronization.
5. Use **Force refresh** only when the broker has corrected historical prices.
6. The UI's **Sync / Update MT5 Data** button can update the database without running a backtest.

## Windows / MT5 requirement

The actual download requires MetaTrader 5 to be installed and connected on the Windows machine. The application does not silently substitute Yahoo/GC=F for broker-native backtests.

## CLI

`tools/export_mt5_data.py` writes to the same database. An optional `--output` also creates a CSV export. `--force-refresh` re-fetches the selected range.

The database is intentionally ignored by Git and should remain local to the trading machine.
