"""Export broker-native MT5 candles to the same local cache used by the UI.
Run on Windows with MetaTrader 5 terminal installed/logged in.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.trading.broker_connector import MT5Broker
from src.backtesting.mt5_data_cache import MT5DataCache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="M15", choices=["M1","M5","M15","M30","H1","H4","D1"])
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--output", default=None, help="Optional CSV export in addition to saving the SQLite database")
    ap.add_argument("--force-refresh", action="store_true", help="Re-fetch the selected range from MT5 and update cached candles")
    ap.add_argument("--status", action="store_true", help="Print the current database status after syncing")
    args = ap.parse_args()

    cfg = load_config(validate=False)
    broker = MT5Broker(login=cfg.mt5.login, password=cfg.mt5.password, server=cfg.mt5.server,
                       symbol_candidates=cfg.mt5.symbol_candidates, leverage=cfg.mt5.leverage)
    if not broker.connect():
        raise SystemExit("MT5 connection failed. Open MT5, log in, and verify the broker server.")
    try:
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)
        cache = MT5DataCache()
        df, first = cache.update(broker, args.symbol, args.timeframe, start, end, force_refresh=args.force_refresh)
        if df.empty:
            raise SystemExit("No historical candles returned by MT5.")
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"Saved {len(df)} candles to {out}")
        else:
            print(f"Database synced: {len(df)} candles in requested range for {args.symbol} {args.timeframe}; initial_download={first}; force_refresh={args.force_refresh}")
        if args.status:
            print(cache.status(args.symbol, args.timeframe))
    finally:
        broker.disconnect()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
