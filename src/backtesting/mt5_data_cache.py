"""Persistent MT5 historical-data database with incremental synchronization.

The database is the source of truth for backtests.  The first request downloads
only the requested history from MT5 and stores it in SQLite.  Later requests
reuse the stored candles and download only missing/extended ranges.  An
explicit ``force_refresh`` can be used when the broker has corrected history.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)

REQUIRED = ["Date", "Open", "High", "Low", "Close", "Volume"]

_TIMEFRAME_MINUTES = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M12": 12,
    "M15": 15,
    "M20": 20,
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H3": 180,
    "H4": 240,
    "H6": 360,
    "H8": 480,
    "H12": 720,
    "D1": 1440,
    "W1": 10080,
}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED)
    out = df.copy()
    rename = {c: c.title() for c in out.columns}
    rename.update({
        "time": "Date", "datetime": "Date", "date": "Date",
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "tick_volume": "Volume", "volume": "Volume",
    })
    out = out.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Historical data missing required columns: {missing}")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    for c in REQUIRED[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=REQUIRED).copy()
    out = out[(out["High"] >= out[["Open", "Close"]].max(axis=1)) &
              (out["Low"] <= out[["Open", "Close"]].min(axis=1)) &
              (out["High"] >= out["Low"]) & (out["Close"] > 0)]
    out = out.sort_values("Date").drop_duplicates("Date", keep="last")
    return out[REQUIRED].reset_index(drop=True)


def timeframe_delta(timeframe: str) -> timedelta:
    key = str(timeframe).upper()
    if key not in _TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported MT5 timeframe for cache: {timeframe}")
    return timedelta(minutes=_TIMEFRAME_MINUTES[key])


def _safe_key(symbol: str, timeframe: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in f"{symbol}_{timeframe}")


class MT5DataCache:
    """SQLite-backed persistent cache for broker-native OHLCV candles.

    ``root`` may be a directory.  The database is stored as
    ``root/market_data.db``.  The public API intentionally remains compatible
    with the previous CSV cache so the rest of the bot does not need to know
    how the data is persisted.
    """

    def __init__(self, root: str | Path = "data/mt5_database") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "market_data.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_candles_lookup
                    ON candles(symbol, timeframe, timestamp);
                CREATE TABLE IF NOT EXISTS sync_metadata (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    first_timestamp TEXT,
                    last_timestamp TEXT,
                    rows INTEGER NOT NULL DEFAULT 0,
                    last_sync_utc TEXT,
                    PRIMARY KEY(symbol, timeframe)
                );
                """
            )

    def path(self, symbol: str, timeframe: str) -> Path:
        """Compatibility helper; the persistent store is SQLite, not one CSV per symbol."""
        return self.db_path

    def _bounds(self, symbol: str, timeframe: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
        if not row or row[0] is None:
            return None, None
        return pd.Timestamp(row[0]), pd.Timestamp(row[1])

    def load(self, symbol: str, timeframe: str, start: Optional[datetime] = None,
             end: Optional[datetime] = None) -> pd.DataFrame:
        query = (
            "SELECT timestamp AS Date, open AS Open, high AS High, low AS Low, "
            "close AS Close, volume AS Volume FROM candles WHERE symbol=? AND timeframe=?"
        )
        params: list[Any] = [symbol, timeframe]
        if start is not None:
            query += " AND timestamp >= ?"
            params.append(pd.Timestamp(start).tz_localize(None).isoformat())
        if end is not None:
            query += " AND timestamp <= ?"
            params.append(pd.Timestamp(end).tz_localize(None).isoformat())
        query += " ORDER BY timestamp ASC"
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return normalize_ohlcv(df)

    def save(self, symbol: str, timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
        clean = normalize_ohlcv(df)
        if clean.empty:
            return clean
        rows = [
            (symbol, timeframe, row.Date.isoformat(), float(row.Open), float(row.High),
             float(row.Low), float(row.Close), float(row.Volume))
            for row in clean.itertuples(index=False)
        ]
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO candles(symbol,timeframe,timestamp,open,high,low,close,volume)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol,timeframe,timestamp) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume""",
                rows,
            )
            bounds = conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM candles WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
            first, last, total = bounds[0], bounds[1], int(bounds[2])
            conn.execute(
                """INSERT INTO sync_metadata(symbol,timeframe,first_timestamp,last_timestamp,rows,last_sync_utc)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(symbol,timeframe) DO UPDATE SET
                     first_timestamp=excluded.first_timestamp,
                     last_timestamp=excluded.last_timestamp,
                     rows=excluded.rows,
                     last_sync_utc=excluded.last_sync_utc""",
                (symbol, timeframe, first, last, total, datetime.now(timezone.utc).isoformat()),
            )
        return clean

    def _internal_gaps(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Return meaningful gaps inside stored coverage.

        Weekends/market closures naturally create gaps.  Only gaps of more than
        two expected candles are returned, avoiding unnecessary MT5 requests.
        """
        df = self.load(symbol, timeframe, start, end)
        if len(df) < 2:
            return []
        expected = timeframe_delta(timeframe)
        deltas = df["Date"].diff()
        gaps: list[Tuple[pd.Timestamp, pd.Timestamp]] = []
        # Do not repeatedly ask MT5 to fill normal weekend/session closures.
        # Only short intraday gaps are treated as missing data automatically.
        max_gap = max(expected * 3, timedelta(hours=6))
        for i in deltas[deltas > expected * 2].index:
            gap_size = deltas.loc[i] - expected
            if gap_size <= max_gap:
                gaps.append((df.loc[i - 1, "Date"] + expected, df.loc[i, "Date"] - expected))
        return gaps

    def update(self, broker: Any, symbol: str, timeframe: str,
               start: datetime, end: datetime, force_refresh: bool = False) -> Tuple[pd.DataFrame, bool]:
        """Synchronize the requested range and return cached data.

        Returns ``(data, initial_download)``.  No broker call is made when the
        requested range is already covered, unless ``force_refresh`` is True.
        """
        if start >= end:
            raise ValueError("start must be before end")
        timeframe_delta(timeframe)  # validate early

        if force_refresh:
            fetched = broker.load_historical_data(start, end, symbol=symbol, timeframe=timeframe)
            self.save(symbol, timeframe, fetched)
            return self.load(symbol, timeframe, start, end), False

        first, last = self._bounds(symbol, timeframe)
        initial_download = first is None
        fetched_parts: list[pd.DataFrame] = []
        step = timeframe_delta(timeframe)

        if first is None:
            fetched_parts.append(broker.load_historical_data(start, end, symbol=symbol, timeframe=timeframe))
        else:
            if start < first:
                older_end = min(end, first + step)
                fetched_parts.append(broker.load_historical_data(start, older_end, symbol=symbol, timeframe=timeframe))
            if end > last:
                newer_start = max(start, last - step)
                fetched_parts.append(broker.load_historical_data(newer_start, end, symbol=symbol, timeframe=timeframe))
            for gap_start, gap_end in self._internal_gaps(symbol, timeframe, start, end):
                if gap_start < gap_end:
                    fetched_parts.append(broker.load_historical_data(gap_start, gap_end, symbol=symbol, timeframe=timeframe))

        if fetched_parts:
            combined = pd.concat([part for part in fetched_parts if part is not None and not part.empty], ignore_index=True)
            if not combined.empty:
                self.save(symbol, timeframe, combined)

        return self.load(symbol, timeframe, start, end), initial_download

    def status(self, symbol: str, timeframe: str) -> dict[str, Any]:
        first, last = self._bounds(symbol, timeframe)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
            ).fetchone()[0]
            meta = conn.execute(
                "SELECT last_sync_utc FROM sync_metadata WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
        return {
            "database": str(self.db_path),
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": int(rows),
            "first": first.isoformat() if first is not None else None,
            "last": last.isoformat() if last is not None else None,
            "last_sync_utc": meta[0] if meta else None,
        }
