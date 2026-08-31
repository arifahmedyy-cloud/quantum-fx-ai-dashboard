"""Broker-native market data service.

Yahoo Finance is intentionally not included. Production trading/backtesting
must use MT5 broker data or the deterministic paper market-data service.
"""
from __future__ import annotations

import pandas as pd
from src.logger import get_logger
from src.exceptions import DataError

log = get_logger(__name__)


class DataService:
    """Standardize and fetch broker-native OHLCV data."""

    def __init__(self, symbol: str = "XAUUSDm") -> None:
        self.symbol = symbol

    def fetch_from_mt5(self, broker, period: str = "5d", interval: str = "15m", bars: int = 500) -> pd.DataFrame:
        try:
            df = broker.get_ohlcv(symbol=self.symbol, timeframe=interval, bars=bars)
        except Exception as exc:
            raise DataError(f"MT5 data fetch failed: {exc}") from exc
        if df is None or df.empty:
            raise DataError("MT5 returned empty data")
        return self._clean_dataframe(df)

    @staticmethod
    def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                " ".join(c).strip() if len(c) > 1 and str(c[1]) not in ("nan", "NaN") else str(c[0])
                for c in df.columns.values
            ]
        if "Date" not in df.columns and "Time" not in df.columns:
            df = df.reset_index()
        col_map = {}
        for c in df.columns:
            cs = str(c).lower()
            if "date" in cs or "time" in cs:
                col_map[c] = "Date"
            elif "open" in cs:
                col_map[c] = "Open"
            elif "high" in cs:
                col_map[c] = "High"
            elif "low" in cs:
                col_map[c] = "Low"
            elif "close" in cs:
                col_map[c] = "Close"
            elif "volume" in cs:
                col_map[c] = "Volume"
        df = df.rename(columns=col_map)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
