from datetime import datetime
import pandas as pd

from src.backtesting.mt5_data_cache import normalize_ohlcv, MT5DataCache


def sample():
    return pd.DataFrame({
        "Date": ["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:15"],
        "Open": [1, 2, 2], "High": [2, 3, 3], "Low": [0.5, 1.5, 1.5],
        "Close": [1.5, 2.5, 2.5], "Volume": [10, 20, 21]
    })


def test_normalize_deduplicates_and_sorts():
    df = normalize_ohlcv(sample())
    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 2
    assert df.iloc[-1]["Volume"] == 21


def test_cache_uses_cached_range_without_download(tmp_path):
    class Broker:
        def __init__(self): self.calls = 0
        def load_historical_data(self, *args, **kwargs):
            self.calls += 1
            return sample().iloc[:2].copy()
    broker = Broker()
    cache = MT5DataCache(tmp_path)
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 1, 0, 15)
    df, first = cache.update(broker, "XAUUSD", "M15", start, end)
    assert first is True and broker.calls == 1
    df2, first2 = cache.update(broker, "XAUUSD", "M15", start, end)
    assert first2 is False
    assert len(df2) == len(df)


def test_incremental_update_only_fetches_new_tail(tmp_path):
    class Broker:
        def __init__(self): self.calls = []
        def load_historical_data(self, start, end, **kwargs):
            self.calls.append((start, end, kwargs))
            return pd.DataFrame({
                "Date": [start, end],
                "Open": [1.0, 2.0], "High": [1.2, 2.2],
                "Low": [0.8, 1.8], "Close": [1.1, 2.1], "Volume": [10, 20],
            })

    broker = Broker()
    cache = MT5DataCache(tmp_path)
    s1 = datetime(2026, 1, 1)
    e1 = datetime(2026, 1, 1, 0, 15)
    cache.update(broker, "XAUUSDm", "M15", s1, e1)
    calls_after_first = len(broker.calls)
    e2 = datetime(2026, 1, 1, 0, 30)
    df, initial = cache.update(broker, "XAUUSDm", "M15", s1, e2)
    assert initial is False
    assert len(broker.calls) == calls_after_first + 1
    assert len(df) >= 2
    assert cache.status("XAUUSDm", "M15")["rows"] >= 2


def test_cache_status_points_to_sqlite_database(tmp_path):
    cache = MT5DataCache(tmp_path)
    status = cache.status("XAUUSD", "M15")
    assert status["database"].endswith("market_data.db")
    assert status["rows"] == 0
