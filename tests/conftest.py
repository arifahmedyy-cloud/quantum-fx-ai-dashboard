import sys
import types
import numpy as np
import pandas as pd
import pytest

# app.py / legacy UI helpers import streamlit, but the supported runtime is
# React + FastAPI. Tests only need the session-state container.
if "streamlit" not in sys.modules:
    st = types.ModuleType("streamlit")
    st.session_state = {}
    sys.modules["streamlit"] = st

from src.config import RiskConfig

@pytest.fixture
def risk_config():
    return RiskConfig()

@pytest.fixture
def sample_ohlcv():
    rng = np.random.default_rng(42)
    n = 220
    close = 2400 + np.cumsum(rng.normal(0, 2.0, n))
    open_ = close + rng.normal(0, 0.8, n)
    high = np.maximum(open_, close) + rng.uniform(1, 4, n)
    low = np.minimum(open_, close) - rng.uniform(1, 4, n)
    volume = rng.integers(1000, 10000, n)
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
