"""Standalone validation of MLSignalModel on synthetic data.

Real market data (yfinance / MT5) isn't reachable from this sandbox, so
this validates the PIPELINE's correctness, not real trading edge:

  Scenario A (pure random walk): no exploitable signal exists by
  construction. A correctly-built, leak-free pipeline should score
  ~50% test accuracy here. If it scores much higher, that's a red flag
  for look-ahead leakage, not a good model.

  Scenario B (injected momentum): candles are constructed so that
  short-term momentum genuinely predicts the next candle's direction.
  A working pipeline should score meaningfully above 50% here,
  confirming the model can actually learn a real pattern when one
  exists.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.trading.indicators import TechnicalIndicators
from src.trading.ml_signal import MLSignalModel


def make_random_walk(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    vol = rng.uniform(100, 1000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                          "Close": close, "Volume": vol}, index=dates)


def make_momentum_signal(n=2000, seed=2):
    """Construct data where next-bar direction is genuinely predictable
    from recent momentum, so the pipeline has something real to learn.

    Horizon matters here: RSI/MACD/trend-score are smoothed over
    ~14-26 bars, so a signal injected at a 3-bar horizon gets washed
    out by that smoothing (this was tried and correctly failed — see
    CHANGELOG note in the accompanying test report). Using a ~20-bar
    momentum window instead means it's actually visible to the features
    the model is given, which is a fairer test of the pipeline.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    ret = np.zeros(n)
    noise = rng.normal(0, 1.0, n)
    window = 20
    for i in range(window, n):
        momentum = ret[i - window:i].sum()
        # next return biased in the direction of recent slow momentum
        ret[i] = 0.45 * np.sign(momentum) * rng.uniform(0.3, 1.2) + noise[i]
    close = 2000 + np.cumsum(ret)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    vol = rng.uniform(100, 1000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                          "Close": close, "Volume": vol}, index=dates)


def run_scenario(name, df):
    print(f"\n=== {name} ===")
    df_ind = TechnicalIndicators.add_all(df)
    X, y = MLSignalModel.build_features_and_labels(df_ind)
    print(f"Rows after feature/label build: {len(X)} (from {len(df)} raw candles)")

    # Time-ordered split — no shuffling, test set is strictly after train set.
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows (test is strictly later in time)")

    model = MLSignalModel(n_estimators=200, max_depth=4)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    baseline = max(y_test.mean(), 1 - y_test.mean())

    print(f"Test accuracy:     {acc:.4f}")
    print(f"Test ROC-AUC:      {auc:.4f}")
    print(f"Majority-class baseline: {baseline:.4f}  (accuracy you'd get by always guessing the more common label)")
    print(f"Class balance (test): up={y_test.mean():.3f}, down={1 - y_test.mean():.3f}")

    latest = model.predict_latest(df_ind)
    print(f"predict_latest() on most recent candle: {latest}")
    return acc, auc, baseline


if __name__ == "__main__":
    acc_a, auc_a, base_a = run_scenario("Scenario A: Pure random walk (no real signal)", make_random_walk())
    acc_b, auc_b, base_b = run_scenario("Scenario B: Injected momentum signal", make_momentum_signal())

    print("\n=== Interpretation ===")
    print(f"A (random walk):  acc={acc_a:.3f} vs baseline={base_a:.3f} "
          f"-> {'OK: near baseline, no leakage detected' if acc_a - base_a < 0.06 else 'WARNING: suspiciously high, check for look-ahead leakage'}")
    print(f"B (momentum):      acc={acc_b:.3f} vs baseline={base_b:.3f} "
          f"-> {'OK: model learned the injected signal' if acc_b - base_b > 0.03 else 'WARNING: model failed to learn an intentionally-injected signal'}")
