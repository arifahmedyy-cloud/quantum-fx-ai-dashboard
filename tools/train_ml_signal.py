"""Train the optional XGBoost directional-confidence model (ml_signal.py)
and save it for the live trading loop to load automatically.

This is what actually makes the "ML filter" risk setting do something —
previously MLSignalModel was fully implemented but had no training script,
no persistence (save/load), and the live loop hardcoded ml_confidence to
None regardless of the setting.

Usage:
    python tools/train_ml_signal.py --broker paper --symbol XAUUSD --timeframe H1 --bars 5000
    python tools/train_ml_signal.py --broker mt5 --symbol XAUUSDm --timeframe H1 --bars 20000

The trained model is written to models/ml_signal_model.json (+ a matching
.meta.json). app.py's init_session_state() automatically tries to load a
model from that path at startup — nothing else needs to be configured
once a model has been trained here (beyond turning on "ML filter" in the
dashboard's Risk Manager settings).

IMPORTANT — this deliberately REFUSES to save a model that doesn't beat a
naive majority-class baseline by a meaningful margin on held-out
(strictly later in time, never shuffled) data. A model that can't beat
"always guess up" (or "always guess down") isn't a useful confluence
filter and would just add noise/false confidence to the live decision —
better to run with no ML filter (the previous, honest default) than a
model masquerading as informative.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import accuracy_score, roc_auc_score

from src.logger import get_logger
from src.trading.indicators import TechnicalIndicators
from src.trading.ml_signal import MLSignalModel

log = get_logger(__name__)

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "ml_signal_model.json")
MIN_ACCURACY_EDGE_OVER_BASELINE = 0.02  # must beat majority-class baseline by at least this much


def fetch_training_data(broker_kind: str, symbol: str, timeframe: str, bars: int):
    if broker_kind == "paper":
        from src.services.paper_market_data import generate_paper_ohlcv
        print(f"Generating {bars} bars of PAPER SYNTHETIC {symbol} {timeframe} data.")
        print("NOTE: a model trained on synthetic data learns the synthetic")
        print("generator's own harmonic pattern, not real market structure —")
        print("this is useful to validate the pipeline works end-to-end, but")
        print("train on real MT5 history (--broker mt5) before trusting this")
        print("model with live paper/MT5 trading.\n")
        return generate_paper_ohlcv(symbol, timeframe, bars)
    elif broker_kind == "mt5":
        from src.trading.broker_connector import MT5Broker
        from src.config import load_config
        config = load_config(validate=False)
        broker = MT5Broker(
            login=config.mt5.login, password=config.mt5.password, server=config.mt5.server,
            leverage=config.mt5.leverage,
        )
        if not broker.connect():
            raise RuntimeError(
                "Could not connect to MT5. Make sure the MT5 terminal is running, "
                "logged in, and your .env has valid MT5_LOGIN/MT5_PASSWORD/MT5_SERVER."
            )
        print(f"Fetching {bars} bars of REAL MT5 {symbol} {timeframe} history.")
        df = broker.get_ohlcv(symbol=symbol, timeframe=timeframe, bars=bars)
        if df.empty:
            raise RuntimeError(
                f"MT5 returned no data for {symbol} {timeframe}. Check the symbol "
                f"name matches what your broker actually exposes (e.g. XAUUSDm)."
            )
        return df
    else:
        raise ValueError(f"Unknown broker kind: {broker_kind!r} (expected 'paper' or 'mt5')")


def train_and_validate(df, n_estimators: int, max_depth: int, learning_rate: float, test_frac: float):
    df_ind = TechnicalIndicators.add_all(df)
    X, y = MLSignalModel.build_features_and_labels(df_ind)
    print(f"Rows after indicator warm-up + feature/label build: {len(X)} (from {len(df)} raw candles)")
    if len(X) < 200:
        raise RuntimeError(
            f"Only {len(X)} usable rows after warm-up/labeling — need at least a "
            f"few hundred for a meaningful train/test split. Fetch more bars."
        )

    # Time-ordered split only — never shuffle a time series before splitting.
    split = int(len(X) * (1 - test_frac))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows (test is strictly later in time)")

    model = MLSignalModel(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    baseline = max(y_test.mean(), 1 - y_test.mean())

    print(f"\nTest accuracy:            {acc:.4f}")
    print(f"Test ROC-AUC:             {auc:.4f}")
    print(f"Majority-class baseline:  {baseline:.4f}  (accuracy from always guessing the more common label)")
    print(f"Edge over baseline:       {acc - baseline:+.4f}")

    # Refit on ALL available data (train+test) for the final saved model —
    # standard practice once validation has already told us whether the
    # pipeline finds a real edge; the held-out split above is only for
    # measuring that edge honestly, not for producing the deployed model.
    final_model = MLSignalModel(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    final_model.fit(X, y)

    return final_model, acc, auc, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--broker", choices=["paper", "mt5"], default="paper",
                         help="Data source: 'paper' (synthetic, always available) or 'mt5' (real broker history, requires a running MT5 terminal).")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="H1", choices=["M5", "M15", "M30", "H1", "H4"])
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--test-frac", type=float, default=0.3, help="Fraction of data held out for testing (time-ordered, not shuffled).")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--force", action="store_true",
                         help="Save the model even if it doesn't beat the baseline accuracy threshold.")
    args = parser.parse_args()

    df = fetch_training_data(args.broker, args.symbol, args.timeframe, args.bars)
    model, acc, auc, baseline = train_and_validate(
        df, args.n_estimators, args.max_depth, args.learning_rate, args.test_frac)

    edge = acc - baseline
    if edge < MIN_ACCURACY_EDGE_OVER_BASELINE and not args.force:
        print(f"\nREFUSING TO SAVE: edge over baseline ({edge:+.4f}) is below the "
              f"minimum ({MIN_ACCURACY_EDGE_OVER_BASELINE:+.4f}) required to trust this "
              f"model as a confluence filter. This usually means there's no learnable "
              f"edge in this data/feature set at this horizon — that's a legitimate, "
              f"common outcome, not necessarily a bug. Options:")
        print("  - Try a different timeframe or more bars")
        print("  - Train on real MT5 history instead of paper synthetic data")
        print("  - Re-run with --force to save anyway (not recommended for live use)")
        sys.exit(1)

    model.save(args.out)
    print(f"\nModel saved to {args.out} (+ {args.out}.meta.json)")
    print("The live trading loop will pick this up automatically on next start.")
    print("Turn on \"ML filter\" in the dashboard's Risk Manager settings to use it.")


if __name__ == "__main__":
    main()
