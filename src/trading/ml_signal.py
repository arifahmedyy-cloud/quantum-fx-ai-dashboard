"""XGBoost-based directional signal model.

This is an OPTIONAL confluence input, not a replacement for the existing
rule-based regime/decision engine. It trains a gradient-boosted classifier
on the same technical indicators `TechnicalIndicators.add_all()` already
computes, to predict the probability that the *next* candle closes higher
than the current one.

Design choices (why they matter for a trading bot specifically):

- Time-ordered train/test split only. Never shuffle. Shuffling a time
  series before splitting leaks future information into training and
  makes backtests lie.
- The label for row i is built from row i+1's close, then the last row
  (no future candle yet) is dropped — no look-ahead.
- Only indicator columns are used as features (never Date/Open/High/Low/
  Close/Volume directly), so the model can't just memorize price levels
  that won't repeat.
- Returns *probabilities*, not hard buy/sell calls. The intended use is
  as one more confluence factor inside `decision_engine.py`, combined
  with the existing regime/SMC/risk checks — not as a standalone trigger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)

FEATURE_COLUMNS = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "BB_Position",
    "ATR_Pct", "ADX", "DI_Plus", "DI_Minus",
    "Volatility", "Trend_Score",
]


@dataclass
class MLSignalResult:
    """Output of a single prediction."""
    prob_up: float          # model's probability the next candle closes higher
    confidence: float       # abs(prob_up - 0.5) * 2, i.e. 0 (coin flip) to 1 (certain)
    n_features_used: int


class MLSignalModel:
    """Thin wrapper around an XGBoost classifier for directional prediction.

    Not fit for use until `.fit()` has been called (or `.load()`, if a
    persistence layer is added later). Calling `.predict_proba()` before
    fitting raises `RuntimeError` rather than silently returning garbage.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 4,
                 learning_rate: float = 0.05, random_state: int = 42):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "xgboost is not installed. Run `pip install xgboost` "
                "(and add it to requirements.txt) to use MLSignalModel."
            ) from e

        self._xgb = xgb
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="logloss",
        )
        self._fitted = False
        self.feature_columns_: list[str] = []

    @staticmethod
    def build_features_and_labels(df_ind: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """From an indicator-enriched OHLC DataFrame (output of
        `TechnicalIndicators.add_all`), build (X, y):

        - X: only the indicator columns present in FEATURE_COLUMNS
        - y: 1 if next candle's Close > this candle's Close, else 0

        Rows with any NaN feature (e.g. the warm-up period) or without a
        next candle (the final row) are dropped.
        """
        available = [c for c in FEATURE_COLUMNS if c in df_ind.columns]
        if not available:
            raise ValueError(
                "None of the expected feature columns are present. "
                "Did you run TechnicalIndicators.add_all() first?"
            )

        X = df_ind[available].copy()
        y = (df_ind["Close"].shift(-1) > df_ind["Close"]).astype(int)
        y.iloc[-1] = np.nan  # no next candle to label the last row with

        combined = pd.concat([X, y.rename("_label")], axis=1).dropna()
        return combined[available], combined["_label"].astype(int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if len(X) == 0:
            raise ValueError("No training rows after dropping NaNs — check warm-up period / data length.")
        self.feature_columns_ = list(X.columns)
        self.model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=int))
        self._fitted = True
        log.info("MLSignalModel fitted on %d rows, %d features.", len(X), len(self.feature_columns_))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MLSignalModel.predict_proba() called before .fit(). "
                                "This is a bug in the caller, not a case to silently handle.")
        X_aligned = X[self.feature_columns_]
        return self.model.predict_proba(X_aligned.to_numpy(dtype=float))[:, 1]

    def predict_latest(self, df_ind: pd.DataFrame) -> Optional[MLSignalResult]:
        """Predict for the most recent fully-formed row only — this is the
        call site `decision_engine.py` would use each cycle."""
        available = [c for c in self.feature_columns_ if c in df_ind.columns]
        if len(available) != len(self.feature_columns_):
            log.warning("Missing feature columns at inference time: %s",
                        set(self.feature_columns_) - set(available))
            return None
        row = df_ind[self.feature_columns_].iloc[[-1]]
        if row.isna().any(axis=None):
            return None  # not enough warm-up history yet for this row
        prob_up = float(self.predict_proba(row)[0])
        return MLSignalResult(
            prob_up=prob_up,
            confidence=abs(prob_up - 0.5) * 2,
            n_features_used=len(self.feature_columns_),
        )

    def save(self, path: str) -> None:
        """Persist the fitted model + feature column order to disk.

        Added as part of the audit: MLSignalModel previously had no
        persistence layer at all (its own docstring said "or .load(), if
        a persistence layer is added later" — it never was), so the model
        could only ever exist in-memory for the lifetime of one Python
        process. That's why `ml_confidence` was hardcoded to None in
        app.py: there was no way to have a trained model ready when the
        live trading loop starts without retraining from scratch inside
        the UI thread every time.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted MLSignalModel.")
        import json
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save_model(path)
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump({"feature_columns": self.feature_columns_}, f)
        log.info("MLSignalModel saved to %s (+ %s)", path, meta_path)

    @classmethod
    def load(cls, path: str) -> "MLSignalModel":
        """Load a model previously saved with .save()."""
        import json
        import os
        meta_path = path + ".meta.json"
        if not os.path.exists(path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"No saved MLSignalModel found at {path} (+ .meta.json)")
        instance = cls()
        instance.model.load_model(path)
        with open(meta_path) as f:
            meta = json.load(f)
        instance.feature_columns_ = meta["feature_columns"]
        instance._fitted = True
        log.info("MLSignalModel loaded from %s (%d features)", path, len(instance.feature_columns_))
        return instance
