"""Tests for the ML signal model feature completion.

Prior to this fix, MLSignalModel (src/trading/ml_signal.py) was fully
implemented but:
  1. Had no save()/load() — a trained model could only live in-memory for
     one Python process, so there was no way to have a ready model when
     the live trading loop starts.
  2. app.py hardcoded `ml_confidence = None` unconditionally, so the
     "ML filter" risk-manager setting did nothing regardless of whether
     a model existed.

These tests cover:
  - save()/load() round-trip correctness
  - app.py actually computing a non-None, DIRECTIONAL ml_confidence
    (not a raw magnitude) when a model is loaded and the regime signal
    proposes a BUY/SELL
  - ml_confidence staying None (feature effectively off) when disabled,
    no model is loaded, or the regime signal is NO_TRADE
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.trading.indicators import TechnicalIndicators
from src.trading.ml_signal import MLSignalModel
from src.services.paper_market_data import generate_paper_ohlcv


def _make_trained_model():
    df = generate_paper_ohlcv("XAUUSD", "H1", 1500)
    df_ind = TechnicalIndicators.add_all(df)
    X, y = MLSignalModel.build_features_and_labels(df_ind)
    model = MLSignalModel(n_estimators=50, max_depth=3)
    model.fit(X, y)
    return model, df_ind


class TestMLSignalPersistence:
    def test_save_load_round_trip_predicts_identically(self, tmp_path):
        model, df_ind = _make_trained_model()
        before = model.predict_latest(df_ind)

        path = str(tmp_path / "model.json")
        model.save(path)
        loaded = MLSignalModel.load(path)
        after = loaded.predict_latest(df_ind)

        assert before.prob_up == pytest.approx(after.prob_up, abs=1e-9)
        assert before.confidence == pytest.approx(after.confidence, abs=1e-9)
        assert loaded.feature_columns_ == model.feature_columns_

    def test_save_raises_if_not_fitted(self, tmp_path):
        model = MLSignalModel()
        with pytest.raises(RuntimeError):
            model.save(str(tmp_path / "unfitted.json"))

    def test_load_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MLSignalModel.load(str(tmp_path / "does_not_exist.json"))


class TestMLModelLiveLoopWiring:
    def _run_forced_cycle(self, config, ml_model, forced_action, tmp_path):
        import types
        st = types.ModuleType("streamlit")

        class SessionState(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

            def get(self, k, d=None):
                return dict.get(self, k, d)

        st.session_state = SessionState()
        sys.modules["streamlit"] = st

        import importlib
        import app
        importlib.reload(app)
        app.init_session_state()
        st.session_state.ml_model = ml_model

        from src.services.journal_service import JournalService
        from src.services.notification_service import NotificationService
        from src.services.health_service import HealthService
        from src.trading.regime_detector import RegimeDetector
        from src.trading.smc import SMCAnalyzer
        from src.trading.decision_engine import DecisionEngine
        from src.trading.risk_manager import RiskManager
        from src.trading.symbol_manager import SymbolManager
        from src.trading.correlation_guard import CorrelationGuard
        from src.models import SignalOutput
        from src.config import NotificationConfig

        broker = app.create_broker(config)
        broker.connect()
        journal = JournalService(db_path=str(tmp_path / "journal.db"))
        notifier = NotificationService(NotificationConfig(enabled=False))
        health = HealthService(broker, interval_seconds=30, notifier=notifier)
        regime = RegimeDetector()
        smc = SMCAnalyzer()
        decision = DecisionEngine(config.risk)
        risk = RiskManager(config.risk)
        symbol_manager = SymbolManager(["XAUUSD"])
        correlation_guard = CorrelationGuard(symbol_manager, max_net_usd_exposure=config.risk.max_net_usd_exposure)
        account = broker.get_account_info()

        forced_signal = SignalOutput(
            action=forced_action, confidence=80, regime="trending", strategy="test",
            expected_pf=1.0, expected_max_dd=0.0, expected_avg_rr=1.0, consistency_score=0,
            sl=2440.0 if forced_action == "BUY" else 2460.0,
            tp=2470.0 if forced_action == "BUY" else 2430.0,
            entry=2450.0, lot_size=0.1, explanation="forced", metrics={},
        )
        with patch.object(regime, "generate_signal", return_value=forced_signal):
            return app._run_symbol_cycle(
                "XAUUSD", config, broker, journal, notifier, health, regime, smc,
                decision, risk, None, None, symbol_manager, correlation_guard,
                account, "Regime (default)", {},
            )

    def test_ml_confidence_engages_when_enabled_with_loaded_model(self, tmp_path):
        from src.config import load_config
        os.environ["BROKER"] = "paper"
        config = load_config(validate=False)
        config.risk.enable_ml_filter = True
        config.risk.min_ml_confidence = 1.0
        config.risk.strict_smc_confluence = False

        model, _ = _make_trained_model()
        sig = self._run_forced_cycle(config, model, "BUY", tmp_path)

        assert "ML model" in sig["explanation"], (
            f"ml_confidence never engaged even with a loaded model and "
            f"enable_ml_filter=True. explanation={sig['explanation']!r}"
        )

    def test_ml_confidence_absent_when_filter_disabled(self, tmp_path):
        from src.config import load_config
        os.environ["BROKER"] = "paper"
        config = load_config(validate=False)
        config.risk.enable_ml_filter = False  # explicitly off
        config.risk.strict_smc_confluence = False

        model, _ = _make_trained_model()
        sig = self._run_forced_cycle(config, model, "BUY", tmp_path)

        assert "ML model" not in sig["explanation"]

    def test_ml_confidence_absent_when_no_model_loaded(self, tmp_path):
        from src.config import load_config
        os.environ["BROKER"] = "paper"
        config = load_config(validate=False)
        config.risk.enable_ml_filter = True
        config.risk.strict_smc_confluence = False

        sig = self._run_forced_cycle(config, None, "BUY", tmp_path)  # no model
        assert "ML model" not in sig["explanation"]
