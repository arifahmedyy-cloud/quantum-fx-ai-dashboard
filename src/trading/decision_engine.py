"""Decision engine that gates signals through confluence checks.

Combines regime signal, SMC bias, and optional ML filter into a final decision.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.logger import get_logger
from src.models import Decision, SignalOutput, SMCResult
from src.config import RiskConfig

log = get_logger(__name__)


class DecisionEngine:
    """Gates trading signals through multi-layer confluence."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._last_decision: Optional[Decision] = None

    def decide(
        self,
        regime_signal: SignalOutput,
        smc: SMCResult,
        ml_confidence: Optional[float] = None,
        ai_reviewer_confidence: Optional[float] = None,
        current_price: float = 0.0,
    ) -> Decision:
        """Make final trading decision.

        Args:
            regime_signal: Signal from regime detector.
            smc: SMC analysis result.
            ml_confidence: Optional confidence from a trained ML model
                (0-100). No such model exists in this codebase yet (V7 on
                the roadmap) — this parameter is reserved for it.
            ai_reviewer_confidence: Optional independent confidence from an
                LLM trade reviewer (Claude/Gemini/GPT via AIService). This
                was previously passed in AS `ml_confidence`, which mislabeled
                every LLM opinion as a trained model's prediction in logs —
                a real audit-trail problem once an actual ML model gets
                added later, since the two would silently collide under one
                threshold. They are now tracked and logged separately, and
                either or both may be present.
            current_price: Current market price.

        Returns:
            Decision object with action and explanation.
        """
        notes: List[str] = []
        action = "NO_TRADE"
        ai_score = regime_signal.confidence
        entry = regime_signal.entry
        sl = regime_signal.sl
        tp = regime_signal.tp
        lot_size = regime_signal.lot_size

        # Trained ML model filter (reserved — no model exists yet)
        if self.config.enable_ml_filter and ml_confidence is not None:
            if ml_confidence < self.config.min_ml_confidence:
                notes.append(f"ML model filter blocked: confidence {ml_confidence:.1f} < {self.config.min_ml_confidence}")
                return Decision(
                    action="NO_TRADE", ai_score=ai_score, regime_signal=regime_signal,
                    smc=smc, confluence_notes=notes,
                    explanation="ML model confidence too low",
                )
            notes.append(f"ML model confidence: {ml_confidence:.1f}")
            ai_score = int((ai_score + ml_confidence) / 2)

        # LLM trade reviewer (Claude/Gemini/GPT) — advisory only. This can
        # only ever LOWER ai_score or gate the trade to NO_TRADE; it can
        # never raise a NO_TRADE regime signal into a trade, and it never
        # touches RiskManager or the broker directly (AIProvider.analyze()
        # returns a plain confidence+reasoning value, nothing more).
        if self.config.enable_ai_reviewer and ai_reviewer_confidence is not None:
            if ai_reviewer_confidence < self.config.min_ai_reviewer_confidence:
                notes.append(f"AI reviewer blocked: confidence {ai_reviewer_confidence:.1f} < {self.config.min_ai_reviewer_confidence}")
                return Decision(
                    action="NO_TRADE", ai_score=ai_score, regime_signal=regime_signal,
                    smc=smc, confluence_notes=notes,
                    explanation="AI reviewer confidence too low",
                )
            notes.append(f"AI reviewer confidence: {ai_reviewer_confidence:.1f}")
            ai_score = int((ai_score + ai_reviewer_confidence) / 2)

        # Minimum AI score
        if ai_score < self.config.min_ai_score:
            notes.append(f"AI score {ai_score} below minimum {self.config.min_ai_score}")
            return Decision(
                action="NO_TRADE", ai_score=ai_score, regime_signal=regime_signal,
                smc=smc, confluence_notes=notes,
                explanation="AI score below threshold",
            )

        # SMC confluence
        if regime_signal.action == "BUY" and smc.bias == "bullish":
            notes.append("SMC confirms bullish bias")
            action = "BUY"
        elif regime_signal.action == "SELL" and smc.bias == "bearish":
            notes.append("SMC confirms bearish bias")
            action = "SELL"
        elif regime_signal.action == "BUY" and smc.bias == "bearish":
            notes.append("SMC contradicts bullish signal")
            if self.config.strict_smc_confluence:
                action = "NO_TRADE"
                notes.append("Strict SMC confluence blocked the trade")
            else:
                ai_score = max(0, ai_score - 15)
                if ai_score >= self.config.min_ai_score:
                    action = "BUY"
        elif regime_signal.action == "SELL" and smc.bias == "bullish":
            notes.append("SMC contradicts bearish signal")
            if self.config.strict_smc_confluence:
                action = "NO_TRADE"
                notes.append("Strict SMC confluence blocked the trade")
            else:
                ai_score = max(0, ai_score - 15)
                if ai_score >= self.config.min_ai_score:
                    action = "SELL"
        elif smc.bias == "neutral":
            notes.append("SMC neutral — using regime signal only")
            action = regime_signal.action if regime_signal.action in ("BUY", "SELL") else "NO_TRADE"
        else:
            notes.append(f"No confluence: regime={regime_signal.action}, SMC={smc.bias}")

        # Zone check
        if action in ("BUY", "SELL"):
            if action == "BUY" and smc.zone == "premium":
                notes.append("Warning: buying in premium zone")
                ai_score = max(0, ai_score - 10)
            elif action == "SELL" and smc.zone == "discount":
                notes.append("Warning: selling in discount zone")
                ai_score = max(0, ai_score - 10)

        # Final validation
        if action in ("BUY", "SELL") and ai_score < self.config.min_ai_score:
            notes.append(f"Final score {ai_score} below threshold after adjustments")
            action = "NO_TRADE"

        explanation = f"Decision: {action} | Score: {ai_score}/100 | " + " | ".join(notes)
        log.info(explanation)

        self._last_decision = Decision(
            action=action, ai_score=ai_score, regime_signal=regime_signal,
            smc=smc, confluence_notes=notes, explanation=explanation,
            sl=sl, tp=tp, entry=entry, lot_size=lot_size,
        )
        return self._last_decision
