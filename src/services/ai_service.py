"""AI trade reviewer facade — picks the configured provider (Claude, Gemini,
or GPT) and exposes one stable interface to the rest of the app.

decision_engine.py and app.py only ever talk to AIService; they never know
or care which underlying provider(s) answered. Adding a 4th provider later
means adding one new file under src/services/ai/ that implements
AIProvider — this file only needs a one-line addition to _PROVIDERS.

Two modes:
    - Single-provider (default): uses `config.provider` only, as before.
    - Consensus (`config.ai_consensus_mode=True`): queries every provider
      that has an API key configured, regardless of `config.provider`, and
      combines their opinions. This is closer to what "AI validation:
      OpenAI, Gemini, Claude" implies — an independent second (and third)
      opinion is more convincing when multiple different models agree than
      when relying on just one. If the available providers disagree
      strongly, the combined confidence is penalized rather than averaged
      naively — strong disagreement between independent reviewers IS
      meaningful uncertainty, and the system defaults to more caution when
      state is uncertain, not less.
"""

from __future__ import annotations

import statistics
from typing import Optional, List

from src.logger import get_logger
from src.config import AIConfig
from src.models import SignalOutput, SMCResult, NewsSentiment, AIAnalysis
from src.services.ai.claude_provider import ClaudeProvider
from src.services.ai.gemini_provider import GeminiProvider
from src.services.ai.gpt_provider import GPTProvider

log = get_logger(__name__)

_PROVIDERS = {
    "claude": lambda cfg: ClaudeProvider(cfg.anthropic_api_key, cfg.model or "claude-sonnet-5"),
    "gemini": lambda cfg: GeminiProvider(cfg.google_api_key, cfg.gemini_model or "gemini-2.0-flash"),
    "gpt": lambda cfg: GPTProvider(cfg.openai_api_key, cfg.gpt_model or "gpt-4o-mini"),
}


class AIService:
    """Wraps the active AI provider(s) to produce an independent trade confidence score."""

    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self._provider = None
        self._all_providers: List = []

        if not self.config.enabled:
            return

        if getattr(self.config, "ai_consensus_mode", False):
            for name, factory in _PROVIDERS.items():
                try:
                    provider = factory(self.config)
                    if provider.is_available:
                        self._all_providers.append((name, provider))
                except Exception as exc:
                    log.warning("Consensus mode: failed to init provider '%s': %s", name, exc)
            if not self._all_providers:
                log.warning("Consensus mode enabled but no provider has a valid API key configured")
        else:
            factory = _PROVIDERS.get(self.config.provider)
            if factory is None:
                log.warning("Unknown AI provider '%s', AI reviewer disabled", self.config.provider)
            else:
                self._provider = factory(self.config)

    @property
    def is_available(self) -> bool:
        if self._all_providers:
            return True
        return self._provider is not None and self._provider.is_available

    @property
    def active_provider_name(self) -> str:
        if self._all_providers:
            return "consensus(" + "+".join(name for name, _ in self._all_providers) + ")"
        return self.config.provider if self.is_available else "none"

    def analyze(
        self,
        regime_signal: SignalOutput,
        smc: SMCResult,
        news: Optional[NewsSentiment],
        current_price: float,
    ) -> AIAnalysis:
        """Ask the active provider(s) for an independent confidence score on the current setup.

        Never raises. Returns AIAnalysis(available=False, error=...) if no
        provider is configured/reachable.
        """
        if self._all_providers:
            return self._analyze_consensus(regime_signal, smc, news, current_price)

        if not self.is_available:
            return AIAnalysis(available=False, error="AI reviewer not configured")
        return self._provider.analyze(regime_signal, smc, news, current_price)

    def _analyze_consensus(
        self,
        regime_signal: SignalOutput,
        smc: SMCResult,
        news: Optional[NewsSentiment],
        current_price: float,
    ) -> AIAnalysis:
        results = []
        for name, provider in self._all_providers:
            r = provider.analyze(regime_signal, smc, news, current_price)
            if r.available:
                results.append((name, r))
            else:
                log.info("Consensus: provider '%s' unavailable this call: %s", name, r.error)

        if not results:
            return AIAnalysis(available=False, error="No provider returned a usable response this call")

        confidences = [r.confidence for _, r in results]
        mean_confidence = statistics.mean(confidences)
        spread = (max(confidences) - min(confidences)) if len(confidences) > 1 else 0.0

        # Strong disagreement between independent reviewers is itself a
        # signal of uncertainty — penalize rather than pretend consensus
        # exists when it doesn't. A 40+ point spread (e.g. one says 85,
        # another says 40) roughly halves the combined confidence.
        disagreement_penalty = min(mean_confidence, spread * 0.5)
        combined_confidence = max(0.0, mean_confidence - disagreement_penalty)

        reasoning_parts = [f"{name}={r.confidence:.0f}" for name, r in results]
        reasoning = f"Consensus of {len(results)} provider(s): " + ", ".join(reasoning_parts)
        if spread >= 40:
            reasoning += f" — high disagreement (spread={spread:.0f}), confidence penalized"

        return AIAnalysis(confidence=round(combined_confidence, 1), reasoning=reasoning[:300], available=True)
