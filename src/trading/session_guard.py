"""Session-based trading guard.

The dashboard already SHOWS which trading session is active
(`src/ui/components.py: render_session_widget`), but nothing previously
enforced anything based on it — it was purely decorative. This module
actually gates new trades during low-liquidity windows, and is the single
source of truth for session boundaries (the display widget imports
TRADING_SESSIONS from here so the two can never drift out of sync).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from src.logger import get_logger

log = get_logger(__name__)

# Session windows in UTC. Gold/Forex conventionally split into three
# overlapping sessions; times are approximate market-open hours. Adjust
# here if your broker's server time differs — both the enforcement guard
# below and the dashboard's session widget read from this one definition.
TRADING_SESSIONS = [
    ("Sydney/Asian", 22, 7),   # 22:00 UTC -> 07:00 UTC (wraps midnight)
    ("London", 7, 16),         # 07:00 UTC -> 16:00 UTC
    ("New York", 12, 21),      # 12:00 UTC -> 21:00 UTC
]


def _in_session(hour: float, start: int, end: int) -> bool:
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end  # wraps past midnight


def active_sessions(current_time: Optional[datetime] = None) -> List[str]:
    now_utc = current_time.astimezone(timezone.utc) if current_time else datetime.now(timezone.utc)
    hour = now_utc.hour + now_utc.minute / 60.0
    return [name for name, start, end in TRADING_SESSIONS if _in_session(hour, start, end)]


@dataclass
class SessionCheckResult:
    allowed: bool
    active_sessions: List[str]
    reason: str = ""


class SessionGuard:
    def __init__(
        self,
        block_between_sessions: bool = True,
        allowed_sessions: Optional[List[str]] = None,
        avoid_first_minutes: int = 0,
        avoid_last_minutes: int = 0,
    ) -> None:
        """
        Args:
            block_between_sessions: If True, block new trades when no major
                session is active at all (the lowest-liquidity, widest-spread
                window — typically ~21:00-22:00 UTC).
            allowed_sessions: If given, ONLY allow trading during these named
                sessions (e.g. ["London", "New York"] to skip the thinner
                Asian session entirely). None = all sessions allowed.
            avoid_first_minutes: Block trading in the first N minutes after
                a session opens (the volatility/spread spike right at open).
            avoid_last_minutes: Block trading in the last N minutes before a
                session closes.
        """
        self.block_between_sessions = block_between_sessions
        self.allowed_sessions = set(allowed_sessions) if allowed_sessions else None
        self.avoid_first_minutes = avoid_first_minutes
        self.avoid_last_minutes = avoid_last_minutes

    def check(self, current_time: Optional[datetime] = None) -> SessionCheckResult:
        now_utc = current_time.astimezone(timezone.utc) if current_time else datetime.now(timezone.utc)
        hour = now_utc.hour + now_utc.minute / 60.0
        active = active_sessions(now_utc)

        if not active:
            if self.block_between_sessions:
                return SessionCheckResult(False, active, "No major session active — low liquidity, blocking new trades.")
            return SessionCheckResult(True, active, "")

        if self.allowed_sessions is not None:
            if not (set(active) & self.allowed_sessions):
                return SessionCheckResult(
                    False, active,
                    f"Active session(s) {active} not in allowed list {sorted(self.allowed_sessions)}.",
                )

        for name, start, end in TRADING_SESSIONS:
            if name not in active:
                continue
            minutes_since_open = ((hour - start) % 24) * 60
            if self.avoid_first_minutes and minutes_since_open < self.avoid_first_minutes:
                return SessionCheckResult(
                    False, active,
                    f"{name} session opened {minutes_since_open:.0f} min ago — "
                    f"waiting out the first {self.avoid_first_minutes} min (open volatility).",
                )
            minutes_to_close = ((end - hour) % 24) * 60
            if self.avoid_last_minutes and minutes_to_close < self.avoid_last_minutes:
                return SessionCheckResult(
                    False, active,
                    f"{name} session closes in {minutes_to_close:.0f} min — "
                    f"blocking new trades in the last {self.avoid_last_minutes} min.",
                )

        return SessionCheckResult(True, active, "")
