"""Economic calendar blackout guard.

Previously there was NO protection against trading through high-impact
news events (NFP, FOMC, CPI, etc.) at all — only price-based news
*sentiment* scoring existed (`news_service.py`), which is a different thing
(what the news says) from calendar-based *event-time* protection (blocking
trading during the volatility spike a scheduled release causes, regardless
of what the number turns out to be).

This project has no network access available while I'm building it, so
this can't be wired to a live economic-calendar API right now. Built as a
generic, extensible guard instead:
    - NFP (US Non-Farm Payrolls): auto-generated. It's always the first
      Friday of the month at 13:30 UTC (08:30 ET) — a fixed, well-known
      rule, so this needs no external data source and is always correct.
    - Everything else (FOMC decisions, CPI, major central bank rate
      decisions): added manually via `add_events()`, or later wired to a
      real calendar API (e.g. Trading Economics, Forex Factory) that
      populates the same `EconomicEvent` list this guard already consumes
      — the guard logic itself doesn't need to change when that's added.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class EconomicEvent:
    name: str
    time_utc: datetime
    impact: str = "high"  # "high" | "medium" | "low"
    currency: str = "USD"


@dataclass
class CalendarCheckResult:
    allowed: bool
    reason: str = ""
    nearest_event: Optional[EconomicEvent] = None


def generate_nfp_dates(year: int) -> List[EconomicEvent]:
    """First Friday of every month at 13:30 UTC — the fixed NFP release
    schedule. Deterministic, needs no external data."""
    events = []
    for month in range(1, 13):
        cal = calendar.monthcalendar(year, month)
        first_week = cal[0]
        # monthcalendar's columns are Mon=0..Sun=6; Friday=4
        friday_day = first_week[4] if first_week[4] != 0 else cal[1][4]
        events.append(EconomicEvent(
            name="US Non-Farm Payrolls (NFP)",
            time_utc=datetime(year, month, friday_day, 13, 30, tzinfo=timezone.utc),
            impact="high",
            currency="USD",
        ))
    return events


class EconomicCalendarGuard:
    def __init__(
        self,
        blackout_minutes_before: int = 30,
        blackout_minutes_after: int = 30,
        min_impact: str = "high",
        auto_generate_nfp: bool = True,
        nfp_years: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            blackout_minutes_before: Block new trades starting this many
                minutes before a qualifying event.
            blackout_minutes_after: ...and continuing this many minutes
                after, since the volatility spike doesn't end the instant
                the number prints.
            min_impact: Only events at this impact level or higher trigger
                a blackout ("high" only, by default — don't blackout on
                every minor data release).
            auto_generate_nfp: Pre-populate NFP dates automatically (see
                module docstring — this is a fixed, always-correct schedule).
            nfp_years: Which years to generate NFP dates for. Defaults to
                the current and next calendar year (correct for LIVE
                trading). For BACKTESTING a historical period, pass the
                actual years being tested — e.g. `nfp_years=[2022, 2023]`
                for a 2022-2023 backtest — otherwise the guard has no
                relevant events loaded and silently allows trading straight
                through every historical NFP release.
        """
        self.blackout_minutes_before = blackout_minutes_before
        self.blackout_minutes_after = blackout_minutes_after
        self.min_impact = min_impact
        self._events: List[EconomicEvent] = []

        if auto_generate_nfp:
            years = nfp_years if nfp_years is not None else [
                datetime.now(timezone.utc).year, datetime.now(timezone.utc).year + 1,
            ]
            for year in years:
                self._events.extend(generate_nfp_dates(year))

    def add_events(self, events: List[EconomicEvent]) -> None:
        """Add manually-known events (FOMC decisions, CPI releases, etc.)."""
        self._events.extend(events)
        log.info("Added %d economic event(s) to calendar guard", len(events))

    def upcoming_events(self, within_days: int = 7, current_time: Optional[datetime] = None) -> List[EconomicEvent]:
        now = current_time or datetime.now(timezone.utc)
        horizon = now + timedelta(days=within_days)
        return sorted(
            [e for e in self._events if now <= e.time_utc <= horizon],
            key=lambda e: e.time_utc,
        )

    _IMPACT_RANK = {"low": 0, "medium": 1, "high": 2}

    def check(self, current_time: Optional[datetime] = None) -> CalendarCheckResult:
        """Returns whether new trades are allowed right now.

        Never raises. An empty or misconfigured event list just means no
        blackout is ever triggered — this guard fails open by design (it's
        a supplementary protection, not the primary risk control), so a
        missing calendar feed degrades to "no calendar-based blocking"
        rather than blocking all trading.
        """
        now = current_time or datetime.now(timezone.utc)
        min_rank = self._IMPACT_RANK.get(self.min_impact, 2)

        nearest_in_window = None
        for event in self._events:
            if self._IMPACT_RANK.get(event.impact, 0) < min_rank:
                continue
            window_start = event.time_utc - timedelta(minutes=self.blackout_minutes_before)
            window_end = event.time_utc + timedelta(minutes=self.blackout_minutes_after)
            if window_start <= now <= window_end:
                nearest_in_window = event
                break

        if nearest_in_window is not None:
            return CalendarCheckResult(
                allowed=False,
                reason=f"Within blackout window of '{nearest_in_window.name}' "
                       f"({nearest_in_window.time_utc.strftime('%Y-%m-%d %H:%M UTC')}, "
                       f"impact={nearest_in_window.impact}) — blocking new trades.",
                nearest_event=nearest_in_window,
            )
        return CalendarCheckResult(allowed=True)
