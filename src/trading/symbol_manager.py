"""Manages the set of symbols the bot actively scans and trades, each with
its own risk profile (pip value, typical spread tolerance).

Adding a new pair later is a one-line addition to DEFAULT_SYMBOLS — nothing
else in the trading loop needs to change, since regime_detector, smc, and
decision_engine already operate on a (symbol, dataframe) pair without
knowing how many symbols exist in total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass(frozen=True)
class SymbolProfile:
    """Per-symbol trading parameters."""
    symbol: str                 # broker symbol, e.g. "XAUUSD", "EURUSD"
    display_name: str
    category: str               # "gold" | "forex" | "crypto"
    pip_size: float              # price move that equals 1 pip
    max_spread_pips: float       # skip trading if live spread exceeds this
    quote_currency: str          # currency this symbol is priced in (USD for most)
    base_currency: str           # the "other side" of the pair, for correlation checks
    contract_size: float = 100000.0  # units per 1.0 lot — used for position sizing


# Vantage-style symbol names. If your broker uses a different suffix
# (e.g. "EURUSDm", "XAUUSD.a"), add a matching profile — the bot tries each
# candidate against symbol_candidates the same way it already does for gold.
DEFAULT_SYMBOLS: List[SymbolProfile] = [
    SymbolProfile("XAUUSD", "Gold / USD", "gold", pip_size=0.1, max_spread_pips=5.0,
                   quote_currency="USD", base_currency="XAU", contract_size=100.0),  # 1.0 lot = 100 oz
    SymbolProfile("EURUSD", "Euro / USD", "forex", pip_size=0.0001, max_spread_pips=2.0,
                   quote_currency="USD", base_currency="EUR", contract_size=100000.0),
    SymbolProfile("GBPUSD", "Pound / USD", "forex", pip_size=0.0001, max_spread_pips=2.5,
                   quote_currency="USD", base_currency="GBP", contract_size=100000.0),
    SymbolProfile("USDJPY", "USD / Yen", "forex", pip_size=0.01, max_spread_pips=2.0,
                   quote_currency="JPY", base_currency="USD", contract_size=100000.0),
    SymbolProfile("AUDUSD", "Aussie / USD", "forex", pip_size=0.0001, max_spread_pips=2.5,
                   quote_currency="USD", base_currency="AUD", contract_size=100000.0),
    # Crypto: "pip" isn't a native unit for BTC, but the bot's spread-filter
    # math (spread_price / pip_size) needs SOME per-symbol unit to compare
    # against a threshold. Using $1 as the unit keeps max_spread_pips
    # readable as "max acceptable spread in dollars" for this one. Without
    # this explicit entry, BTCUSD previously fell through to the generic
    # forex fallback (pip_size=0.0001), which would compute a spread of e.g.
    # $20 as 200,000 "pips" — always failing any sane max_spread_pips
    # threshold and silently blocking BTCUSD from ever trading.
    SymbolProfile("BTCUSD", "Bitcoin / USD", "crypto", pip_size=1.0, max_spread_pips=50.0,
                   quote_currency="USD", base_currency="BTC", contract_size=1.0),  # 1.0 lot = 1 BTC (broker-dependent, verify against your broker's spec)
]

_BY_SYMBOL: Dict[str, SymbolProfile] = {p.symbol: p for p in DEFAULT_SYMBOLS}


class SymbolManager:
    """Holds the set of symbols currently active for scanning/trading."""

    def __init__(self, active_symbols: Optional[List[str]] = None) -> None:
        self.active_symbols: List[str] = active_symbols or ["XAUUSD"]

    def profile(self, symbol: str) -> SymbolProfile:
        if symbol in _BY_SYMBOL:
            return _BY_SYMBOL[symbol]
        # Fallback for symbols not in DEFAULT_SYMBOLS: derive base/quote from
        # standard 6-character forex naming (e.g. EURGBP -> base=EUR, quote=GBP).
        if len(symbol) == 6 and symbol.isalpha():
            base, quote = symbol[:3].upper(), symbol[3:].upper()
        else:
            base, quote = symbol.upper(), "USD"
        return SymbolProfile(symbol, symbol, "forex", pip_size=0.0001, max_spread_pips=3.0,
                              quote_currency=quote, base_currency=base)

    def is_usd_exposed(self, symbol: str) -> bool:
        p = self.profile(symbol)
        return p.quote_currency == "USD" or p.base_currency == "USD" or p.category == "gold"

    @staticmethod
    def all_known_symbols() -> List[str]:
        return [p.symbol for p in DEFAULT_SYMBOLS]
