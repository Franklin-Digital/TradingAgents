"""FMP daily OHLCV — the licensed last resort when QuestDB has no bars.

WHY THIS EXISTS
---------------
`franklinfinancial.get_stock_data` used to fall back to **yfinance** when neither
QuestDB instance had bars. That violates the house rule (CLAUDE.md):

    No silent fallback vendors in production — licensed vendors only; fail LOUD,
    never substitute a scraped source. yfinance = academic only.

And it was not hypothetical. Because the historical port pointed at the FROZEN
Databento archive (fixed 2026-08-15), QuestDB legitimately had no recent bars,
so the fallback fired constantly and nothing failed:

    2026-08-14 00:16:15 INFO QuestDB has no data for SCHY
                             [2026-08-03 → 2026-08-15], falling back to yfinance

14 such lines in one nightly run. `ai-score` emitted Buy/Hold/Overweight ratings
built on scraped prices, and the fallback is exactly what hid it — an empty
primary looked like a quiet symbol rather than a misrouted query.

We pay for FMP. Use it.

WHAT THIS IS NOT
----------------
Not a silent substitute. It is a LICENSED tier, logged at WARNING when used, and
when FMP has nothing either the caller reports no data rather than inventing it.

`fmp_ohlcv_1m` in QuestDB is deliberately NOT used as a daily source: it has
proven split discontinuities (7 splits confirmed unapplied, 8 symbols corrupt),
so resampling it to daily would import that corruption. The API endpoint below
returns split-adjusted daily bars directly.
"""
import json
import logging

from .fmp_common import _make_request

log = logging.getLogger(__name__)

# Proven endpoint: already used by DayTradingAgent/common/fmp_tier3.py for the
# index series, and it returns the whole requested range in one call.
_ENDPOINT = "historical-price-eod/full"


def get_daily_bars(symbol: str, start_date: str, end_date: str) -> list[dict]:
    """Daily OHLCV from FMP, shaped like a QuestDB row so callers need no change.

    Returns rows with keys: ts, open, high, low, close, volume — ASCENDING by
    date, matching `ORDER BY ts ASC` in the QuestDB queries. Returns [] when FMP
    has nothing; the caller decides how loudly to complain.
    """
    sym = symbol.upper()
    text = _make_request(_ENDPOINT, {"symbol": sym,
                                     "from": start_date,
                                     "to": end_date})
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        log.error("FMP returned non-JSON for %s [%s → %s]", sym, start_date, end_date)
        return []

    # /stable returns a bare list; older shapes nest under "historical".
    raw = payload if isinstance(payload, list) else payload.get("historical", [])
    if not raw:
        return []

    rows = []
    for r in raw:
        date = r.get("date")
        if not date:
            continue
        # A bar with no close is unusable; skip rather than emit a zero, which
        # would be indistinguishable from a real price of 0 downstream.
        if r.get("close") in (None, ""):
            continue
        rows.append({
            "ts": str(date)[:10],
            "open": r.get("open"),
            "high": r.get("high"),
            "low": r.get("low"),
            "close": r.get("close"),
            "volume": r.get("volume") or 0,
        })

    # FMP returns newest-first; every caller here expects ascending.
    rows.sort(key=lambda x: x["ts"])
    return rows
