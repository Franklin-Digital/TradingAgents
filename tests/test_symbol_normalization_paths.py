"""Symbol normalization must apply on every yfinance path, not just price fetch.

Regression tests for #983 (instrument identity), #984 (reflection returns), and
the news path: a broker symbol like XAUUSD must resolve to the same Yahoo symbol
(GC=F) that the price path uses, so identity, realized-return, and news lookups
hit the right instrument instead of failing/mismatching.
"""
import pandas as pd

import tradingagents.agents.utils.agent_utils as au
import tradingagents.dataflows.yfinance_news as ynews
from tradingagents.graph.trading_graph import TradingAgentsGraph


def test_identity_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        @property
        def info(self):
            return {"longName": "Gold Futures", "quoteType": "FUTURE"}

    monkeypatch.setattr(au.yf, "Ticker", FakeTicker)
    au.resolve_instrument_identity.cache_clear()

    identity = au.resolve_instrument_identity("XAUUSD")

    assert seen["symbol"] == "GC=F"  # normalized, not the raw broker symbol
    assert identity.get("company_name") == "Gold Futures"


def test_fetch_returns_uses_the_same_symbol_the_data_path_priced(monkeypatch):
    """Franklin fork: the realized-return lookup goes through the licensed data
    interface (QuestDB/FMP), not yfinance, and passes the ticker through
    unnormalized.

    ``normalize_symbol`` maps to *Yahoo* conventions (XAUUSD -> GC=F), which the
    Franklin vendors do not speak. The invariant upstream's version protects —
    the reflection path must price the SAME instrument the analysis priced — is
    preserved here because both paths hand the raw ticker to
    ``franklinfinancial.get_stock_data``.
    """
    queried = []

    def fake_get_stock_data(symbol, start, end, *a, **k):
        queried.append(symbol)
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
        idx = pd.date_range(start="2025-01-02", periods=len(prices), freq="D")
        rows = "\n".join(
            f"{d.strftime('%Y-%m-%d')},{p},{p},{p},{p},1000"
            for d, p in zip(idx, prices)
        )
        return "Datetime,Open,High,Low,Close,Volume\n" + rows + "\n"

    monkeypatch.setattr(
        "tradingagents.dataflows.franklinfinancial.get_stock_data",
        fake_get_stock_data,
    )

    # _fetch_returns does not use ``self``; call unbound to avoid building the graph.
    raw, alpha, days, resolved = TradingAgentsGraph._fetch_returns(
        None, "XAUUSD", "2025-01-02", holding_days=5, benchmark="SPY"
    )

    assert queried[0] == "XAUUSD"  # passed through, NOT mapped to Yahoo's GC=F
    assert queried[1] == "SPY"     # benchmark used as configured
    assert raw is not None and days is not None
    assert resolved == "2025-01-07"  # resolution date recorded (#1251)


def test_news_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def get_news(self, count):
            return []

    monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(ynews, "yf_retry", lambda fn: fn())

    out = ynews.get_news_yfinance("XAUUSD", "2025-01-01", "2025-01-10")

    assert seen["symbol"] == "GC=F"   # news queried with the canonical symbol
    assert "XAUUSD" in out            # the user's ticker stays in the report
    assert "GC=F" in out              # provenance noted
