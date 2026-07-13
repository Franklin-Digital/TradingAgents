"""FMP fundamentals — same function contract as alpha_vantage_fundamentals /
yfinance_fundamentals so it drops into VENDOR_METHODS unchanged.

Endpoints (FMP Starter, `/stable/`):
  profile                    → company overview (marketCap, sector, industry, beta, price…)
  income-statement           → income statement
  balance-sheet-statement    → balance sheet
  cash-flow-statement        → cash flow statement
Statements are look-ahead-filtered on filingDate when curr_date is provided.
"""
from .fmp_common import _make_request, filter_statements_by_date


def _period(freq: str) -> str:
    """Map the shared freq arg ('quarterly'/'annual') to FMP's period value."""
    return "quarter" if str(freq).lower().startswith("q") else "annual"


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """Company overview / profile (market cap, sector, industry, ratios, price)."""
    return _make_request("profile", {"symbol": ticker})


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    text = _make_request("balance-sheet-statement",
                         {"symbol": ticker, "period": _period(freq), "limit": 8})
    return filter_statements_by_date(text, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    text = _make_request("cash-flow-statement",
                         {"symbol": ticker, "period": _period(freq), "limit": 8})
    return filter_statements_by_date(text, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    text = _make_request("income-statement",
                         {"symbol": ticker, "period": _period(freq), "limit": 8})
    return filter_statements_by_date(text, curr_date)
