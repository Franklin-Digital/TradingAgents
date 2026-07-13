"""Unit tests for the FMP fundamentals vendor (DAYTRADE-179).

All network access is mocked — these tests never hit financialmodelingprep.com.
Covers:
  - freq → FMP period mapping (quarterly→quarter, annual→annual)
  - filter_statements_by_date look-ahead guard on filingDate
  - FMPRateLimitError on "limit exceeded" dict bodies and HTTP 429
  - get_fundamentals hits the `profile` endpoint
  - route_to_vendor calls ONLY fmp for fundamentals (no yfinance/alpha_vantage)
    and surfaces RuntimeError when FMP rate-limits
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.dataflows import fmp_common
from tradingagents.dataflows.fmp_common import (
    FMPRateLimitError,
    filter_statements_by_date,
)
from tradingagents.dataflows.fmp_fundamentals import (
    _period,
    get_fundamentals,
    get_income_statement,
)
from tradingagents.dataflows.interface import VENDOR_METHODS, route_to_vendor


def _mock_response(status_code=200, body="[]"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = body if isinstance(body, str) else json.dumps(body)
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture(autouse=True)
def _fmp_key(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key-not-real")


# ---------------------------------------------------------------------------
# Period mapping
# ---------------------------------------------------------------------------

def test_period_mapping():
    assert _period("quarterly") == "quarter"
    assert _period("Quarterly") == "quarter"
    assert _period("q") == "quarter"
    assert _period("annual") == "annual"
    assert _period("Annual") == "annual"


def test_income_statement_sends_quarter_period():
    with patch.object(fmp_common.requests, "get",
                      return_value=_mock_response(body="[]")) as mock_get:
        get_income_statement("AAPL", freq="quarterly")
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["period"] == "quarter"
    assert kwargs["params"]["symbol"] == "AAPL"


def test_income_statement_sends_annual_period():
    with patch.object(fmp_common.requests, "get",
                      return_value=_mock_response(body="[]")) as mock_get:
        get_income_statement("AAPL", freq="annual")
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["period"] == "annual"


# ---------------------------------------------------------------------------
# Look-ahead guard
# ---------------------------------------------------------------------------

def test_filter_statements_drops_future_filings():
    rows = [
        {"date": "2026-03-31", "filingDate": "2026-05-01", "revenue": 1},
        {"date": "2026-06-30", "filingDate": "2026-08-01", "revenue": 2},
        {"date": "2025-12-31", "filingDate": "2026-02-01", "revenue": 3},
    ]
    kept = json.loads(filter_statements_by_date(json.dumps(rows), "2026-07-13"))
    assert [r["revenue"] for r in kept] == [1, 3]


def test_filter_statements_keeps_boundary_date():
    rows = [{"filingDate": "2026-07-13", "revenue": 1}]
    kept = json.loads(filter_statements_by_date(json.dumps(rows), "2026-07-13"))
    assert len(kept) == 1


def test_filter_statements_noop_without_curr_date():
    text = json.dumps([{"filingDate": "2099-01-01"}])
    assert filter_statements_by_date(text, None) == text


def test_filter_statements_falls_back_to_date_field():
    rows = [{"date": "2026-08-01", "revenue": 1},
            {"date": "2026-06-30", "revenue": 2}]
    kept = json.loads(filter_statements_by_date(json.dumps(rows), "2026-07-13"))
    assert [r["revenue"] for r in kept] == [2]


# ---------------------------------------------------------------------------
# Rate-limit / entitlement errors
# ---------------------------------------------------------------------------

def test_limit_exceeded_dict_body_raises_rate_limit():
    body = {"Error Message": "Limit Reach . Please upgrade your plan — exceeded"}
    with patch.object(fmp_common.requests, "get",
                      return_value=_mock_response(body=body)):
        with pytest.raises(FMPRateLimitError):
            get_fundamentals("AAPL")


def test_http_429_raises_rate_limit():
    with patch.object(fmp_common.requests, "get",
                      return_value=_mock_response(status_code=429, body="Too Many Requests")):
        with pytest.raises(FMPRateLimitError):
            get_fundamentals("AAPL")


def test_normal_array_body_passes_through():
    body = [{"symbol": "AAPL", "marketCap": 3_000_000_000_000}]
    with patch.object(fmp_common.requests, "get",
                      return_value=_mock_response(body=body)):
        result = get_fundamentals("AAPL")
    assert json.loads(result)[0]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# Endpoint selection
# ---------------------------------------------------------------------------

def test_get_fundamentals_hits_profile_endpoint():
    with patch.object(fmp_common.requests, "get",
                      return_value=_mock_response(body="[]")) as mock_get:
        get_fundamentals("AAPL")
    args, kwargs = mock_get.call_args
    assert args[0].endswith("/stable/profile")
    assert kwargs["params"]["symbol"] == "AAPL"
    assert "apikey" in kwargs["params"]


# ---------------------------------------------------------------------------
# Vendor routing — FMP only, no yfinance/alpha_vantage fallback
# ---------------------------------------------------------------------------

FUNDAMENTALS_METHODS = (
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
)

_FMP_CONFIG = {
    "data_vendors": {"fundamental_data": "fmp"},
    "tool_vendors": {},
}


def test_fundamentals_maps_are_fmp_only():
    for method in FUNDAMENTALS_METHODS:
        assert list(VENDOR_METHODS[method].keys()) == ["fmp"], method


def test_other_method_maps_untouched():
    assert set(VENDOR_METHODS["get_stock_data"]) == {
        "franklin", "questdb", "alpha_vantage", "yfinance"}
    assert set(VENDOR_METHODS["get_indicators"]) == {
        "franklin", "alpha_vantage", "yfinance"}
    assert set(VENDOR_METHODS["get_news"]) == {"alpha_vantage", "yfinance"}
    assert set(VENDOR_METHODS["get_global_news"]) == {"yfinance", "alpha_vantage"}
    assert set(VENDOR_METHODS["get_insider_transactions"]) == {
        "alpha_vantage", "yfinance"}


def test_route_to_vendor_calls_only_fmp():
    fmp_impl = MagicMock(return_value='[{"symbol": "AAPL"}]')
    with patch("tradingagents.dataflows.interface.get_config",
               return_value=_FMP_CONFIG), \
         patch.dict(VENDOR_METHODS["get_fundamentals"],
                    {"fmp": fmp_impl}, clear=True), \
         patch("tradingagents.dataflows.y_finance.get_fundamentals") as yf_impl:
        result = route_to_vendor("get_fundamentals", "AAPL")
    fmp_impl.assert_called_once_with("AAPL")
    yf_impl.assert_not_called()
    assert result == '[{"symbol": "AAPL"}]'


def test_route_to_vendor_raises_runtime_error_when_fmp_rate_limited():
    fmp_impl = MagicMock(side_effect=FMPRateLimitError("limit exceeded"))
    with patch("tradingagents.dataflows.interface.get_config",
               return_value=_FMP_CONFIG), \
         patch.dict(VENDOR_METHODS["get_fundamentals"],
                    {"fmp": fmp_impl}, clear=True), \
         patch("tradingagents.dataflows.y_finance.get_fundamentals") as yf_impl:
        with pytest.raises(RuntimeError, match="No available vendor"):
            route_to_vendor("get_fundamentals", "AAPL")
    fmp_impl.assert_called_once()
    yf_impl.assert_not_called()
