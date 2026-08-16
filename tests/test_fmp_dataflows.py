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


# ---------------------------------------------------------------------------
# FMP daily OHLCV — the LICENSED fallback that replaced yfinance (2026-08-15)
#
# The old fallback was yfinance, which the house rules ban in production. It was
# firing constantly (14 times in one nightly ai-score run) because the historical
# QuestDB port pointed at the frozen archive, so ratings were being produced on
# scraped prices with no error anywhere.
# ---------------------------------------------------------------------------

from unittest.mock import patch  # noqa: E402

from tradingagents.dataflows.fmp_ohlcv import get_daily_bars  # noqa: E402

_FMP_PAYLOAD = """[
  {"symbol":"SCHY","date":"2026-08-14","open":33.20,"high":33.23,"low":33.13,"close":33.18,"volume":402634},
  {"symbol":"SCHY","date":"2026-08-12","open":33.26,"high":33.27,"low":33.16,"close":33.18,"volume":456675},
  {"symbol":"SCHY","date":"2026-08-13","open":33.19,"high":33.20,"low":33.10,"close":33.19,"volume":488327}
]"""


def test_daily_bars_are_returned_ascending():
    """FMP returns newest-first; every caller here expects ORDER BY ts ASC."""
    with patch("tradingagents.dataflows.fmp_ohlcv._make_request",
               return_value=_FMP_PAYLOAD):
        rows = get_daily_bars("SCHY", "2026-08-12", "2026-08-14")
    assert [r["ts"] for r in rows] == ["2026-08-12", "2026-08-13", "2026-08-14"]


def test_rows_match_the_questdb_shape():
    """Shaped like a QuestDB row so _format_ohlcv_csv needs no change."""
    with patch("tradingagents.dataflows.fmp_ohlcv._make_request",
               return_value=_FMP_PAYLOAD):
        rows = get_daily_bars("SCHY", "2026-08-12", "2026-08-14")
    assert set(rows[0]) == {"ts", "open", "high", "low", "close", "volume"}


def test_nested_historical_shape_is_accepted():
    payload = '{"symbol":"X","historical":[{"date":"2026-08-14","open":1,"high":2,"low":1,"close":2,"volume":5}]}'
    with patch("tradingagents.dataflows.fmp_ohlcv._make_request", return_value=payload):
        assert len(get_daily_bars("X", "2026-08-01", "2026-08-14")) == 1


def test_bar_without_a_close_is_dropped_not_zeroed():
    """A zero close is indistinguishable from a real price of 0 downstream."""
    payload = '[{"date":"2026-08-14","open":1,"high":2,"low":1,"close":null,"volume":5}]'
    with patch("tradingagents.dataflows.fmp_ohlcv._make_request", return_value=payload):
        assert get_daily_bars("X", "2026-08-01", "2026-08-14") == []


def test_non_json_response_yields_no_rows():
    with patch("tradingagents.dataflows.fmp_ohlcv._make_request", return_value="<html>502</html>"):
        assert get_daily_bars("X", "2026-08-01", "2026-08-14") == []


def test_empty_payload_yields_no_rows():
    with patch("tradingagents.dataflows.fmp_ohlcv._make_request", return_value="[]"):
        assert get_daily_bars("X", "2026-08-01", "2026-08-14") == []


def test_module_does_not_import_yfinance():
    """The whole point: this tier is licensed."""
    import tradingagents.dataflows.fmp_ohlcv as m
    src = open(m.__file__).read()
    assert "import yfinance" not in src
    assert "y_finance" not in src.replace("# ", "")
