"""Tests for the franklinfinancial data vendor.

Covers the full fallback chain (QuestDB live → QuestDB historical → yfinance)
for both OHLCV and indicator functions, plus CSV parsing, error handling, and
the trading_graph integration.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows.franklinfinancial import (
    _HIST_OHLCV_SQL,
    _LIVE_OHLCV_SQL,
    _format_ohlcv_csv,
    _http_query,
    _load_ohlcv_df,
    get_indicators,
    get_stock_data,
)
from tradingagents.dataflows.interface import VENDOR_LIST, VENDOR_METHODS, route_to_vendor
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return {
        "questdb_host": "192.168.1.41",
        "questdb_http_port": 9000,
        "questdb_historical_host": "192.168.1.25",
        "questdb_historical_http_port": 39000,
    }


@pytest.fixture
def patch_config(mock_config):
    with patch(
        "tradingagents.dataflows.franklinfinancial._get_config",
        return_value=mock_config,
    ):
        yield mock_config


def _mock_urlopen(response_data):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


SAMPLE_HIST_RESPONSE = {
    "columns": [
        {"name": "ts", "type": "TIMESTAMP"},
        {"name": "open", "type": "DOUBLE"},
        {"name": "high", "type": "DOUBLE"},
        {"name": "low", "type": "DOUBLE"},
        {"name": "close", "type": "DOUBLE"},
        {"name": "volume", "type": "LONG"},
    ],
    "dataset": [
        ["2026-01-10T00:00:00.000000Z", 120.00, 122.00, 119.50, 121.50, 2000000],
        ["2026-01-11T00:00:00.000000Z", 121.50, 123.00, 121.00, 122.80, 2200000],
    ],
}

EMPTY_RESPONSE = {"columns": [], "dataset": []}


# ---------------------------------------------------------------------------
# _http_query
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHttpQuery:
    def test_returns_list_of_dicts(self):
        with patch("tradingagents.dataflows.franklinfinancial.urllib.request.urlopen") as mock_url:
            mock_url.return_value = _mock_urlopen(SAMPLE_HIST_RESPONSE)
            rows = _http_query("SELECT 1", "localhost", 9000)

        assert len(rows) == 2
        assert rows[0]["ts"] == "2026-01-10T00:00:00.000000Z"
        assert rows[0]["close"] == 121.50
        assert rows[1]["volume"] == 2200000

    def test_returns_empty_on_network_error(self):
        with patch("tradingagents.dataflows.franklinfinancial.urllib.request.urlopen") as mock_url:
            mock_url.side_effect = ConnectionError("refused")
            assert _http_query("SELECT 1", "localhost", 9000) == []

    def test_returns_empty_on_timeout(self):
        with patch("tradingagents.dataflows.franklinfinancial.urllib.request.urlopen") as mock_url:
            mock_url.side_effect = TimeoutError("timed out")
            assert _http_query("SELECT 1", "localhost", 9000, timeout=1) == []

    def test_empty_dataset(self):
        with patch("tradingagents.dataflows.franklinfinancial.urllib.request.urlopen") as mock_url:
            mock_url.return_value = _mock_urlopen(EMPTY_RESPONSE)
            assert _http_query("SELECT 1", "localhost", 9000) == []

    def test_url_encodes_sql(self):
        with patch("tradingagents.dataflows.franklinfinancial.urllib.request.urlopen") as mock_url:
            mock_url.return_value = _mock_urlopen(EMPTY_RESPONSE)
            _http_query("SELECT * FROM t WHERE x = 'a b'", "host", 9000)
        call_url = mock_url.call_args[0][0]
        assert "query=" in call_url


# ---------------------------------------------------------------------------
# get_stock_data — OHLCV with fallback chain
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetStockData:
    def test_returns_live_prod_data_first(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.return_value = [
                {"ts": "2026-07-10T14:30:00Z", "open": 130.5, "high": 132.1, "low": 129.8, "close": 131.25, "volume": 1500000},
            ]
            result = get_stock_data("NVDA", "2026-07-10", "2026-07-11")

        assert "QuestDB live-prod" in result
        assert "131.25" in result
        assert "Datetime,Open,High,Low,Close,Volume" in result
        mock_q.assert_called_once()

    def test_falls_back_to_historical_when_live_empty(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.side_effect = [
                [],
                [{"ts": "2026-01-10T00:00:00Z", "open": 120, "high": 122, "low": 119.5, "close": 121.5, "volume": 2000000}],
            ]
            result = get_stock_data("NVDA", "2026-01-10", "2026-01-11")

        assert "QuestDB historical" in result
        assert "121.5" in result
        assert mock_q.call_count == 2

    def test_falls_back_to_FMP_not_yfinance_when_both_questdb_empty(self, patch_config):
        """This test used to assert the opposite, and that is the bug.

        It pinned a yfinance fallback, which CLAUDE.md bans in production
        (licensed vendors only, fail loud). The path was live: while the
        historical port pointed at the frozen archive, QuestDB had no recent
        bars and ai-score rated positions on scraped prices without one error.
        """
        fmp_rows = [{"ts": "2026-07-10", "open": 10, "high": 11,
                     "low": 9, "close": 10.5, "volume": 100}]
        with patch("tradingagents.dataflows.franklinfinancial._http_query", return_value=[]), \
             patch("tradingagents.dataflows.fmp_ohlcv.get_daily_bars", return_value=fmp_rows) as mock_fmp:
            result = get_stock_data("RKLB", "2026-07-10", "2026-07-11")

        mock_fmp.assert_called_once()
        assert "FMP historical-price-eod" in result
        assert "yfinance" not in result.lower()

    def test_reports_NO_DATA_when_questdb_and_FMP_are_both_empty(self, patch_config):
        """Empty beats unlicensed. The agent must see "no data", not a price."""
        with patch("tradingagents.dataflows.franklinfinancial._http_query", return_value=[]), \
             patch("tradingagents.dataflows.fmp_ohlcv.get_daily_bars", return_value=[]):
            result = get_stock_data("RKLB", "2026-07-10", "2026-07-11")

        assert "NO DATA" in result
        assert "Total records: 0" in result

    def test_symbol_uppercased(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.return_value = [
                {"ts": "2026-07-10T14:30:00Z", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
            ]
            get_stock_data("nvda", "2026-07-10", "2026-07-11")

        sql_arg = mock_q.call_args[0][0]
        assert "symbol = 'NVDA'" in sql_arg

    def test_csv_format_has_correct_header(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.return_value = [
                {"ts": "2026-07-10T14:30:00Z", "open": 130.5, "high": 132.1, "low": 129.8, "close": 131.25, "volume": 1500000},
            ]
            result = get_stock_data("NVDA", "2026-07-10", "2026-07-11")

        lines = [line for line in result.splitlines() if line and not line.startswith("#")]
        assert lines[0] == "Datetime,Open,High,Low,Close,Volume"

    def test_csv_metadata_comments(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.return_value = [
                {"ts": "2026-07-10T14:30:00Z", "open": 130.5, "high": 132.1, "low": 129.8, "close": 131.25, "volume": 1500000},
            ]
            result = get_stock_data("NVDA", "2026-07-10", "2026-07-11")

        assert "# Stock data for NVDA" in result
        assert "# Source:" in result
        assert "# Total records: 1" in result

    def test_live_query_uses_correct_hosts(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.return_value = [
                {"ts": "2026-07-10T14:30:00Z", "open": 130, "high": 132, "low": 129, "close": 131, "volume": 1500000},
            ]
            get_stock_data("NVDA", "2026-07-10", "2026-07-11")

        assert mock_q.call_args[0][1] == "192.168.1.41"
        assert mock_q.call_args[0][2] == 9000

    def test_historical_query_uses_dgx(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.side_effect = [[], [{"ts": "2026-01-10", "open": 120, "high": 122, "low": 119, "close": 121, "volume": 2000000}]]
            get_stock_data("NVDA", "2026-01-10", "2026-01-11")

        assert mock_q.call_args_list[1][0][1] == "192.168.1.25"
        # 39000 = live ibkr-fmp historical. This asserted 29000 (the FROZEN
        # Databento archive) and so PINNED the routing bug in place.
        assert mock_q.call_args_list[1][0][2] == 39000


# ---------------------------------------------------------------------------
# _format_ohlcv_csv
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFormatOhlcvCsv:
    def test_handles_none_values_gracefully(self):
        rows = [{"ts": "2026-07-10T14:30:00Z", "open": None, "high": 132, "low": None, "close": 131, "volume": None}]
        result = _format_ohlcv_csv(rows, "TEST", "2026-07-10", "2026-07-10", "test")
        assert "0.0" in result
        assert "131.0" in result

    def test_empty_rows(self):
        result = _format_ohlcv_csv([], "TEST", "2026-07-10", "2026-07-10", "test")
        assert "Total records: 0" in result
        assert "Datetime,Open,High,Low,Close,Volume" in result

    def test_multiple_rows_ordered(self):
        rows = [
            {"ts": "2026-07-10T00:00:00Z", "open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000},
            {"ts": "2026-07-11T00:00:00Z", "open": 103, "high": 108, "low": 102, "close": 107, "volume": 1200},
        ]
        result = _format_ohlcv_csv(rows, "TEST", "2026-07-10", "2026-07-11", "test")
        data_lines = [line for line in result.splitlines()
                      if line and not line.startswith("#") and line != "Datetime,Open,High,Low,Close,Volume"]
        assert len(data_lines) == 2
        assert data_lines[0].startswith("2026-07-10")

    def test_rounding(self):
        rows = [{"ts": "2026-07-10", "open": 100.456, "high": 101.789, "low": 99.123, "close": 100.999, "volume": 1234}]
        result = _format_ohlcv_csv(rows, "T", "2026-07-10", "2026-07-10", "test")
        assert "100.46" in result


# ---------------------------------------------------------------------------
# _load_ohlcv_df — DataFrame loading with merge and yfinance fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadOhlcvDf:
    def test_merges_historical_and_live(self, patch_config):
        hist_rows = [
            {"ts": "2026-07-08T00:00:00Z", "open": 125, "high": 127, "low": 124, "close": 126, "volume": 1000000},
            {"ts": "2026-07-09T00:00:00Z", "open": 126, "high": 128, "low": 125, "close": 127, "volume": 1100000},
        ]
        live_rows = [
            {"ts": "2026-07-09T15:00:00Z", "open": 127, "high": 129, "low": 126, "close": 128.5, "volume": 1200000},
            {"ts": "2026-07-10T15:00:00Z", "open": 128.5, "high": 130, "low": 128, "close": 129.75, "volume": 1300000},
        ]

        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.side_effect = [hist_rows, live_rows]
            df = _load_ohlcv_df("NVDA", "2026-07-10")

        assert len(df) == 3
        jul9 = df[df["Date"] == pd.Timestamp("2026-07-09")]
        assert float(jul9["Close"].iloc[0]) == 128.5

    def test_live_overwrites_historical_same_date(self, patch_config):
        hist_rows = [{"ts": "2026-07-10T00:00:00Z", "open": 100, "high": 105, "low": 99, "close": 101, "volume": 500}]
        live_rows = [{"ts": "2026-07-10T15:30:00Z", "open": 100, "high": 106, "low": 99, "close": 104, "volume": 800}]

        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.side_effect = [hist_rows, live_rows]
            df = _load_ohlcv_df("TEST", "2026-07-10")

        assert len(df) == 1
        assert float(df["Close"].iloc[0]) == 104

    def test_falls_back_to_FMP_not_yfinance_when_both_empty(self, patch_config):
        """Indicators built on scraped prices are as unlicensed as the prices.

        stockstats_utils.load_ohlcv is a yfinance path; this asserted it was
        called.
        """
        fmp_rows = [{"ts": "2026-07-10", "open": 130.0, "high": 132.0,
                     "low": 129.0, "close": 131.0, "volume": 1500000}]
        with patch("tradingagents.dataflows.franklinfinancial._http_query", return_value=[]), \
             patch("tradingagents.dataflows.fmp_ohlcv.get_daily_bars", return_value=fmp_rows) as mock_fmp:
            df = _load_ohlcv_df("RKLB", "2026-07-10")

        mock_fmp.assert_called_once()
        assert len(df) == 1
        assert float(df.iloc[0]["Close"]) == 131.0

    def test_returns_an_empty_frame_when_questdb_and_FMP_are_both_empty(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query", return_value=[]), \
             patch("tradingagents.dataflows.fmp_ohlcv.get_daily_bars", return_value=[]):
            df = _load_ohlcv_df("RKLB", "2026-07-10")

        assert df.empty
        assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]

    def test_filters_zero_close_rows(self, patch_config):
        rows = [
            {"ts": "2026-07-10T00:00:00Z", "open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000},
            {"ts": "2026-07-11T00:00:00Z", "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0},
        ]

        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.side_effect = [rows, []]
            df = _load_ohlcv_df("TEST", "2026-07-11")

        assert len(df) == 1
        assert float(df["Close"].iloc[0]) == 103

    def test_lookback_is_365_days(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.return_value = []
            with patch("tradingagents.dataflows.stockstats_utils.load_ohlcv", return_value=pd.DataFrame()):
                _load_ohlcv_df("NVDA", "2026-07-10")

        hist_sql = mock_q.call_args_list[0][0][0]
        assert "2025-07-10" in hist_sql

    def test_historical_queried_before_live(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.return_value = [
                {"ts": "2026-07-10T00:00:00Z", "open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000},
            ]
            _load_ohlcv_df("NVDA", "2026-07-10")

        assert mock_q.call_count == 2
        assert mock_q.call_args_list[0][0][1] == "192.168.1.25"  # historical first
        assert mock_q.call_args_list[1][0][1] == "192.168.1.41"  # live second


# ---------------------------------------------------------------------------
# get_indicators — technical indicators via stockstats
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetIndicators:
    def _make_ohlcv_df(self, days=60):
        dates = pd.date_range("2026-05-01", periods=days, freq="B")
        return pd.DataFrame({
            "Date": dates,
            "Open": [100 + i * 0.5 for i in range(len(dates))],
            "High": [101 + i * 0.5 for i in range(len(dates))],
            "Low": [99 + i * 0.5 for i in range(len(dates))],
            "Close": [100 + i * 0.5 for i in range(len(dates))],
            "Volume": [1000000 + i * 10000 for i in range(len(dates))],
        })

    def test_invalid_indicator_raises(self, patch_config):
        with pytest.raises(ValueError, match="not supported"):
            get_indicators("NVDA", "fake_indicator", "2026-07-10")

    def test_empty_df_returns_message(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._load_ohlcv_df", return_value=pd.DataFrame()):
            result = get_indicators("RKLB", "rsi", "2026-07-10")
        assert "No OHLCV data" in result

    def test_rsi_computes_successfully(self, patch_config):
        df = self._make_ohlcv_df()
        with patch("tradingagents.dataflows.franklinfinancial._load_ohlcv_df", return_value=df):
            result = get_indicators("NVDA", "rsi", "2026-07-10", 5)
        assert "rsi" in result.lower()
        assert "QuestDB" in result

    def test_macd_computes_successfully(self, patch_config):
        df = self._make_ohlcv_df()
        with patch("tradingagents.dataflows.franklinfinancial._load_ohlcv_df", return_value=df):
            result = get_indicators("NVDA", "macd", "2026-07-10", 5)
        assert "macd" in result.lower()

    def test_bollinger_computes_successfully(self, patch_config):
        df = self._make_ohlcv_df()
        with patch("tradingagents.dataflows.franklinfinancial._load_ohlcv_df", return_value=df):
            result = get_indicators("NVDA", "boll_ub", "2026-07-10", 5)
        assert "boll_ub" in result.lower()

    def test_all_valid_indicators_accepted(self, patch_config):
        valid = [
            "close_50_sma", "close_200_sma", "close_10_ema",
            "macd", "macds", "macdh", "rsi",
            "boll", "boll_ub", "boll_lb", "atr",
            "vwma", "mfi",
        ]
        for ind in valid:
            try:
                get_indicators("NVDA", ind, "2026-07-10")
            except ValueError:
                pytest.fail(f"Indicator '{ind}' rejected by validation")
            except Exception:
                pass

    def test_look_back_days_controls_output_range(self, patch_config):
        df = self._make_ohlcv_df()
        with patch("tradingagents.dataflows.franklinfinancial._load_ohlcv_df", return_value=df):
            result_5 = get_indicators("NVDA", "rsi", "2026-07-10", 5)
            result_30 = get_indicators("NVDA", "rsi", "2026-07-10", 30)

        lines_5 = [line for line in result_5.splitlines() if line.startswith("2026-")]
        lines_30 = [line for line in result_30.splitlines() if line.startswith("2026-")]
        assert len(lines_5) < len(lines_30)

    def test_indicator_description_included(self, patch_config):
        df = self._make_ohlcv_df()
        with patch("tradingagents.dataflows.franklinfinancial._load_ohlcv_df", return_value=df):
            result = get_indicators("NVDA", "rsi", "2026-07-10", 5)
        assert "RSI" in result


# ---------------------------------------------------------------------------
# Vendor registration in interface.py
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVendorRegistration:
    def test_franklin_in_vendor_list(self):
        assert "franklin" in VENDOR_LIST

    def test_franklin_first_in_vendor_list(self):
        assert VENDOR_LIST[0] == "franklin"

    def test_franklin_registered_for_stock_data(self):
        assert "franklin" in VENDOR_METHODS["get_stock_data"]

    def test_franklin_registered_for_indicators(self):
        assert "franklin" in VENDOR_METHODS["get_indicators"]

    def test_franklin_implementation_is_correct_function(self):
        assert VENDOR_METHODS["get_stock_data"]["franklin"] is get_stock_data
        assert VENDOR_METHODS["get_indicators"]["franklin"] is get_indicators


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDefaultConfig:
    def test_core_stock_apis_defaults_to_franklin(self):
        assert DEFAULT_CONFIG["data_vendors"]["core_stock_apis"] == "franklin"

    def test_technical_indicators_defaults_to_franklin(self):
        assert DEFAULT_CONFIG["data_vendors"]["technical_indicators"] == "franklin"

    def test_historical_questdb_config_present(self):
        assert "questdb_historical_host" in DEFAULT_CONFIG
        assert "questdb_historical_http_port" in DEFAULT_CONFIG

    def test_historical_questdb_defaults_to_dgx(self):
        """The DGX runs TWO QuestDB instances; naming the host is not enough.

        29000 is the frozen Databento archive (ibkr_* stuck at 2026-08-05,
        2,753 symbols); 39000 is the live ibkr-fmp instance (2026-08-14,
        11,558 symbols). This test asserted 29000 and pinned the bug.
        """
        assert DEFAULT_CONFIG["questdb_historical_host"] == "192.168.1.25"
        assert DEFAULT_CONFIG["questdb_historical_http_port"] == 39000

    def test_fundamentals_is_fmp(self):
        # DAYTRADE-179: FMP is the only fundamentals vendor
        assert DEFAULT_CONFIG["data_vendors"]["fundamental_data"] == "fmp"

    def test_news_still_yfinance(self):
        assert DEFAULT_CONFIG["data_vendors"]["news_data"] == "yfinance"


# ---------------------------------------------------------------------------
# trading_graph._parse_ohlcv_csv
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseOhlcvCsv:
    def test_parses_standard_csv(self):
        csv = (
            "# Stock data for NVDA\n"
            "# Source: QuestDB\n"
            "# Total records: 2\n\n"
            "Datetime,Open,High,Low,Close,Volume\n"
            "2026-07-10T14:30:00Z,130.5,132.1,129.8,131.25,1500000\n"
            "2026-07-11T14:30:00Z,131.25,133.0,130.5,132.75,1800000\n"
        )
        df = TradingAgentsGraph._parse_ohlcv_csv(csv)
        assert len(df) == 2
        assert "Close" in df.columns
        assert float(df["Close"].iloc[0]) == 131.25

    def test_skips_comment_lines(self):
        csv = (
            "# comment 1\n"
            "# comment 2\n"
            "Datetime,Open,High,Low,Close,Volume\n"
            "2026-07-10,130,132,129,131,1500000\n"
        )
        df = TradingAgentsGraph._parse_ohlcv_csv(csv)
        assert len(df) == 1

    def test_empty_string_returns_empty_df(self):
        assert TradingAgentsGraph._parse_ohlcv_csv("").empty

    def test_only_comments_returns_empty_df(self):
        assert TradingAgentsGraph._parse_ohlcv_csv("# No data\n# Nothing\n").empty

    def test_parses_dates(self):
        csv = "Datetime,Open,High,Low,Close,Volume\n2026-07-10,130,132,129,131,1500000\n"
        df = TradingAgentsGraph._parse_ohlcv_csv(csv)
        assert pd.api.types.is_datetime64_any_dtype(df["Datetime"])

    def test_parses_yfinance_date_column(self):
        csv = "Date,Open,High,Low,Close,Volume\n2026-07-10,130,132,129,131,1500000\n"
        df = TradingAgentsGraph._parse_ohlcv_csv(csv)
        assert len(df) == 1
        assert "Date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])


# ---------------------------------------------------------------------------
# trading_graph._fetch_returns
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFetchReturns:
    def _make_graph(self):
        return TradingAgentsGraph.__new__(TradingAgentsGraph)

    def test_returns_alpha_when_data_available(self):
        stock_csv = (
            "Datetime,Open,High,Low,Close,Volume\n"
            "2026-07-07,100,105,99,100,1000\n"
            "2026-07-08,101,106,100,102,1100\n"
            "2026-07-09,102,107,101,104,1200\n"
        )
        spy_csv = (
            "Datetime,Open,High,Low,Close,Volume\n"
            "2026-07-07,450,455,449,450,5000000\n"
            "2026-07-08,451,456,450,452,5100000\n"
            "2026-07-09,452,457,451,453,5200000\n"
        )

        with patch("tradingagents.dataflows.franklinfinancial.get_stock_data") as mock_get:
            mock_get.side_effect = [stock_csv, spy_csv]
            raw, alpha, days, resolved = self._make_graph()._fetch_returns(
                "NVDA", "2026-07-07", holding_days=2
            )

        assert raw is not None
        assert alpha is not None
        assert days == 2
        # The resolution date is the last bar used — when the outcome became
        # known (#1251). _parse_ohlcv_csv yields a Datetime column, not an index.
        assert resolved == "2026-07-09"
        expected_raw = (104 - 100) / 100
        expected_spy = (453 - 450) / 450
        assert abs(raw - expected_raw) < 0.001
        assert abs(alpha - (expected_raw - expected_spy)) < 0.001

    def test_returns_none_when_insufficient_data(self):
        with patch("tradingagents.dataflows.franklinfinancial.get_stock_data") as mock_get:
            mock_get.return_value = "# No data\nDatetime,Open,High,Low,Close,Volume\n"
            raw, alpha, days, resolved = self._make_graph()._fetch_returns(
                "RKLB", "2026-07-12", holding_days=5
            )

        assert raw is None
        assert alpha is None
        assert days is None
        assert resolved is None

    def test_returns_none_when_start_close_is_zero(self):
        stock_csv = (
            "Datetime,Open,High,Low,Close,Volume\n"
            "2026-07-07,0,0,0,0,0\n"
            "2026-07-08,101,106,100,102,1100\n"
        )
        spy_csv = (
            "Datetime,Open,High,Low,Close,Volume\n"
            "2026-07-07,450,455,449,450,5000000\n"
            "2026-07-08,451,456,450,452,5100000\n"
        )
        with patch("tradingagents.dataflows.franklinfinancial.get_stock_data") as mock_get:
            mock_get.side_effect = [stock_csv, spy_csv]
            raw, alpha, days, resolved = self._make_graph()._fetch_returns(
                "NVDA", "2026-07-07", holding_days=1
            )
        assert raw is None
        assert resolved is None

    def test_returns_none_on_exception(self):
        with patch("tradingagents.dataflows.franklinfinancial.get_stock_data", side_effect=RuntimeError("boom")):
            raw, alpha, days, resolved = self._make_graph()._fetch_returns("NVDA", "2026-07-07")
        assert raw is None
        assert resolved is None


# ---------------------------------------------------------------------------
# SQL generation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSqlGeneration:
    def test_live_sql_timestamp_format(self):
        sql = _LIVE_OHLCV_SQL.format(symbol="NVDA", start="2026-07-10", end="2026-07-11")
        assert "'2026-07-10T00:00:00.000000Z'" in sql
        assert "'2026-07-11T23:59:59.999999Z'" in sql
        assert "symbol = 'NVDA'" in sql

    def test_hist_sql_date_format(self):
        sql = _HIST_OHLCV_SQL.format(symbol="NVDA", start="2026-01-01", end="2026-07-10")
        assert "'2026-01-01'" in sql
        assert "'2026-07-10'" in sql

    def test_live_sql_uses_last_as_close(self):
        assert "last AS close" in _LIVE_OHLCV_SQL

    def test_hist_sql_uses_close_directly(self):
        assert "close" in _HIST_OHLCV_SQL
        assert "last" not in _HIST_OHLCV_SQL


# ---------------------------------------------------------------------------
# route_to_vendor — franklin is used by default
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRouteToVendor:
    def test_stock_data_routes_to_franklin(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._http_query") as mock_q:
            mock_q.return_value = [
                {"ts": "2026-07-10T14:30:00Z", "open": 130, "high": 132, "low": 129, "close": 131, "volume": 1500000},
            ]
            result = route_to_vendor("get_stock_data", "NVDA", "2026-07-10", "2026-07-11")

        assert "NVDA" in result
        assert "QuestDB" in result

    def test_indicators_route_to_franklin(self, patch_config):
        with patch("tradingagents.dataflows.franklinfinancial._load_ohlcv_df", return_value=pd.DataFrame()):
            result = route_to_vendor("get_indicators", "NVDA", "rsi", "2026-07-10", 30)

        assert "No OHLCV data" in result
