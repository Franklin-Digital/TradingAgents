"""Franklin Financial data vendor — IBKR real-time + QuestDB historical.

Replaces yfinance as the primary OHLCV and indicators source.  Data flows:

  IBKR Gateway (salcobian-md) → scanner → QuestDB active_universe (live)
  IBKR Gateway (historical)   → ibkr_historical_feed → QuestDB on DGX (deep)

This vendor reads from QuestDB directly — no yfinance, no Yahoo Finance,
no delayed data.  The same real-time IBKR prices that triggered the scanner
are used for the LLM analysis.

Query priority:
  1. QuestDB live-prod (mac-pro) — active_universe, real-time snapshots
  2. QuestDB historical (DGX)   — IBKR historical bars for deep history
  3. Error message (no silent fallback to delayed sources)

Registered in interface.py as vendor "franklin".
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Annotated

import pandas as pd

log = logging.getLogger(__name__)

# --- QuestDB connection defaults (overridable via config or env) ---

_LIVE_HOST = os.getenv("QUESTDB_HOST", "192.168.1.41")
_LIVE_PORT = int(os.getenv("QUESTDB_HTTP_PORT", "9000"))

_HIST_HOST = os.getenv("QUESTDB_HISTORICAL_HOST", "192.168.1.25")
_HIST_PORT = int(os.getenv("QUESTDB_HISTORICAL_HTTP_PORT", "29000"))


def _get_config():
    """Lazy import to avoid circular dependency."""
    from .config import get_config
    return get_config()


def _http_query(sql: str, host: str, port: int, timeout: int = 15) -> list[dict]:
    """Execute a QuestDB HTTP query and return rows as list of dicts."""
    url = f"http://{host}:{port}/exec?" + urllib.parse.urlencode({"query": sql})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        columns = [col["name"] for col in data.get("columns", [])]
        return [dict(zip(columns, row)) for row in data.get("dataset", [])]
    except Exception as e:
        log.warning("QuestDB query failed (%s:%s): %s", host, port, e)
        return []


# ---------------------------------------------------------------------------
# OHLCV — get_stock_data(symbol, start_date, end_date) -> str
# ---------------------------------------------------------------------------

_LIVE_OHLCV_SQL = """\
SELECT
    ts,
    open,
    high,
    low,
    last AS close,
    volume,
    vwap
FROM active_universe
WHERE symbol = '{symbol}'
  AND ts >= '{start}T00:00:00.000000Z'
  AND ts <= '{end}T23:59:59.999999Z'
ORDER BY ts ASC
"""

_HIST_OHLCV_SQL = """\
SELECT
    ts,
    open,
    high,
    low,
    close,
    volume
FROM ibkr_ohlcv_1d
WHERE symbol = '{symbol}'
  AND ts >= '{start}'
  AND ts <= '{end}'
ORDER BY ts ASC
"""


def _format_ohlcv_csv(rows: list[dict], symbol: str, start: str, end: str, source: str) -> str:
    """Format OHLCV rows into the CSV string expected by the agent pipeline."""
    header = (
        f"# Stock data for {symbol} from {start} to {end}\n"
        f"# Source: {source}\n"
        f"# Total records: {len(rows)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    csv_lines = ["Datetime,Open,High,Low,Close,Volume"]
    for r in rows:
        csv_lines.append(
            f"{r.get('ts', '')},"
            f"{round(float(r.get('open') or 0), 2)},"
            f"{round(float(r.get('high') or 0), 2)},"
            f"{round(float(r.get('low') or 0), 2)},"
            f"{round(float(r.get('close') or 0), 2)},"
            f"{int(r.get('volume') or 0)}"
        )
    return header + "\n".join(csv_lines)


def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Return OHLCV data from QuestDB (live-prod first, then historical on DGX).

    Output format matches get_YFin_data_online so the agent pipeline needs
    no changes.
    """
    cfg = _get_config()
    sym = symbol.upper()

    live_host = cfg.get("questdb_host", _LIVE_HOST)
    live_port = cfg.get("questdb_http_port", _LIVE_PORT)

    # 1. Try live-prod (active_universe — real-time IBKR data)
    sql = _LIVE_OHLCV_SQL.format(symbol=sym, start=start_date, end=end_date)
    rows = _http_query(sql, live_host, live_port)
    if rows:
        return _format_ohlcv_csv(
            rows, sym, start_date, end_date,
            f"QuestDB live-prod ({live_host}:{live_port})"
        )

    # 2. Try historical (DGX — IBKR historical bars)
    hist_host = cfg.get("questdb_historical_host", _HIST_HOST)
    hist_port = cfg.get("questdb_historical_http_port", _HIST_PORT)
    sql = _HIST_OHLCV_SQL.format(symbol=sym, start=start_date, end=end_date)
    rows = _http_query(sql, hist_host, hist_port)
    if rows:
        return _format_ohlcv_csv(
            rows, sym, start_date, end_date,
            f"QuestDB historical ({hist_host}:{hist_port})"
        )

    return (
        f"No OHLCV data for '{sym}' between {start_date} and {end_date}. "
        f"Symbol may not be in active_universe or historical tables."
    )


# ---------------------------------------------------------------------------
# Technical Indicators — uses QuestDB OHLCV → stockstats
# ---------------------------------------------------------------------------

def _load_ohlcv_df(symbol: str, curr_date: str) -> pd.DataFrame:
    """Load OHLCV as a pandas DataFrame for stockstats indicator calculation.

    Fetches up to 1 year of daily data ending at curr_date. Tries live-prod
    first (active_universe snapshots), then historical on DGX (ibkr_ohlcv_1d).
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    cfg = _get_config()
    sym = symbol.upper()
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=365)
    start_str = start_dt.strftime("%Y-%m-%d")

    live_host = cfg.get("questdb_host", _LIVE_HOST)
    live_port = cfg.get("questdb_http_port", _LIVE_PORT)
    hist_host = cfg.get("questdb_historical_host", _HIST_HOST)
    hist_port = cfg.get("questdb_historical_http_port", _HIST_PORT)

    rows = []

    # Try historical first for deep data (more rows = better indicators)
    sql = _HIST_OHLCV_SQL.format(symbol=sym, start=start_str, end=curr_date)
    rows = _http_query(sql, hist_host, hist_port)

    # Supplement with live-prod for the most recent days
    live_sql = _LIVE_OHLCV_SQL.format(symbol=sym, start=start_str, end=curr_date)
    live_rows = _http_query(live_sql, live_host, live_port)

    if not rows and not live_rows:
        return pd.DataFrame()

    # Merge: historical as base, live overlays for recent dates
    all_rows = {r["ts"][:10]: r for r in rows}  # key by date
    for r in live_rows:
        date_key = r["ts"][:10]
        all_rows[date_key] = r  # live overwrites historical for same date

    records = []
    for r in sorted(all_rows.values(), key=lambda x: x["ts"]):
        records.append({
            "Date": pd.to_datetime(r["ts"][:10]),
            "Open": float(r.get("open") or 0),
            "High": float(r.get("high") or 0),
            "Low": float(r.get("low") or 0),
            "Close": float(r.get("close") or 0),
            "Volume": int(r.get("volume") or 0),
        })

    df = pd.DataFrame(records)
    df = df[df["Close"] > 0]  # drop zero-price rows
    return df


def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator name"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "days to look back"] = 30,
) -> str:
    """Calculate a technical indicator from QuestDB OHLCV data via stockstats.

    Same output format as get_stock_stats_indicators_window.
    """
    from stockstats import wrap as stockstats_wrap

    valid_indicators = {
        "close_50_sma", "close_200_sma", "close_10_ema",
        "macd", "macds", "macdh", "rsi",
        "boll", "boll_ub", "boll_lb", "atr",
        "vwma", "mfi",
    }
    if indicator not in valid_indicators:
        raise ValueError(
            f"Indicator '{indicator}' not supported. Choose from: {sorted(valid_indicators)}"
        )

    df = _load_ohlcv_df(symbol, curr_date)
    if df.empty:
        log.warning("No QuestDB OHLCV data for %s — cannot compute %s", symbol, indicator)
        return f"No OHLCV data available for {symbol} to compute {indicator}."

    try:
        sdf = stockstats_wrap(df.copy())
        sdf["Date"] = sdf["Date"].dt.strftime("%Y-%m-%d")
        sdf[indicator]  # trigger calculation

        end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=look_back_days)

        ind_lines = []
        current = end_dt
        while current >= start_dt:
            date_str = current.strftime("%Y-%m-%d")
            match = sdf[sdf["Date"] == date_str]
            if not match.empty:
                val = match[indicator].values[0]
                ind_lines.append(f"{date_str}: {val if pd.notna(val) else 'N/A'}")
            else:
                ind_lines.append(f"{date_str}: N/A: Not a trading day (weekend or holiday)")
            current -= timedelta(days=1)

        best_ind_params = {
            "close_50_sma": "50 SMA: Medium-term trend indicator.",
            "close_200_sma": "200 SMA: Long-term trend benchmark.",
            "close_10_ema": "10 EMA: Responsive short-term average.",
            "macd": "MACD: Momentum via EMA differences.",
            "macds": "MACD Signal: EMA smoothing of MACD.",
            "macdh": "MACD Histogram: Gap between MACD and signal.",
            "rsi": "RSI: Overbought/oversold momentum.",
            "boll": "Bollinger Middle: 20 SMA basis.",
            "boll_ub": "Bollinger Upper Band: 2 std dev above.",
            "boll_lb": "Bollinger Lower Band: 2 std dev below.",
            "atr": "ATR: Average true range volatility.",
            "vwma": "VWMA: Volume-weighted moving average.",
            "mfi": "MFI: Money Flow Index (price + volume).",
        }

        result = (
            f"## {indicator} values from {start_dt.strftime('%Y-%m-%d')} to {curr_date}:\n"
            f"# Source: QuestDB (Franklin Financial)\n\n"
            + "\n".join(ind_lines)
            + "\n\n"
            + best_ind_params.get(indicator, "")
        )
        return result

    except Exception as e:
        log.error("Failed to compute %s for %s from QuestDB data: %s", indicator, symbol, e)
        return f"Error computing {indicator} for {symbol}: {e}"
