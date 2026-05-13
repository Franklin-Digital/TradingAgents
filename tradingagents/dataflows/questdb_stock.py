"""
QuestDB price data source for TauricTradingAgents.

Reads OHLCV data directly from the DayTradingAgent active_universe table,
eliminating redundant Alpha Vantage / yfinance calls for names the scanner
already tracks. Returns data in the same CSV format as get_YFin_data_online
so the rest of the agent pipeline needs no changes.

The QuestDB connection defaults to the Franklin prod instance at 192.168.1.41:9000.
Override via the 'questdb_host' and 'questdb_http_port' keys in DEFAULT_CONFIG,
or via QUESTDB_HOST / QUESTDB_HTTP_PORT environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Annotated

log = logging.getLogger(__name__)

_DEFAULT_HOST = os.getenv("QUESTDB_HOST", "192.168.1.41")
_DEFAULT_PORT = int(os.getenv("QUESTDB_HTTP_PORT", "9000"))

# active_universe columns available for OHLCV reconstruction:
#   open, high, low, last (≈ close), volume, vwap, ts
_OHLCV_SQL = """
SELECT
    ts,
    open,
    high,
    low,
    last  AS close,
    volume,
    vwap
FROM active_universe
WHERE symbol = '{symbol}'
  AND ts >= '{start_date}T00:00:00.000000Z'
  AND ts <= '{end_date}T23:59:59.999999Z'
ORDER BY ts ASC
"""


def _http_query(sql: str, host: str, port: int) -> list[dict]:
    url = f"http://{host}:{port}/exec?" + urllib.parse.urlencode({"query": sql})
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        columns = [col["name"] for col in data.get("columns", [])]
        return [dict(zip(columns, row)) for row in data.get("dataset", [])]
    except Exception as e:
        log.warning("QuestDB query failed: %s", e)
        return []


def get_questdb_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Return OHLCV data for symbol from QuestDB active_universe as a CSV string.

    Falls back to a clear error message (not an exception) when the symbol
    is not in active_universe or the date range has no data — the caller can
    detect the empty result and fall through to yfinance.

    Output format matches get_YFin_data_online:
        # Stock data for NVDA from 2026-05-12 to 2026-05-12
        # Total records: N
        ...
        Datetime,Open,High,Low,Close,Volume,Vwap
        2026-05-12T09:34:00.000000Z,...
    """
    from .config import get_config
    cfg = get_config()
    host = cfg.get("questdb_host", _DEFAULT_HOST)
    port = cfg.get("questdb_http_port", _DEFAULT_PORT)

    rows = _http_query(
        _OHLCV_SQL.format(symbol=symbol.upper(), start_date=start_date, end_date=end_date),
        host,
        port,
    )

    if not rows:
        return (
            f"No QuestDB data for '{symbol}' between {start_date} and {end_date}. "
            f"Symbol may not be in active_universe for this period."
        )

    header = (
        f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
        f"# Source: QuestDB active_universe ({host}:{port})\n"
        f"# Total records: {len(rows)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    csv_lines = ["Datetime,Open,High,Low,Close,Volume,Vwap"]
    for r in rows:
        csv_lines.append(
            f"{r['ts']},"
            f"{round(float(r.get('open') or 0), 2)},"
            f"{round(float(r.get('high') or 0), 2)},"
            f"{round(float(r.get('low') or 0), 2)},"
            f"{round(float(r.get('close') or 0), 2)},"
            f"{int(r.get('volume') or 0)},"
            f"{round(float(r.get('vwap') or 0), 2)}"
        )

    return header + "\n".join(csv_lines)
