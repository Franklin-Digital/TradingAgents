from typing import Annotated

from langchain_core.tools import tool

from tradingagents.agents.utils.agent_utils import MAX_TOOL_RESULT_CHARS
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.text_truncation import truncate_series_text


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    result = route_to_vendor("get_stock_data", symbol, start_date, end_date)
    # NOT truncate_text: the vendor returns bars in ascending time order, so
    # head-truncation kept the OLDEST 1% and dropped every recent row (see
    # truncate_series_text).  This keeps the recent rows verbatim and
    # downsamples the older ones so the full span still reaches the model.
    return truncate_series_text(str(result), MAX_TOOL_RESULT_CHARS)
