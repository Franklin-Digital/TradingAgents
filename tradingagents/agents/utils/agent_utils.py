from langchain_core.messages import HumanMessage, RemoveMessage

# --- Context-budget constants ---
# DeepSeek context = 131K tokens ≈ 524K chars.  Each debate prompt embeds
# 4 analyst reports + history + instructions.  These caps keep the total
# prompt well under the limit even for mega-cap symbols with rich data.
MAX_TOOL_RESULT_CHARS = 20_000  # per tool call return (news, fundamentals, OHLCV)
MAX_REPORT_CHARS = 8_000      # per analyst report in debate prompts
MAX_HISTORY_CHARS = 15_000    # debate history (investment or risk)
MAX_PAST_CONTEXT_CHARS = 5_000  # memory log past_context


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to *max_chars*, appending a notice when trimmed."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[... truncated to {max_chars:,} chars]"


# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
