import functools
import logging
from collections.abc import Mapping
from typing import Any

import yfinance as yf
from langchain_core.messages import HumanMessage, RemoveMessage

# --- Context-budget constants ---
#
# CEILING = the smallest context window in the serving chain, NOT the primary's.
#
# Production runs Nemotron 3.5 Lightning (262,144 ctx) as primary with
# BIFROST_FALLBACK_MODELS=deepseek/deepseek-chat (131,072 ctx) behind it.
# Bifrost fallback is resolved by the GATEWAY, after the request body is
# already built and sent -- we cannot know which model will serve a call at
# prompt-construction time. Sizing to Nemotron's 262K would therefore build a
# prompt that hard-400s the instant the fallback fires, i.e. exactly when
# robustness matters most. So the budget is DeepSeek's 131,072.
#
# When DeepSeek leaves the fallback chain, this is the one number to change.
CONTEXT_CEILING_TOKENS = 131_072   # DeepSeek V3 -- smallest ctx in the chain
COMPLETION_RESERVE_TOKENS = 16_000  # observed completion peak 11,117 + margin
#
# Caps below are derived from measurement, not round numbers (2026-08-31, 202
# real symbol runs in ~/.tradingagents/logs + 550 Bifrost calls):
#
#   observed max analyst report      14,862 chars  -> MAX_REPORT_CHARS 16,000
#   observed max debate history      31,520 chars  -> MAX_HISTORY_CHARS 36,000
#   observed max past_context        21,548 chars  -> MAX_PAST_CONTEXT_CHARS 24,000
#
# These are CIRCUIT BREAKERS, not routine filters: sized so the ordinary case
# passes whole and only a pathological outlier is clipped. Measured prose
# density is 3.99-4.86 chars/token; the budget below uses a conservative 3.5.
#
# MAX_TOOL_RESULT_CHARS deliberately stays at 20,000. Measured on NVDA/MSFT/
# GOOGL/AMZN/TSLA, eight of the nine analyst tools return LESS than 20,000
# chars uncapped (largest: get_balance_sheet at 15,742), so raising this buys
# them nothing.
#
# The ninth, get_stock_data, is a different problem that a bigger cap cannot
# solve. Measured at the vendor boundary (below route_to_vendor's _truncate,
# production routing core_stock_apis="questdb"), NVDA for the requested range
# 2026-01-01..2026-08-30 returns:
#
#   raw            2,140,847 chars / 34,130 rows, spanning 2026-06-22..2026-08-29
#   after head-cut    20,000 chars /    329 rows, spanning 2026-06-22..2026-06-23
#
# i.e. 0.96% of the rows survive, and because truncation keeps the HEAD of a
# chronologically ascending series, what survives is the OLDEST day. A run on
# 2026-08-30 reasons about prices that stop on 2026-06-23. Filling all of
# Nemotron's 262K context would still keep only a few percent, so the fix is
# downsampling in the tool, not a bigger cap here.
#
# That downsampling now EXISTS: dataflows/text_truncation.truncate_series_text
# keeps the recent rows verbatim and thins the older ones, and it is applied in
# BOTH places a series is capped -- route_to_vendor's _truncate (which runs
# first, below the tool wrappers, and used to discard the recent rows before any
# agent saw them) and get_stock_data itself. MAX_TOOL_RESULT_CHARS stays 20,000:
# the analyst-phase estimate below is anchored on it, and the fix changes WHICH
# rows survive, not how many chars do.
#
# Separately and independently: the vendor returned nothing before 2026-06-22
# despite the request starting 2026-01-01. That is a data-path gap, not a
# truncation effect, and is being tracked outside this change.
#
# Beware when re-measuring: truncate_text's notice is exactly 32 chars, so a
# fully-truncated tool result is 20,032 chars. Reading length at or above the
# tool wrapper therefore reports "20,032 raw, 32 chars dropped, 0 rows lost"
# and hides the 99% loss entirely. Measure at the vendor impl, as above.
#
# Worst case with every cap saturated at once, vs the 131,072 ceiling:
#
#   Analyst phase (unchanged): empirical peak 49,242 tok over 550 calls;
#                              x1.5 tool-call headroom            ~73,900 tok
#   Debate phase:   4 x 16,000 + 36,000 = 100,000 chars / 3.5      ~29,500 tok
#   PM phase:       36,000 + 24,000 + plans ~4,400 + scaffold      ~19,100 tok
#
#   peak = analyst phase ~73,900 + 16,000 completion reserve = 89,900 tok
#        = 69% of 131,072, leaving 31% margin.
#
# Note the two phases being raised (29,500 / 19,100) sit far below the analyst
# phase, so this change does not move the system's peak prompt at all -- it
# only stops clipping the ~5% of symbols whose reports, debate history, or
# accumulated lessons overflowed.
MAX_TOOL_RESULT_CHARS = 20_000  # per tool call return (news, fundamentals, OHLCV)
MAX_REPORT_CHARS = 16_000     # per analyst report in debate prompts
MAX_HISTORY_CHARS = 36_000    # debate history (investment or risk)
MAX_PAST_CONTEXT_CHARS = 24_000  # memory log past_context


# Import tools from separate utility files.  These MUST stay below the
# constants above: each tool module does
#   from tradingagents.agents.utils.agent_utils import MAX_TOOL_RESULT_CHARS
# so the constants have to exist before the tool modules are executed.
# Hoisting these to the top of the file is a circular ImportError.
from tradingagents.agents.utils.core_stock_tools import get_stock_data  # noqa: E402
from tradingagents.agents.utils.fundamental_data_tools import (  # noqa: E402
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from tradingagents.agents.utils.macro_data_tools import get_macro_indicators  # noqa: E402
from tradingagents.agents.utils.market_data_validation_tools import (  # noqa: E402
    get_verified_market_snapshot,
)
from tradingagents.agents.utils.news_data_tools import (  # noqa: E402
    get_global_news,
    get_insider_transactions,
    get_news,
)
from tradingagents.agents.utils.prediction_markets_tools import get_prediction_markets  # noqa: E402
from tradingagents.agents.utils.technical_indicators_tools import get_indicators  # noqa: E402

# Truncation helpers live in dataflows.text_truncation so the vendor-routing
# layer (dataflows.interface.route_to_vendor) shares one implementation without
# a circular import.  Re-exported here because the debate/researcher agents all
# import them from agent_utils.
from tradingagents.dataflows.text_truncation import (  # noqa: E402
    truncate_series_text,
    truncate_text,
)

# Public surface: the data tools are imported here so agents and the graph
# import them from one place, plus the instrument/language helpers defined below.
__all__ = [
    "get_stock_data",
    "get_indicators",
    "truncate_text",
    "truncate_series_text",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_macro_indicators",
    "get_prediction_markets",
    "get_verified_market_snapshot",
    "build_instrument_context",
    "resolve_instrument_identity",
    "get_instrument_context_from_state",
    "get_language_instruction",
    "create_msg_delete",
]

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def opponent_argument_or_opening(text: str, opponent: str) -> str:
    """Opponent's latest argument, or an explicit opening marker when empty.

    The first speaker in each debate round receives an empty opponent response;
    interpolating it into a "refute the opponent" prompt makes the model
    fabricate the other side's position. Returning a clear "has not spoken yet"
    marker instead lets it open with its own case (#1176).
    """
    text = (text or "").strip()
    if text:
        return text
    return f"(The {opponent} has not spoken yet — open the debate with your own case.)"


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@functools.lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Best-effort by design: if yfinance is unavailable, rate-limited, or doesn't
    recognise the ticker, we return ``{}`` and the caller falls back to
    ticker-only context rather than failing before analysis starts. Cached so
    the lookup happens at most once per ticker per process.

    The symbol is normalized first (e.g. ``XAUUSD`` -> ``GC=F``) so identity
    resolves for the same instrument the price path actually fetches (#983).
    """
    from tradingagents.dataflows.symbol_utils import normalize_symbol

    try:
        info = yf.Ticker(normalize_symbol(ticker)).info or {}
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}

    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).
    """
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    details = []
    if identity:
        name = identity.get("company_name") or identity.get("name")
        if name:
            details.append(f"{'Name' if is_crypto else 'Company'}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``TradingAgentsGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(
        str(state["company_of_interest"]),
        state.get("asset_type", "stock"),
    )


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add a context-anchored placeholder.

        The placeholder must not be a bare ``"Continue"``: some
        OpenAI-compatible providers interpret that literally as the user task
        and produce output about the word "continue" instead of analysing the
        instrument (#888). Anchoring it to the resolved instrument context and
        date keeps the next analyst on-task even if the provider treats the
        placeholder as a standalone request.
        """
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        instrument_context = get_instrument_context_from_state(state)
        trade_date = state.get("trade_date", "the requested date")
        placeholder = HumanMessage(
            content=(
                f"Proceed with your assigned analysis for this workflow. "
                f"{instrument_context} The analysis date is {trade_date}."
            )
        )
        return {"messages": removal_operations + [placeholder]}

    return delete_messages



