import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# --- Franklin AI Gateway (Bifrost) ------------------------------------------
# Bifrost is the single entry point for ALL LLM traffic so token usage is
# metered per request (Bifrost logs prompt/completion/total tokens to its
# logs.db). Point Tauric at the gateway and let Bifrost route to whatever
# model is requested — llama-4-scout (vLLM on the DGX) today, and Claude / GPT
# / others as those providers are enabled in Bifrost.
#
# Nothing here is a hardcode: the endpoint and model are env-overridable so a
# different model (e.g. "claude-sonnet-4-6", "gpt-5.4") can be selected with no
# code change — just set BIFROST_MODEL (or the deep/quick split below).
BIFROST_BASE_URL = os.getenv(
    "BIFROST_BASE_URL",
    # VLLM_BASE_URL kept as a fallback for backward compatibility with the
    # previous vllm-only configuration.
    os.getenv("VLLM_BASE_URL", "https://ai-gateway.franklinfinancial.ai/v1"),
)
# Default model id Bifrost routes to the DGX vLLM deployment. A DEFAULT, not a
# hardcode — override with BIFROST_MODEL, or per-tier with BIFROST_DEEP_MODEL /
# BIFROST_QUICK_MODEL, to point at any model Bifrost serves.
BIFROST_DEFAULT_MODEL = os.getenv(
    "BIFROST_MODEL", "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"
)
BIFROST_DEEP_MODEL = os.getenv("BIFROST_DEEP_MODEL", BIFROST_DEFAULT_MODEL)
BIFROST_QUICK_MODEL = os.getenv("BIFROST_QUICK_MODEL", BIFROST_DEFAULT_MODEL)
# Provider Tauric uses to reach Bifrost. "vllm" is OpenAI-Chat-Completions
# compatible and keyless, which is how the gateway is exposed.
BIFROST_PROVIDER = os.getenv("BIFROST_PROVIDER", "vllm")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance, questdb
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "questdb",  # Use QuestDB for scanner-universe names
    },
    # QuestDB connection (Franklin prod instance — used when data_vendors.core_stock_apis = "questdb")
    "questdb_host": os.getenv("QUESTDB_HOST", "192.168.1.41"),
    "questdb_http_port": int(os.getenv("QUESTDB_HTTP_PORT", "9000")),
    # -- Confluence report publishing --------------------------------------
    # Every call to ta.propagate() auto-publishes a timestamped report page:
    #   Reports -> yyyy -> yyyy-mm -> yyyy-mm-dd HH:MM:SS ET · SYMBOL · Signal
    # Set confluence_publish=False to disable without removing these keys.
    # Required env vars in franklin.env:
    #   CONFLUENCE_USER_EMAIL   sal.cobian@franklinfinancial.ai
    #   CONFLUENCE_API_TOKEN    <atlassian api token>
    "confluence_publish":         True,
    "confluence_base_url":        "https://franklindigitalcorp.atlassian.net/wiki",
    # The Trading space's active key is "trading" (it was renamed from the
    # legacy "TradingAge"; both still work via Confluence's alias mechanism).
    "confluence_space_key":       "trading",
    "confluence_parent_page_id":  "1376579",   # Reports root page
}
