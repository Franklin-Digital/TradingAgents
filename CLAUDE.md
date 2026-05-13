# CLAUDE.md — TauricTradingAgents (Franklin fork)

This is the Franklin fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).
It runs multi-agent LLM analysis on scanner-filtered symbols and writes decisions to QuestDB.

## Franklin-Specific Configuration

### LLM Provider

Franklin runs **llama-4-scout** via the Bifrost AI Gateway (vLLM on the DGX).
No API key required — the gateway is on the internal network via Cloudflare tunnel.

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"]    = "vllm"
config["deep_think_llm"]  = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"
config["quick_think_llm"] = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"
```

Gateway endpoint: `https://ai-gateway.franklinfinancial.ai/v1` (default).
Override via env: `VLLM_BASE_URL`, `VLLM_API_KEY` (optional — only if gateway adds auth).

Anthropic Claude is still supported as a fallback:

```python
config["llm_provider"]    = "anthropic"
config["deep_think_llm"]  = "claude-sonnet-4-6"
config["quick_think_llm"] = "claude-haiku-4-5-20251001"
# Requires ANTHROPIC_API_KEY
```

### QuestDB Data Source

Scanner-universe names use QuestDB `active_universe` as the price data source
(eliminates redundant Alpha Vantage / yfinance calls for names already tracked by the scanner).

```python
config["data_vendors"]["core_stock_apis"] = "questdb"
config["questdb_host"]      = "192.168.1.41"
config["questdb_http_port"] = 9000
```

The QuestDB vendor is wired in `tradingagents/dataflows/questdb_stock.py` and registered
in `tradingagents/dataflows/interface.py`. It falls back to yfinance automatically when
the symbol is not in active_universe for the requested date range.

Override host/port via environment:
```bash
export QUESTDB_HOST=192.168.1.41
export QUESTDB_HTTP_PORT=9000
```

### Running via the DayTradingAgent dispatcher

The normal production path is through `DayTradingAgent/integration/universe_dispatcher.py`,
which pulls the top 10 priority symbols from `active_universe` and calls `ta.propagate()`:

```bash
# From DayTradingAgent/
PYTHONPATH=../TauricTradingAgents python -m integration.universe_dispatcher
```

Decisions are written back to QuestDB `trading_decisions` by `integration/decisions_writer.py`.

### Running standalone

```bash
cd /Users/Sal/Projects/Franklin/TauricTradingAgents
pip install -e .
python -m cli.main   # interactive CLI — select "vllm" provider

# or programmatically via Bifrost:
python -c "
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
config = DEFAULT_CONFIG.copy()
config['llm_provider']    = 'vllm'
config['deep_think_llm']  = 'llama-4-scout'
config['quick_think_llm'] = 'llama-4-scout'
ta = TradingAgentsGraph(config=config)
state, decision = ta.propagate('NVDA', '2026-05-13')
print(decision)
"
```

## Repository Layout

```
tradingagents/
  agents/          — Analyst, Researcher, Trader, Risk, Portfolio Manager agents
  dataflows/       — Data vendor routing (yfinance, alpha_vantage, questdb)
    questdb_stock.py   ← Franklin addition: QuestDB OHLCV source
    interface.py       ← vendor routing table (questdb registered here)
    config.py          ← runtime config accessor
  graph/           — LangGraph workflow (trading_graph.py, propagation.py)
  default_config.py  ← all config defaults including questdb_host/port
cli/               — Interactive CLI
```

## Key Interfaces

- `TradingAgentsGraph.propagate(symbol, date)` → `(final_state, signal_string)`
  - `final_state`: dict with `final_trade_decision` (markdown), `investment_debate_state`, `risk_debate_state`
  - `signal_string`: one of `Buy | Overweight | Hold | Underweight | Sell`

## Integration with DayTradingAgent

Data flow:
```
QuestDB active_universe (scanner output)
    ↓  DayTradingAgent/integration/universe_dispatcher.py
TradingAgentsGraph.propagate(symbol, date)
    ↓  DayTradingAgent/integration/decisions_writer.py
QuestDB trading_decisions
```

QuestDB is the only integration point. No direct imports between repos.

## Deployment

Same MacBook → git push → mac-pro git pull workflow as all Franklin repos.
No systemd service yet — dispatcher is run on-demand or via cron on mac-pro.
