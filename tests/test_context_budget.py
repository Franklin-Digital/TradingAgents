"""Guard the ai-score prompt budget against the SMALLEST context in the chain.

Production sends every call to Bifrost with Nemotron 3.5 Lightning (262,144
ctx) as primary and ``BIFROST_FALLBACK_MODELS=deepseek/deepseek-chat``
(131,072 ctx) behind it. Bifrost resolves the fallback itself, after the body
is built and sent -- so the prompt has to fit the *smallest* window in the
chain, or it hard-400s the moment the fallback fires.

This recomputes the worst-case budget from the constants themselves rather
than restating the numbers, so raising any cap past the ceiling fails here
instead of in production at 02:00.
"""

from tradingagents.agents.utils.agent_utils import (
    COMPLETION_RESERVE_TOKENS,
    CONTEXT_CEILING_TOKENS,
    MAX_HISTORY_CHARS,
    MAX_PAST_CONTEXT_CHARS,
    MAX_REPORT_CHARS,
    MAX_TOOL_RESULT_CHARS,
    truncate_text,
)

# Measured 2026-08-31 on real payloads (tiktoken cl100k_base).
# Prose (analyst reports, debate history, past_context) measured 3.99-4.86
# chars/token; dense OHLCV CSV measured 1.55. Both rounded against us.
PROSE_CHARS_PER_TOKEN = 3.5
DENSE_CHARS_PER_TOKEN = 1.5

# Empirical analyst-phase peak: 49,242 prompt tokens over 550 Bifrost calls,
# at MAX_TOOL_RESULT_CHARS = 20,000. The phase scales linearly in that cap.
OBSERVED_ANALYST_PEAK_TOKENS = 49_242
OBSERVED_AT_TOOL_CHARS = 20_000
# Design headroom: assume an unluckier symbol makes 1.5x the tool calls.
TOOL_CALL_HEADROOM = 1.5

# Prompt scaffolding (instructions, instrument context, plans) in chars.
DEBATE_SCAFFOLD_CHARS = 3_000
PM_PLAN_AND_SCAFFOLD_CHARS = 6_900  # max research_plan 3,441 + trader 948 + text


def _analyst_phase_tokens() -> float:
    scale = MAX_TOOL_RESULT_CHARS / OBSERVED_AT_TOOL_CHARS
    return OBSERVED_ANALYST_PEAK_TOKENS * scale * TOOL_CALL_HEADROOM


def _debate_phase_tokens() -> float:
    chars = 4 * MAX_REPORT_CHARS + MAX_HISTORY_CHARS + DEBATE_SCAFFOLD_CHARS
    return chars / PROSE_CHARS_PER_TOKEN


def _pm_phase_tokens() -> float:
    chars = MAX_HISTORY_CHARS + MAX_PAST_CONTEXT_CHARS + PM_PLAN_AND_SCAFFOLD_CHARS
    return chars / PROSE_CHARS_PER_TOKEN


def test_worst_case_prompt_fits_the_smallest_context_in_the_chain():
    peak = max(_analyst_phase_tokens(), _debate_phase_tokens(), _pm_phase_tokens())
    budget = CONTEXT_CEILING_TOKENS - COMPLETION_RESERVE_TOKENS
    assert peak <= budget, (
        f"worst-case prompt {peak:,.0f} tok exceeds the {budget:,} tok budget "
        f"(ceiling {CONTEXT_CEILING_TOKENS:,} - completion reserve "
        f"{COMPLETION_RESERVE_TOKENS:,}). analyst={_analyst_phase_tokens():,.0f} "
        f"debate={_debate_phase_tokens():,.0f} pm={_pm_phase_tokens():,.0f}"
    )


def test_ceiling_is_deepseek_not_nemotron():
    """The fallback pins the budget. Sizing to Nemotron's 262,144 is the bug
    this constant exists to prevent."""
    assert CONTEXT_CEILING_TOKENS == 131_072


def test_analyst_phase_is_the_binding_one():
    """The phases raised for Nemotron must stay off the critical path, so the
    raise provably cannot move the system's peak prompt."""
    assert _debate_phase_tokens() < _analyst_phase_tokens()
    assert _pm_phase_tokens() < _analyst_phase_tokens()


def test_caps_clear_the_measured_maxima():
    """Caps are circuit breakers: the ordinary case must pass through whole."""
    assert MAX_REPORT_CHARS > 14_862       # largest analyst report seen
    assert MAX_HISTORY_CHARS > 31_520      # largest debate history seen
    assert MAX_PAST_CONTEXT_CHARS > 21_548  # largest past_context seen


def test_truncation_is_still_enforced():
    """Raising caps must never become removing them."""
    for cap in (MAX_TOOL_RESULT_CHARS, MAX_REPORT_CHARS, MAX_HISTORY_CHARS,
                MAX_PAST_CONTEXT_CHARS):
        assert cap > 0
        oversized = "x" * (cap + 10_000)
        out = truncate_text(oversized, cap)
        assert len(out) < len(oversized)
        assert "truncated" in out
