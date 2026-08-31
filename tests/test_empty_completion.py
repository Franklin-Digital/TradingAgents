"""Regression tests: an empty or truncated completion must never become a Hold.

Nemotron is a reasoning model — it spends its token budget on internal reasoning
before writing anything to ``content``. Under a constrained budget the provider
returns HTTP 200 with ``finish_reason: length`` and the ``content`` key ABSENT.
Before this fix that produced ``""`` -> ``parse_rating("")`` -> ``"Hold"``: a
fabricated decision, indistinguishable from a real one.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import RATING_REVIEW, is_review
from tradingagents.agents.utils.structured import (
    EmptyCompletionError,
    completion_finish_reason,
    completion_text,
    ensure_usable_completion,
    invoke_structured_or_freetext,
)
from tradingagents.graph.signal_processing import SignalProcessor


def _msg(content=None, finish_reason=None, **kw):
    """A minimal stand-in for a LangChain AIMessage."""
    ns = SimpleNamespace(response_metadata={}, additional_kwargs={}, **kw)
    if content is not None:
        ns.content = content
    if finish_reason is not None:
        ns.response_metadata["finish_reason"] = finish_reason
    return ns


@pytest.mark.unit
class TestCompletionInspection:
    def test_absent_content_key_reads_as_empty(self):
        # The measured Nemotron failure: no `content` attribute at all.
        assert completion_text(_msg()) == ""

    def test_empty_string_content_reads_as_empty(self):
        assert completion_text(_msg(content="")) == ""

    def test_content_blocks_are_flattened(self):
        assert completion_text(_msg(content=[{"type": "text", "text": "Rating: Buy"}])) == "Rating: Buy"

    def test_finish_reason_from_response_metadata(self):
        assert completion_finish_reason(_msg(content="x", finish_reason="length")) == "length"

    def test_finish_reason_from_additional_kwargs(self):
        m = _msg(content="x")
        m.additional_kwargs["finish_reason"] = "length"
        assert completion_finish_reason(m) == "length"


@pytest.mark.unit
class TestEnsureUsableCompletion:
    def test_absent_content_raises(self):
        with pytest.raises(EmptyCompletionError, match="empty completion"):
            ensure_usable_completion(_msg(), "Portfolio Manager")

    def test_whitespace_only_content_raises(self):
        with pytest.raises(EmptyCompletionError):
            ensure_usable_completion(_msg(content="   \n  "), "Portfolio Manager")

    def test_truncated_response_raises_even_with_content(self):
        # finish_reason: length is a first-class signal — a truncated response
        # is not a rating even when it happens to contain a rating word.
        with pytest.raises(EmptyCompletionError, match="truncated"):
            ensure_usable_completion(
                _msg(content="Rating: Buy because", finish_reason="length"),
                "Portfolio Manager",
            )

    def test_good_completion_passes_through(self):
        out = ensure_usable_completion(
            _msg(content="**Rating**: Overweight", finish_reason="stop"), "Portfolio Manager"
        )
        assert out == "**Rating**: Overweight"


@pytest.mark.unit
class TestFreeTextFallbackNeverReturnsEmpty:
    def test_empty_free_text_fallback_raises_instead_of_returning_hold(self):
        plain = MagicMock()
        plain.invoke.return_value = _msg(finish_reason="length")  # no content key
        with pytest.raises(EmptyCompletionError):
            invoke_structured_or_freetext(None, plain, "prompt", lambda r: r, "Portfolio Manager")

    def test_structured_miss_then_empty_free_text_raises(self):
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad json")
        plain = MagicMock()
        plain.invoke.return_value = _msg(content="")
        with pytest.raises(EmptyCompletionError):
            invoke_structured_or_freetext(structured, plain, "p", lambda r: r, "Portfolio Manager")

    def test_structured_rendering_to_empty_falls_back_to_free_text(self):
        structured = MagicMock()
        structured.invoke.return_value = object()
        plain = MagicMock()
        plain.invoke.return_value = _msg(content="Rating: Sell")
        out = invoke_structured_or_freetext(structured, plain, "p", lambda r: "", "Portfolio Manager")
        assert out == "Rating: Sell"


@pytest.mark.unit
class TestEmptyDecisionSurfacesAsReview:
    def test_signal_processor_maps_empty_decision_to_review(self):
        assert SignalProcessor().process_signal("") == RATING_REVIEW
        assert is_review(SignalProcessor().process_signal(""))

    def test_memory_log_tags_empty_decision_review_not_hold(self, tmp_path):
        log_path = tmp_path / "trading_memory_NVDA.md"
        mem = TradingMemoryLog({"memory_log_path": str(log_path)})
        mem.store_decision("NVDA", "2026-08-31", "")
        raw = log_path.read_text()
        assert "| REVIEW | pending]" in raw
        assert "| Hold | pending]" not in raw

    def test_memory_log_still_tags_a_real_rating(self, tmp_path):
        log_path = tmp_path / "trading_memory_NVDA.md"
        mem = TradingMemoryLog({"memory_log_path": str(log_path)})
        mem.store_decision("NVDA", "2026-08-31", "**Rating**: Overweight\n\nReasoning.")
        assert "| Overweight | pending]" in log_path.read_text()
