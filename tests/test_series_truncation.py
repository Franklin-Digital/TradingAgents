"""Chronological tool results must keep the RECENT rows, not the oldest ones.

Regression for the production bug where ``get_stock_data`` — which returns
OHLCV bars in ASCENDING time order — was passed through ``truncate_text``.
That helper keeps the head, so the model received the OLDEST 1% of the series
and none of the data near the analysis date.  Measured on the deployed code
for NVDA 2026-01-01..2026-08-30: 34,124 rows / 1,950,553 chars in, 354 rows
ending 2026-06-23 out, two months of the most recent data silently dropped.
"""

from datetime import date, timedelta

import pytest

from tradingagents.agents.utils.agent_utils import (
    MAX_TOOL_RESULT_CHARS,
    truncate_series_text,
    truncate_text,
)

MAX = 20_000


def _series(n_rows: int = 34_000) -> tuple[str, list[str]]:
    """Build an ascending OHLCV CSV shaped exactly like the vendor's output."""
    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=i)).isoformat()}T20:00:00.000000Z,"
        f"{100 + i % 50}.0,{101 + i % 50}.0,{99 + i % 50}.0,"
        f"{100 + i % 50}.5,{1_000_000 + i}"
        for i in range(n_rows)
    ]
    text = (
        "# Stock data for NVDA from 2026-01-01 to 2026-08-30\n"
        "# Source: QuestDB live-prod (192.168.1.41:9000)\n"
        f"# Total records: {n_rows}\n"
        "# Data retrieved on: 2026-08-31 01:43:00\n"
        "\n"
        "Datetime,Open,High,Low,Close,Volume\n" + "\n".join(rows)
    )
    return text, rows


def _data_rows(out: str) -> list[str]:
    return [
        ln
        for ln in out.split("\n")
        if ln.strip() and not ln.lstrip().startswith("#") and not ln.startswith("Datetime")
    ]


def test_old_head_truncation_loses_the_recent_rows():
    """Pin the defect: plain truncate_text drops everything recent.

    This is the behaviour that shipped.  It documents *why* get_stock_data may
    never go back to truncate_text.
    """
    text, rows = _series()
    out = truncate_text(text, MAX)
    assert rows[-1] not in out
    assert rows[0] in out


def test_truncated_series_retains_the_most_recent_row():
    """The core regression: the newest observation must survive truncation."""
    text, rows = _series()
    out = truncate_series_text(text, MAX)
    assert len(out) <= MAX
    assert rows[-1] in out, "most recent row was dropped by truncation"


def test_truncated_series_retains_a_verbatim_recent_tail():
    """Not just the last row — a contiguous, unmodified recent window."""
    text, rows = _series()
    out = truncate_series_text(text, MAX)
    tail = _data_rows(out)[-50:]
    assert tail == rows[-50:]


def test_truncated_series_preserves_the_full_span():
    """Downsampling, not tail-clipping: the oldest row is still present.

    A pure tail would give the analyst ~2 weeks and destroy the 200-SMA /
    support-resistance context the market-analyst prompt asks for.
    """
    text, rows = _series()
    out = truncate_series_text(text, MAX)
    assert rows[0] in out


def test_every_emitted_row_is_an_unmodified_vendor_row():
    """Rows are dropped, never merged — no synthetic bars.

    ``active_universe`` rows are session snapshots whose Volume is cumulative,
    so aggregating them would fabricate values.
    """
    text, rows = _series()
    out = truncate_series_text(text, MAX)
    original = set(rows)
    for row in _data_rows(out):
        assert row in original


def test_rows_stay_in_ascending_order():
    text, rows = _series()
    emitted = _data_rows(truncate_series_text(text, MAX))
    assert emitted == sorted(emitted)


def test_header_and_provenance_preamble_survive():
    text, _ = _series()
    out = truncate_series_text(text, MAX)
    assert "# Source: QuestDB live-prod (192.168.1.41:9000)" in out
    assert "Datetime,Open,High,Low,Close,Volume" in out


def test_notice_reports_downsampling():
    text, _ = _series()
    out = truncate_series_text(text, MAX)
    assert "DOWNSAMPLED" in out
    assert "VERBATIM" in out


def test_short_series_is_returned_unchanged():
    text, _ = _series(n_rows=10)
    assert truncate_series_text(text, MAX) == text


@pytest.mark.parametrize(
    "prose",
    [
        "A" * 50_000,
        "Analyst note: revenue grew, margins held, guidance raised.\n" * 800,
    ],
)
def test_prose_falls_back_to_head_truncation(prose):
    """Non-tabular text keeps its head — the thesis lives at the top."""
    out = truncate_series_text(prose, MAX)
    assert out == truncate_text(prose, MAX)


def test_get_stock_data_tool_uses_series_truncation(monkeypatch):
    """End-to-end through the tool the market analyst actually calls."""
    import tradingagents.agents.utils.core_stock_tools as cst

    text, rows = _series()
    monkeypatch.setattr(cst, "route_to_vendor", lambda *a, **k: text)
    out = cst.get_stock_data.invoke(
        {"symbol": "NVDA", "start_date": "2026-01-01", "end_date": "2026-08-30"}
    )
    assert len(out) <= MAX_TOOL_RESULT_CHARS
    assert rows[-1] in out


# --- The routing layer truncates too, and it runs FIRST ---------------------
#
# dataflows.interface.route_to_vendor caps every vendor return before any agent
# tool wrapper sees it. It also kept the head, so the recent rows were already
# gone by the time the tool wrapper truncated again — fixing only the wrapper
# would have been cosmetic. Found by running the real market analyst end to end
# against AMD: the model received rows ending 2026-06-26 for a 2026-08-31 run,
# even with the wrapper fixed.


def test_route_to_vendor_truncation_keeps_recent_rows(monkeypatch):
    from tradingagents.dataflows import interface

    text, rows = _series()
    assert len(text) > interface._MAX_TOOL_CHARS

    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_stock_data"], "questdb", lambda *a, **k: text
    )
    monkeypatch.setattr(interface, "get_vendor", lambda *a, **k: "questdb")

    out = interface.route_to_vendor("get_stock_data", "NVDA", "2026-01-01", "2026-08-30")
    assert len(out) <= interface._MAX_TOOL_CHARS
    assert rows[-1] in out, "routing layer dropped the most recent row"
    assert rows[0] in out


def test_route_to_vendor_still_head_truncates_prose(monkeypatch):
    """News/fundamentals must be unaffected: the lede stays."""
    from tradingagents.dataflows import interface

    prose = "".join(
        f"Headline {i}: quarterly results beat expectations, guidance raised.\n"
        for i in range(2000)
    )
    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_news"], "questdb", lambda *a, **k: prose
    )
    monkeypatch.setattr(interface, "get_vendor", lambda *a, **k: "questdb")
    interface.VENDOR_METHODS["get_news"].setdefault("questdb", lambda *a, **k: prose)

    out = interface.route_to_vendor("get_news", "NVDA", "2026-01-01", "2026-08-30")
    assert out.startswith("Headline 0:")
    assert len(out) <= interface._MAX_TOOL_CHARS + 64


def test_double_truncation_cannot_resurrect_dropped_rows():
    """Wrapper-level truncation is powerless once the router has dropped rows.

    Documents why the fix had to land in BOTH layers.
    """
    text, rows = _series()
    already_lost = truncate_text(text, MAX)  # what the old router produced
    assert rows[-1] not in truncate_series_text(already_lost, MAX)


# --- The cap must hold for real -------------------------------------------
#
# The note quotes row keys, so its length is data-dependent; the internal
# reserve is an estimate. #12's context-budget guard (tests/test_context_budget.py)
# derives the analyst-phase worst case from MAX_TOOL_RESULT_CHARS on the
# assumption that a tool result never exceeds it, so this is load-bearing.


@pytest.mark.parametrize("key_width", [10, 40, 200, 2_000])
def test_output_never_exceeds_max_chars_for_any_key_width(key_width):
    """A pathologically wide first column must not push the result over."""
    rows = [
        f"{str(i).rjust(key_width, 'K')},{100 + i}.0,{101 + i}.0,{99 + i}.0,{100 + i}.5,{1000 + i}"
        for i in range(5_000)
    ]
    text = "# hdr\n\nKey,Open,High,Low,Close,Volume\n" + "\n".join(rows)
    out = truncate_series_text(text, MAX)
    assert len(out) <= MAX, f"overflowed by {len(out) - MAX} chars at key_width={key_width}"
    assert rows[-1][:MAX] in out or rows[-1] in out


def test_series_result_respects_the_tool_result_cap():
    """The budget guard's premise: a series result fits MAX_TOOL_RESULT_CHARS."""
    text, _ = _series()
    assert len(truncate_series_text(text, MAX_TOOL_RESULT_CHARS)) <= MAX_TOOL_RESULT_CHARS


def test_series_truncation_does_not_loosen_the_context_budget():
    """Downsampling changes WHICH rows survive, not how many chars do.

    #12 anchors _analyst_phase_tokens() on MAX_TOOL_RESULT_CHARS. If series
    truncation could return more than that, the analyst-phase estimate would be
    understated and the worst-case prompt could exceed DeepSeek's context.
    """
    from tradingagents.dataflows import interface

    text, _ = _series()
    head = truncate_text(text, MAX_TOOL_RESULT_CHARS)
    series = truncate_series_text(text, MAX_TOOL_RESULT_CHARS)
    # The old head path actually ran slightly OVER the cap (notice appended
    # after the slice); the new path stays at or under it.
    assert len(series) <= MAX_TOOL_RESULT_CHARS <= len(head)
    assert len(series) <= interface._MAX_TOOL_CHARS
