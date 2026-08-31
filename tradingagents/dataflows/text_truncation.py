"""Truncation helpers shared by the dataflow routing layer and the agents.

Lives in ``dataflows`` (a leaf module with no intra-package imports) because
BOTH layers truncate: ``dataflows.interface.route_to_vendor`` caps every vendor
result, and the agent tool wrappers cap again.  Importing this from
``agent_utils`` instead would be a circular import.
"""

def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to *max_chars*, appending a notice when trimmed."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[... truncated to {max_chars:,} chars]"


# --- Time-series-aware truncation -------------------------------------------
#
# ``truncate_text`` keeps the HEAD, which is correct for prose (reports, debate
# history, memory) where the opening carries the thesis.  It is catastrophically
# wrong for a chronological table.  ``get_stock_data`` returns bars in ASCENDING
# time order, so head-truncation kept the OLDEST rows and dropped everything
# recent.  Measured on NVDA 2026-01-01..2026-08-30 against the deployed code:
#
#     vendor returned    1,950,553 chars  34,124 rows  2026-06-22 .. 2026-08-29
#     after truncation      20,000 chars     354 rows  2026-06-22 .. 2026-06-23
#
# 1.04% of rows survived and the two most recent months — including the
# analysis date itself — never reached the model.  Every rating produced that
# way reasoned over stale prices.
#
# Why not simply keep the TAIL: the analyst prompt asks for the 50/200 SMA,
# MACD, Bollinger context and support/resistance levels.  A pure tail keeps
# ~354 of 34,124 rows, i.e. the last day or two of snapshots, and destroys all
# longer-horizon structure.  That is a different wrong answer, not a fix.
#
# Why not AGGREGATE older rows into synthetic bars: the live-prod rows are
# real-time snapshots of ``active_universe``, not incremental bars.  ``Volume``
# is the cumulative session volume (NVDA reaches ~195,000,000 by the close) and
# ``Open/High/Low`` are session-to-date values that read 0.0 in the pre-open.
# Summing volume or OHLC-merging those rows would fabricate numbers that look
# authoritative and are simply false — exactly the failure mode this codebase
# has been burned by before.
#
# So: DOWNSAMPLE BY DROPPING ROWS.  Keep the most recent rows verbatim at full
# resolution, and keep an evenly-spaced sample of the older rows so the full
# span — including the very first and very last observation — still reaches the
# model.  Every row the model sees is an unmodified vendor observation.

_SERIES_RECENT_SHARE = 0.5  # half the row budget goes to the verbatim tail


def _split_series(text: str) -> tuple[list[str], str, list[str]] | None:
    """Split a CSV-shaped tool result into (preamble, column_header, data_rows).

    Returns ``None`` when the text does not look like a delimited table, so the
    caller can fall back to plain head-truncation rather than mangling prose.
    """
    lines = text.split("\n")
    i = 0
    preamble: list[str] = []
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        preamble.append(lines[i])
        i += 1
    if i >= len(lines):
        return None
    header = lines[i]
    if "," not in header:
        return None
    rows = [ln for ln in lines[i + 1:] if ln.strip()]
    if len(rows) < 3:
        return None
    # Guard against prose that happens to contain commas ("revenue grew,
    # margins held, guidance raised"). Two checks, both required:
    #   1. data rows carry the same field count as the header, and
    #   2. at least one non-leading column is numeric across the sample.
    # Every series we truncate (OHLCV, indicator tables) has numeric columns;
    # comma-laden prose does not.
    want = header.count(",")
    sample = rows[:20]
    if sum(1 for r in sample if r.count(",") == want) < len(sample):
        return None
    if not _has_numeric_column(header, sample):
        return None
    return preamble, header, rows


def _is_number(value: str) -> bool:
    try:
        float(value.strip())
    except (TypeError, ValueError):
        return False
    return True


def _has_numeric_column(header: str, sample: list[str]) -> bool:
    """True when some non-leading column is numeric in every sampled row.

    The leading column is excluded because it is the timestamp/date key.
    """
    n_cols = header.count(",") + 1
    for col in range(1, n_cols):
        if all(_is_number(r.split(",")[col]) for r in sample):
            return True
    return False


def _first_field(row: str) -> str:
    return row.split(",", 1)[0].strip()


def _evenly_spaced(items: list, n: int) -> list:
    """Pick *n* items spread across *items*, always including first and last."""
    total = len(items)
    if n >= total:
        return list(items)
    if n <= 1:
        return [items[0]]
    idx = sorted({round(k * (total - 1) / (n - 1)) for k in range(n)})
    return [items[i] for i in idx]


def truncate_series_text(
    text: str,
    max_chars: int,
    recent_share: float = _SERIES_RECENT_SHARE,
) -> str:
    """Fit a chronological table into *max_chars* while keeping the RECENT rows.

    Assumes ascending time order (oldest first), which is what the OHLCV
    vendors return.  The newest rows are kept verbatim; older rows are
    downsampled by dropping rows at an even stride, never by merging values.
    Falls back to :func:`truncate_text` for anything that is not table-shaped.
    """
    if not text or len(text) <= max_chars:
        return text

    parsed = _split_series(text)
    if parsed is None:
        return truncate_text(text, max_chars)
    preamble, header, rows = parsed

    fixed = "\n".join([*preamble, header])
    # Reserve generous room for the explanatory note we prepend to the rows.
    note_reserve = 600
    budget = max_chars - len(fixed) - note_reserve
    if budget <= 0:
        return truncate_text(text, max_chars)

    # 1. Fill the recent half of the budget with verbatim tail rows.
    recent_budget = int(budget * recent_share)
    recent: list[str] = []
    used = 0
    for row in reversed(rows):
        cost = len(row) + 1
        if used + cost > recent_budget:
            break
        recent.append(row)
        used += cost
    recent.reverse()
    if not recent:  # a single row wider than half the budget
        recent = [rows[-1]]
        used = len(rows[-1]) + 1

    older = rows[: len(rows) - len(recent)]
    older_budget = budget - used

    sampled: list[str] = []
    stride = 1
    if older and older_budget > 0:
        avg = sum(len(r) for r in older) / len(older) + 1
        n_keep = max(1, int(older_budget // avg))
        sampled = _evenly_spaced(older, n_keep)
        while sampled and sum(len(r) + 1 for r in sampled) > older_budget:
            sampled = sampled[1:]
        stride = max(1, round(len(older) / len(sampled))) if sampled else 1

    kept = len(sampled) + len(recent)
    note = [
        f"# NOTE: {len(rows):,} rows exceeded the {max_chars:,}-char tool-result "
        f"budget; {kept:,} rows are shown.",
        f"# Most recent {len(recent):,} rows are VERBATIM and complete "
        f"({_first_field(recent[0])} .. {_first_field(recent[-1])}).",
    ]
    if sampled:
        how = (
            "kept in full" if len(sampled) == len(older)
            else f"DOWNSAMPLED to {len(sampled):,} of {len(older):,}, "
                 f"roughly 1 row in {stride:,}"
        )
        note.append(
            f"# Older rows ({_first_field(older[0])} .. {_first_field(older[-1])}) "
            f"are {how}. Rows were dropped, never merged or recomputed — every "
            f"value below is an unmodified observation."
        )
    else:
        note.append(
            f"# {len(older):,} older rows were dropped entirely — no budget "
            f"remained for them."
        )
    note.append("# The last row is the most recent observation available.")

    parts = [fixed, *note]
    if sampled:
        parts.extend(sampled)
        parts.append("# --- full resolution from here ---")
    parts.extend(recent)
    return "\n".join(parts)
