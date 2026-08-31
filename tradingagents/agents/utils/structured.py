"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ``finish_reason`` values that mean the model never finished writing its answer.
# A reasoning model (e.g. Nemotron) can spend its whole budget on internal
# reasoning and stop before emitting a single character of ``content`` — the
# provider still returns HTTP 200, so only the finish reason gives it away.
TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens", "MAX_TOKENS"})


class EmptyCompletionError(RuntimeError):
    """The provider returned a 200 with no usable completion.

    Raised when the completion text is empty/absent, or when the response was
    cut off mid-generation (``finish_reason: length``). It is deliberately an
    exception rather than an empty string: an empty decision parses to a
    fabricated ``Hold`` downstream, which is indistinguishable from a real one.
    Callers should treat it as a *transient* failure and retry.
    """


def completion_finish_reason(response: Any) -> str | None:
    """Best-effort ``finish_reason`` for a LangChain message, or ``None``.

    Providers put it in different places depending on the integration, so check
    both the standard ``response_metadata`` and the passthrough
    ``additional_kwargs``.
    """
    for attr in ("response_metadata", "additional_kwargs"):
        meta = getattr(response, attr, None)
        if isinstance(meta, dict):
            for key in ("finish_reason", "stop_reason"):
                value = meta.get(key)
                if value:
                    return str(value)
    return None


def completion_text(response: Any) -> str:
    """Return the text of a LangChain message, tolerating absent/blocked content.

    ``content`` is ``""`` when the provider omitted the key entirely, and a list
    of content blocks for multimodal/reasoning responses. Both shapes collapse
    to a plain string here so emptiness can be tested in one place.
    """
    content = getattr(response, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def ensure_usable_completion(response: Any, agent_name: str) -> str:
    """Return the completion text, or raise :class:`EmptyCompletionError`.

    Guards the two silent-failure modes measured with Nemotron under a
    constrained token budget: an absent/empty ``content`` key, and a response
    truncated mid-generation. Neither is a rating, and neither must be allowed
    to reach ``parse_rating`` where it would become a fabricated ``Hold``.
    """
    text = completion_text(response)
    finish_reason = completion_finish_reason(response)
    if finish_reason in TRUNCATED_FINISH_REASONS:
        raise EmptyCompletionError(
            f"{agent_name}: response truncated (finish_reason={finish_reason!r}, "
            f"{len(text)} chars of content) — a truncated response is not a rating"
        )
    if not text.strip():
        raise EmptyCompletionError(
            f"{agent_name}: provider returned an empty completion "
            f"(finish_reason={finish_reason!r})"
        )
    return text

# Schema-only structured output binds exactly one tool (the schema itself), so a
# model that reaches for a search tool emits an unknown tool call and the whole
# structured attempt is discarded for a free-text retry. Agents on this path
# state the constraint explicitly rather than relying on the binding alone
# (#1130).
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                # A thinking model can answer in plain text instead of calling
                # the tool, leaving the parser with nothing to return. Treat it
                # as a structured miss and fall back, with a clear reason.
                raise ValueError("structured output returned no parsed result")
            rendered = render(result)
            if not (rendered or "").strip():
                raise ValueError("structured output rendered to empty text")
            return rendered
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    # Do NOT return ``response.content`` blindly: an absent content key yields
    # "", which parses to a fabricated ``Hold`` downstream. Fail loudly instead
    # so the run is retried and, failing that, surfaced as REVIEW.
    return ensure_usable_completion(response, agent_name)
