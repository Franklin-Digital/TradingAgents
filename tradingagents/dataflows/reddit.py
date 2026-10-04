"""Reddit discussion posts for a ticker.

DEFAULT SOURCE: Franklin's Reddit store (DAYTRADE-1040). The franklin-reddit
collector polls r/wallstreetbets, r/stocks and r/investing every 10 minutes
into Postgres, and franklin-api serves ``GET /social/reddit/{symbol}``. The
live per-symbol search below made ~2,300 Reddit requests per 777-symbol run
from one public IP and was HTTP-429'd on most of them, so most symbols were
scored with no Reddit input at all. ``REDDIT_SOURCE=live`` restores it.

If the store cannot be read, the analyst gets an explicit "unavailable"
placeholder. It NEVER falls back to live search on its own: a silent
fallback would bring back the 429 storm with nothing saying so.

LIVE SOURCE (``REDDIT_SOURCE=live``): the original path. Its default is Reddit's public Atom/RSS search feed
(``reddit.com/r/{sub}/search.rss``). The richer JSON search endpoint
(``/search.json``) is reliably WAF-blocked (``HTTP 403``) for public clients
(issue #862), and probing it on every call only doubled our request volume
against Reddit's per-IP rate limit — tripping ``429`` on the RSS fallback — so
it is kept (``_fetch_subreddit_json``) but not used by default. On a 429 we back
off once (honouring ``Retry-After``). RSS lacks score / comment counts, so those
posts are marked and the formatter omits the metrics rather than printing fake
zeros.

No API key required. Returns formatted plaintext blocks ready for prompt
injection and degrades gracefully — returns a placeholder string rather than
raising, so callers never special-case missing data.
"""

from __future__ import annotations

import html
import http.client
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .date_window import in_window
from .symbol_utils import crypto_base

logger = logging.getLogger(__name__)


def _within_window(posts, start_date, end_date):
    """Keep only posts published in [start_date, end_date] (look-ahead safe).

    No window (both None) leaves the list untouched for live callers. A post with
    no ``created_utc`` epoch is dropped in a historical window (#1220).
    """
    if not (start_date and end_date):
        return posts
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    kept = []
    for p in posts:
        ts = p.get("created_utc")
        created = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        if in_window(created, start_dt, end_dt):
            kept.append(p)
    return kept

_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
# A descriptive, identified User-Agent (per Reddit's API etiquette). Reddit
# blocks generic/anonymous tokens like bare "Mozilla/5.0" or "curl/…" but
# serves this one on both endpoints; the RSS feed accepts it even when the
# JSON search endpoint 403s, so no browser-spoofing is needed.
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })


def _iso_to_timestamp(iso_str: str | None) -> float | None:
    """Parse an Atom ``published`` timestamp to a UTC epoch, or None."""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """Reduce the HTML body Reddit embeds in an Atom entry to plain text."""
    if not content:
        return ""
    # Reddit wraps the real selftext between SC_OFF / SC_ON markers.
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


def _retry_after_seconds(exc: HTTPError) -> float | None:
    """Seconds to wait from a 429's ``Retry-After`` header, capped at 30s."""
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        return min(float(val), 30.0) if val else None
    except (ValueError, TypeError, AttributeError):
        return None


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict]:
    """Default path: parse the public Atom search feed for a subreddit.

    Carries no score / comment counts, so those fields are left None and the
    post is tagged ``source="rss"`` for honest display. On a 429 (Reddit's
    per-IP rate limit) we back off once — honouring ``Retry-After`` when
    present — before giving up, so a transient burst doesn't blank the feed.
    """
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            wait = _retry_after_seconds(exc) or 5.0
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(ticker, sub, limit, timeout, _retry=False)
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
        })
    return posts


def _fetch_subreddit_json(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Richer JSON search path (carries score / comment counts).

    Reddit's WAF currently returns ``403 Blocked`` on this endpoint for
    non-OAuth clients (issue #862), so it is NOT used by default — calling it on
    every request only doubled our volume against the per-IP rate limit and
    triggered 429s on the RSS fallback. Kept for the day the WAF relaxes or an
    OAuth token is wired in; degrades to RSS on failure.
    """
    url = _API.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning(
            "Reddit JSON fetch failed for r/%s · %s: %s — falling back to RSS feed.",
            sub, ticker, exc,
        )
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def _fetch_subreddit(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Fetch one subreddit, RSS-first.

    The JSON search endpoint is reliably WAF-blocked (403) for public clients,
    so we go straight to the RSS feed — which serves our identified User-Agent
    reliably — halving our request volume against Reddit's per-IP rate limit.
    """
    return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def _fetch_reddit_posts_live(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 1.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    ``inter_request_delay`` paces the (now RSS-only) per-subreddit requests to
    stay under Reddit's public per-IP rate limit; combined with the RSS-first
    path it makes 429s rare even when several analyses run back-to-back.

    When ``start_date``/``end_date`` (yyyy-mm-dd) are given, posts are trimmed to
    that window so a historical run does not leak current discussion into a
    backtest (#1220).
    """
    # Crypto reaches us as a Yahoo pair (BTC-USD); search Reddit for the base
    # ("BTC") so the query actually matches discussion instead of near-nothing.
    ticker = crypto_base(ticker) or ticker
    blocks = []
    total_posts = 0
    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _within_window(_fetch_subreddit(ticker, sub, limit_per_sub, timeout),
                               start_date, end_date)
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>")
            continue

        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}"
        header += " (via RSS feed; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            # Score / comment counts are absent on the RSS fallback path —
            # show them only when present rather than printing fake zeros.
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{meta}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"<no Reddit posts found mentioning {ticker.upper()} across "
            f"{', '.join(f'r/{s}' for s in subreddits)} in the past 7 days>"
        )
    return "\n\n".join(blocks)


# ── Franklin's Reddit store (default) ─────────────────────────────────────────

_STORE_DEFAULT_URL = "http://localhost:8070"      # franklin-api prod on mac-pro


def _store_get(symbol: str, params: dict, timeout: float) -> dict:
    base = os.getenv("FRANKLIN_API_URL", _STORE_DEFAULT_URL).rstrip("/")
    url = f"{base}/social/reddit/{quote(symbol)}?{urlencode(params)}"
    with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as r:
        return json.loads(r.read())


def _fetch_reddit_posts_store(
    ticker: str,
    subreddits: Iterable[str],
    limit_per_sub: int,
    timeout: float,
    start_date: str | None,
    end_date: str | None,
) -> str:
    symbol = (crypto_base(ticker) or ticker).upper()
    subs = [s.lower() for s in subreddits]
    params = {"limit": 500}
    if start_date:
        params["from"] = f"{start_date}T00:00:00Z"
    if end_date:
        params["to"] = f"{end_date}T23:59:59.999999Z"
        # A historical run reads the store AS IT WAS at the end of its window:
        # posts we only collected later must not leak into a backtest (#1220).
        if end_date < datetime.now(timezone.utc).strftime("%Y-%m-%d"):
            params["as_of"] = f"{end_date}T23:59:59.999999Z"
    try:
        payload = _store_get(symbol, params, timeout)
    except HTTPError as e:
        logger.warning("Reddit store HTTP %s for %s", e.code, symbol)
        return f"<Reddit data unavailable: Franklin's Reddit store returned HTTP {e.code}>"
    except (OSError, ValueError, http.client.HTTPException) as e:
        logger.warning("Reddit store unreachable for %s: %s", symbol, e)
        return "<Reddit data unavailable: Franklin's Reddit store could not be reached>"

    by_sub: dict[str, list[dict]] = {s: [] for s in subs}
    for p in payload.get("data") or []:
        sub = (p.get("subreddit") or "").lower()
        if sub in by_sub:
            by_sub[sub].append(p)

    blocks, total = [], 0
    for sub in subs:
        posts = sorted(by_sub[sub], key=lambda p: p.get("created_utc") or "", reverse=True)
        posts = posts[:limit_per_sub]
        total += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {symbol} in the past 7 days>")
            continue
        no_metrics = all(p.get("score") is None for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {symbol}"
        header += " (scores/comments unavailable):" if no_metrics else ":"
        lines = [header]
        for p in posts:
            meta = (p.get("created_utc") or "?")[:10]
            if p.get("score") is not None and p.get("num_comments") is not None:
                meta += f" · {p['score']:>4}↑ · {p['num_comments']:>3}c"
            if p.get("matched_in") == "body":
                # A body-only mention is often a passing reference, not the topic.
                meta += " · mentioned in body only"
            title = (p.get("title") or "").replace("\n", " ").strip()
            body = (p.get("body") or "").replace("\n", " ").strip()
            if len(body) > 240:
                body = body[:240] + "…"
            lines.append(f"  [{meta}] {title}" + (f"\n    body excerpt: {body}" if body else ""))
        blocks.append("\n".join(lines))

    if total == 0:
        return (f"<no Reddit posts found mentioning {symbol} across "
                f"{', '.join(f'r/{s}' for s in subs)} in the past 7 days>")
    return "\n\n".join(blocks)


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 1.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Recent Reddit posts mentioning ``ticker``, as a plaintext block.

    Reads Franklin's Reddit store unless ``REDDIT_SOURCE=live``. Never raises:
    a failure becomes a clear placeholder string.
    """
    source = os.getenv("REDDIT_SOURCE", "store").strip().lower()
    if source == "live":
        return _fetch_reddit_posts_live(ticker, subreddits, limit_per_sub, timeout,
                                        inter_request_delay, start_date, end_date)
    if source != "store":
        logger.warning("REDDIT_SOURCE=%r is not 'store' or 'live'; using the store", source)
    return _fetch_reddit_posts_store(ticker, subreddits, limit_per_sub, timeout,
                                     start_date, end_date)
