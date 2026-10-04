"""Reddit from Franklin's store (DAYTRADE-1040): the default source.

franklin-api GET /social/reddit/{symbol} on mac-pro serves what the
franklin-reddit collector stored. These tests check the request built, the
formatting, the look-ahead guard, and that a store failure NEVER falls back
to live Reddit search (which is what 429'd every run).
"""
from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from tradingagents.dataflows import reddit


def _post(sub, created, title, body="", matched_in="title", score=None, comments=None):
    return {"post_id": f"t3_{abs(hash((sub, created, title))) % 10**8}", "symbol": "NVDA",
            "subreddit": sub, "title": title, "body": body, "created_utc": created,
            "matched_in": matched_in, "match_type": "bare",
            "score": score, "num_comments": comments}


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


class _Seen(list):
    """Requested URLs, plus the payload (or exception) the fake store returns."""


@pytest.fixture
def store(monkeypatch):
    seen = _Seen()
    seen.payload = {"data": []}

    def fake_urlopen(req, timeout=None):
        seen.append(req.full_url)
        if isinstance(seen.payload, Exception):
            raise seen.payload
        return _Resp(seen.payload)

    monkeypatch.setattr(reddit, "urlopen", fake_urlopen)
    monkeypatch.delenv("REDDIT_SOURCE", raising=False)
    monkeypatch.setenv("FRANKLIN_API_URL", "http://localhost:8071/")

    def live_must_not_run(*a, **k):
        raise AssertionError("live Reddit search must not run on the store path")
    monkeypatch.setattr(reddit, "_fetch_subreddit", live_must_not_run)
    return seen


def _query(url):
    u = urlparse(url)
    return u.path, {k: v[0] for k, v in parse_qs(u.query).items()}


def test_default_source_is_the_store_with_one_request(store):
    reddit.fetch_reddit_posts("NVDA", start_date="2026-09-27", end_date="2099-01-01")
    assert len(store) == 1
    path, q = _query(store[0])
    assert store[0].startswith("http://localhost:8071/social/reddit/NVDA?")
    assert q["from"] == "2026-09-27T00:00:00Z" and q["to"] == "2099-01-01T23:59:59.999999Z"
    assert "as_of" not in q                      # a live (current) window reads current data


def test_historical_window_reads_the_store_as_it_was(store):
    reddit.fetch_reddit_posts("NVDA", start_date="2026-05-13", end_date="2026-05-20")
    _, q = _query(store[0])
    assert q["as_of"] == "2026-05-20T23:59:59.999999Z"


def test_posts_grouped_newest_first_capped_and_marked(store):
    store.payload = {"data": [
        _post("stocks", "2026-10-01T10:00:00+00:00", "older stocks"),
        _post("stocks", "2026-10-02T10:00:00+00:00", "newer stocks", body="x" * 300),
        _post("wallstreetbets", "2026-10-02T09:00:00+00:00", "wsb", matched_in="body"),
        _post("options", "2026-10-02T09:00:00+00:00", "not a configured sub"),
    ]}
    out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks", "wallstreetbets", "investing"),
                                    limit_per_sub=1, end_date="2099-01-01")
    assert "r/stocks — 1 recent posts mentioning NVDA (scores/comments unavailable):" in out
    assert "newer stocks" in out and "older stocks" not in out          # newest, capped at 1
    assert "x" * 240 + "…" in out and "x" * 241 not in out               # excerpt cut at 240
    assert "[2026-10-02 · mentioned in body only] wsb" in out
    assert "r/investing: <no posts found mentioning NVDA" in out
    assert "not a configured sub" not in out


def test_metrics_shown_when_the_store_has_them(store):
    store.payload = {"data": [_post("stocks", "2026-10-02T10:00:00+00:00", "t", score=12, comments=3)]}
    out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",), end_date="2099-01-01")
    assert "  12↑ ·   3c" in out and "unavailable" not in out


def test_no_posts_anywhere(store):
    out = reddit.fetch_reddit_posts("NVDA", end_date="2099-01-01")
    assert out.startswith("<no Reddit posts found mentioning NVDA across r/wallstreetbets")


@pytest.mark.parametrize("exc, text", [
    (HTTPError("u", 503, "Service Unavailable", {}, None), "returned HTTP 503"),
    (OSError("connection refused"), "could not be reached"),
])
def test_store_failure_is_explicit_and_never_falls_back_to_live(store, exc, text):
    store.payload = exc
    out = reddit.fetch_reddit_posts("NVDA", end_date="2099-01-01")
    assert out.startswith("<Reddit data unavailable") and text in out


def test_crypto_pair_reads_the_base_symbol(store):
    reddit.fetch_reddit_posts("BTC-USD", end_date="2099-01-01")
    assert urlparse(store[0]).path == "/social/reddit/BTC"


def test_live_switch_uses_the_old_search_and_not_the_store(monkeypatch):
    monkeypatch.setenv("REDDIT_SOURCE", "live")
    monkeypatch.setattr(reddit, "_fetch_subreddit", lambda *a, **k: [])
    monkeypatch.setattr(reddit, "_store_get", lambda *a, **k: pytest.fail("store must not be read"))
    out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",), inter_request_delay=0)
    assert "<no Reddit posts found" in out


def test_unknown_source_value_uses_the_store(store, monkeypatch):
    monkeypatch.setenv("REDDIT_SOURCE", "bogus")
    reddit.fetch_reddit_posts("NVDA", end_date="2099-01-01")
    assert len(store) == 1
