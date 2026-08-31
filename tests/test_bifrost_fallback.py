"""Bifrost per-request fallback chain (BIFROST_FALLBACK_MODELS).

Bifrost holds NO server-side fallback config: if the request body omits the
``fallbacks`` key, nothing falls back. The OpenAI SDK silently drops unknown
top-level kwargs, so the chain has to ride in ``extra_body``.

The load-bearing property here is the no-op: with the env var unset the
request body must be byte-identical to the pre-fallback shape, so nobody who
has not opted in sees a changed request.
"""

import pytest

from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import bifrost_fallback_models

FALLBACK = "nemotron/nemotron-3.5-lightning"


def _client(**kwargs):
    return create_llm_client(
        provider="openrouter",
        model="deepseek/deepseek-chat",
        base_url="http://localhost:8080/v1",
        api_key="test-key",
        **kwargs,
    )


class TestParsing:
    def test_unset_is_empty(self):
        assert bifrost_fallback_models({}) == []

    def test_empty_string_is_empty(self):
        assert bifrost_fallback_models({"BIFROST_FALLBACK_MODELS": ""}) == []

    def test_whitespace_only_is_empty(self):
        assert bifrost_fallback_models({"BIFROST_FALLBACK_MODELS": "  ,  , "}) == []

    def test_single_entry(self):
        env = {"BIFROST_FALLBACK_MODELS": FALLBACK}
        assert bifrost_fallback_models(env) == [FALLBACK]

    def test_chain_is_ordered_and_stripped(self):
        env = {"BIFROST_FALLBACK_MODELS": " a/one , b/two ,, c/three "}
        assert bifrost_fallback_models(env) == ["a/one", "b/two", "c/three"]


class TestRequestBody:
    def test_no_env_means_no_fallbacks_key_at_all(self, monkeypatch):
        monkeypatch.delenv("BIFROST_FALLBACK_MODELS", raising=False)
        llm = _client().get_llm()
        payload = llm._get_request_payload([("human", "hi")])
        assert "extra_body" not in payload
        assert "fallbacks" not in payload
        assert sorted(payload) == ["messages", "model", "stream"]

    def test_empty_env_is_also_a_no_op(self, monkeypatch):
        monkeypatch.setenv("BIFROST_FALLBACK_MODELS", "")
        payload = _client().get_llm()._get_request_payload([("human", "hi")])
        assert "extra_body" not in payload

    def test_env_set_puts_fallbacks_in_extra_body(self, monkeypatch):
        monkeypatch.setenv("BIFROST_FALLBACK_MODELS", FALLBACK)
        payload = _client().get_llm()._get_request_payload([("human", "hi")])
        assert payload["extra_body"] == {"fallbacks": [FALLBACK]}
        # Primary model is unchanged — DeepSeek stays first in line.
        assert payload["model"] == "deepseek/deepseek-chat"
        # And no max_tokens is introduced.
        assert "max_tokens" not in payload

    def test_multi_entry_chain_preserved_in_order(self, monkeypatch):
        monkeypatch.setenv("BIFROST_FALLBACK_MODELS", f"{FALLBACK},openai/gpt-5.4")
        payload = _client().get_llm()._get_request_payload([("human", "hi")])
        assert payload["extra_body"]["fallbacks"] == [FALLBACK, "openai/gpt-5.4"]

    def test_explicit_extra_body_from_config_wins(self, monkeypatch):
        monkeypatch.setenv("BIFROST_FALLBACK_MODELS", FALLBACK)
        llm = _client(extra_body={"fallbacks": ["other/model"]}).get_llm()
        payload = llm._get_request_payload([("human", "hi")])
        assert payload["extra_body"] == {"fallbacks": ["other/model"]}

    def test_applies_to_the_deepseek_subclass_too(self, monkeypatch):
        monkeypatch.setenv("BIFROST_FALLBACK_MODELS", FALLBACK)
        llm = create_llm_client(
            provider="deepseek", model="deepseek-chat", api_key="test-key"
        ).get_llm()
        payload = llm._get_request_payload([("human", "hi")])
        assert payload["extra_body"] == {"fallbacks": [FALLBACK]}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
