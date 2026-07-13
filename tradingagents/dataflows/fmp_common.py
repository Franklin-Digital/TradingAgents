"""Financial Modeling Prep (FMP) shared client — mirrors alpha_vantage_common.

Franklin uses the FMP **Starter** plan, which serves the newer `/stable/` API
(the legacy `/api/v3/` returns empty on Starter). Key is FMP_API_KEY in franklin.env.
Rate-limit / entitlement errors raise FMPRateLimitError (same contract as
AlphaVantageRateLimitError). FMP is the ONLY fundamentals vendor (DAYTRADE-179),
so route_to_vendor surfaces this as a RuntimeError rather than silently
falling back to an unlicensed vendor.
"""
import json
import os

import requests

API_BASE_URL = "https://financialmodelingprep.com/stable"


def get_api_key() -> str:
    """Retrieve the FMP API key from the environment."""
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("FMP_API_KEY environment variable is not set.")
    return api_key


class FMPRateLimitError(Exception):
    """Raised when the FMP API rate limit / plan entitlement is exceeded."""
    pass


def _make_request(endpoint: str, params: dict | None = None) -> str:
    """GET `/stable/<endpoint>` and return the raw JSON text.

    Raises:
        FMPRateLimitError: on HTTP 429 or an entitlement/limit error body — the
            only signal `route_to_vendor` treats as "try the next vendor".
    """
    api_params = dict(params or {})
    api_params["apikey"] = get_api_key()

    resp = requests.get(f"{API_BASE_URL}/{endpoint}", params=api_params, timeout=30)

    if resp.status_code == 429:
        raise FMPRateLimitError(f"FMP rate limit (HTTP 429): {resp.text[:200]}")
    resp.raise_for_status()

    text = resp.text
    # FMP error/limit bodies come back as a JSON object with an error message
    # (successful data payloads are JSON *arrays*), so only dicts can be errors.
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            msg = (payload.get("Error Message") or payload.get("message")
                   or payload.get("error") or "")
            if any(w in msg.lower() for w in ("limit", "exceeded", "api key",
                                              "not available", "upgrade", "subscription")):
                raise FMPRateLimitError(f"FMP entitlement/limit error: {msg}")
    except json.JSONDecodeError:
        pass

    return text


def filter_statements_by_date(text: str, curr_date: str | None) -> str:
    """Drop statement rows not yet PUBLIC as of curr_date (look-ahead guard).

    FMP statement rows carry `filingDate` (when the report was filed) and `date`
    (fiscal period end). We gate on `filingDate` — the numbers can't be known
    before they were filed — which is stricter (more correct) than gating on the
    fiscal-period-end date. No-op when curr_date is None or the body isn't a list.
    """
    if not curr_date:
        return text
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(rows, list):
        return text
    kept = [r for r in rows
            if str(r.get("filingDate") or r.get("date") or "") <= curr_date]
    return json.dumps(kept)
