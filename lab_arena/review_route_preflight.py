"""Check, before reviews start, that every pinned reviewer route still exists.

Each review request allows exactly one OpenRouter provider route
(``code_review.REVIEW_PROVIDER_ROUTES``). When that provider withdraws its
endpoint, OpenRouter refuses every review with 404 and the round cannot admit
any miner. The public endpoint catalog shows the withdrawal hours earlier, so
the driver reads it while a round is still open and alerts through the
existing Sentry failure path.

The catalog is free and needs no key. It is an observation only: it never
changes routing, admission, retries or spend.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from lab_arena import code_review

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/%s/endpoints"
FETCH_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_INTERVAL_SECONDS = 600.0
PINNED_REVIEW_MODELS = (
    code_review.DEFAULT_REVIEW_MODEL,
    code_review.FALLBACK_REVIEW_MODEL,
    code_review.PROSPECTIVE_REVIEW_MODEL,
    code_review.PROSPECTIVE_ALTERNATE_MODEL,
)
ALERT_CODE = "arena.review_route_missing"


def fetch_endpoints(model: str) -> Any:
    """Read one model's public endpoint catalog, bounded in time and size."""

    url = ENDPOINTS_URL % urllib.parse.quote(model, safe="/")
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "lab-arena-route-preflight"},
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("endpoint catalog too large")
    return json.loads(body.decode("utf-8"))


def route_present(document: Any, route: str) -> bool:
    """Return whether ``route`` matches an endpoint tag in the catalog.

    A provider slug (``together``) matches its own tag and its variants
    (``together/fp8``); a variant slug (``moonshotai/mxfp4``) matches only
    itself. Raise ``ValueError`` when the document is not a catalog.
    """

    data = document.get("data") if isinstance(document, Mapping) else None
    endpoints = data.get("endpoints") if isinstance(data, Mapping) else None
    if not isinstance(endpoints, list):
        raise ValueError("endpoint catalog malformed")
    for endpoint in endpoints:
        tag = endpoint.get("tag") if isinstance(endpoint, Mapping) else None
        if isinstance(tag, str) and (tag == route or tag.startswith(route + "/")):
            return True
    return False


def check_routes(
    fetch: Callable[[str], Any], models: Sequence[str] = PINNED_REVIEW_MODELS,
) -> List[Dict[str, str]]:
    """Return one ``ok``/``missing``/``unavailable`` result per pinned route."""

    results = []
    for model in models:
        route = code_review.review_provider_only(model)[0]
        try:
            status = "ok" if route_present(fetch(model), route) else "missing"
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
            status = "unavailable"
        results.append({"model": model, "route": route, "status": status})
    return results


def alert_missing(result: Mapping[str, str]) -> bool:
    """Send one Sentry failure per missing route, coalesced per process."""

    try:
        from leadpoet_observability import capture_failure
    except Exception:  # Observability is optional and must never stop the driver.
        return False
    return capture_failure(
        ALERT_CODE,
        component="arena-service",
        stage="review_route_preflight",
        terminal=True,
        retryable=True,
        fail_closed=False,
        correlation_id="review-route:%s:%s" % (result["model"], result["route"]),
        dependency="openrouter",
        decision_code="review_route_missing",
    )


class ReviewRoutePreflight:
    """Throttled route check that the driver tick runs while a round is open."""

    def __init__(
        self,
        fetch: Callable[[str], Any] = fetch_endpoints,
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        alert: Callable[[Mapping[str, str]], Any] = alert_missing,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._interval = float(interval_seconds)
        self._alert = alert
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._last: Optional[float] = None

    def run(self) -> Optional[List[Dict[str, str]]]:
        """Check all routes at most once per interval; ``None`` when skipped."""

        with self._lock:
            now = self._monotonic()
            if self._last is not None and now - self._last < self._interval:
                return None
            self._last = now
        results = check_routes(self._fetch)
        for result in results:
            if result["status"] == "missing":
                self._alert(result)
        return results


__all__ = [
    "ALERT_CODE",
    "PINNED_REVIEW_MODELS",
    "ReviewRoutePreflight",
    "check_routes",
    "fetch_endpoints",
    "route_present",
]
