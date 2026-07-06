"""Host-side provider evidence proxy: record-once, replay-for-all.

Scoring containers send their provider traffic here instead of calling
providers directly (the in-container bootstrap rewrites provider hosts to
this proxy, and containers carry no provider credentials). For each request
the proxy answers from, in order:

1. the baseline tape — the reference run's recorded evidence for the day;
2. the shared day cache — responses already recorded today for the same
   request by any earlier run;
3. the live provider — called once, and the response is recorded into the
   shared day cache so every later identical request replays it.

Because recording happens at the host boundary, the day cache only ever
contains what providers actually returned.
Everything expires at 00:00 UTC (the recording's utc_day must match today),
after which the same inputs are recorded fresh.

Credentials live only in this process: inbound requests have credential
parameters stripped, and upstream calls are re-authenticated from the
proxy's own environment, so a container never needs (or sees) a real key.
"""

from __future__ import annotations

import base64
from decimal import Decimal
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from research_lab.eval.provider_evidence_cache import (
    canonical_request_fingerprint,
    load_evidence_cache,
)
from research_lab.eval.provider_costs import (
    DEFAULT_PROVIDER_COST_CAP_USD_PER_ICP,
    DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    ProviderCostEstimate,
    ProviderCostLedger,
    decimal_from_env,
    estimate_provider_cost,
    extract_openrouter_cost_dollars,
    redacted_endpoint,
    scrapingdog_credits_for_path,
)

PROXY_URL_ENV = "RESEARCH_LAB_EVIDENCE_PROXY_URL"
logger = logging.getLogger(__name__)

# Upstream routing: first path segment selects the provider. The proxy
# re-authenticates each upstream call from its own environment.
_UPSTREAMS: dict[str, dict[str, Any]] = {
    "exa": {
        "base": "https://api.exa.ai",
        "auth": lambda: {"x-api-key": os.getenv("EXA_API_KEY") or ""},
    },
    "sd": {
        "base": "https://api.scrapingdog.com",
        "auth_param": ("api_key", lambda: os.getenv("SCRAPINGDOG_API_KEY") or os.getenv("QUALIFICATION_SCRAPINGDOG_API_KEY") or ""),
    },
    "or": {
        "base": "https://openrouter.ai",
        "auth": lambda: {"Authorization": "Bearer " + (os.getenv("OPENROUTER_API_KEY") or os.getenv("QUALIFICATION_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY") or "")},
    },
}

_HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "host", "content-length", "authorization", "x-api-key"}


def _utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


class EvidenceStore:
    """Baseline tape + shared day cache with single-flight live calls.

    A Condition guards all state. Single-flight guarantees that for any one
    request fingerprint, exactly one caller makes the live provider call while
    every concurrent identical caller waits and then replays the recorded
    result — so parallel same-request work (e.g. two candidates on the same
    ICP) shares one live call instead of each calling the provider.
    """

    def __init__(self, baseline_dir: str = "", day_cache_path: str = "") -> None:
        # Reentrant so a lookup that triggers a midnight rollover can reload
        # under the same held lock without deadlocking.
        self._cond = threading.Condition(threading.RLock())
        self._baseline: dict[str, dict[str, Any]] = {}
        self._day: dict[str, dict[str, Any]] = {}
        self._inflight: set[str] = set()
        self._cost_ledgers: dict[str, ProviderCostLedger] = {}
        self._day_path = day_cache_path
        self._loaded_day = ""
        self._baseline_dir = baseline_dir
        with self._cond:
            self._reload_locked()

    def _reload_locked(self) -> None:
        self._loaded_day = _utc_day()
        self._baseline = {}
        self._inflight.clear()
        self._cost_ledgers.clear()
        if self._baseline_dir and os.path.isdir(self._baseline_dir):
            for name in sorted(os.listdir(self._baseline_dir)):
                if not name.endswith(".json"):
                    continue
                try:
                    loaded = load_evidence_cache(os.path.join(self._baseline_dir, name))
                except Exception:
                    continue
                for key, record in loaded.items():
                    self._baseline.setdefault(key, record)
        self._day = {}
        if self._day_path and os.path.isfile(self._day_path):
            try:
                with open(self._day_path, "r", encoding="utf-8") as handle:
                    doc = json.load(handle)
                if str(doc.get("utc_day") or "") == self._loaded_day:
                    entries = doc.get("entries")
                    if isinstance(entries, Mapping):
                        self._day = {
                            str(k): dict(v)
                            for k, v in entries.items()
                            if isinstance(v, Mapping) and isinstance(v.get("status"), int)
                        }
            except Exception:
                self._day = {}

    def reload(self) -> None:
        with self._cond:
            self._reload_locked()

    def _rollover_if_needed_locked(self) -> None:
        if self._loaded_day != _utc_day():
            # Midnight UTC: all recorded evidence expires; start the day empty
            # and wake any waiters so they re-lead against the fresh day.
            self._reload_locked()
            self._cond.notify_all()

    def _cached_locked(self, fingerprint: str) -> dict[str, Any] | None:
        record = self._baseline.get(fingerprint)
        if record is None:
            record = self._day.get(fingerprint)
        return dict(record) if record else None

    def lookup(self, fingerprint: str) -> dict[str, Any] | None:
        with self._cond:
            self._rollover_if_needed_locked()
            return self._cached_locked(fingerprint)

    def acquire_or_wait(self, fingerprint: str, timeout: float = 175.0) -> tuple[dict[str, Any] | None, bool]:
        """Single-flight gate.

        Returns (record, is_leader):
        - (record, False): already recorded — replay it, do not call live.
        - (None, True): you are the leader — make the live call, then call
          record() (or release_lead() on failure).
        - (None, False): timed out waiting for the leader — fall back to a
          live call without leadership (rare; keeps a stuck leader from
          blocking forever).
        """
        with self._cond:
            deadline = None
            while True:
                self._rollover_if_needed_locked()
                cached = self._cached_locked(fingerprint)
                if cached is not None:
                    return cached, False
                if fingerprint not in self._inflight:
                    self._inflight.add(fingerprint)
                    return None, True
                # Another caller is leading this fingerprint; wait for it.
                if deadline is None:
                    deadline = timeout
                if not self._cond.wait(timeout=deadline):
                    return None, False

    def release_lead(self, fingerprint: str) -> None:
        """Leader's live call failed with no recordable result: let a waiter
        take over rather than block."""
        with self._cond:
            self._inflight.discard(fingerprint)
            self._cond.notify_all()

    def record(self, fingerprint: str, status: int, body: bytes) -> None:
        with self._cond:
            self._rollover_if_needed_locked()
            self._inflight.discard(fingerprint)
            self._cond.notify_all()
            if fingerprint in self._baseline or fingerprint in self._day:
                return
            self._day[fingerprint] = {
                "status": int(status),
                "body_b64": base64.b64encode(body).decode("ascii"),
                "outcome": "error" if status >= 400 else "success",
            }
            if self._day_path:
                doc = {
                    "schema_version": "1.1",
                    "utc_day": self._loaded_day,
                    "entries": self._day,
                }
                tmp = f"{self._day_path}.tmp.{os.getpid()}"
                try:
                    with open(tmp, "w", encoding="utf-8") as handle:
                        json.dump(doc, handle, sort_keys=True, separators=(",", ":"))
                    os.replace(tmp, self._day_path)
                except Exception:
                    pass

    def cost_ledger(self, scope: str, cap_usd: Decimal) -> ProviderCostLedger:
        with self._cond:
            self._rollover_if_needed_locked()
            key = str(scope or "unscoped")
            ledger = self._cost_ledgers.get(key)
            if ledger is None:
                ledger = ProviderCostLedger(scope=key, cap_usd=cap_usd)
                self._cost_ledgers[key] = ledger
            return ledger


class _ProxyHandler(BaseHTTPRequestHandler):
    store: EvidenceStore
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # quiet; the gateway logs enough
        pass

    def _provider(self) -> tuple[str, str] | None:
        parts = self.path.lstrip("/").split("/", 1)
        name = parts[0] if parts else ""
        upstream = _UPSTREAMS.get(name)
        if not upstream:
            return None
        rest = "/" + (parts[1] if len(parts) > 1 else "")
        return name, rest

    def _respond(
        self,
        status: int,
        body: bytes,
        evidence: str = "",
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            if evidence:
                self.send_header("X-Research-Lab-Evidence", evidence)
            for key, value in dict(headers or {}).items():
                if key and value is not None:
                    self.send_header(str(key), str(value))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            pass

    def _handle(self) -> None:
        routed = self._provider()
        if routed is None:
            self._respond(404, b'{"error":"unknown provider route"}')
            return
        name, rest = routed
        upstream = _UPSTREAMS[name]
        length = int(self.headers.get("Content-Length") or 0)
        request_body = self.rfile.read(length) if length else b""
        # Fingerprint on the UPSTREAM shape of the request so it matches
        # evidence recorded from direct provider calls (baseline tapes).
        upstream_url = upstream["base"] + rest
        fingerprint = canonical_request_fingerprint(self.command, upstream_url, request_body or None)
        scope = str(self.headers.get("X-Research-Lab-Cost-Scope") or "unscoped").strip() or "unscoped"
        cap_usd = decimal_from_env(
            "RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP",
            DEFAULT_PROVIDER_COST_CAP_USD_PER_ICP,
        )
        header_cap = str(self.headers.get("X-Research-Lab-Cost-Cap-Usd") or "").strip()
        if header_cap:
            try:
                parsed_cap = Decimal(header_cap)
                if parsed_cap >= 0:
                    cap_usd = parsed_cap
            except Exception:
                pass
        credit_price = decimal_from_env(
            "RESEARCH_LAB_SCRAPINGDOG_COST_PER_CREDIT_USD",
            DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
        )
        ledger = self.store.cost_ledger(scope, cap_usd)
        # Global read-through with single-flight: the first run of the day to
        # make a request calls the provider live while every concurrent
        # identical request (e.g. another candidate on the same ICP) waits and
        # replays the recorded result, so one live call is shared by all.
        cached, is_leader = self.store.acquire_or_wait(fingerprint)
        if cached is not None:
            try:
                body = base64.b64decode(cached.get("body_b64") or "")
            except Exception:
                body = b""
            status = int(cached.get("status") or 502)
            event = ledger.cache_hit_event(
                provider=name,
                endpoint=redacted_endpoint(name, upstream_url),
                request_fingerprint=fingerprint,
                status_code=status,
            )
            self._respond(status, body, evidence="hit", headers=event.to_headers())
            return
        endpoint = redacted_endpoint(name, upstream_url)
        if name == "sd" and scrapingdog_credits_for_path(urllib.parse.urlsplit(upstream_url).path) is None:
            event = ledger.block_event(
                provider=name,
                endpoint=endpoint,
                request_fingerprint=fingerprint,
                reason="unknown_scrapingdog_endpoint",
            )
            body = json.dumps(
                {
                    "error": "research_lab_provider_cost_unknown_scrapingdog_endpoint",
                    "endpoint": endpoint,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if is_leader:
                self.store.release_lead(fingerprint)
            self._respond(402, body, evidence="blocked", headers=event.to_headers())
            return
        if ledger.should_block_paid_call():
            event = ledger.block_event(
                provider=name,
                endpoint=endpoint,
                request_fingerprint=fingerprint,
                reason="cost_cap_reached",
            )
            body = json.dumps(
                {
                    "error": "research_lab_provider_cost_cap_exceeded",
                    "provider": name,
                    "endpoint": endpoint,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if is_leader:
                self.store.release_lead(fingerprint)
            self._respond(402, body, evidence="blocked", headers=event.to_headers())
            return
        # Live call, re-authenticated from the proxy's own credentials.
        target = upstream_url
        auth_param = upstream.get("auth_param")
        if auth_param:
            param, getter = auth_param
            split = urllib.parse.urlsplit(target)
            pairs = [(k, v) for k, v in urllib.parse.parse_qsl(split.query, keep_blank_values=True) if k.lower() != param]
            pairs.append((param, getter()))
            target = urllib.parse.urlunsplit(split._replace(query=urllib.parse.urlencode(pairs)))
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in _HOP_HEADERS and not k.lower().startswith("x-research-lab")
        }
        auth = upstream.get("auth")
        if auth:
            for key, value in auth().items():
                if value:
                    headers[key] = value
        request = urllib.request.Request(target, data=request_body or None, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = int(getattr(response, "status", None) or getattr(response, "code", 0) or 0)
                body = response.read()
        except urllib.error.HTTPError as exc:
            # An HTTP error status is a real, recordable provider outcome.
            status = int(exc.code)
            try:
                body = exc.read()
            except Exception:
                body = b""
        except Exception:
            # No recordable result (transport failure): release leadership so a
            # waiting caller can retry rather than block on us.
            if is_leader:
                self.store.release_lead(fingerprint)
            event = ledger.record_live_event(
                provider=name,
                request_fingerprint=fingerprint,
                status_code=502,
                estimate=ProviderCostEstimate(
                    provider=name,
                    endpoint=endpoint,
                    billable=False,
                    cost_source="transport_failure_zero_cost",
                ),
            )
            self._respond(502, b'{"error":"upstream unreachable"}', evidence="error", headers=event.to_headers())
            return
        self.store.record(fingerprint, status, body)
        estimate = estimate_provider_cost(
            provider=name,
            upstream_url=upstream_url,
            status=status,
            response_body=body,
            request_body=request_body or None,
            scrapingdog_credit_price_usd=credit_price,
        )
        if (
            name == "or"
            and estimate.tracking_failed
            and estimate.tracking_reason == "missing_openrouter_cost"
            and estimate.generation_id
        ):
            try:
                gen_url = (
                    _UPSTREAMS["or"]["base"]
                    + "/api/v1/generation?id="
                    + urllib.parse.quote(estimate.generation_id, safe="")
                )
                auth_getter = _UPSTREAMS["or"].get("auth")
                gen_headers = {
                    key: value
                    for key, value in (auth_getter() if auth_getter else {}).items()
                    if value
                }
                gen_req = urllib.request.Request(gen_url, headers=gen_headers, method="GET")
                with urllib.request.urlopen(gen_req, timeout=30) as gen_response:
                    if int(getattr(gen_response, "status", None) or getattr(gen_response, "code", 0) or 0) < 400:
                        generation_body = gen_response.read()
                        reconciled_cost, reconciled_metadata = extract_openrouter_cost_dollars(generation_body)
                        if reconciled_cost is not None:
                            estimate = ProviderCostEstimate(
                                provider="or",
                                endpoint=estimate.endpoint,
                                model=estimate.model,
                                billable=True,
                                cost_usd=reconciled_cost,
                                cost_source="openrouter_generation_reconciliation",
                                prompt_tokens=int(reconciled_metadata.get("prompt_tokens") or 0),
                                completion_tokens=int(reconciled_metadata.get("completion_tokens") or 0),
                                generation_id=estimate.generation_id,
                            )
            except Exception as exc:
                logger.warning(
                    "research_lab_openrouter_generation_cost_reconcile_failed generation_id_prefix=%s error=%s",
                    str(estimate.generation_id or "")[:16],
                    exc,
                )
        event = ledger.record_live_event(
            provider=name,
            request_fingerprint=fingerprint,
            status_code=status,
            estimate=estimate,
        )
        self._respond(status, body, evidence="recorded", headers=event.to_headers())

    do_GET = _handle
    do_POST = _handle


def serve_evidence_proxy(
    *,
    host: str = "0.0.0.0",
    port: int = 0,
    baseline_dir: str = "",
    day_cache_path: str = "",
) -> tuple[ThreadingHTTPServer, EvidenceStore, threading.Thread]:
    """Start the proxy; returns (server, store, thread). Caller owns shutdown."""
    store = EvidenceStore(baseline_dir=baseline_dir, day_cache_path=day_cache_path)
    handler = type("BoundProxyHandler", (_ProxyHandler,), {"store": store})
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="evidence-proxy", daemon=True)
    thread.start()
    return server, store, thread


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Provider evidence proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--baseline-dir", default=os.getenv("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR") or "")
    parser.add_argument("--day-cache", default=os.getenv("RESEARCH_LAB_PROVIDER_EVIDENCE_DAY_CACHE") or "")
    args = parser.parse_args()
    server, _store, thread = serve_evidence_proxy(
        host=args.host, port=args.port, baseline_dir=args.baseline_dir, day_cache_path=args.day_cache
    )
    print(json.dumps({"listening": f"{args.host}:{args.port}"}))
    thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
