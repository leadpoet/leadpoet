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
import json
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

PROXY_URL_ENV = "RESEARCH_LAB_EVIDENCE_PROXY_URL"

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
    """Baseline tape (read-only) + shared day cache (append, host-recorded)."""

    def __init__(self, baseline_dir: str = "", day_cache_path: str = "") -> None:
        self._lock = threading.Lock()
        self._baseline: dict[str, dict[str, Any]] = {}
        self._day: dict[str, dict[str, Any]] = {}
        self._day_path = day_cache_path
        self._loaded_day = ""
        self._baseline_dir = baseline_dir
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._loaded_day = _utc_day()
            self._baseline = {}
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

    def _rollover_if_needed(self) -> None:
        if self._loaded_day != _utc_day():
            # Midnight UTC: all recorded evidence expires; start the day empty.
            self.reload()

    def lookup(self, fingerprint: str) -> dict[str, Any] | None:
        with self._lock:
            self._rollover_if_needed()
            record = self._baseline.get(fingerprint)
            if record is None:
                record = self._day.get(fingerprint)
            return dict(record) if record else None

    def record(self, fingerprint: str, status: int, body: bytes) -> None:
        with self._lock:
            self._rollover_if_needed()
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

    def _respond(self, status: int, body: bytes, evidence: str = "") -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            if evidence:
                self.send_header("X-Research-Lab-Evidence", evidence)
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
        # Global read-through for every run, including baseline reruns: the
        # first run of the day to make a request populates the cache live, and
        # every later identical request — candidate or a mid-day baseline
        # rerun — replays the same recorded response.
        cached = self.store.lookup(fingerprint)
        if cached is not None:
            try:
                body = base64.b64decode(cached.get("body_b64") or "")
            except Exception:
                body = b""
            self._respond(int(cached.get("status") or 502), body, evidence="hit")
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
            status = int(exc.code)
            try:
                body = exc.read()
            except Exception:
                body = b""
        except Exception:
            self._respond(502, b'{"error":"upstream unreachable"}')
            return
        self.store.record(fingerprint, status, body)
        self._respond(status, body, evidence="recorded")

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
