#!/usr/bin/env python3
"""Publish a gateway restart marker to the gateway's own OTel destination.

The restart script already records every stage to a host-local JSONL ledger and
summarises the outcome to Sentry.  Neither of those is visible in the telemetry
store that holds the gateway's spans, so a gap in gateway traffic cannot be told
apart from an unplanned process death: both look like silence.

This emitter closes that gap.  It posts one zero-duration span per restart
boundary -- ``started`` when the restart script is invoked, ``finished`` when it
exits -- to the same OTLP endpoint and token the gateway already uses for its
request spans, so no additional host provisioning is required.  Both markers of
one restart share a trace id derived from the restart invocation id, so the pair
reads as a single restart with a duration and an outcome.

Everything here is best effort.  The restart path must never fail, slow down, or
change behaviour because telemetry was unreachable, so every error is swallowed
and the exit code is always zero unless ``--strict`` is passed (tests only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from read_gateway_otel_env import resolve_value  # noqa: E402

DEFAULT_ENV_FILE = "/home/ec2-user/.config/leadpoet/gateway.env"
SERVICE_NAME = "leadpoet-gateway-restart"
SPAN_NAME = "gateway.restart"
SCHEMA_VERSION = "leadpoet.gateway_restart_marker.v1"
EVENTS = ("started", "finished")


def _hex_digest(*parts: str, length: int) -> str:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


def trace_id_for(invocation_id: str) -> str:
    """Both markers of one restart share a trace, so the pair joins up."""
    seed = invocation_id or f"anonymous-{os.getpid()}"
    return _hex_digest("leadpoet.gateway.restart", seed, length=32)


def span_id_for(invocation_id: str, event: str, now_ns: int) -> str:
    return _hex_digest(invocation_id, event, str(now_ns), length=16)


def _attribute(key: str, value: Any) -> Optional[Dict[str, Any]]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _attributes(pairs: Dict[str, Any]) -> List[Dict[str, Any]]:
    built = (_attribute(key, value) for key, value in pairs.items())
    return [attribute for attribute in built if attribute is not None]


def build_payload(
    *,
    event: str,
    status: str,
    stage: str,
    invocation_id: str,
    candidate_sha: str,
    elapsed_seconds: Optional[float],
    now_ns: int,
) -> Dict[str, Any]:
    # status_code 2 (ERROR) on a failed restart, so a failed restart is visible
    # without having to read the attributes.
    status_code = 2 if (event == "finished" and status not in ("", "passed")) else 0
    span = {
        "traceId": trace_id_for(invocation_id),
        "spanId": span_id_for(invocation_id, event, now_ns),
        "name": SPAN_NAME,
        "kind": 1,
        "startTimeUnixNano": str(now_ns),
        "endTimeUnixNano": str(now_ns),
        "status": {"code": status_code},
        "attributes": _attributes(
            {
                "restart.event": event,
                "restart.component": "gateway",
                "restart.status": status,
                "restart.stage": stage,
                "restart.invocation_id": invocation_id,
                "restart.candidate_sha": candidate_sha,
                "restart.elapsed_seconds": elapsed_seconds,
                "schema.version": SCHEMA_VERSION,
            }
        ),
    }
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _attributes({"service.name": SERVICE_NAME})
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "leadpoet.gateway.restart"},
                        "spans": [span],
                    }
                ],
            }
        ]
    }


def post(endpoint: str, token: str, payload: Dict[str, Any], timeout: float) -> int:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(getattr(response, "status", 0) or 0)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", choices=EVENTS, required=True)
    parser.add_argument("--status", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--invocation-id", default="")
    parser.add_argument("--candidate-sha", default="")
    parser.add_argument("--elapsed-seconds", default="")
    parser.add_argument("--env-file", type=Path, default=Path(DEFAULT_ENV_FILE))
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code on failure. For tests only; the "
        "restart path always calls this without it.",
    )
    args = parser.parse_args(argv)

    try:
        endpoint = resolve_value(args.env_file, "GATEWAY_OTEL_ENDPOINT")
        token = resolve_value(args.env_file, "GATEWAY_OTEL_TOKEN")
        if not endpoint or not token:
            raise RuntimeError("gateway OTel endpoint or token is unavailable")
        try:
            elapsed: Optional[float] = round(float(args.elapsed_seconds), 3)
        except (TypeError, ValueError):
            elapsed = None
        payload = build_payload(
            event=args.event,
            status=args.status,
            stage=args.stage,
            invocation_id=args.invocation_id,
            candidate_sha=args.candidate_sha,
            elapsed_seconds=elapsed,
            now_ns=time.time_ns(),
        )
        post(endpoint, token, payload, args.timeout)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(f"gateway restart marker not published: {error}", file=sys.stderr)
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
