#!/usr/bin/env python3
"""Verify Langfuse observability redaction/no-op contracts."""

from __future__ import annotations

from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.observability.langfuse_client import langfuse_enabled, observation  # noqa: E402
from research_lab.observability.redaction import RedactionBlocked, redact_for_langfuse  # noqa: E402
from research_lab.observability.score_export import score_payload_from_bundle  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _must_block(payload: object) -> None:
    try:
        redact_for_langfuse(payload)
    except RedactionBlocked:
        return
    raise AssertionError(f"payload was not blocked: {payload!r}")


def main() -> int:
    os.environ["LANGFUSE_ENABLED"] = "false"
    _assert(not langfuse_enabled(), "Langfuse must be disabled by default in verifier")
    with observation("contract.noop", metadata={"run_id": "00000000-0000-0000-0000-000000000001"}) as obs:
        _assert(obs is None, "disabled Langfuse observation should be a no-op")

    _must_block({"openrouter_api_key": "sk-or-secret"})
    _must_block({"event_doc": {"prompt": "hidden prompt"}})
    _must_block({"event_doc": {"llm_response": "private response"}})
    _must_block({"event_doc": {"page_content": "private page"}})
    _must_block({"event_doc": "hidden_icp marker"})
    safe = redact_for_langfuse(
        {
            "run_id": "00000000-0000-0000-0000-000000000001",
            "email": "person@example.com",
            "score_bundle_hash": "sha256:" + "a" * 64,
        }
    )
    _assert("[REDACTED:email:" in safe["email"], "email should be redacted")
    scores = score_payload_from_bundle(
        {
            "aggregates": {
                "base_score": 10.0,
                "candidate_score": 12.0,
                "mean_delta": 2.0,
                "delta_lcb": 1.1,
                "hard_failure_count": 0,
                "successful_icp_count": 20,
            },
            "reward_path": {"eligible_for_probation": True},
        }
    )
    _assert(scores["mean_delta"] == 2.0, "mean_delta score export missing")
    _assert(scores["eligible_for_probation"] is True, "reward path score export missing")
    print("langfuse observability contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
