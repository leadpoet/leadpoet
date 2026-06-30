"""Verify OpenRouter-402 worker handling: classify -> CreditBlockedHostedRunError,
_mark_blocked_for_credit pauses (non-terminal), preflight gate, staleness restart.

Run: python3 scripts/verify_research_lab_credit_block.py  (exit 0 == pass)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gateway.research_lab.worker as worker  # noqa: E402


@contextlib.contextmanager
def patched(**overrides):
    sentinel = object()
    originals = {k: getattr(worker, k, sentinel) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(worker, k, v)
        yield
    finally:
        for k, v in originals.items():
            if v is sentinel:
                delattr(worker, k)
            else:
                setattr(worker, k, v)


def _bare_worker():
    w = worker.ResearchLabHostedWorker.__new__(worker.ResearchLabHostedWorker)
    w.worker_ref = "test-worker"
    w.config = SimpleNamespace(resolve_auto_research_model=lambda tier: ("t", "openai/gpt-x", {}))
    return w


def check(errors, cond, msg):
    if not cond:
        errors.append(msg)


def test_classification(errors):
    # body-error 402 -> CreditBlockedHostedRunError
    try:
        worker._raise_openrouter_generation_response_error(
            {"error": {"code": 402, "message": "This request requires more credits"}},
            failure="no choices", default_retryable=True)
        errors.append("[classify] expected raise")
    except worker.CreditBlockedHostedRunError:
        pass
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[classify] wrong type: {type(exc).__name__}")
    # marker-only message
    check(errors, worker._is_openrouter_credit_block(None, "HTTP 402: insufficient credits"),
          "[classify] marker not detected")
    # non-credit error should NOT be credit-block
    check(errors, not worker._is_openrouter_credit_block({"error": {"code": 500}}, "server error"),
          "[classify] 500 wrongly flagged as credit block")


def test_mark_blocked(errors):
    writes = {"receipt": [], "queue": [], "ticket": [], "project": []}

    async def fake_receipt(**k): writes["receipt"].append(k); return {"event_id": "r"}
    async def fake_queue(**k): writes["queue"].append(k); return {"event_id": "q"}
    async def fake_ticket(**k): writes["ticket"].append(k); return {"event_id": "t"}
    async def fake_project(*a, **k): writes["project"].append(k)
    async def fake_ckpt(run_id): return {"checkpoint_hash": "sha256:ck"}

    ctx = SimpleNamespace(run_id="run-1", ticket_id="tk-1", receipt_id="rc-1",
                          ticket={"miner_hotkey": "5HMiner"}, queue_row={"queue_priority": 0})
    w = _bare_worker()
    w._run_budget_context = lambda c: {"research_model_tier": "default"}
    with patched(create_receipt_event=fake_receipt, create_queue_event=fake_queue,
                 create_ticket_event=fake_ticket, safe_project_public_loop_activity=fake_project,
                 latest_auto_research_checkpoint=fake_ckpt):
        out = asyncio.run(w._mark_blocked_for_credit(ctx, "OpenRouter 402 insufficient credits"))
    check(errors, out.status == "blocked_for_credit", f"[mark] status={out.status}")
    check(errors, writes["receipt"] and writes["receipt"][0]["receipt_status"] == "queued",
          "[mark] receipt must stay queued (non-terminal)")
    check(errors, writes["queue"] and writes["queue"][0]["event_type"] == "paused"
          and writes["queue"][0]["reason"] == "blocked_for_credit", "[mark] queue must be paused/blocked_for_credit")
    check(errors, writes["ticket"] and writes["ticket"][0]["event_type"] == "running",
          "[mark] ticket must stay running (not cancelled)")
    if writes["queue"]:
        doc = writes["queue"][0]["event_doc"]
        check(errors, doc.get("checkpoint_hash") == "sha256:ck", "[mark] missing checkpoint_hash")
        check(errors, doc.get("miner_hotkey") == "5HMiner", "[mark] missing miner_hotkey")
        check(errors, doc.get("model") == "openai/gpt-x", "[mark] missing model")
        check(errors, doc.get("max_tokens") == 1800, "[mark] missing max_tokens")
        check(errors, doc.get("resume_mode") == "resume_from_checkpoint", "[mark] resume_mode wrong")

    # no checkpoint -> still paused, resume_mode restart_from_scratch
    writes["queue"].clear()

    async def fake_ckpt_none(run_id): return None

    with patched(create_receipt_event=fake_receipt, create_queue_event=fake_queue,
                 create_ticket_event=fake_ticket, safe_project_public_loop_activity=fake_project,
                 latest_auto_research_checkpoint=fake_ckpt_none):
        out = asyncio.run(w._mark_blocked_for_credit(ctx, "402"))
    check(errors, out.status == "blocked_for_credit"
          and writes["queue"][0]["event_doc"]["resume_mode"] == "restart_from_scratch",
          "[mark/no-ckpt] should still pause, restart_from_scratch")


def test_preflight(errors):
    w = _bare_worker()
    ctx = SimpleNamespace(run_id="run-2")

    # depleted -> raises CreditBlockedHostedRunError
    def pf_depleted(raw, **k): return {"limit_remaining": 0}
    with patched(preflight_openrouter_key=pf_depleted):
        try:
            asyncio.run(w._preflight_openrouter_credit(ctx, {"OPENROUTER_API_KEY": "sk-or-v1-xxx"}))
            errors.append("[preflight] expected credit block on limit_remaining=0")
        except worker.CreditBlockedHostedRunError:
            pass

    # funded -> no raise
    def pf_funded(raw, **k): return {"limit_remaining": 5.0}
    with patched(preflight_openrouter_key=pf_funded):
        asyncio.run(w._preflight_openrouter_credit(ctx, {"OPENROUTER_API_KEY": "sk-or-v1-xxx"}))  # no raise

    # unmetered (None) -> no raise
    def pf_none(raw, **k): return {"limit_remaining": None}
    with patched(preflight_openrouter_key=pf_none):
        asyncio.run(w._preflight_openrouter_credit(ctx, {"OPENROUTER_API_KEY": "sk-or-v1-xxx"}))

    # preflight transport error -> skipped (no raise; not a credit block)
    def pf_raises(raw, **k): raise RuntimeError("network down")
    with patched(preflight_openrouter_key=pf_raises):
        asyncio.run(w._preflight_openrouter_credit(ctx, {"OPENROUTER_API_KEY": "sk-or-v1-xxx"}))


def test_staleness(errors):
    w = _bare_worker()
    fresh = {"artifact_hash": "sha256:A", "manifest_hash": "sha256:M"}
    stale = {"artifact_hash": "sha256:OLD", "manifest_hash": "sha256:OLDM"}
    artifact = SimpleNamespace(model_artifact_hash="sha256:A", manifest_hash="sha256:M")
    check(errors, w._validate_resume_state_freshness(fresh, artifact, "run-3") is fresh,
          "[stale] fresh checkpoint should be kept")
    check(errors, w._validate_resume_state_freshness(stale, artifact, "run-3") is None,
          "[stale] stale checkpoint should be discarded (restart from scratch)")


def main() -> int:
    errors: list[str] = []
    for t in (test_classification, test_mark_blocked, test_preflight, test_staleness):
        try:
            t(errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[{t.__name__}] raised {type(exc).__name__}: {exc}")
    if errors:
        print("FAIL — credit-block verification")
        for e in errors:
            print("  -", e)
        return 1
    print("PASS — credit-block verification (402 classify, mark_blocked non-terminal pause w/ "
          "checkpoint+model+max_tokens, no-checkpoint restart policy, preflight gate, staleness restart)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
