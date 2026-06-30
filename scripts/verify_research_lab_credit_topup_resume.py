"""Verify resume_credit_blocked_runs_for_miner: re-queues a miner's blocked_for_credit
runs as credit_topup_resume when funded, leaves them still_blocked when depleted,
ignores other miners' runs, and honors explicit run_ids. (Endpoint is a thin wrapper
over this fastapi-free function.)

Run: python3 scripts/verify_research_lab_credit_topup_resume.py  (exit 0 == pass)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gateway.research_lab.recovery as recovery  # noqa: E402


@contextlib.contextmanager
def patched(**overrides):
    sentinel = object()
    originals = {k: getattr(recovery, k, sentinel) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(recovery, k, v)
        yield
    finally:
        for k, v in originals.items():
            if v is sentinel:
                delattr(recovery, k)
            else:
                setattr(recovery, k, v)


class FakeConfig:
    @staticmethod
    def from_env():
        return SimpleNamespace()


def check(errors, cond, msg):
    if not cond:
        errors.append(msg)


def _common(blocked_rows, tickets, credit_ready, created):
    async def sa(table, **kwargs):
        return list(blocked_rows)

    async def so(table, **kwargs):
        filters = dict((f[0], f[1]) for f in kwargs.get("filters", ()))
        return tickets.get(filters.get("ticket_id"))

    async def cqe(**kwargs):
        created.append(kwargs)
        return {"event_id": "e", "seq": 1, "anchored_hash": "h"}

    async def credit(config, *, ticket_id):
        return credit_ready

    return dict(
        ResearchLabGatewayConfig=FakeConfig,
        autoresearch_queue_capacity_doc=lambda c: {"autoresearch_capacity": 5},
        select_all=sa,
        select_one=so,
        create_queue_event=cqe,
        _openrouter_credit_ready=credit,
    )


def _row(run_id, ticket_id):
    return {"run_id": run_id, "ticket_id": ticket_id, "current_queue_status": "paused",
            "current_reason": "blocked_for_credit", "queue_priority": 0}


def test_funded(errors):
    created = []
    with patched(**_common([_row("run-1", "tk-1")],
                           {"tk-1": {"miner_hotkey": "5HMINER"}}, (True, "funded"), created)):
        out = asyncio.run(recovery.resume_credit_blocked_runs_for_miner("5HMINER"))
    check(errors, out["requeued"] == 1 and out["still_blocked"] == 0, f"[funded] {out}")
    check(errors, created and created[0]["reason"] == "credit_topup_resume", "[funded] wrong reason")
    check(errors, created and created[0]["event_doc"]["resume_source"] == "miner_credit_topup_resume",
          "[funded] missing resume_source")


def test_depleted(errors):
    created = []
    with patched(**_common([_row("run-1", "tk-1")],
                           {"tk-1": {"miner_hotkey": "5HMINER"}}, (False, "limit_remaining=0"), created)):
        out = asyncio.run(recovery.resume_credit_blocked_runs_for_miner("5HMINER"))
    check(errors, out["requeued"] == 0 and out["still_blocked"] == 1, f"[depleted] {out}")
    check(errors, len(created) == 0, "[depleted] must not requeue depleted")


def test_other_miner(errors):
    created = []
    with patched(**_common([_row("run-1", "tk-1")],
                           {"tk-1": {"miner_hotkey": "5HOTHER"}}, (True, "funded"), created)):
        out = asyncio.run(recovery.resume_credit_blocked_runs_for_miner("5HMINER"))
    check(errors, out["requeued"] == 0 and len(created) == 0, "[other-miner] must ignore other miner's run")


def test_run_ids(errors):
    created = []
    rows = [_row("run-1", "tk-1"), _row("run-2", "tk-2")]
    tickets = {"tk-1": {"miner_hotkey": "5HMINER"}, "tk-2": {"miner_hotkey": "5HMINER"}}
    with patched(**_common(rows, tickets, (True, "funded"), created)):
        out = asyncio.run(recovery.resume_credit_blocked_runs_for_miner("5HMINER", run_ids=["run-2"]))
    check(errors, out["requeued"] == 1 and created and created[0]["run_id"] == "run-2",
          "[run_ids] only requested run should requeue")


def main() -> int:
    errors: list[str] = []
    for t in (test_funded, test_depleted, test_other_miner, test_run_ids):
        try:
            t(errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[{t.__name__}] raised {type(exc).__name__}: {exc}")
    if errors:
        print("FAIL — credit-topup-resume verification")
        for e in errors:
            print("  -", e)
        return 1
    print("PASS — credit-topup-resume verification (funded requeue w/ credit_topup_resume, depleted "
          "still_blocked, other-miner ignored, run_ids filter)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
