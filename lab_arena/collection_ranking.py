"""Elect one comparison-pack judge on the active, funded worker lease."""

from __future__ import annotations

import hashlib
import json
import re
from lab_arena import collection_judge as judge, store
from lab_arena.collection_workflow import store_artifact

RPC = "collection_rank_pack_v1"
store.FUNCTION_SIGNATURES[RPC] = (
    ("p_job", "uuid"),
    ("p_lease", "text"),
    ("p_packet", "text"),
    ("p_assessment", "text"),
)


def rank_pack_if_ready(
    job, packet, assessment, policy, prompt, broker, objects, database
):
    """Return the private ranking artifact reference, or None for other pack members.

    The worker must persist generator and judge artifacts before entering this hook.
    Terminal output and payout eligibility never depend on winning a ranking.
    """
    run_id = job["job_id"]
    batch = policy.batch_id

    def identity(s):
        return type(s) is str and re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", s)

    if (
        not identity(run_id)
        or not identity(batch)
        or job["batch_id"] != batch
        or packet["run_id"] != run_id
        or assessment["run_id"] != run_id
        or packet["scenario_id"] != job["scenario_id"]
    ):
        raise ValueError("invalid rank job")
    ready = database.rpc(
        RPC,
        {
            "p_job": run_id,
            "p_lease": store.hash_lease_token(job["lease_token"]),
            "p_packet": "sha256:" + judge.digest(packet),
            "p_assessment": "sha256:" + judge.digest(assessment),
        },
    )
    if ready.get("status") == "waiting":
        return None
    if ready.get("status") != "ready":
        raise ValueError("invalid pack authority")
    members = ready["members"]
    if not 1 <= len(members) <= 6 or run_id not in {m["job_id"] for m in members}:
        raise ValueError("invalid pack members")
    # This pointer is owned by the deterministic elected worker. A lost reply after
    # successful persistence can be replayed without spending or dereferencing a new input.
    key = f"collection/{batch}/{run_id}/ranking-result-{judge.digest(members)}.json"
    existing = objects.get_optional_bounded(key, 32768)
    if existing is not None:
        return store_artifact(objects, batch, run_id, "ranking", json.loads(existing))
    candidates = []
    for m in members:
        rid = m["job_id"]
        art = []
        for kind, field, limit in [
            ("judge-input", "packet_sha256", 16 * 1024 * 1024),
            ("judge-output", "assessment_sha256", 1024 * 1024),
        ]:
            sha = m[field]
            if not identity(rid) or not re.fullmatch(r"sha256:[a-f0-9]{64}", sha):
                raise ValueError("invalid rank artifact")
            source = objects.get_bounded(
                f"collection/{batch}/{rid}/{kind}-{sha[7:]}.json", limit
            )
            if len(source) > limit or hashlib.sha256(source).hexdigest() != sha[7:]:
                raise ValueError("rank artifact changed")
            art.append(json.loads(source))
        if art[0]["run_id"] != rid or art[0]["scenario_id"] != job["scenario_id"]:
            raise ValueError("rank packet identity")
        candidates.append(tuple(art))
    try:
        record = judge.rank_scenario(policy, candidates, broker.judge, prompt=prompt)
    except Exception as exc:
        # A failed paid dispatch may already be charged. Persist an explicit gap;
        # this never grants a winner or emits an unverified training preference.
        record = {
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "record": None,
        }
    result = record or {"status": "insufficient_eligible_runs", "record": None}
    result = {
        **result,
        "batch_id": batch,
        "scenario_id": job["scenario_id"],
        "pack_digest": judge.digest(members),
    }
    payload = judge._json(result).encode()
    objects.put(key, payload)
    if objects.get_bounded(key, len(payload)) != payload:
        raise ValueError("ranking receipt readback")
    return store_artifact(objects, batch, run_id, "ranking", result)
