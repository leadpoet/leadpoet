"""A fixed, committed confirmation set and cohort for Arena integrity rounds."""

from __future__ import annotations

import asyncio
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import Any, Mapping, Sequence

from lab_arena import contracts, integrity, verify

BANK_SCHEMA = "leadpoet.lab_arena.confirmation_bank.v1"
COHORT_SCHEMA = "leadpoet.lab_arena.confirmation_cohort.v1"


def build_bank(round_id: str, icps: Sequence[Mapping[str, Any]], main_icps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(icps) != contracts.CONFIRMATION_ICP_COUNT:
        raise ValueError("confirmation requires exactly five fresh ICPs")
    seen = {integrity.requirement_fingerprint(icp) for icp in main_icps}
    prepared = []
    for index, icp in enumerate(icps):
        projected = integrity.agent_visible_icp(icp)
        fingerprint = integrity.requirement_fingerprint(projected)
        if fingerprint in seen:
            raise ValueError("confirmation ICP repeats an evaluated requirement set")
        seen.add(fingerprint)
        projected["icp_id"] = "confirmation_%s_%03d" % (round_id, index + 21)
        # Reuse the real scorer's input validation without making provider calls.
        from qualification.scoring.competition import _normalized_icp
        _normalized_icp(projected)
        prepared.append(projected)
    return {"schema_version": BANK_SCHEMA, "round_id": round_id, "nonce": secrets.token_hex(32), "icps": prepared}


def read_bank(payload: bytes, *, round_id: str, digest: str) -> dict[str, Any]:
    if contracts.hash_bytes(payload) != digest:
        raise ValueError("confirmation artifact hash mismatch")
    document = json.loads(payload)
    if not isinstance(document, dict) or set(document) != {"schema_version", "round_id", "nonce", "icps"} or document["schema_version"] != BANK_SCHEMA or document["round_id"] != round_id:
        raise ValueError("confirmation artifact scope mismatch")
    if not isinstance(document["icps"], list) or len(document["icps"]) != contracts.CONFIRMATION_ICP_COUNT or not all(isinstance(icp, dict) for icp in document["icps"]):
        raise ValueError("confirmation artifact invalid")
    nonce = document["nonce"]
    if not isinstance(nonce, str) or len(nonce) != 64 or any(c not in "0123456789abcdef" for c in nonce):
        raise ValueError("confirmation artifact invalid")
    return document


def select_cohort(entries: Sequence[Mapping[str, Any]], eligibility: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baselines = [dict(entry) for entry in entries if entry.get("is_king")]
    if len(baselines) != 1 or baselines[0].get("final_score") is None:
        raise ValueError("confirmation requires a valid baseline")
    baseline = baselines[0]
    floor = Fraction(repr(float(baseline["final_score"]))) + verify.PROMOTION_THRESHOLD_POINTS
    contenders = [dict(entry) for entry in entries if not entry.get("is_king")
        and entry.get("final_score") is not None
        and eligibility.get(str(entry["submission_id"]), {}).get("eligible") is True
        and Fraction(repr(float(entry["final_score"]))) >= floor]
    contenders.sort(key=lambda entry: (-float(entry["final_score"]), str(entry["submission_id"])))
    chosen = contenders[:contracts.CONFIRMATION_FINALIST_COUNT]
    ids = [str(baseline["submission_id"])] + [str(entry["submission_id"]) for entry in chosen]
    return {"schema_version": COHORT_SCHEMA, "submission_ids": ids,
        "baseline_submission_id": str(baseline["submission_id"]),
        "main_entries": [baseline] + chosen,
        "main_eligibility": {submission_id: dict(eligibility[submission_id]) for submission_id in ids},
        "required": bool(chosen)}


def fresh_confirmation_icps(*, round_id: str, evaluation_date: str, main_icps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Generate a separate private bank with the existing trusted generator.

    No template fallback: a missing provider result delays commitment rather
    than substituting a predictable or previously disclosed benchmark.
    """
    from gateway.tasks.icp_generator import generate_icps_with_openrouter

    # Do not transmit the main benchmark. Duplicate checks stay local.
    context = (
        "Generate a separate private Arena confirmation draw. Draw identity: "
        + secrets.token_hex(32)
        + ". Choose varied buyer requirement combinations while preserving "
        "the same realism and breadth requirements."
    )
    def generate():
        return asyncio.run(asyncio.wait_for(generate_icps_with_openrouter(int(evaluation_date.replace("-", "")), total_icps=20, generation_context=context), timeout=600))

    # Service methods may be called from an ASGI event loop or a sync worker.
    with ThreadPoolExecutor(max_workers=1) as pool:
        generated = pool.submit(generate).result(timeout=600)
    if not generated:
        raise ValueError("confirmation generation unavailable")
    seen = {integrity.requirement_fingerprint(icp) for icp in main_icps}
    candidates = list(generated[0])
    secrets.SystemRandom().shuffle(candidates)
    selected = []
    for icp in candidates:
        fingerprint = integrity.requirement_fingerprint(icp)
        if fingerprint not in seen:
            selected.append(dict(icp))
            seen.add(fingerprint)
        if len(selected) == contracts.CONFIRMATION_ICP_COUNT:
            return selected
    raise ValueError("confirmation generator returned insufficient distinct ICPs")
