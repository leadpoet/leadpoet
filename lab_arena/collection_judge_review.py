"""Authenticate a real human's blind pilot marks and bind them to immutable evidence."""

from __future__ import annotations

import base64
import re
from dataclasses import asdict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)

from lab_arena import collection_judge as judge

BUNDLE_SCHEMA = "collection.review.v1"
RECEIPT_SCHEMA = "collection.human-review.v1"


def _read_bundle(bundle):
    judge._fields(
        bundle, ("schema", "policy", "assignments", "runs", "planted", "drift")
    )
    judge._require(bundle["schema"] == BUNDLE_SCHEMA, "bundle schema")
    policy = judge.BatchPolicy(**bundle["policy"])
    schedule = judge.assign_judges(policy, bundle["assignments"])
    assessments, packets = [], {}
    for item in bundle["runs"]:
        judge._fields(item, ("packet", "assessment", "manifest_sha256"))
        judge._require(
            isinstance(item["manifest_sha256"], str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", item["manifest_sha256"]),
            "immutable run manifest digest",
        )
        p, a = item["packet"], item["assessment"]
        rid = p["run_id"]
        judge._require(
            rid in schedule
            and rid not in packets
            and a["run_id"] == rid
            and a["scenario_id"] == p["scenario_id"],
            "wrong or duplicate run",
        )
        packets[rid] = p
        assessments.append(a)
        checks = p["checks"]
        judge._require(
            isinstance(checks, dict)
            and checks
            and all(type(v) is bool for v in checks.values()),
            "invalid recorded checks",
        )
        judge._require(
            a["batch_id"] == policy.batch_id
            and a["policy_sha256"] == judge.digest(asdict(policy))
            and a["prompt_sha256"] == policy.prompt_sha256,
            "assessment policy",
        )
        if not all(checks.values()):
            judge._require(
                a["status"] == "code_rejected"
                and a["records"] == []
                and a["keep_steps"] == 0
                and a["hard_steps"] == [],
                "invalid deterministic rejection",
            )
            continue
        if p.get("projection_error"):
            judge._require(
                a["status"] == "cannot_judge"
                and a.get("projection_error") == p["projection_error"]
                and a["records"] == []
                and a["keep_steps"] == 0
                and a["hard_steps"] == [],
                "invalid input gap",
            )
            continue
        scenario, run = judge._packet(p)
        judge._require(
            a["packet_sha256"] == judge.digest([scenario, run]),
            "assessment packet digest",
        )
        n = len(run["steps"])
        judge._require(
            type(a["records"]) is list and 1 <= len(a["records"]) <= 2,
            "judge record count",
        )
        for v in a["records"]:
            identity = {
                "spec_version": judge.SPEC,
                "run_id": rid,
                "scenario_id": p["scenario_id"],
                "judge": v["judge"],
                "judge_version": policy.version(v["judge"]),
            }
            judge._require(
                v == judge.validate_verdict(v, identity, n), "noncanonical verdict"
            )
        decision = judge._decide(policy, a["records"], n, {})
        decision["claim_review"] = judge.claim_review(p, a["records"])
        judge._require(
            all(a[k] == v for k, v in decision.items()), "assessment decision"
        )
    judge._require(set(schedule) == set(packets), "incomplete batch")
    control_ids = set(bundle["planted"]) | set(bundle["drift"])
    reviewable = [
        packets[a["run_id"]]
        for a in assessments
        if a["records"] and a["run_id"] not in control_ids
    ]
    return policy, assessments, reviewable


def prepare_review(bundle):
    policy, _, packets = _read_bundle(bundle)
    return judge.blind_human_sample(policy, packets)


def claim_review_jobs(bundle):
    """Only packets with incomplete/disputed independent claim evidence need a human."""
    policy, assessments, _ = _read_bundle(bundle)
    by_id = {a["run_id"]: a for a in assessments}
    aliases, jobs = {}, {}
    for item in bundle["runs"]:
        p = item["packet"]
        a = by_id[p["run_id"]]
        if (
            p["run_id"] in bundle["planted"]
            or p["run_id"] in bundle["drift"]
            or a["status"] not in ("accepted", "salvaged")
            or a["claim_review"] not in ("missing", "disputed")
        ):
            continue
        alias = judge.digest([policy.seed, "human claim review", p["run_id"]])[:24]
        scenario, run = judge._packet(p)
        aliases[alias] = p["run_id"]
        jobs[alias] = {
            "scenario": scenario,
            "run": run,
            "claim_units": (p.get("claim_reader") or {}).get("units"),
            "unsupported_claims": [v["unsupported_claims"] for v in a["records"]],
        }
    return aliases, jobs


def _validate_marks(marks):
    fields = {"sample_sha256", "marks"}
    if "claim_decisions" in marks:
        fields.update(("claim_decisions", "claim_sample_sha256"))
    judge._fields(marks, fields)


def _report(bundle, form, reviewer):
    scores = form["marks"]
    policy, assessments, _ = _read_bundle(bundle)
    marks = {
        alias: {"source": "human_blind", "reviewer": reviewer, "scores": parts}
        for alias, parts in scores.items()
    }
    report = judge.quality_report(
        policy,
        assignments=bundle["assignments"],
        assessments=assessments,
        planted=bundle["planted"],
        drift=bundle["drift"],
        human_marks=marks,
    )

    approved = []
    if "claim_decisions" in form:
        aliases, jobs = claim_review_jobs(bundle)
        decisions = form["claim_decisions"]
        judge._require(
            form["claim_sample_sha256"] == judge.digest(jobs)
            and type(decisions) is dict
            and set(decisions) <= set(aliases),
            "claim sample changed",
        )
        for alias, submitted in decisions.items():
            judge._require(
                type(submitted) is bool or submitted is None,
                "claim adjudication must be boolean or pending",
            )
            if submitted is True:
                approved.append(aliases[alias])
    report["approved_claim_runs"] = sorted(approved)
    return report


def sign_review(bundle, blind_marks, reviewer_id, private_key_pem):
    """Called only by the reviewer after completing the 50 blind marks. Store failures too."""
    judge._require(judge._text(reviewer_id, 128), "reviewer identity")
    aliases, jobs = prepare_review(bundle)
    _validate_marks(blind_marks)
    judge._require(
        blind_marks["sample_sha256"] == judge.digest(jobs)
        and set(blind_marks["marks"]) == set(aliases),
        "incomplete or mismatched blind marks",
    )
    report = _report(bundle, blind_marks, reviewer_id)
    body = {
        "schema": RECEIPT_SCHEMA,
        "bundle_sha256": judge.digest(bundle),
        "reviewer": reviewer_id,
        "blind_marks": blind_marks,
        "report": report,
    }
    key = load_pem_private_key(private_key_pem, password=None)
    judge._require(isinstance(key, Ed25519PrivateKey), "expected Ed25519 reviewer key")
    return {
        **body,
        "signature": base64.b64encode(key.sign(judge._json(body).encode())).decode(
            "ascii"
        ),
    }


def verify_review(bundle, receipt, public_key_pem):
    """Verify against the deployment's pinned reviewer key before releasing any batch."""
    judge._fields(
        receipt,
        ("schema", "bundle_sha256", "reviewer", "blind_marks", "report", "signature"),
    )
    judge._require(
        receipt["schema"] == RECEIPT_SCHEMA
        and judge._text(receipt["reviewer"], 128)
        and receipt["bundle_sha256"] == judge.digest(bundle),
        "review receipt identity",
    )
    body = {k: v for k, v in receipt.items() if k != "signature"}
    key = load_pem_public_key(public_key_pem)
    judge._require(
        isinstance(key, Ed25519PublicKey), "expected pinned Ed25519 reviewer key"
    )
    try:
        key.verify(
            base64.b64decode(receipt["signature"], validate=True),
            judge._json(body).encode(),
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise ValueError("review signature invalid") from exc
    marks = receipt["blind_marks"]
    _validate_marks(marks)
    aliases, jobs = prepare_review(bundle)
    judge._require(
        marks["sample_sha256"] == judge.digest(jobs)
        and set(marks["marks"]) == set(aliases),
        "review sample changed",
    )
    report = _report(bundle, marks, receipt["reviewer"])
    judge._require(
        report == receipt["report"] and report["quality_ready"],
        "batch did not pass review",
    )
    return report


def registered_bundle(policy, assignments, *, planted, drift, jobs, objects):
    """Reconstruct from server-owned *complete* batch records at release/sign-off.

    The caller supplies enrollment/assignment metadata and controls frozen at batch open,
    plus persisted job rows from the DB. A supplied reviewer copy grants no authority.
    """
    import hashlib
    import json

    def ident(value):
        return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value)

    def verified(rid, kind, ref, limit):
        sha = ref.get("sha256")
        judge._require(
            isinstance(sha, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", sha),
            "registered artifact digest",
        )
        key = f"collection/{policy.batch_id}/{rid}/{kind}-{sha[7:]}.json"
        judge._require(ref.get("key", key) == key, "registered artifact path")
        raw = objects.get_bounded(key, limit)
        judge._require(
            len(raw) <= limit and hashlib.sha256(raw).hexdigest() == sha[7:],
            "registered artifact content hash",
        )
        return json.loads(raw)

    judge._require(ident(policy.batch_id), "invalid batch identifier")
    by_id = {a["run_id"]: a for a in assignments}
    known = {r["job_id"]: r for r in jobs}
    judge._require(
        len(by_id) == len(assignments)
        and len(known) == len(jobs)
        and known.keys() == by_id.keys(),
        "registered job set",
    )
    runs = []
    for rid in sorted(by_id):
        record = known[rid]
        a = by_id[rid]
        judge._require(
            ident(rid)
            and record["batch_id"] == policy.batch_id
            and record["model"] == a["generator"]
            and record["harness_commit"] == a["harness"]
            and record["miner_hotkey"] == a["miner"],
            "registered job metadata",
        )
        manifest = verified(
            rid, "manifest", {"sha256": record["artifact_sha256"]}, 1024 * 1024
        )
        judge._require(
            manifest["job_id"] == rid
            and manifest["batch_id"] == record["batch_id"]
            and manifest["judge_policy"] == asdict(policy)
            and manifest["model"] == record["model"]
            and manifest["harness_commit"] == record["harness_commit"]
            and manifest["scenario_id"] == record["scenario_id"],
            "registered manifest identity",
        )
        packet = verified(rid, "judge-input", manifest["packet"], 16 * 1024 * 1024)
        assessment = verified(rid, "judge-output", manifest["assessment"], 1024 * 1024)
        judge._require(
            packet["run_id"] == rid and packet["scenario_id"] == record["scenario_id"],
            "registered packet identity",
        )
        runs.append(
            {
                "packet": packet,
                "assessment": assessment,
                "manifest_sha256": record["artifact_sha256"],
            }
        )
    result = {
        "schema": BUNDLE_SCHEMA,
        "policy": asdict(policy),
        "assignments": sorted(assignments, key=lambda a: a["run_id"]),
        "runs": runs,
        "planted": planted,
        "drift": drift,
    }
    _read_bundle(result)
    return result
