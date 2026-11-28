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
    Encoding,
    PublicFormat,
    load_pem_private_key,
    load_pem_public_key,
)

from lab_arena import collection_judge as judge

BUNDLE_SCHEMA = "collection.review.v1"
RECEIPT_SCHEMA = "collection.human-review.v1"
BLIND_RECEIPT_SCHEMA = "collection.human-review.v2"


class QualificationSampleUnavailable(ValueError):
    """Only the fixed gate names are safe to print outside private review."""


def _read_bundle(bundle, *, require_split=True):
    fields = {"schema", "policy", "assignments", "runs", "planted", "drift"}
    if "capture_version" in bundle:
        from lab_arena.collection_workflow import comparison_revisions

        comparison_revisions(bundle)
        fields.add("capture_version")
        if bundle.get("failed_attempts"):
            fields.add("failed_journals")
    if "claim_protocol" in bundle:
        from lab_arena.collection_claims import protocol_of
        protocol_of(bundle)
        fields.add("claim_protocol")
    if require_split or "split_manifest_sha256" in bundle:
        fields.add("split_manifest_sha256")
    if "failed_attempts" in bundle:
        fields.add("failed_attempts")
    judge._fields(bundle, fields)
    judge._require(bundle["schema"] == BUNDLE_SCHEMA, "bundle schema")
    policy = judge.BatchPolicy(**bundle["policy"])
    schedule = judge.assign_judges(policy, bundle["assignments"])
    assessments, packets = [], {}
    for item in bundle["runs"]:
        keys = {"packet", "assessment", "manifest_sha256"}
        if "control_sha256" in item:
            keys.add("control_sha256")
            judge._require(
                isinstance(item["control_sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", item["control_sha256"]),
                "control digest",
            )
        judge._fields(item, keys)
        judge._require(
            isinstance(item["manifest_sha256"], str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", item["manifest_sha256"]),
            "immutable run manifest digest",
        )
        p, a = item["packet"], item["assessment"]
        if require_split:
            if any(x["schema"] != bundle.get("claim_protocol","collection.fresh-claims.v1")
                   for x in p.get("fresh_claim_proof", {}).values()):
                raise ValueError("stored claim protocol differs from review batch")
            stamp = p.get("split") or {}
            judge._require(
                re.fullmatch(r"[a-f0-9]{64}", str(bundle["split_manifest_sha256"]))
                and stamp.get("manifest_sha256") == bundle["split_manifest_sha256"]
                and p["checks"].get("training_split") is True
                and p["checks"].get("privacy") is True,
                "unverified review split",
            )
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
        gate = judge.claim_verdict(p)
        if gate in ("failed", "missing"):
            judge._require(
                a.get("claim_gate") == gate
                and a["claim_review"] == gate
                and a["status"]
                == ("claim_rejected" if gate == "failed" else "cannot_judge")
                and not a["records"]
                and a["keep_steps"] == 0
                and not a["hard_steps"],
                "invalid claim gate",
            )
            continue
        scenario, run = judge._packet(p)
        from lab_arena.collection_judge_segments import verify_notes

        verify_notes(p, a)
        judge._require(
            a["packet_sha256"] == judge.digest([scenario, run]),
            "assessment packet digest",
        )
        n = len(run["steps"])
        unavailable = "verifier_unavailable" in a
        judge._require(
            type(a["records"]) is list
            and (0 if unavailable else 1) <= len(a["records"]) <= 2,
            "judge record count",
        )
        generator = schedule[rid]["generator"]
        if policy.qualification:
            judge._require(a.get("generator_family") == judge.model_family(generator), "wrong generator family")
        for v in a["records"]:
            judge._require(v["judge"] in judge.independent_roles(policy, generator), "judge family conflicts with generator")
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
        if unavailable:
            judge.validate_unavailable(policy, a, schedule[rid]["primary"])
            continue
        decision = judge._decide(policy, a["records"], n, {})
        decision["claim_review"] = judge.claim_review(p, a["records"])
        judge._require(
            all(a[k] == v for k, v in decision.items()), "assessment decision"
        )
    failures = bundle.get("failed_attempts", [])
    judge._require(type(failures) is list, "failed attempt list required")
    failed_ids = set()
    attempts = set()
    slots = {a.get("slot"): a for a in bundle["assignments"]}
    for f in failures:
        judge._fields(
            f,
            (
                "run_id",
                "slot",
                "attempt",
                "miner",
                "generator",
                "harness",
                "scenario_id",
            ),
        )
        a = slots.get(f["slot"])
        rid = f["run_id"]
        judge._require(
            policy.pilot
            and a
            and judge._integer(f["slot"], 0, 24575)
            and judge._integer(f["attempt"], 1, 2)
            and (f["slot"], f["attempt"]) not in attempts
            and judge._text(rid, 128)
            and rid not in failed_ids
            and rid not in packets
            and judge._text(f["miner"], 256)
            and judge._text(f["scenario_id"], 128)
            and all(f[k] == a[k] for k in ("generator", "harness"))
            and (rid == a["run_id"] or (f["attempt"] == 1 and rid not in schedule)),
            "invalid failed pilot attempt",
        )
        failed_ids.add(rid)
        attempts.add((f["slot"], f["attempt"]))
    judge._require(
        all(n == 1 or (s, 1) in attempts for s, n in attempts),
        "missing previous failed attempt",
    )
    judge._require(set(schedule) - failed_ids == set(packets), "incomplete batch")
    control_ids = set(bundle["planted"]) | set(bundle["drift"])
    reviewable = [
        packets[a["run_id"]]
        for a in assessments
        if a["records"]
        and not a.get("verifier_unavailable")
        and a["run_id"] not in control_ids
        and all(not v["cannot_judge"] for v in a["records"])
    ]
    return policy, assessments, reviewable


def prepare_review(bundle, *, for_qualification=False):
    policy, assessments, packets = _read_bundle(bundle)
    if for_qualification:
        judge._require(policy.pilot, "qualification requires a pilot")
        report = judge.quality_report(
            policy, assignments=bundle["assignments"], assessments=assessments,
            human_marks={}, planted=bundle["planted"], drift=bundle["drift"],
            failed_run_ids={f["run_id"] for f in bundle.get("failed_attempts", [])},
        )
        holds = set(report["holds"]) - {"human_sample_incomplete", "human_agreement", "judge_qualification"}
        if holds:
            raise QualificationSampleUnavailable("qualification cohort: " + ", ".join(sorted(holds)))
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
        failed_run_ids={f["run_id"] for f in bundle.get("failed_attempts", [])},
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
    return _sign(body, private_key_pem)


def _sign(body, private_key_pem):
    key = load_pem_private_key(private_key_pem, password=None)
    judge._require(isinstance(key, Ed25519PrivateKey), "expected Ed25519 reviewer key")
    return {
        **body,
        "signature": base64.b64encode(key.sign(judge._json(body).encode())).decode(
            "ascii"
        ),
    }


def sign_blind_review(jobs, marks, batch_sha256, reviewer, private_key_pem):
    """Sign the human's decisions on the reviewer machine without disclosing the batch."""
    _validate_marks(marks)
    judge._require(
        judge._text(reviewer, 128) and re.fullmatch(r"[0-9a-f]{64}", str(batch_sha256))
        and type(jobs) is dict and len(jobs) == 50 and set(jobs) == set(marks["marks"])
        and judge.digest(jobs) == marks["sample_sha256"],
        "incomplete or mismatched blind sample",
    )
    for scores in marks["marks"].values():
        judge._fields(scores, judge.PARTS)
        judge._require(all(judge._integer(score, 0, 3) for score in scores.values()), "human scores")
    return _sign({
        "schema": BLIND_RECEIPT_SCHEMA, "bundle_sha256": batch_sha256,
        "reviewer": reviewer, "blind_marks": marks,
    }, private_key_pem)


def _verified_report(bundle, receipt, public_key_pem):
    """Verify against the deployment's pinned reviewer key before releasing any batch."""
    fields = {"schema", "bundle_sha256", "reviewer", "blind_marks", "signature"}
    if receipt.get("schema") == RECEIPT_SCHEMA:
        fields.add("report")
    judge._fields(receipt, fields)
    judge._require(
        receipt["schema"] in (RECEIPT_SCHEMA, BLIND_RECEIPT_SCHEMA)
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
    if receipt["schema"] == RECEIPT_SCHEMA:
        judge._require(report == receipt["report"], "batch did not pass review")
    return report


def verify_review(bundle, receipt, public_key_pem):
    report = _verified_report(bundle, receipt, public_key_pem)
    judge._require(report["quality_ready"], "batch did not pass review")
    return report


def _qualification_body(bundle, receipt, reviewer_key, approver, allow_live):
    from lab_arena.collection_claims import CLAIM_RUBRIC

    policy = judge.BatchPolicy(**bundle["policy"])
    report = _verified_report(bundle, receipt, reviewer_key)
    judge._require(
        type(allow_live) is bool
        and policy.pilot
        and policy.qualification
        and report["holds"] == ["judge_qualification"]
        and judge._text(approver, 128)
        and approver != receipt["reviewer"],
        "independent qualification review required",
    )
    return {
        "schema": "collection.judge-qualification.v1",
        "claim_rubric_sha256": judge.digest(CLAIM_RUBRIC),
        "bundle_sha256": judge.digest(bundle),
        "review_sha256": judge.digest(receipt),
        "prompt_sha256": policy.prompt_sha256,
        "approver": approver,
        **({"claim_protocol":bundle["claim_protocol"]} if "claim_protocol" in bundle else {}),
        "allow_live": allow_live,
        "judges": {
            m: {"revision": spec.revision, "route": spec.route}
            for m, spec in judge.QUALIFICATION_JUDGES.items()
        },
    }


def approve_qualification(
    bundle,
    receipt,
    reviewer_key_pem,
    approver,
    approver_private_pem,
    *,
    allow_live=False,
):
    """Explicit offline human decision after signed blind calibration."""
    body = _qualification_body(bundle, receipt, reviewer_key_pem, approver, allow_live)
    key = load_pem_private_key(approver_private_pem, password=None)
    judge._require(
        isinstance(key, Ed25519PrivateKey), "expected qualification signing key"
    )
    pub = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    judge._require(
        pub
        != load_pem_public_key(reviewer_key_pem).public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ),
        "separate approval key required",
    )
    return {
        **body,
        "signature": base64.b64encode(key.sign(judge._json(body).encode())).decode(),
    }


def verify_qualification(
    bundle, receipt, reviewer_key_pem, attestation, approver_key_pem
):
    """Authenticate the operator's explicit decision against the registered pilot evidence."""
    body = _qualification_body(
        bundle,
        receipt,
        reviewer_key_pem,
        attestation.get("approver"),
        attestation.get("allow_live"),
    )
    judge._require(
        {k: v for k, v in attestation.items() if k != "signature"} == body,
        "qualification approval differs from measured evidence",
    )
    pub = load_pem_public_key(approver_key_pem)
    judge._require(
        isinstance(pub, Ed25519PublicKey)
        and pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
        != load_pem_public_key(reviewer_key_pem).public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ),
        "separate approval key required",
    )
    try:
        pub.verify(
            base64.b64decode(attestation["signature"], validate=True),
            judge._json(body).encode(),
        )
    except (InvalidSignature, ValueError, KeyError, TypeError) as exc:
        raise ValueError("qualification approval signature invalid") from exc
    return body


def terminal_pilot_attempts(rows):
    """Keep the terminal job of each logical slot and every recorded failure."""
    by_slot = {}
    failed = []
    seen = set()
    for row in sorted(rows, key=lambda r: (r["slot"], r.get("attempt", 1))):
        slot, n, rid = row["slot"], row.get("attempt", 1), row["job_id"]
        previous = by_slot.get(slot)
        judge._require(
            rid not in seen
            and judge._integer(slot, 0, 24575)
            and (
                n == 1
                if previous is None
                else (
                    n == 2
                    and previous.get("attempt", 1) == 1
                    and previous["status"] == "failed"
                    and all(
                        previous[k] == row[k]
                        for k in ("batch_id", "scenario_id", "model", "harness_commit")
                    )
                )
            )
            and type(n) is int
            and row["status"] in ("accepted", "rejected", "failed"),
            "invalid pilot slot attempts",
        )
        if row["status"] == "failed":
            judge._require(
                row["artifact_sha256"] is None,
                "failed attempt has conflicting evidence",
            )
            failed.append(
                dict(
                    run_id=rid,
                    slot=slot,
                    attempt=n,
                    miner=row["miner_hotkey"],
                    generator=row["model"],
                    harness=row["harness_commit"],
                    scenario_id=row["scenario_id"],
                )
            )
        else:
            judge._require(
                bool(row["artifact_sha256"]), "missing terminal pilot manifest"
            )
        by_slot[slot] = row
        seen.add(rid)
    return list(by_slot.values()), failed


def registered_bundle(
    policy,
    assignments,
    *,
    planted,
    drift,
    jobs,
    objects,
    expected_builder=None,
    failed_attempts=(),
    capture_version=None,
):
    """Reconstruct from server-owned *complete* batch records at release/sign-off.

    The caller supplies enrollment/assignment metadata and controls frozen at batch open,
    plus persisted job rows from the DB. A supplied reviewer copy grants no authority.
    """
    import hashlib
    import json
    from lab_arena.collection_journal import WorkflowJournal
    from lab_arena.collection_workflow import comparison_revisions

    capture = (
        {"capture_version": capture_version} if capture_version is not None else {}
    )
    agent, builder = comparison_revisions(capture)
    if capture and expected_builder != builder:
        raise ValueError("comparison capture builder differs from the frozen policy")

    def ident(value):
        return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value)

    def verified(rid, kind, ref, limit, batch=None):
        batch = policy.batch_id if batch is None else batch
        judge._require(
            ident(rid) and ident(batch), "invalid registered artifact coordinates"
        )
        sha = ref.get("sha256")
        judge._require(
            isinstance(sha, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", sha),
            "registered artifact digest",
        )
        key = f"collection/{batch}/{rid}/{kind}-{sha[7:]}.json"
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
    control_templates = set()
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
        if record.get("status") == "failed" and policy.pilot:
            judge._require(
                record["artifact_sha256"] is None, "conflicting failed manifest"
            )
            continue
        effective = record.get("pilot_review_sha256", record["artifact_sha256"])
        if "pilot_review_sha256" in record:
            judge._require(
                policy.pilot
                and record["status"] == "accepted"
                and effective != record["artifact_sha256"],
                "invalid pilot review overlay",
            )
        manifest = verified(rid, "manifest", {"sha256": effective}, 1024 * 1024)
        if "pilot_review_sha256" in record:
            judge._require(
                manifest.get("judge_recovery", {}).get("prior_manifest_sha256")
                == record["artifact_sha256"],
                "pilot review is not linked to its original manifest",
            )
        judge._require(
            manifest["job_id"] == rid
            and manifest["batch_id"] == record["batch_id"]
            and manifest["judge_policy"] == asdict(policy)
            and manifest.get("capture_version") == capture_version
            and (not capture or manifest["harness_commit"] == agent)
            and manifest["model"] == record["model"]
            and manifest["harness_commit"] == record["harness_commit"]
            and (
                expected_builder is None
                or manifest.get("builder_commit") == expected_builder
            )
            and manifest["scenario_id"] == record["scenario_id"],
            "registered manifest identity",
        )
        packet = verified(rid, "judge-input", manifest["packet"], 16 * 1024 * 1024)
        assessment = verified(rid, "judge-output", manifest["assessment"], 1024 * 1024)
        if "judge_recovery" in manifest:
            recovery = manifest["judge_recovery"]
            judge._fields(recovery, ("attempt_id", "prior_manifest_sha256"))
            prior = verified(
                rid,
                "manifest",
                {"sha256": recovery["prior_manifest_sha256"]},
                1024 * 1024,
            )
            judge._require(
                (not policy.pilot or "pilot_review_sha256" in record)
                and "judge_recovery" not in prior
                and not prior.get("control")
                and re.fullmatch(
                    r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}",
                    recovery["attempt_id"],
                )
                and {
                    k: v
                    for k, v in manifest.items()
                    if k not in ("judge_recovery", "assessment")
                }
                == {k: v for k, v in prior.items() if k != "assessment"},
                "recovered generator evidence differs from original",
            )
            old = verified(rid, "judge-output", prior["assessment"], 1024 * 1024)
            primary = judge.assign_judges(policy, [a])[rid]["primary"]
            judge.validate_unavailable(policy, old, primary)
            judge._require(
                assessment["records"], "recovered judgment needs genuine verdicts"
            )
            required_votes = (
                2
                if old["overlap"] or assessment["records"][0]["confidence"] == "low"
                else 1
            )
            judge._require(
                assessment["records"][: len(old["records"])] == old["records"]
                and len({v["judge"] for v in assessment["records"]})
                == len(assessment["records"])
                == required_votes
                and old["verifier_unavailable"]["model"]
                in {v["judge_version"] for v in assessment["records"]}
                and not assessment.get("verifier_unavailable")
                and all(not v["cannot_judge"] for v in assessment["records"])
                and all(
                    old[k] == assessment[k]
                    for k in (
                        "packet_sha256",
                        "policy_sha256",
                        "prompt_sha256",
                        "run_id",
                        "batch_id",
                        "scenario_id",
                        "overlap",
                    )
                ),
                "incomplete recovered judgment or changed previous vote",
            )
        judge._require(
            packet["run_id"] == rid and packet["scenario_id"] == record["scenario_id"],
            "registered packet identity",
        )
        control = manifest.get("control")
        if rid in planted or rid in drift:
            from lab_arena import collection_controls as controls

            judge._require(
                control
                and manifest.get("generator") is None
                and not manifest.get("ranking"),
                "control missing dedicated private case",
            )
            case = verified(rid, "control-case", control, 16 * 1024 * 1024)
            checked = controls.validate_case(case)
            if policy.qualification:
                controls.verify_control_family(case, [policy.version(r) for r in policy.roles])
            expected = planted[rid] if rid in planted else drift[rid]
            judge._require(
                control["sha256"] == "sha256:" + checked
                and packet == {**case["packet"], "run_id": rid}
                and case["kind"] == ("planted" if rid in planted else "drift")
                and judge.digest(case["expected"]) == judge.digest(expected),
                "control case or expected answer differs",
            )
            packet_hash = judge.digest(case["packet"])
            judge._require(
                packet_hash not in control_templates, "duplicate control template"
            )
            control_templates.add(packet_hash)
            if case["kind"] == "drift":
                orig = case["provenance"]
                src = orig["source_run_id"]
                batch = orig["source_batch_id"]
                judge._require(
                    batch != policy.batch_id, "drift reference must predate evaluation"
                )
                sh = orig["source_manifest_sha256"]
                baseline = verified(src, "manifest", {"sha256": sh}, 1024 * 1024, batch)
                judge._require(
                    not baseline.get("control"),
                    "drift baseline must be an original measured run",
                )
                original_packet = verified(
                    src, "judge-input", baseline["packet"], 16 * 1024 * 1024, batch
                )
                original_assessment = verified(
                    src, "judge-output", baseline["assessment"], 1024 * 1024, batch
                )
                reproduced = controls.drift_case(
                    original_packet,
                    original_assessment,
                    baseline,
                    source_manifest_sha256=sh,
                )
                judge._require(
                    reproduced == case, "drift anchor differs from recorded baseline"
                )
        else:
            judge._require(control is None, "unexpected control case on generation job")
            judge._require(
                isinstance(manifest.get("generator"), dict), "missing generator upload"
            )
            row = verified(rid, "generator", manifest["generator"], 128 * 1024 * 1024)
            if capture:
                WorkflowJournal(objects, policy.batch_id, rid).comparison_index(
                    complete=True,
                    model=record["model"],
                    manifest_events=manifest.get("events"),
                    generator=row,
                )
            judge._require(
                row.get("run") == rid and row.get("teacher") == record["model"],
                "generator identity differs from registered job",
            )
        runs.append(
            {
                "packet": packet,
                "assessment": assessment,
                "manifest_sha256": effective,
                **({"control_sha256": control["sha256"][7:]} if control else {}),
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
    result.update(capture)
    if failed_attempts:
        result["failed_attempts"] = failed_attempts
        if capture:
            result["failed_journals"] = {
                failed["run_id"]: WorkflowJournal(
                    objects, policy.batch_id, failed["run_id"]
                ).comparison_index(complete=False, model=failed["generator"])
                for failed in failed_attempts
            }
    _read_bundle(result, require_split=False)
    return result
