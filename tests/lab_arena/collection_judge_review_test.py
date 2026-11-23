from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from lab_arena import collection_judge as j, collection_judge_review as review
from scripts import collection_review as cli
from tests.lab_arena.collection_judge_test import (
    POLICY,
    PROMPT,
    Caller,
    fail_truth,
    packet,
)


@pytest.fixture(scope="module")
def batch():
    policy = replace(POLICY, pilot=True)
    ids = [f"r-{i}" for i in range(132)]
    assignments = [
        dict(run_id=r, generator=f"g{i%6}", harness="shared", miner="private-payer")
        for i, r in enumerate(ids)
    ]
    schedule = j.assign_judges(policy, assignments)
    runs = []
    for rid in ids:
        p = packet(rid)
        c = Caller((lambda v, n: fail_truth(v)) if rid == ids[-1] else None)
        runs.append(
            {
                "manifest_sha256": "sha256:" + j.digest([rid, "fixture manifest"]),
                "packet": p,
                "assessment": j.judge_run(policy, schedule[rid], p, c, prompt=PROMPT),
            }
        )
    return {
        "schema": review.BUNDLE_SCHEMA,
        "policy": asdict(policy),
        "assignments": assignments,
        "runs": runs,
        "planted": {ids[-1]: False, ids[-2]: True},
        "drift": {
            r: {j_: dict.fromkeys(j.PARTS, 3) for j_ in j.JUDGES} for r in ids[100:130]
        },
    }


def keys():
    key = Ed25519PrivateKey.generate()
    return (
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ),
    )


def complete_marks(bundle):
    aliases, jobs = review.prepare_review(bundle)
    return {
        "sample_sha256": j.digest(jobs),
        "marks": {alias: dict.fromkeys(j.PARTS, 3) for alias in aliases},
    }


def test_signed_marks_are_bound_to_exact_run_receipts_and_review_authority(batch):
    private, public = keys()
    # The signing key and marks in these tests exist for the fixture only.
    receipt = review.sign_review(
        batch, complete_marks(batch), "fixture human identity", private
    )
    assert review.verify_review(batch, receipt, public)["quality_ready"]
    wrong = deepcopy(receipt)
    wrong["report"]["blind_human_marks"] = 49
    with pytest.raises(ValueError, match="signature"):
        review.verify_review(batch, wrong, public)
    with pytest.raises(ValueError, match="signature"):
        review.verify_review(batch, receipt, keys()[1])
    mutated = deepcopy(batch)
    mutated["runs"][0]["packet"]["steps"][0]["action"] = "modified"
    with pytest.raises(ValueError, match="identity"):
        review.verify_review(mutated, receipt, public)
    with pytest.raises(ValueError, match="packet"):
        review.prepare_review(mutated)
    forged = deepcopy(batch)
    forged["runs"][0]["assessment"]["keep_steps"] = 1
    with pytest.raises(ValueError, match="decision"):
        review.prepare_review(forged)
    unrelated = complete_marks(batch)
    unrelated["sample_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="blind marks"):
        review.sign_review(batch, unrelated, "fixture", private)


def test_cli_sample_to_real_mark_import_then_signature_check(batch, tmp_path, capsys):
    private, public = keys()
    paths = {
        k: tmp_path / k for k in ["bundle", "jobs", "marks", "key", "pub", "receipt"]
    }
    cli.write_new(paths["bundle"], batch)
    for k, b in [("key", private), ("pub", public)]:
        paths[k].write_bytes(b)
        paths[k].chmod(0o600)
    assert (
        cli.main(
            [
                "sample",
                "--bundle",
                str(paths["bundle"]),
                "--jobs",
                str(paths["jobs"]),
                "--marks-template",
                str(paths["marks"]),
            ]
        )
        == 0
    )
    sample = json.loads(paths["jobs"].read_bytes())
    assert len(sample) == 50 and "private-payer" not in json.dumps(sample)
    marks = json.loads(paths["marks"].read_bytes())
    assert all(v is None for doc in marks["marks"].values() for v in doc.values())
    sign = [
        "sign",
        "--bundle",
        str(paths["bundle"]),
        "--reviewer",
        "test reviewer",
        "--private-key",
        str(paths["key"]),
        "--marks",
        str(paths["marks"]),
        "--receipt",
        str(paths["receipt"]),
    ]
    assert cli.main(sign) == 2 and not paths["receipt"].exists()
    paths["marks"].write_text(json.dumps(complete_marks(batch)))
    assert cli.main(sign) == 0 and paths["receipt"].stat().st_mode & 0o077 == 0
    assert (
        cli.main(
            [
                "verify",
                "--bundle",
                str(paths["bundle"]),
                "--receipt",
                str(paths["receipt"]),
                "--public-key",
                str(paths["pub"]),
            ]
        )
        == 0
    )
    logged = capsys.readouterr()
    assert "private-payer" not in logged.out and "reference_answer" not in logged.out


def test_failed_control_persists_signed_hold_and_cannot_release(batch):
    b = deepcopy(batch)
    b["planted"][next(iter(b["planted"]))] = True
    private, public = keys()
    receipt = review.sign_review(b, complete_marks(b), "fixture", private)
    assert "planted_control_mismatch" in receipt["report"]["holds"]
    with pytest.raises(ValueError, match="did not pass"):
        review.verify_review(b, receipt, public)


def test_registered_manifest_refuses_relabeling_input_and_omitted_runs():
    from lab_arena.collection_workflow import store_artifact

    class Objects(dict):
        def get_bounded(self, key, size):
            return self[key]

        def put(self, key, data):
            self[key] = data

    objects = Objects()
    rid = "job-abc"
    p = packet(rid)
    assignments = [
        dict(
            run_id=rid,
            generator="gen",
            harness="commit",
            miner="payer",
            slot=0,
            cohort=1,
        )
    ]
    a = j.judge_run(
        POLICY, j.assign_judges(POLICY, assignments)[rid], p, Caller(), prompt=PROMPT
    )
    job = {
        "job_id": rid,
        "model": "gen",
        "harness_commit": "commit",
        "miner_hotkey": "payer",
        "batch_id": POLICY.batch_id,
        "scenario_id": p["scenario_id"],
    }
    m = {
        **job,
        "judge_policy": asdict(POLICY),
        "generator": store_artifact(objects, POLICY.batch_id, rid, "generator", {"run": rid, "teacher": "gen"}),
        "packet": store_artifact(objects, POLICY.batch_id, rid, "judge-input", p),
        "assessment": store_artifact(objects, POLICY.batch_id, rid, "judge-output", a),
    }
    job["artifact_sha256"] = store_artifact(
        objects, POLICY.batch_id, rid, "manifest", m
    )["sha256"]
    kw = {"jobs": [job], "objects": objects, "planted": {}, "drift": {}}
    assert review.registered_bundle(POLICY, assignments, **kw)["runs"] == [
        {"packet": p, "assessment": a, "manifest_sha256": job["artifact_sha256"]}
    ]
    with pytest.raises(ValueError, match="registered job set"):
        review.registered_bundle(POLICY, [], **kw)
    with pytest.raises(ValueError, match="metadata"):
        review.registered_bundle(
            POLICY, [dict(assignments[0], generator="spoof")], **kw
        )
    objects[m["packet"]["key"]] += b"\n"
    with pytest.raises(ValueError, match="content hash"):
        review.registered_bundle(POLICY, assignments, **kw)


def test_bound_claim_adjudication_is_the_only_override_for_missing_reader(batch):
    b = deepcopy(batch)
    item = b["runs"][0]
    p = item["packet"]
    pol = j.BatchPolicy(**b["policy"])
    sched = j.assign_judges(pol, b["assignments"])
    p["claim_reader"] = {
        "units": [
            {
                "id": "email",
                "text": "Check pricing.",
                "sources": {"file": "Pricing is preliminary."},
            }
        ],
        "results": [],
    }
    item["assessment"] = j.judge_run(
        pol, sched[p["run_id"]], p, Caller(), prompt=PROMPT
    )
    aliases, jobs = review.claim_review_jobs(b)
    (alias,) = aliases
    assert aliases[alias] == p["run_id"] and "private-payer" not in json.dumps(jobs)
    private, public = keys()
    form = {
        **complete_marks(b),
        "claim_sample_sha256": j.digest(jobs),
        "claim_decisions": {alias: True},
    }
    rec = review.sign_review(b, form, "fixture reviewer", private)
    report = review.verify_review(b, rec, public)
    assert not j.training_selection(item["assessment"])
    assert j.training_selection(
        item["assessment"], approved_claim_runs=report["approved_claim_runs"]
    )
