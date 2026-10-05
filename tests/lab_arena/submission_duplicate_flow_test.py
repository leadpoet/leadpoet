"""Signed intake through the duplicate gate on disposable current-schema PostgreSQL."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lab_arena import broker, contracts, service as svc, submission_runtime
from lab_arena.api import create_app
from lab_arena.code_review_runtime import SubmissionCodeReviewer
from lab_arena.store import hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_service_round import (
    CANARY_DEEPLINE_KEY, CANARY_OPENROUTER_KEY,
    CANARY_OPENROUTER_MANAGEMENT_KEY, Harness, keypair, price_table,
)
from tests.lab_arena.test_queued_submission_replacement_flow import (
    ChecksumObjects, retime_disposable_round, signed,
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database
    return lambda: psycopg2.connect(**dsn)


LICENSE = Path(__file__).resolve().parents[2].joinpath("LICENSE").read_bytes()
MINER_CREDENTIALS = {
    "openrouter_api_key": CANARY_OPENROUTER_KEY,
    "openrouter_management_key": CANARY_OPENROUTER_MANAGEMENT_KEY,
    "deepline_api_key": CANARY_DEEPLINE_KEY,
}


def source(
    code: str, *, prompt: str = "Use verified company evidence.\n",
    include_license: bool = True,
) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            members = [
                ("harness.py", code.encode()),
                ("prompts/evidence.txt", prompt.encode()),
            ]
            if include_license:
                members.append(("LICENSE", LICENSE))
            for path, payload in members:
                member = tarfile.TarInfo(path)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


class LabeledReviewTransport:
    """A fixture judge with explicit expected labels, independent of similarity code."""

    def __init__(self):
        self.calls = []

    def send(self, **request):
        assert request["headers"]["Authorization"] == "Bearer " + CANARY_OPENROUTER_KEY
        assert "organizer-only-key" not in repr(request)
        params = json.loads(request["body"])
        submitted = json.loads(params["messages"][1]["content"])
        files = {item["path"]: item["content"] for item in submitted["submission_files"]}
        contexts = submitted["similarity_contexts"]
        code = files["harness.py"]
        # This case is a deliberate local-identifier rename with no behavior
        # change. The other submitted programs have expected pass labels.
        duplicate = (
            "target = query.industry" in code
            or '"""Documentation."""' in code
            or files["prompts/evidence.txt"] == "Use verified company evidence!\n"
        )
        if duplicate:
            assert contexts
            prompt_duplicate = files["prompts/evidence.txt"].endswith("evidence!\n")
            doc_duplicate = '"""Documentation."""' in code
            findings = [{
                "category": "duplicate_submission",
                "file": "prompts/evidence.txt" if prompt_duplicate else "harness.py",
                "evidence": "evidence!" if prompt_duplicate else (
                    "Documentation." if doc_duplicate else "return target"
                ),
                "explanation": "Only punctuation or local names changed.",
                "comparison_id": contexts[0]["comparison_id"], "confidence": 0.99,
            }]
        else:
            findings = []
        self.calls.append((request, submitted))
        document = {
            "verdict": "reject" if duplicate else "pass",
            "reviewed_files": [item["path"] for item in submitted["submission_files"]],
            "findings": findings,
        }
        if contexts:
            document["reviewed_comparisons"] = [item["comparison_id"] for item in contexts]
        answer = {
            "model": params["model"], "usage": {"cost": "0.001"},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(document)}}],
        }
        return broker.ProviderResponse(200, {"content-type": "application/json"}, json.dumps(answer).encode())


def submit(http, h, round_id, miner, payload, *, duplicate=False, rejection_code=None):
    checksum = base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode()
    presign = http.post("/arena/v1/submissions/presign", json=signed(
        h, miner, round_id, contracts.SCOPE_SUBMISSION_PRESIGN,
        {"source_size_bytes": len(payload), "source_content_md5": checksum,
         "consent": {"public_rerun": True}},
    ))
    assert presign.status_code == 200, presign.text
    target = presign.json()
    assert target["upload_headers"]["content-md5"] == checksum
    h.objects.put(target["source_ref"], payload)
    finalized = http.post(
        f"/arena/v1/submissions/{target['submission_id']}/finalize",
        json=signed(h, miner, round_id, contracts.SCOPE_SUBMISSION_FINALIZE, {
            "submission_id": target["submission_id"],
            "source_ref": target["source_ref"], "source_size_bytes": len(payload),
            "credentials": MINER_CREDENTIALS,
        }),
    )
    if rejection_code:
        assert finalized.status_code == 400, finalized.text
        assert finalized.json()["code"] == rejection_code
    elif duplicate:
        assert finalized.status_code == 409, finalized.text
        assert finalized.json()["code"] == "submission_rejected:duplicate_submission"
    else:
        assert finalized.status_code == 200, finalized.text
        assert finalized.json()["status"] == "accepted"
    return target["submission_id"]


def test_signed_duplicate_gate_review_and_freeze(connect, tmp_path, monkeypatch):
    h = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    h.objects = ChecksumObjects(h.objects_root)
    h.service = h.build_service()
    h.service._submission_request_limiter = SimpleNamespace(
        check=lambda _hotkey: SimpleNamespace(allowed=True)
    )
    h.clock.now = datetime.now(timezone.utc)
    h.chain.epoch = 40601
    round_id = "arena-2026-12-15"
    h.round_id = round_id
    h.service.create_round(datetime.now(timezone.utc) + timedelta(hours=12), round_id=round_id)
    manager = h.service.config.credential_manager
    payer = submission_runtime.SubmissionProviderKeys(
        store=h.service.store, credentials=manager,
        organizer_keys={"openrouter": "organizer-only-key"},
    )
    transport = LabeledReviewTransport()
    h.service.config.code_reviewer = SubmissionCodeReviewer(
        store=h.service.store, objects=h.objects,
        credential_for=payer.code_review_key,
        price_table=price_table(), transport=transport,
        similarity_references_for=h.service.submission_similarity_references,
    )

    base_code = "def run_icp(icp):\n    privatecompetitor = icp.industry\n    return privatecompetitor\n"
    variants = (
        ("base", source(base_code)),
        ("exact", source(base_code)),
        ("comments", source("# changed comment\n" + base_code.replace("    ", "        "))),
        ("rename", source("def run_icp(query):\n    target = query.industry\n    return target\n")),
        ("algorithm", source("def run_icp(icp):\n    return icp.company\n")),
        ("threshold", source("def run_icp(icp):\n    return icp.industry if icp.score >= 7 else None\n")),
        ("noop_doc", source(base_code.replace("    privatecompetitor", "    \"\"\"Documentation.\"\"\"\n    privatecompetitor"))),
        ("prompt_punctuation", source(base_code, prompt="Use verified company evidence!\n")),
        ("prompt_words", source(base_code, prompt="Use recent company evidence.\n")),
    )
    ids = {}
    with TestClient(create_app(h.service)) as http:
        unauthenticated = http.post("/arena/v1/submissions/presign", json={})
        assert unauthenticated.status_code in (400, 401, 403)
        bad_signature = signed(h, keypair("duplicate-flow-invalid"), round_id,
                               contracts.SCOPE_SUBMISSION_PRESIGN,
                               {"source_size_bytes": len(variants[0][1]),
                                "consent": {"public_rerun": True}})
        bad_signature["signature"] = "00" * 64
        assert http.post("/arena/v1/submissions/presign", json=bad_signature).status_code == 401
        for label, payload in variants:
            ids[label] = submit(
                http, h, round_id, keypair("duplicate-flow-" + label), payload,
                duplicate=label in ("exact", "comments"),
            )
        # One normal replacement before the cutoff keeps only the replacement.
        owner = keypair("duplicate-flow-replacement")
        original = submit(http, h, round_id, owner, source("def run_icp(icp):\n    return icp.old\n"))
        replacement = submit(http, h, round_id, owner, source("def run_icp(icp):\n    return icp.new\n"))
        assert h.service.store.get_submission(original)["rejection_rule"] == "source_replaced"
        limit = http.post("/arena/v1/submissions/presign", json=signed(
            h, owner, round_id, contracts.SCOPE_SUBMISSION_PRESIGN,
            {"source_size_bytes": len(source("def run_icp(icp):\n    return icp.third\n")),
             "consent": {"public_rerun": True}},
        ))
        assert limit.status_code == 409
        assert limit.json()["code"] == "submission_replacement_limit_reached"
        assert http.get(f"/arena/v1/submissions/{ids['base']}/code").status_code == 403

    assert h.service.review_pending_submissions()["reviewed"] == 0
    schedule = retime_disposable_round(connect, h, round_id, datetime.now(timezone.utc) + timedelta(seconds=30))
    h.clock.now = datetime.fromisoformat(schedule["submission_cutoff"].replace("Z", "+00:00")) - timedelta(hours=1)
    assert h.service.submission_similarity_references(h.service.store.get_submission(ids["base"]))
    for label in ("base", "rename", "algorithm", "threshold", "noop_doc", "prompt_punctuation", "prompt_words"):
        result = h.service.config.code_reviewer.review(h.service.store.get_submission(ids[label]))
        assert result["status"] == (
            "rejected" if label in ("rename", "noop_doc", "prompt_punctuation") else "passed"
        ), (label, result, h.service.store.get_submission(ids[label])["code_review_doc"])
    assert h.service.config.code_reviewer.review(h.service.store.get_submission(replacement))["status"] == "passed"
    assert len(transport.calls) == 8
    assert all("privatecompetitor" not in repr(submitted["similarity_contexts"])
               for _, submitted in transport.calls)
    for label in ("exact", "comments"):
        row = h.service.store.get_submission(ids[label])
        assert row["status"] == "rejected"
        assert row["rejection_rule"] == "duplicate_submission"
        assert row["code_review_attempts"] == 0
    assert h.service.store.get_submission(ids["rename"])["code_review_doc"]["categories"] == ["duplicate_submission"]
    assert h.service.store.get_submission(ids["prompt_punctuation"])["code_review_doc"]["categories"] == ["duplicate_submission"]
    rename_calls = [request for request, submitted in transport.calls
                    if "target = query.industry" in next(file["content"] for file in submitted["submission_files"]
                                                     if file["path"] == "harness.py")]
    assert len(rename_calls) == 1
    assert b"privatecompetitor" not in rename_calls[0]["body"]
    assert h.service.store.get_submission(original)["code_review_status"] != "passed"

    retime_disposable_round(connect, h, round_id, datetime.now(timezone.utc) - timedelta(seconds=2))
    h.clock.now = datetime.now(timezone.utc)
    assert h.service.advance_round(round_id)["status"] == "ok"
    participants = {item["submission_id"] for item in h.service.store.get_round(round_id)["participants"]}
    assert ids["exact"] not in participants
    assert ids["comments"] not in participants
    assert ids["rename"] not in participants
    assert ids["noop_doc"] not in participants
    assert ids["prompt_punctuation"] not in participants
    assert {ids["base"], ids["algorithm"], ids["threshold"], ids["prompt_words"], replacement} <= participants
    assert all(run["submission_id"] not in {ids["exact"], ids["comments"], ids["rename"], ids["noop_doc"], ids["prompt_punctuation"]}
               for run in h.service.store.list_runs(round_id))

    # The same frozen source is checked again at delivery, under a valid
    # execute lease. A same-size object mutation must not reach a runner.
    token = "a" * 64
    run = {
        "run_id": "fixture-source-integrity", "round_id": round_id,
        "submission_id": ids["base"], "kind": "execute", "status": "leased",
        "lease_token_hash": hash_lease_token(token),
        "lease_expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    monkeypatch.setattr(h.service.store, "get_run", lambda _run_id: run)
    row = h.service.store.get_submission(ids["base"])
    original_source = h.objects.get(row["source_ref"])
    assert h.service.handle_source(run["run_id"], token) == original_source
    changed = bytearray(original_source)
    changed[-1] ^= 1
    h.objects._path(row["source_ref"]).write_bytes(bytes(changed))
    with pytest.raises(svc.ServiceError) as failure:
        h.service.handle_source(run["run_id"], token)
    assert (failure.value.code, failure.value.status) == ("run_source_integrity_failed", 500)
    h.service.cancel(round_id, sorted(svc.CANCEL_REASONS.values())[0])


def test_public_bootstrap_champion_copy_across_two_hotkeys(connect, tmp_path):
    h = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    h.objects = ChecksumObjects(h.objects_root)
    h.baseline_source = source("def run_icp(icp):\n    return icp.public_champion\n")
    h.service = h.build_service()
    h.service._submission_request_limiter = SimpleNamespace(
        check=lambda _hotkey: SimpleNamespace(allowed=True)
    )
    h.clock.now = datetime.now(timezone.utc)
    h.chain.epoch = 40602
    round_id = "arena-2026-12-16"
    h.service.create_round(datetime.now(timezone.utc) + timedelta(hours=12), round_id=round_id)
    payer = submission_runtime.SubmissionProviderKeys(
        store=h.service.store, credentials=h.service.config.credential_manager,
        organizer_keys={"openrouter": "organizer-only-key"},
    )
    transport = LabeledReviewTransport()
    h.service.config.code_reviewer = SubmissionCodeReviewer(
        store=h.service.store, objects=h.objects,
        credential_for=payer.code_review_key, price_table=price_table(),
        transport=transport,
        similarity_references_for=h.service.submission_similarity_references,
    )
    with TestClient(create_app(h.service)) as http:
        owner = keypair("bootstrap-copy-one")
        first = submit(http, h, round_id, owner, h.baseline_source)
        failed_replacement = submit(
            http, h, round_id, owner,
            source("def run_icp(icp):\n    return icp.invalid\n", include_license=False),
            rejection_code="submission_rejected:source_license_missing",
        )
        assert h.service.store.get_submission(failed_replacement)["status"] == "rejected"
        assert h.service.store.get_submission(first)["status"] == "accepted"
        second_attempt = http.post("/arena/v1/submissions/presign", json=signed(
            h, owner, round_id, contracts.SCOPE_SUBMISSION_PRESIGN,
            {"source_size_bytes": len(source("def run_icp(icp):\n    return icp.second_attempt\n")),
             "consent": {"public_rerun": True}},
        ))
        assert second_attempt.status_code == 409
        assert second_attempt.json()["code"] == "submission_replacement_limit_reached"
        second = submit(http, h, round_id, keypair("bootstrap-copy-two"),
                        h.baseline_source, duplicate=True)
    schedule = retime_disposable_round(connect, h, round_id, datetime.now(timezone.utc) + timedelta(seconds=30))
    h.clock.now = datetime.fromisoformat(schedule["submission_cutoff"].replace("Z", "+00:00")) - timedelta(hours=1)
    assert h.service.config.code_reviewer.review(h.service.store.get_submission(first))["status"] == "rejected"
    assert transport.calls == []
    assert h.service.store.get_submission(first)["code_review_doc"]["cost_microusd"] == 0
    assert h.service.store.get_submission(first)["code_review_doc"]["categories"] == ["duplicate_submission"]
    assert h.service.store.get_submission(second)["rejection_rule"] == "duplicate_submission"
    assert h.service.store.get_submission(second)["code_review_attempts"] == 0
