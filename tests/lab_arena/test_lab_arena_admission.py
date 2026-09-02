"""Submission admission on disposable PostgreSQL (labarena.md 6.3, step 6 exit gate)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_arena import admission, build, contracts
from tests.lab_arena.lab_arena_pg_harness import LAB_ARENA_MIGRATION, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_service_round import Harness, keypair, package_bytes


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration((LAB_ARENA_MIGRATION,))


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database
    return lambda: psycopg2.connect(**dsn)


def upload(harness, flavor: str, round_id: str, *, archive=None) -> str:
    miner = keypair("adm-miner-" + flavor)
    archive = archive if archive is not None else package_bytes(flavor)
    envelope = contracts.build_signed_request(scope=contracts.SCOPE_SUBMISSION, round_id=round_id, hotkey=miner.ss58_address, body={"package_hash": contracts.hash_bytes(archive), "consent": {"source_publication": True, "public_rerun": True}}, timestamp=int(harness.clock().timestamp()), sign_message=lambda m: miner.sign(m.encode()).hex())
    result = harness.service.handle_submission(envelope, archive)
    return result["submission_id"], result["status"]


def company(icp, index):
    return {"company_name": "%s Co %d" % (icp["industry"], index), "company_website": "https://%s-%d.example.com" % (icp["industry"].lower(), index), "industry": icp["industry"], "employee_count": "51-200", "country": "United States", "intent_signals": [{"source": "news", "description": "Funding", "url": "https://news.example.com/%d" % index, "date": "2026-08-01", "snippet": "raised", "matched_icp_signal": 0}]}


def test_admission_builds_screens_and_freezes_or_rejects_for_published_rules(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    service = harness.service
    for row in service.store.list_rounds():
        if row["status"] not in ("published", "cancelled"):
            service.store.cancel_round(row["round_id"], "operator")
    configuration = service.create_round(datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc))
    round_id = configuration["round_id"]
    good_id, status = upload(harness, "Good", round_id)
    hardcoded_id, _ = upload(harness, "Hardcoded", round_id)
    silent_id, _ = upload(harness, "Silent", round_id)
    duplicate_id, dup_status = upload(harness, "Duplicate", round_id, archive=package_bytes("Good"))
    assert status == "uploaded" and dup_status == "uploaded"
    builds = []

    def image_builder(inspection, submission_id):
        builds.append((submission_id, inspection.source_tree_hash))
        return "sha256:" + hashlib.sha256(inspection.source_tree_hash.encode()).hexdigest()

    def run_model(image_digest, icp, providers_enabled):
        flavor = {"sha256:" + hashlib.sha256(build.inspect_package(package_bytes(name)).source_tree_hash.encode()).hexdigest(): name for name in ("Good", "Hardcoded", "Silent")}.get(image_digest, "Good")
        if flavor == "Hardcoded":
            return [company(icp, 1)]  # answers without provider access: a hardcoded table
        if flavor == "Silent":
            return []  # never produces a valid company
        return [company(icp, index) for index in range(3)] if providers_enabled else []

    outcomes = admission.admit_uploaded_submissions(service, round_id=round_id, image_builder=image_builder, run_model=run_model)
    by_id = {outcome.submission_id: outcome for outcome in outcomes}
    assert by_id[good_id].status == "accepted" and by_id[good_id].image_digest.startswith("sha256:")
    assert by_id[hardcoded_id].status == "rejected" and by_id[hardcoded_id].rule == "screening.companies_without_providers"
    assert by_id[silent_id].status == "rejected" and by_id[silent_id].rule is not None
    assert by_id[duplicate_id].status == "rejected" and by_id[duplicate_id].rule == "package.duplicate_artifact"
    assert len(builds) == 4  # every package is built once; the builder never sees a rejected package twice
    stored = service.store.get_submission(good_id)
    assert stored["status"] == "accepted" and stored["screening_result"]["accepted"] is True
    rejected = service.store.get_submission(hardcoded_id)
    assert rejected["status"] == "rejected" and rejected["rejection_rule"] == "screening.companies_without_providers"
    # Replaying admission is idempotent: nothing is rebuilt or re-screened.
    again = admission.admit_uploaded_submissions(service, round_id=round_id, image_builder=image_builder, run_model=run_model)
    assert again == [] and len(builds) == 4
    replay = admission.admit_submission(service, round_id=round_id, submission_id=good_id, image_builder=image_builder, run_model=run_model)
    assert replay.status == "accepted" and len(builds) == 4


def test_docker_image_builder_uses_an_offline_credential_free_build(tmp_path):
    archive = package_bytes("Offline")
    inspection = build.inspect_package(archive)
    commands = []

    class Completed:
        def __init__(self, argv):
            self.args = argv
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""
            if argv[1] == "image" and argv[2] == "inspect":
                self.stdout = json.dumps([{"Id": "sha256:" + "d" * 64, "RepoDigests": ["arena/model@sha256:" + "e" * 64]}])

    def docker_runner(argv, timeout=None, **kwargs):
        commands.append(list(argv))
        if argv[1] == "build":
            iid = [arg for arg in argv if arg.startswith("--iidfile=")]
            path = iid[0].split("=", 1)[1] if iid else argv[argv.index("--iidfile") + 1]
            Path(path).write_text("sha256:" + "d" * 64)
        return Completed(argv)

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    builder = admission.docker_image_builder(base_image="lab-arena-base", base_image_digest="sha256:" + "b" * 64, wheelhouse_dir=wheelhouse, docker_runner=docker_runner, work_dir=tmp_path / "builds")
    digest = builder(inspection, "sub-offline")
    assert admission.IMAGE_DIGEST_RE.match(digest) and digest.endswith("sha256:" + "e" * 64)
    build_argv = [argv for argv in commands if argv[1] == "build"][0]
    assert "--network=none" in build_argv or "--network" in build_argv
    assert not any(arg.startswith("--build-arg") or arg.startswith("--secret") for arg in build_argv)
    dockerfile = (tmp_path / "builds" / "sub-offline" / "Dockerfile").read_text()
    assert "RUN pip install --no-index" in dockerfile and "python3 /model/main.py" not in dockerfile.split("ENTRYPOINT")[0]
