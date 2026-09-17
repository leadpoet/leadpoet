from __future__ import annotations

import copy
import gzip
import io
import json
import tarfile
import stat
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts, source_bundle
from lab_arena.service import S3ObjectStore
from scripts import arena_sep17_baseline_recovery285 as r
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration


PRIVATE = Path("/private/tmp/leadpoet-native-rebaseline-recovery-20260916/private")
REVIEWED_LAB_ARCHIVE = PRIVATE / "model-7ccf69c-lab-source.tar.gz"
REVIEWED_LAB_FACTS = {
    "source_size_bytes": 585058,
    "source_sha256": "7085faa34ddae9994e9f281d5529413611f7e9664457cae6ad5743598982639c",
    "source_commit": "7ccf69c3a0cd3eab4339f57272650c3353a71b9a",
}


class S3Error(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class Client:
    def __init__(self, data):
        self.data = data
        self.writes = []

    def head_object(self, *, Key, **_):
        if Key not in self.data:
            raise S3Error("NoSuchKey")
        return {"ContentLength": len(self.data[Key])}

    def get_object(self, *, Key, **_):
        return {"Body": io.BytesIO(self.data[Key])}

    def put_object(self, *, Key, Body, IfNoneMatch, **_):
        assert IfNoneMatch == "*"
        self.writes.append(Key)
        if Key in self.data:
            raise S3Error("PreconditionFailed")
        self.data[Key] = bytes(Body)


def archive(commit):
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT,
                          pax_headers={"comment": commit}) as tar:
            for name, data in (
                ("harness.py", b"def run_icp(icp): return []\n"),
                ("LICENSE", Path(__file__).parents[1].joinpath("LICENSE").read_bytes()),
            ):
                member = tarfile.TarInfo("leadpoet-sales-agent-lab/" + name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
    return raw.getvalue()


@pytest.fixture
def setup(monkeypatch):
    old = archive(r.TERMINAL_SOURCE_COMMIT)
    new = archive("a" * 40)
    bank = {"schema_version": "leadpoet.lab_arena.benchmark.v1",
            "round_id": r.ROUND, "icps": daily_icps()}
    for name, value in {
        "TERMINAL_SOURCE_SIZE": len(old),
        "TERMINAL_SOURCE_SHA256": r._digest(old),
        "BANK_SHA256": r._digest(contracts.canonical_json(bank["icps"]).encode()),
    }.items():
        monkeypatch.setattr(r, name, value)
    config = base_round_configuration()
    config.update(round_id=r.ROUND, schedule={
        **r.FORWARD_SCHEDULE,
        "benchmark_deadline": "2026-09-17T14:00:00Z",
        "stage_1_start": "2026-09-17T14:00:01Z",
    }, runner_slot_ceiling=20)
    config["call_quotas"] = {"openrouter": 200, "deepline": 30, "scrapingdog": 30}
    row = {"round_id": r.ROUND, "evaluation_date": "2026-09-17",
           "icp_set_date": "2026-09-16", "benchmark_ref": r.BANK_REF,
           "status": "cancelled", "configuration_doc": config}
    baseline = {"round_id": r.ROUND, "source_ref": r.TERMINAL_SOURCE_REF}
    client = Client({r.BANK_REF: contracts.canonical_json(bank).encode(),
                     r.TERMINAL_SOURCE_REF: old})
    calls, fetches = [], []

    def rpc(name, args):
        calls.append((name, copy.deepcopy(args)))
        return {"status": "prepared", "round_id": r.ROUND,
                "baseline_execute_assignments": 20,
                "openrouter_calls_per_icp": 200,
                "execute_namespace": "rerun285", **r._archive_facts(new)}

    def fetch(url, limit):
        fetches.append((url, limit))
        return new

    service = SimpleNamespace(
        store=SimpleNamespace(get_round=lambda _: row, get_submission=lambda _: baseline,
                              _transport=SimpleNamespace(rpc=rpc)),
        config=SimpleNamespace(object_store=S3ObjectStore("test", client=client),
                               baseline_source_fetcher=fetch),
    )
    return SimpleNamespace(service=service, row=row, baseline=baseline, client=client,
                           calls=calls, fetches=fetches, new=new)


def invoke(setup, dry_run=False, hour=15):
    facts = r._archive_facts(setup.new)
    return r.prepare(setup.service, dry_run=dry_run,
                     expected_source_size_bytes=facts["source_size_bytes"],
                     expected_source_sha256=facts["source_sha256"],
                     expected_source_commit=facts["source_commit"],
                     now=datetime(2026, 9, 17, hour, tzinfo=timezone.utc))


def test_dynamic_lab_archive_is_validated_staged_once_and_passed_to_rpc(setup):
    result = invoke(setup)
    facts = r._archive_facts(setup.new)
    assert result == {**result, **facts, "execute_namespace": "rerun285"}
    assert setup.fetches == [(r.SOURCE_URL, source_bundle.MAX_SOURCE_ARCHIVE_BYTES)]
    assert setup.client.data[r.SOURCE_REF] == setup.new
    assert setup.calls == [("lab_arena_prepare_sep17_baseline_recovery285_v1", {
        "p_source_size_bytes": facts["source_size_bytes"],
        "p_source_sha256": facts["source_sha256"],
        "p_source_commit": facts["source_commit"],
        "p_bank_sha256": r.BANK_SHA256,
        "p_forward_schedule": r.FORWARD_SCHEDULE,
    })]


def test_dry_run_fetches_and_validates_but_does_not_write(setup):
    result = invoke(setup, dry_run=True)
    assert result["status"] == "recovery285_preflight_ok"
    assert not setup.calls and not setup.client.writes


def test_replay_reads_frozen_recovery285_object_without_fetch_or_rewrite(setup):
    setup.baseline["source_ref"] = r.SOURCE_REF
    setup.row["status"] = "stage2_scoring"
    setup.client.data[r.SOURCE_REF] = setup.new
    result = invoke(setup, hour=23)
    assert result["replay"] is True
    assert not setup.fetches and not setup.client.writes
    assert len(setup.calls) == 1


@pytest.mark.parametrize("defect", [
    "day", "bank", "terminal_source", "latest_source", "intake", "quota", "status",
])
def test_drift_fails_before_write_or_rpc(setup, defect):
    if defect == "day":
        setup.row["evaluation_date"] = "2026-09-16"
    elif defect == "bank":
        setup.client.data[r.BANK_REF] = b"{}"
    elif defect == "terminal_source":
        setup.client.data[r.TERMINAL_SOURCE_REF] = b"wrong"
    elif defect == "latest_source":
        setup.service.config.baseline_source_fetcher = lambda *_: b"wrong"
    elif defect == "intake":
        setup.row["configuration_doc"]["schedule"]["submission_cutoff"] = "2026-09-16T00:00:00Z"
    elif defect == "quota":
        setup.row["configuration_doc"]["call_quotas"]["deepline"] = 29
    else:
        setup.row["status"] = "stage1"
    with pytest.raises((
        r.RecoveryRefused, contracts.ArenaContractError,
        source_bundle.SourceBundleError,
    )):
        invoke(setup)
    assert not setup.calls and not setup.client.writes


def test_first_prepare_expires_but_replay_remains_available(setup):
    with pytest.raises(r.RecoveryRefused, match="window"):
        invoke(setup, hour=19)
    setup.baseline["source_ref"] = r.SOURCE_REF
    setup.row["status"] = "stage1"
    setup.client.data[r.SOURCE_REF] = setup.new
    assert invoke(setup, hour=23)["status"] == "prepared"


def test_changed_lab_payload_is_refused_before_write_or_rpc(setup):
    reviewed = r._archive_facts(setup.new)
    changed = archive("c" * 40)
    setup.service.config.baseline_source_fetcher = lambda *_: changed
    with pytest.raises(r.RecoveryRefused, match="reviewed identity"):
        r.prepare(
            setup.service,
            dry_run=False,
            expected_source_size_bytes=reviewed["source_size_bytes"],
            expected_source_sha256=reviewed["source_sha256"],
            expected_source_commit=reviewed["source_commit"],
            now=datetime(2026, 9, 17, 15, tzinfo=timezone.utc),
        )
    assert not setup.client.writes and not setup.calls


def test_source_url_and_schedule_are_exact():
    assert r.SOURCE_URL == (
        "https://github.com/leadpoet/leadpoet-sales-agent/"
        "archive/refs/heads/lab.tar.gz"
    )
    assert r.FORWARD_SCHEDULE == {
        "submission_open": "2026-09-16T00:00:00Z",
        "submission_cutoff": "2026-09-17T00:00:00Z",
        "benchmark_deadline": "2026-09-17T18:30:00Z",
        "stage_1_start": "2026-09-17T18:30:01Z",
        "stage_1_close": "2026-09-17T22:00:00Z",
        "stage_1_scoring_close": "2026-09-18T00:00:00Z",
        "stage_2_start": "2026-09-18T00:00:01Z",
        "stage_2_close": "2026-09-18T00:20:00Z",
        "final_scoring_close": "2026-09-18T02:20:00Z",
        "publication_deadline": "2026-09-18T02:50:00Z",
    }


@pytest.mark.skipif(
    not REVIEWED_LAB_ARCHIVE.is_file(),
    reason="Recovery285 release gate requires the protected reviewed lab archive.",
)
def test_reviewed_lab_archive_matches_operator_facts():
    assert stat.S_IMODE(REVIEWED_LAB_ARCHIVE.stat().st_mode) == 0o600
    assert r._archive_facts(REVIEWED_LAB_ARCHIVE.read_bytes()) == REVIEWED_LAB_FACTS
