from __future__ import annotations

import copy
import gzip
import io
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts
from lab_arena.service import S3ObjectStore
from scripts import arena_sep17_baseline_recovery as r
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration


class S3Error(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class Client:
    def __init__(self, data):
        self.data = data
        self.writes = []
        self.failure = None

    def head_object(self, *, Key, **_):
        if Key not in self.data:
            raise S3Error("NoSuchKey")
        return {"ContentLength": len(self.data[Key])}

    def get_object(self, *, Key, **_):
        return {"Body": io.BytesIO(self.data[Key])}

    def put_object(self, *, Key, Body, IfNoneMatch, **_):
        assert IfNoneMatch == "*"
        self.writes.append(Key)
        if self.failure:
            raise self.failure
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
                member = tarfile.TarInfo("champion_model-lab/" + name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
    return raw.getvalue()


@pytest.fixture
def setup(monkeypatch):
    old = archive(r.TERMINAL_SOURCE_COMMIT)
    new = archive(r.SOURCE_COMMIT)
    bank = {"schema_version": "leadpoet.lab_arena.benchmark.v1",
            "round_id": r.ROUND, "icps": daily_icps()}
    for name, value in {
        "SOURCE_SIZE": len(new), "SOURCE_SHA256": r._digest(new),
        "TERMINAL_SOURCE_SIZE": len(old), "TERMINAL_SOURCE_SHA256": r._digest(old),
        "BANK_SHA256": r._digest(contracts.canonical_json(bank["icps"]).encode()),
    }.items():
        monkeypatch.setattr(r, name, value)
    config = base_round_configuration()
    config.update(round_id=r.ROUND, schedule=dict(r.FORWARD_SCHEDULE), runner_slot_ceiling=20)
    config["call_quotas"]["openrouter"] = 60
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
                "baseline_execute_assignments": 20, "openrouter_calls_per_icp": 200,
                "execute_namespace": "rerun278"}

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


def invoke(setup, dry_run=False, hour=4):
    return r.prepare(setup.service, dry_run=dry_run,
                     now=datetime(2026, 9, 17, hour, tzinfo=timezone.utc))


def test_preflight_is_read_only_and_preserves_frozen_config(setup):
    before = copy.deepcopy(setup.row)
    result = invoke(setup, dry_run=True)
    assert result["icp_set_date"] == "2026-09-16"
    assert result["evaluation_date"] == "2026-09-17"
    assert setup.fetches[0][0] == r.SOURCE_URL
    assert setup.row == before
    assert not setup.calls and not setup.client.writes


def test_prepare_stages_conditional_source_then_exact_rpc(setup):
    result = invoke(setup)
    assert result["status"] == "prepared"
    assert setup.client.data[r.SOURCE_REF] == setup.new
    assert setup.calls == [("lab_arena_prepare_sep17_baseline_recovery278_v1", {
        "p_source_size_bytes": r.SOURCE_SIZE, "p_source_sha256": r.SOURCE_SHA256,
        "p_source_commit": r.SOURCE_COMMIT, "p_bank_sha256": r.BANK_SHA256,
        "p_forward_schedule": r.FORWARD_SCHEDULE,
    })]
    invoke(setup)  # Response-loss retry verifies the existing object's exact bytes.
    assert setup.client.data[r.SOURCE_REF] == setup.new


@pytest.mark.parametrize("defect", ["day", "bank", "terminal_source", "latest_source", "intake", "status"])
def test_drift_fails_before_any_write(setup, defect):
    if defect == "day":
        setup.row["evaluation_date"] = "2026-09-16"
    elif defect == "bank":
        setup.client.data[r.BANK_REF] = setup.client.data[r.BANK_REF].replace(b'2026-09-17', b'2026-09-16')
    elif defect == "terminal_source":
        setup.client.data[r.TERMINAL_SOURCE_REF] = b"wrong"
    elif defect == "latest_source":
        setup.service.config.baseline_source_fetcher = lambda *_: b"wrong"
    elif defect == "intake":
        setup.row["configuration_doc"]["schedule"]["submission_cutoff"] = "2026-09-16T00:00:00Z"
    else:
        setup.row["status"] = "stage1"
    with pytest.raises(r.RecoveryRefused):
        invoke(setup)
    assert not setup.calls and not setup.client.writes


def test_existing_different_source_never_overwrites_or_prepares(setup):
    setup.client.data[r.SOURCE_REF] = b"other-source"
    with pytest.raises(contracts.ArenaContractError):
        invoke(setup)
    assert setup.client.data[r.SOURCE_REF] == b"other-source"
    assert not setup.calls


def test_access_error_never_becomes_missing_object_or_preparation(setup):
    setup.client.failure = S3Error("AccessDenied")
    with pytest.raises(S3Error):
        invoke(setup)
    assert len(setup.client.writes) == 1
    assert r.SOURCE_REF not in setup.client.data
    assert not setup.calls


def test_first_prepare_expires_but_progress_replay_uses_staged_identity(setup):
    with pytest.raises(r.RecoveryRefused, match="window"):
        invoke(setup, hour=7)
    setup.baseline["source_ref"] = r.SOURCE_REF
    setup.row["status"] = "stage2_scoring"
    setup.client.data[r.SOURCE_REF] = setup.new
    result = invoke(setup, hour=17)
    assert result["replay"] is True
    assert not setup.fetches  # A later lab push cannot change this recovery's source.
    assert len(setup.calls) == 1


def test_archive_commit_and_license_are_checked(setup):
    with pytest.raises(r.RecoveryRefused, match="commit"):
        r._verify_archive(setup.new, size=len(setup.new), digest=r._digest(setup.new),
                          commit=r.TERMINAL_SOURCE_COMMIT)
