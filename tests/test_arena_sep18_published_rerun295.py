from __future__ import annotations

import copy
import gzip
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts, source_bundle
from lab_arena.service import DEFAULT_BASELINE_SOURCE_URL, S3ObjectStore
from lab_arena.store import FUNCTION_SIGNATURES
from scripts import arena_sep18_published_rerun295 as r
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.sep18_open_quota287_postgres_test import _configuration
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration


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
        with tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT,
            pax_headers={"comment": commit},
        ) as tar:
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
    terminal = archive("7" * 40)
    new = archive("3" * 40)
    bank = {
        "schema_version": "leadpoet.lab_arena.benchmark.v1",
        "round_id": r.ROUND,
        "icps": daily_icps(),
    }
    monkeypatch.setattr(
        r, "BANK_SHA256",
        r._digest(contracts.canonical_json(bank["icps"]).encode()),
    )
    schedule = {
        "submission_open": "2026-09-17T00:00:00Z",
        "submission_cutoff": "2026-09-18T00:00:00Z",
        "benchmark_deadline": "2099-09-18T23:00:00Z",
        "stage_1_start": "2099-09-18T23:00:01Z",
        "stage_1_close": "2099-09-19T01:00:00Z",
        "stage_1_scoring_close": "2099-09-19T03:00:00Z",
        "stage_2_start": "2099-09-19T03:00:01Z",
        "stage_2_close": "2099-09-19T05:00:00Z",
        "final_scoring_close": "2099-09-19T07:00:00Z",
        "publication_deadline": "2099-09-19T07:30:00Z",
    }
    config = _configuration()
    config.update(
        round_id=r.ROUND,
        schedule={**schedule, "benchmark_deadline": "2026-09-18T18:30:00Z"},
        runner_slot_ceiling=20,
        icp_wall_clock_seconds=2700,
        lease_ttl_seconds=3600,
        parallel_twenty_icp_execution=True,
        checkpoint_deadline_policy="atomic_checkpoint_45m_v1",
    )
    config["call_quotas"] = {
        "openrouter": 200, "deepline": 30, "scrapingdog": 30,
    }
    config.update(
        sourcing_cost_eligibility_policy="successful_calls_per_icp_v1",
        execution_icp_cap_microusd=4_000_000,
        cost_per_company_microusd=800_000,
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry.example/scorer@sha256:" + "a" * 64,
        reward_constants=base_round_configuration()["reward_constants"],
    )
    monkeypatch.setattr(
        r, "CONFIG_SHA256", r._digest(contracts.canonical_json(config).encode())
    )
    terminal_facts = r._source_facts(terminal, r.TERMINAL_SOURCE_REF)
    monkeypatch.setattr(r, "TERMINAL_SOURCE_SIZE_BYTES", len(terminal))
    monkeypatch.setattr(r, "TERMINAL_SOURCE_SHA256", terminal_facts["source_sha256"])
    monkeypatch.setattr(r, "TERMINAL_SOURCE_COMMIT", terminal_facts["source_commit"])
    baseline = {
        "submission_id": r.BASELINE,
        "round_id": r.ROUND,
        "source_ref": r.TERMINAL_SOURCE_REF,
        "source_size_bytes": len(terminal),
        "submission_doc": {
            "source_ref": r.TERMINAL_SOURCE_REF,
            "source_size_bytes": len(terminal),
        },
    }
    miners = [
        {
            "submission_id": f"miner-{index}", "round_id": r.ROUND,
            "source_ref": f"arena/{r.ROUND}/sources/miner-{index}.tar.gz",
            "source_size_bytes": 100 + index,
        }
        for index in range(4)
    ]
    row = {
        "round_id": r.ROUND,
        "status": "published",
        "evaluation_date": "2026-09-18",
        "icp_set_date": "2026-09-17",
        "benchmark_ref": r.BANK_REF,
        "configuration_doc": config,
        "publication_doc": {"king_decision": {
            "outcome": "no_king", "king_submission_id": None,
            "king_hotkey": "", "winner_submission_id": None,
        }},
        "reward_activated_at": "2026-09-18T23:30:00Z",
        "reward_basis_hash": "sha256:" + "a" * 64,
        "reward_basis_doc": {"king_outcome": "defended", "king_hotkey": "incumbent"},
        "signing_key_doc": {"public_key_hash": "sha256:" + "b" * 64},
        "effective_reward_epoch": 8248000, "king_outcome": "no_king",
        "king_hotkey": None, "king_start_epoch": 8247000,
        "promotion_required": True, "promotion_doc": None,
        "baseline_promoted_at": None, "champion_submission_id": "incumbent-submission",
        "champion_hotkey": "incumbent", "champion_funding_frozen": True,
        "champion_fallback_providers": ["openrouter"],
    }
    runs = [
        {"run_id": "old-execute", "round_id": r.ROUND,
         "submission_id": r.BASELINE, "status": "accepted", "kind": "execute"},
        {"run_id": "old-score", "round_id": r.ROUND,
         "submission_id": r.BASELINE, "status": "accepted", "kind": "score"},
    ]
    ledger = [{"entry_id": 1, "round_id": r.ROUND, "submission_id": r.BASELINE}]
    client = Client({
        r.BANK_REF: contracts.canonical_json(bank).encode(),
        r.TERMINAL_SOURCE_REF: terminal,
    })
    calls, fetches = [], []

    def rpc(name, args):
        calls.append((name, copy.deepcopy(args)))
        facts = r._source_facts(new, r.SOURCE_REF)
        return {
            "status": "prepared", "round_id": r.ROUND,
            "baseline_execute_assignments": 20,
            "execute_namespace": "rerun295", "score_namespace": "score:rerun295",
            **{key: facts[key] for key in (
                "source_size_bytes", "source_sha256", "source_commit"
            )},
        }

    def fetch(url, limit):
        fetches.append((url, limit))
        return new

    costs = {
        "schema_version": "leadpoet.lab_arena.submission_costs.v1",
        "submission_id": r.BASELINE,
        "providers": [
            {"kind": kind, "provider": "openrouter", "inflight_calls": 0,
             "success_unresolved_calls": 0}
            for kind in ("execute", "score")
        ],
    }
    costs_by_submission = {r.BASELINE: costs}
    for miner in miners:
        miner_costs = copy.deepcopy(costs)
        miner_costs["submission_id"] = miner["submission_id"]
        costs_by_submission[miner["submission_id"]] = miner_costs
    tables = {
        "lab_arena_submissions": [baseline, *miners],
        "lab_arena_runs": runs,
        "lab_arena_ledger": ledger,
    }

    def select(table, *, filters, order, limit, offset):
        selected = [
            item for item in tables[table]
            if all(item.get(key) == value for key, value in filters.items())
        ]
        selected.sort(key=lambda item: item.get(order))
        return [copy.deepcopy(item) for item in selected[offset:offset + limit]]

    store = SimpleNamespace(
        get_round=lambda _: row,
        get_submission=lambda _: baseline,
        list_runs=lambda _, **kwargs: list(runs),
        list_ledger=lambda **kwargs: list(ledger),
        list_submissions=lambda _: [baseline],
        submission_costs=lambda submission_id: copy.deepcopy(
            costs_by_submission[submission_id]
        ),
        _transport=SimpleNamespace(rpc=rpc, select=select),
    )
    service = SimpleNamespace(
        store=store,
        config=SimpleNamespace(
            object_store=S3ObjectStore("test", client=client),
            baseline_source_fetcher=fetch,
        ),
    )
    preflight = r.collect_preflight(service)
    calls.clear()
    return SimpleNamespace(
        service=service, row=row, baseline=baseline, runs=runs, ledger=ledger,
        client=client, calls=calls, fetches=fetches, terminal=terminal, new=new,
        preflight=preflight, schedule=schedule, costs=costs,
        costs_by_submission=costs_by_submission,
    )


def invoke(setup, dry_run=False):
    facts = r._source_facts(setup.new, r.SOURCE_REF)
    return r.prepare(
        setup.service,
        preflight=setup.preflight,
        forward_schedule=setup.schedule,
        dry_run=dry_run,
        expected_source_size_bytes=facts["source_size_bytes"],
        expected_source_sha256=facts["source_sha256"],
        expected_source_commit=facts["source_commit"],
        now=datetime(2026, 9, 17, 19, tzinfo=timezone.utc),
    )


def test_collect_preflight_is_read_only_and_accepts_missing_optional_source_metadata(setup):
    assert setup.preflight["schema_version"] == r.PREFLIGHT_SCHEMA
    assert setup.preflight["read_only"] is True
    assert setup.preflight["production_writes"] == 0
    assert setup.preflight["baseline"]["submission_doc"] == {
        "source_ref": r.TERMINAL_SOURCE_REF,
        "source_size_bytes": len(setup.terminal),
    }
    assert not setup.client.writes


def test_collect_preflight_pages_more_than_postgrest_default_limit(setup):
    setup.ledger.extend(
        {"entry_id": entry_id, "round_id": r.ROUND, "submission_id": r.BASELINE}
        for entry_id in range(2, 1203)
    )
    assert len(r.collect_preflight(setup.service)["baseline_ledger"]) == 1202


def test_collect_preflight_fails_closed_on_missing_cost_kind(setup):
    setup.costs["providers"] = [
        item for item in setup.costs["providers"] if item["kind"] != "score"
    ]
    with pytest.raises(r.RerunRefused, match="score cost state is incomplete"):
        r.collect_preflight(setup.service)


def test_collect_preflight_fails_closed_on_cost_schema_drift(setup):
    setup.costs["schema_version"] = "leadpoet.lab_arena.submission_costs.v2"
    with pytest.raises(r.RerunRefused, match="cost state schema differs"):
        r.collect_preflight(setup.service)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_date", "2026-09-19"),
        ("icp_set_date", "2026-09-18"),
        ("benchmark_ref", "arena/arena-2026-09-18/other.json"),
    ],
)
def test_collect_preflight_requires_frozen_evaluation_identity(setup, field, value):
    setup.row[field] = value
    with pytest.raises(r.RerunRefused, match="evaluation, ICP set, or benchmark"):
        r.collect_preflight(setup.service)


def test_prepare_stages_exact_archive_and_calls_typed_rpc(setup):
    result = invoke(setup)
    facts = r._source_facts(setup.new, r.SOURCE_REF)
    assert result["execute_namespace"] == "rerun295"
    assert result["score_namespace"] == "score:rerun295"
    assert setup.fetches == [(r.SOURCE_URL, source_bundle.MAX_SOURCE_ARCHIVE_BYTES)]
    assert setup.client.data[r.SOURCE_REF] == setup.new
    prepare_calls = [call for call in setup.calls if call[0].endswith("rerun295_v1")]
    assert prepare_calls == [(
        "lab_arena_prepare_sep18_published_rerun295_v1",
        {
            "p_source_size_bytes": facts["source_size_bytes"],
            "p_source_sha256": facts["source_sha256"],
            "p_source_commit": facts["source_commit"],
            "p_bank_sha256": r.BANK_SHA256,
            "p_forward_schedule": setup.schedule,
        },
    )]


def test_dry_run_has_no_write_or_prepare_rpc(setup):
    assert invoke(setup, dry_run=True)["status"] == "rerun295_preflight_ok"
    assert not setup.client.writes
    assert not [call for call in setup.calls if call[0].endswith("rerun295_v1")]


@pytest.mark.parametrize("defect", ["round", "baseline", "run", "ledger", "claim", "cost"])
def test_sealed_or_quiescent_drift_fails_before_write(setup, defect):
    if defect == "round":
        setup.row["cancel_reason"] = "changed"
    elif defect == "baseline":
        setup.baseline["source_size_bytes"] += 1
    elif defect == "run":
        setup.runs.append({
            "run_id": "late", "round_id": r.ROUND,
            "submission_id": r.BASELINE, "status": "failed",
        })
    elif defect == "ledger":
        setup.ledger.append({
            "entry_id": 2, "round_id": r.ROUND, "submission_id": r.BASELINE,
        })
    elif defect == "claim":
        setup.runs[0]["status"] = "leased"
    else:
        setup.costs["providers"][0]["success_unresolved_calls"] = 1
    with pytest.raises(r.RerunRefused):
        invoke(setup)
    assert not setup.client.writes
    assert not [call for call in setup.calls if call[0].endswith("rerun295_v1")]


def test_rpc_signature_is_role_bound_and_exact():
    assert FUNCTION_SIGNATURES["lab_arena_prepare_sep18_published_rerun295_v1"] == (
        ("p_source_size_bytes", "bigint"), ("p_source_sha256", "text"),
        ("p_source_commit", "text"), ("p_bank_sha256", "text"),
        ("p_forward_schedule", "jsonb"),
    )


def test_source_url_is_current_default_while_terminal_metadata_stays_frozen():
    assert r.SOURCE_URL == (
        "https://github.com/leadpoet/leadpoet-sales-agent/"
        "archive/refs/heads/lab.tar.gz"
    )
    assert r.SOURCE_URL == DEFAULT_BASELINE_SOURCE_URL
    assert r.TERMINAL_SOURCE_URL == (
        "https://github.com/leadpoet/champion_model/"
        "archive/refs/heads/lab.tar.gz"
    )
    assert r.TERMINAL_SOURCE_REF.endswith("baseline-2026-09-18.tar.gz")
    assert r.SOURCE_REF.endswith("baseline-2026-09-18-rerun295.tar.gz")
    assert r.TERMINAL_SOURCE_SIZE_BYTES == 604_847
    assert r.TERMINAL_SOURCE_SHA256 == (
        "7e1bb0747014a57bc50f48f9f822d1a7c936682d63f06564e978d23f54eb7fc1"
    )
    assert r.TERMINAL_SOURCE_COMMIT == "e5341f85829ad196b4a1cb58b38a34155697c8d4"
    assert r.BANK_SHA256 == "6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91"
    assert r.CONFIG_SHA256 == "91f218af9cea06d5816876ba5e14c7791f1821569cd2caf593d820af622fdb2a"


@pytest.mark.parametrize("field", ["source_sha256", "source_commit"])
def test_optional_natural_source_identity_conflict_is_rejected(setup, field):
    setup.baseline["submission_doc"][field] = "0" * (64 if field.endswith("sha256") else 40)
    with pytest.raises(r.RerunRefused, match="optional source identity conflicts"):
        r.collect_preflight(setup.service)


@pytest.mark.parametrize("field", ["reward_basis_hash", "reward_basis_doc", "reward_activated_at"])
def test_published_round_requires_complete_activated_reward(setup, field):
    setup.row[field] = None
    with pytest.raises(r.RerunRefused, match="reward authority is incomplete"):
        r.collect_preflight(setup.service)


def test_published_round_requires_completed_crowned_promotion(setup):
    setup.row["publication_doc"]["king_decision"] = {
        "outcome": "crowned", "king_submission_id": "miner-0",
        "king_hotkey": "winner", "winner_submission_id": "miner-0",
    }
    setup.row["king_outcome"] = "crowned"
    with pytest.raises(r.RerunRefused, match="promotion authority is incomplete"):
        r.collect_preflight(setup.service)


def test_prepare_rejects_commit_not_embedded_in_archive(setup):
    facts = r._source_facts(setup.new, r.SOURCE_REF)
    with pytest.raises(r.RerunRefused, match="lab source archive differs from reviewed identity"):
        r.prepare(
            setup.service, preflight=setup.preflight,
            forward_schedule=setup.schedule, dry_run=True,
            expected_source_size_bytes=facts["source_size_bytes"],
            expected_source_sha256=facts["source_sha256"],
            expected_source_commit="4" * 40,
            now=datetime(2026, 9, 17, 19, tzinfo=timezone.utc),
        )
