from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from scripts import arena_sep16_native_rerun as rerun
from tests.lab_arena.icp_fixtures import daily_icps


class _Objects:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.puts = []

    def get_bounded(self, ref, limit):
        value = self.values[ref]
        if len(value) > limit:
            raise ValueError("object exceeds limit")
        return value

    def put(self, ref, value):
        self.puts.append((ref, bytes(value)))
        self.values[ref] = bytes(value)


def _benchmark_service(icps):
    from lab_arena import contracts

    document = {
        "schema_version": "leadpoet.lab_arena.benchmark.v1",
        "round_id": rerun.ROUND,
        "icps": icps,
    }
    payload = contracts.canonical_json(document).encode("utf-8")
    objects = _Objects({
        "arena/arena-2026-09-16/benchmark.json": payload,
    })
    service = SimpleNamespace(config=SimpleNamespace(object_store=objects))
    row = {"benchmark_ref": "arena/arena-2026-09-16/benchmark.json"}
    return service, row, document


class _Transport:
    def __init__(self):
        self.calls = []

    def rpc(self, name, arguments):
        self.calls.append((name, arguments))
        return {"status": "ok", "assignments": 7}


class _Store:
    def __init__(self):
        self._transport = _Transport()
        self.open_scoring = self._unexpected_open

    def _unexpected_open(self, *args, **kwargs):
        raise AssertionError("generic scorer route was not replaced")

    def get_round(self, round_id):
        return {
            "round_id": round_id,
            "status": "stage1_closed",
            "reward_basis_hash": rerun.BASIS_HASH,
        }


class _Service:
    def __init__(self):
        self.store = _Store()

    def open_scoring(self, round_id, stage):
        items = [
            {"submission_id": rerun.BASELINE, "scored_run_id": f"baseline-{index}"}
            for index in range(7)
        ] + [
            {"submission_id": "challenger", "scored_run_id": "challenger-0"}
        ]
        return self.store.open_scoring(
            round_id, stage, items,
            integrity_cache=True, company_quality_cache=False,
        )


def test_open_scoring_routes_only_baseline_items_and_restores_store_method():
    service = _Service()
    original = service.store.open_scoring

    result = rerun._open_scoring(service, 1)

    assert result == {"status": "ok", "assignments": 7}
    assert service.store.open_scoring == original
    assert service.store._transport.calls == [
        (
            "lab_arena_open_sep16_baseline_scoring_v1",
            {
                "p_round_id": rerun.ROUND,
                "p_stage": 1,
                "p_work_items": [
                    {"submission_id": rerun.BASELINE, "scored_run_id": f"baseline-{index}"}
                    for index in range(7)
                ] + [
                    {"submission_id": "challenger", "scored_run_id": "challenger-0"}
                ],
            },
        )
    ]


def test_source_proof_uses_explicit_champion_model_lab_url(monkeypatch):
    from lab_arena import source_bundle

    expected_commit = "a" * 40
    observed = []
    payload = b"sealed-native-source"

    def fetch(url, limit):
        observed.append((url, limit))
        return payload

    monkeypatch.setattr(source_bundle, "validate_source_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(source_bundle, "source_archive_commit", lambda _payload: expected_commit)
    service = SimpleNamespace(
        config=SimpleNamespace(baseline_source_fetcher=fetch)
    )

    source, digest, commit = rerun._source_proof(service, expected_commit)

    assert source == payload
    assert digest == rerun._sha256(payload)
    assert commit == expected_commit
    assert observed == [(rerun.SOURCE_URL, source_bundle.MAX_SOURCE_ARCHIVE_BYTES)]


def test_bank_proof_hashes_authoritative_icp_array(monkeypatch):
    from lab_arena import contracts

    icps = daily_icps()
    service, row, document = _benchmark_service(icps)
    array_digest = rerun._sha256(
        contracts.canonical_json(icps).encode("utf-8")
    )
    envelope_digest = rerun._sha256(
        contracts.canonical_json(document).encode("utf-8")
    )
    assert array_digest != envelope_digest
    monkeypatch.setattr(rerun, "BANK_HASH", array_digest)

    assert rerun._bank_proof(service, row) == array_digest


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document["icps"][0].update(prompt="changed"),
        lambda document: document.update(icps=list(reversed(document["icps"]))),
        lambda document: document["icps"].pop(),
        lambda document: document["icps"].__setitem__(
            -1, copy.deepcopy(document["icps"][0])
        ),
        lambda document: document.update(round_id="arena-2026-09-17"),
        lambda document: document.update(schema_version="leadpoet.lab_arena.benchmark.v2"),
        lambda document: document.update(extra="drift"),
    ],
    ids=(
        "changed-icp", "reordered-icps", "missing-icp", "duplicate-icp",
        "wrong-round", "wrong-schema", "extra-envelope-field",
    ),
)
def test_bank_proof_rejects_any_sealed_bank_or_envelope_drift(monkeypatch, mutate):
    from lab_arena import contracts

    original = daily_icps()
    sealed_digest = rerun._sha256(
        contracts.canonical_json(original).encode("utf-8")
    )
    document = {
        "schema_version": "leadpoet.lab_arena.benchmark.v1",
        "round_id": rerun.ROUND,
        "icps": copy.deepcopy(original),
    }
    mutate(document)
    objects = _Objects({
        "arena/arena-2026-09-16/benchmark.json": json.dumps(document).encode(),
    })
    service = SimpleNamespace(config=SimpleNamespace(object_store=objects))
    row = {"benchmark_ref": "arena/arena-2026-09-16/benchmark.json"}
    monkeypatch.setattr(rerun, "BANK_HASH", sealed_digest)

    with pytest.raises(rerun.ExactRerunRefused, match="benchmark"):
        rerun._bank_proof(service, row)


def test_stage_source_dry_run_does_not_write_and_live_stage_reads_back(
    monkeypatch, tmp_path,
):
    from lab_arena import contracts, source_bundle

    expected_commit = "b" * 40
    source = b"validated-champion-source"
    icps = daily_icps()
    bank_digest = rerun._sha256(
        contracts.canonical_json(icps).encode("utf-8")
    )
    service, row, _document = _benchmark_service(icps)
    row.update(
        status="published",
        round_id=rerun.ROUND,
        reward_basis_hash=rerun.BASIS_HASH,
    )
    service.store = SimpleNamespace(get_round=lambda round_id: row)
    fetches = []

    def fetch(url, limit):
        fetches.append((url, limit))
        return source

    service.config.baseline_source_fetcher = fetch
    monkeypatch.setattr(rerun, "BANK_HASH", bank_digest)
    monkeypatch.setattr(source_bundle, "validate_source_archive", lambda *_a, **_k: None)
    monkeypatch.setattr(source_bundle, "source_archive_commit", lambda payload: expected_commit)
    monkeypatch.setattr(rerun, "_schedule_proof", lambda _row, _path: {"sealed": True})
    args = SimpleNamespace(
        expected_lab_commit=expected_commit,
        forward_schedule_file=tmp_path / "schedule.json",
        dry_run=True,
    )

    dry_run = rerun._stage_source(service, args)
    assert dry_run["status"] == "source_preflight_ok"
    assert dry_run["source_commit"] == expected_commit
    assert dry_run["source_sha256"] == rerun._sha256(source)
    assert dry_run["bank_sha256"] == bank_digest
    assert service.config.object_store.puts == []

    args.dry_run = False
    staged = rerun._stage_source(service, args)
    assert staged["status"] == "source_staged"
    assert service.config.object_store.puts == [(rerun.SOURCE_REF, source)]
    assert service.config.object_store.values[rerun.SOURCE_REF] == source
    assert fetches == [
        (rerun.SOURCE_URL, source_bundle.MAX_SOURCE_ARCHIVE_BYTES),
        (rerun.SOURCE_URL, source_bundle.MAX_SOURCE_ARCHIVE_BYTES),
    ]


def test_open_scoring_rejects_non_integrity_mode():
    service = _Service()

    def wrong_mode(round_id, stage):
        return service.store.open_scoring(
            round_id, stage, [], integrity_cache=False,
            company_quality_cache=False,
        )

    service.open_scoring = wrong_mode
    with pytest.raises(rerun.ExactRerunRefused, match="integrity mode"):
        rerun._open_scoring(service, 1)


def _recovery_service(payload: bytes):
    source_hash = rerun._sha256(payload)

    class RecoveryStore:
        def __init__(self):
            self._transport = _Transport()

        def get_round(self, round_id):
            return {
                "round_id": round_id,
                "status": "cancelled",
                "reward_basis_hash": rerun.BASIS_HASH,
            }

        def get_submission(self, submission_id):
            assert submission_id == rerun.BASELINE
            return {
                "source_ref": rerun.SOURCE_REF,
                "source_size_bytes": len(payload),
                "submission_doc": {
                    "source_ref": rerun.SOURCE_REF,
                    "source_sha256": source_hash,
                    "source_commit": "3" * 40,
                },
            }

    objects = _Objects({rerun.SOURCE_REF: payload})
    return SimpleNamespace(
        store=RecoveryStore(),
        config=SimpleNamespace(object_store=objects),
    )


def test_recover_proves_staged_source_and_calls_exact_rpc(monkeypatch, tmp_path):
    from lab_arena import source_bundle

    payload = b"latest-tested-champion-source"
    service = _recovery_service(payload)
    schedule = {"sealed": "forward"}
    monkeypatch.setattr(rerun, "_bank_proof", lambda *_args: rerun.BANK_HASH)
    monkeypatch.setattr(rerun, "_schedule_proof", lambda *_args: schedule)
    monkeypatch.setattr(source_bundle, "validate_source_archive", lambda *_a, **_k: None)
    monkeypatch.setattr(source_bundle, "source_archive_commit", lambda _payload: "3" * 40)
    args = SimpleNamespace(
        forward_schedule_file=tmp_path / "schedule.json", dry_run=True
    )

    preflight = rerun._recover(service, args)

    assert preflight == {
        "status": "recovery_preflight_ok",
        "round_id": rerun.ROUND,
        "bank_sha256": rerun.BANK_HASH,
        "source_sha256": rerun._sha256(payload),
        "source_commit": "3" * 40,
        "source_size_bytes": len(payload),
        "execute_namespace": "rerun269",
        "verified_parallel_runner_slots": rerun.VERIFIED_RUNNER_SLOTS,
    }
    assert service.store._transport.calls == []

    service.store._transport.rpc = lambda name, arguments: (
        service.store._transport.calls.append((name, arguments))
        or {"status": "prepared", "baseline_execute_assignments": 20}
    )
    args.dry_run = False
    assert rerun._recover(service, args)["status"] == "prepared"
    assert service.store._transport.calls == [
        (
            "lab_arena_prepare_sep16_baseline_recovery_v1",
            {"p_forward_schedule": schedule},
        )
    ]


@pytest.mark.parametrize("drift", ["ref", "size", "hash", "commit"])
def test_recover_refuses_unsealed_source(drift):
    payload = b"latest-tested-champion-source"
    service = _recovery_service(payload)
    submission = service.store.get_submission(rerun.BASELINE)
    if drift == "ref":
        submission["source_ref"] += ".changed"
    elif drift == "size":
        submission["source_size_bytes"] += 1
    elif drift == "hash":
        submission["submission_doc"]["source_sha256"] = "0" * 64
    else:
        submission["submission_doc"]["source_commit"] = "short"
    service.store.get_submission = lambda _submission_id: submission

    with pytest.raises(rerun.ExactRerunRefused, match="source"):
        rerun._recovery_source_proof(service)


def test_recover_refuses_an_active_round(monkeypatch, tmp_path):
    service = _recovery_service(b"latest-tested-champion-source")
    row = service.store.get_round(rerun.ROUND)
    row["status"] = "stage1"
    service.store.get_round = lambda _round_id: row
    monkeypatch.setattr(
        rerun, "_bank_proof", lambda *_args: pytest.fail("bank proof must not run")
    )

    with pytest.raises(rerun.ExactRerunRefused, match="status"):
        rerun._recover(
            service,
            SimpleNamespace(
                forward_schedule_file=tmp_path / "schedule.json", dry_run=True
            ),
        )


def test_recovery_source_proof_revalidates_archive_and_commit(monkeypatch):
    from lab_arena import source_bundle

    payload = b"latest-tested-champion-source"
    service = _recovery_service(payload)
    validations = []
    monkeypatch.setattr(
        source_bundle,
        "validate_source_archive",
        lambda observed, *, require_license: validations.append(
            (observed, require_license)
        ),
    )
    monkeypatch.setattr(
        source_bundle, "source_archive_commit", lambda _payload: "4" * 40
    )

    with pytest.raises(rerun.ExactRerunRefused, match="commit differs"):
        rerun._recovery_source_proof(service)
    assert validations == [(payload, True)]


@pytest.mark.parametrize("namespace", ["rerun265", "rerun269"])
def test_audit_counts_scores_for_the_active_rerun_namespace(namespace):
    runs = [
        {
            "assignment_id": f"assignment-{position}:{namespace}",
            "submission_id": rerun.BASELINE,
            "kind": "execute",
        }
        for position in range(2)
    ] + [
        {
            "assignment_id": f"assignment-{position}:score:{namespace}",
            "submission_id": rerun.BASELINE,
            "kind": "score",
        }
        for position in range(2)
    ] + [
        {
            "assignment_id": "unrelated:score:" + (
                "rerun269" if namespace == "rerun265" else "rerun265"
            ),
            "submission_id": rerun.BASELINE,
            "kind": "score",
        }
    ]

    class AuditStore:
        def get_round(self, round_id):
            assert round_id == rerun.ROUND
            return {
                "round_id": round_id,
                "status": "cancelled",
                "reward_basis_hash": rerun.BASIS_HASH,
            }

        def get_submission(self, submission_id):
            assert submission_id == rerun.BASELINE
            return {"source_ref": rerun.SOURCE_REF}

        def list_runs(self, round_id):
            return runs if round_id == rerun.ROUND else []

    result = rerun._audit(SimpleNamespace(store=AuditStore()))

    assert result["active_rerun_namespace"] == namespace
    assert result["rerun_execute_assignments"] == 2
    assert result["rerun_score_assignments"] == 2
