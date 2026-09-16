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
