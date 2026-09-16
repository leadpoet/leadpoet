from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import arena_sep16_native_rerun as rerun


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
