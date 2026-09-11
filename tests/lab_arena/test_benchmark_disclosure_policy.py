"""Focused checks for the frozen delayed benchmark disclosure policy."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lab_arena import contracts, icp_disclosure, integrity, scoring, source_disclosure
from lab_arena.service import ArenaService, RoundDefaults, ServiceConfig, ServiceError


CUTOFF = datetime(2026, 9, 13, tzinfo=timezone.utc)
PUBLIC_AT = CUTOFF + timedelta(hours=24)


def _row(*, status="published", policy=icp_disclosure.DELAYED_DISCLOSURE_POLICY):
    configuration = {
        "schedule": {
            "submission_open": "2026-09-12T00:00:00Z",
            "submission_cutoff": "2026-09-13T00:00:00Z",
        }
    }
    if policy is not ...:
        configuration["benchmark_disclosure_policy"] = policy
    return {
        "round_id": "arena-2026-09-13",
        "status": status,
        "configuration_doc": configuration,
        "benchmark_ref": "arena/arena-2026-09-13/benchmark.json",
        "icp_set_date": "2026-09-12",
        "evaluation_date": "2026-09-13",
        "participants": [],
    }


def test_legacy_round_keeps_its_existing_cutoff_disclosure_behavior():
    row = _row(status="committed", policy=...)

    metadata = icp_disclosure.disclosure_metadata(row)

    assert metadata["public_at"] == "2026-09-13T00:00:00Z"
    assert metadata["disclosure_policy"] == icp_disclosure.DISCLOSURE_POLICY
    assert icp_disclosure.baseline_disclosure(row, [], CUTOFF) is not None


@pytest.mark.parametrize("status", ["open", "committed", "scored", "confirmed"])
def test_delayed_round_stays_private_after_time_boundary_until_terminal(status):
    assert icp_disclosure.baseline_disclosure(
        _row(status=status), [], PUBLIC_AT + timedelta(hours=12)
    ) is None


@pytest.mark.parametrize("status", ["published", "cancelled"])
def test_delayed_round_releases_at_exact_boundary_only_when_terminal(status):
    row = _row(status=status)

    assert icp_disclosure.baseline_disclosure(
        row, [], PUBLIC_AT - timedelta(microseconds=1)
    ) is None
    disclosure = icp_disclosure.baseline_disclosure(row, [], PUBLIC_AT)

    assert disclosure is not None
    assert disclosure["public_at"] == "2026-09-14T00:00:00Z"
    assert disclosure["disclosure_policy"] == "after_scoring_day2_v1"
    assert disclosure["public_positions"] == list(
        range(contracts.BENCHMARK_ICP_COUNT)
    )


def test_delayed_cancelled_round_without_committed_bank_stays_private():
    row = _row(status="cancelled")
    row["benchmark_ref"] = None

    assert icp_disclosure.baseline_disclosure(row, [], PUBLIC_AT) is None


def test_delayed_cancelled_round_reveals_only_its_committed_benchmark():
    row = _row(status="cancelled")
    row["configuration_doc"].update(
        {
            "integrity_policy": integrity.POLICY,
            "scorer_policy": {
                "scoring_adapter_version": integrity.SCORING_ADAPTER
            },
        }
    )
    row["confirmation_bank_hash"] = "sha256:" + "2" * 64
    service = object.__new__(ArenaService)
    service._round = lambda _round_id: row
    service._store = SimpleNamespace(list_runs=lambda *_args, **_kwargs: [])
    service._clock = lambda: PUBLIC_AT
    service.benchmark_icps = lambda _round_id: [
        {"icp_id": "icp-%02d" % position}
        for position in range(contracts.BENCHMARK_ICP_COUNT)
    ]
    service.confirmation_bank = lambda _round_id: {
        "icps": [
            {"icp_id": "confirmation-%02d" % position}
            for position in range(contracts.CONFIRMATION_ICP_COUNT)
        ]
    }

    benchmark = service.public_benchmark(row["round_id"])

    assert len(benchmark["icps"]) == contracts.BENCHMARK_ICP_COUNT
    assert benchmark["public_at"] == "2026-09-14T00:00:00Z"
    assert benchmark["disclosure_policy"] == "after_scoring_day2_v1"
    assert len(benchmark["confirmation_bank"]["icps"]) == (
        contracts.CONFIRMATION_ICP_COUNT
    )
    assert benchmark["private_icp_count"] == 0


def test_delayed_nonopen_round_without_explicit_bank_date_fails_closed():
    row = _row(status="published")
    row["icp_set_date"] = None

    assert icp_disclosure.disclosure_metadata(row) is None
    assert icp_disclosure.baseline_disclosure(row, [], PUBLIC_AT) is None


@pytest.mark.parametrize("policy", [None, "future_policy"])
def test_unknown_or_null_persisted_policy_fails_closed(policy):
    row = _row(policy=policy)

    with pytest.raises(
        icp_disclosure.IcpDisclosureError,
        match="benchmark_disclosure_policy_invalid",
    ):
        icp_disclosure.disclosure_metadata(row)


@pytest.mark.parametrize("policy", [None, "future_policy"])
def test_public_round_reader_translates_invalid_policy_to_service_failure(policy):
    row = _row(policy=policy)
    row["configuration_doc"].update(
        {
            "mode": "shadow",
            "network_name": "test",
            "netuid": 401,
            "scorer_policy": scoring.build_scorer_policy(),
        }
    )
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(mode="shadow", network_name="test", netuid=401)

    with pytest.raises(ServiceError) as caught:
        service._require_round_mode(row)

    assert caught.value.status == 503
    assert caught.value.code == "benchmark_disclosure_policy_invalid"


@pytest.mark.parametrize(
    "value",
    [
        "2026-09-13T00:00:00",
        "not-a-time",
        "",
        "2026-09-13T00:00:00Z\nINJECTED=value",
        "2026-09-13T00:00:00Z\x00",
    ],
)
def test_activation_rejects_naive_invalid_or_control_character_timestamps(value):
    defaults = RoundDefaults(benchmark_disclosure_from=value)

    with pytest.raises(ServiceError, match="benchmark_disclosure_activation_invalid"):
        ServiceConfig(
            mode="shadow",
            store=object(),
            object_store=object(),
            signer=None,
            chain=object(),
            verify_signature=lambda *_args: False,
            daily_icp_source=lambda **_kwargs: {},
            banned_hotkeys_source=lambda: (),
            broker_factory=lambda *_args: None,
            defaults=defaults,
        )


def test_activation_normalizes_an_aware_offset_timestamp_to_utc():
    assert icp_disclosure.parse_activation(
        "2026-09-13T01:00:00+01:00"
    ) == datetime(2026, 9, 13, tzinfo=timezone.utc)


def test_source_release_keeps_the_completed_evaluation_cutoff():
    row = _row()
    row.update(
        {
            "published_at": "2026-09-13T12:00:00Z",
            "publication_doc": {
                "participants": [{"submission_id": "submission-1"}]
            },
        }
    )
    submission = {
        "submission_id": "submission-1",
        "round_id": row["round_id"],
        "status": "frozen",
        "consent": {"public_rerun": True},
        "source_ref": "arena/source.tar.gz",
    }

    status = source_disclosure.disclosure_status(
        submission, datetime(2026, 9, 13, 12, tzinfo=timezone.utc), round_row=row
    )

    assert status["available"] is True
    assert status["available_at"] == "2026-09-13T12:00:00Z"
    assert icp_disclosure.disclosure_metadata(row)["public_at"] == (
        "2026-09-14T00:00:00Z"
    )

    row["status"] = "cancelled"
    assert source_disclosure.disclosure_status(
        submission, PUBLIC_AT, round_row=row
    )["available"] is False


def test_confirmation_results_cannot_bypass_delayed_main_disclosure():
    row = _row()
    row["configuration_doc"].update(
        {
            "integrity_policy": integrity.POLICY,
            "scorer_policy": {
                "scoring_adapter_version": integrity.SCORING_ADAPTER
            },
        }
    )
    row["publication_doc"] = {
        "participants": [
            {
                "submission_id": "submission-1",
                "miner_hotkey": "5" + "A" * 47,
                "is_baseline": False,
            }
        ],
        "stage1_ranking": [],
        "final_ranking": [],
    }

    class Store:
        @staticmethod
        def list_runs(_round_id, **_filters):
            return [
                {
                    "run_id": "confirmation-run",
                    "submission_id": "submission-1",
                    "stage": 3,
                    "kind": "execute",
                    "icp_position": contracts.stage_positions(3)[0],
                    "per_icp_score": 99.0,
                    "output_ref": "private-output.json",
                    "result_doc": {"private": "result"},
                }
            ]

    service = object.__new__(ArenaService)
    service._round = lambda _round_id: row
    service._store = Store()
    service._objects = SimpleNamespace(
        get_bounded=lambda *_args: pytest.fail("private output was read")
    )
    service._clock = lambda: PUBLIC_AT - timedelta(microseconds=1)

    result = service.public_results(row["round_id"], "submission-1")

    assert result["outputs"] == {}
    assert result["run_results"] == []
    assert result["scores"] == {
        "stage_1": [],
        "stage_2": [],
        "confirmation": [],
    }
    assert result["public_icp_status"] == "pending"
    assert result["public_icp_count"] == 0


class _RoundStore:
    def __init__(self):
        self.rows = {}

    def create_round(self, round_id, configuration):
        if round_id in self.rows:
            return {"status": "existing"}
        self.rows[round_id] = {
            "round_id": round_id,
            "status": "open",
            "configuration_doc": configuration,
        }
        return {"status": "created"}

    def get_round(self, round_id):
        return self.rows.get(round_id)


def test_activation_only_marks_new_qualifying_rounds_and_existing_config_wins():
    store = _RoundStore()
    defaults = RoundDefaults(
        baseline_hotkey="5" + "B" * 47,
        baseline_source_url="https://example.com/baseline.tar.gz",
        scorer_image_digest="sha256:" + "1" * 64,
        scorer_image_reference="registry.example/scorer@sha256:" + "1" * 64,
        benchmark_disclosure_from="2026-09-14T00:00:00Z",
    )
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(
        defaults=defaults,
        mode="shadow",
        network_name="test",
        netuid=401,
        pinned_round_id=None,
    )
    service._store = store
    service._scorer_policy = scoring.build_scorer_policy()
    service.runner_settings = lambda: (["5" + "C" * 47], [])

    historical = service.create_round(
        CUTOFF, round_id="arena-2026-09-13-historical"
    )
    assert "benchmark_disclosure_policy" not in historical

    service._config.defaults = replace(
        defaults, benchmark_disclosure_from="2026-09-13T00:00:00Z"
    )
    existing = service.create_round(
        CUTOFF, round_id="arena-2026-09-13-historical"
    )
    assert existing == historical
    assert "benchmark_disclosure_policy" not in existing

    future = service.create_round(
        CUTOFF + timedelta(days=1), round_id="arena-2026-09-14-new"
    )
    assert future["benchmark_disclosure_policy"] == "after_scoring_day2_v1"
