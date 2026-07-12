from __future__ import annotations

import pytest

from gateway.tee.provider_outcome_v2 import (
    ProviderOutcomeLedgerV2,
    ProviderOutcomeV2Error,
    validate_provider_outcome_snapshot_v2,
)


class _Clock:
    def __init__(self, value: str) -> None:
        self.value = value

    def __call__(self) -> str:
        return self.value


def test_measured_outcome_snapshot_preserves_counts_costs_and_digest():
    clock = _Clock("2026-07-10T12:00:00Z")
    ledger = ProviderOutcomeLedgerV2(clock=clock)
    ledger.record(
        provider_id="exa",
        endpoint_class="/search",
        evidence="recorded",
        status=200,
        live_call=True,
        cost_event={
            "billable": True,
            "cost_usd": 0.005,
            "cost_source": "exa_cost_dollars",
        },
    )
    ledger.record(
        provider_id="exa",
        endpoint_class="/search",
        evidence="hit",
        status=200,
        live_call=False,
        cost_event={
            "billable": False,
            "cost_usd": 0,
            "cost_source": "cache_hit_zero_cost",
        },
    )

    snapshot = validate_provider_outcome_snapshot_v2(ledger.snapshot())
    digest = snapshot["provider_outcome_digest"]
    exa = digest["providers"]["exa"]
    assert exa["call_count"] == 2
    assert exa["live_call_count"] == 1
    assert exa["cache_hit_count"] == 1
    assert exa["measured_spend_microusd"] == 5000
    assert digest["day_cache_outcomes"] == {"hit": 1, "live": 1}
    assert snapshot["provider_outcome_digest_hash"] == digest["digest_hash"]
    assert snapshot["source_state_hash"] == digest["sidecar_document_hash"]


def test_transport_failure_is_visible_and_not_billable():
    ledger = ProviderOutcomeLedgerV2(
        clock=lambda: "2026-07-10T12:00:00Z"
    )
    ledger.record(
        provider_id="or",
        endpoint_class="/api/v1/chat/completions",
        evidence="error",
        status=502,
        live_call=True,
        cost_event={
            "billable": True,
            "cost_usd": 9,
            "cost_source": "openrouter_response_usage",
        },
    )
    digest = ledger.snapshot()["provider_outcome_digest"]
    openrouter = digest["providers"]["or"]
    assert openrouter["call_count"] == 1
    assert openrouter["error_count"] == 1
    assert openrouter["status_histogram"] == {"502": 1}
    assert openrouter["measured_spend_microusd"] == 0


def test_snapshot_rolls_over_at_utc_day_without_carrying_old_counts():
    clock = _Clock("2026-07-10T23:59:59Z")
    ledger = ProviderOutcomeLedgerV2(clock=clock)
    ledger.record(
        provider_id="exa",
        endpoint_class="/search",
        evidence="recorded",
        status=200,
        live_call=True,
        cost_event={},
    )
    clock.value = "2026-07-11T00:00:00Z"
    snapshot = validate_provider_outcome_snapshot_v2(ledger.snapshot())
    digest = snapshot["provider_outcome_digest"]
    assert digest["utc_day"] == "2026-07-11"
    assert digest["sidecar_sequence"] == 0
    assert digest["providers"] == {}


def test_snapshot_validation_rejects_digest_or_source_state_tampering():
    snapshot = ProviderOutcomeLedgerV2(
        clock=lambda: "2026-07-10T12:00:00Z"
    ).snapshot()
    snapshot["provider_outcome_digest"] = dict(
        snapshot["provider_outcome_digest"]
    )
    snapshot["provider_outcome_digest"]["sidecar_sequence"] = 1
    with pytest.raises(ProviderOutcomeV2Error, match="commitments differ"):
        validate_provider_outcome_snapshot_v2(snapshot)


def test_invalid_clock_and_negative_cost_fail_closed():
    with pytest.raises(ProviderOutcomeV2Error, match="clock"):
        ProviderOutcomeLedgerV2(clock=lambda: "not-a-clock")

    ledger = ProviderOutcomeLedgerV2(
        clock=lambda: "2026-07-10T12:00:00Z"
    )
    with pytest.raises(ProviderOutcomeV2Error, match="negative"):
        ledger.record(
            provider_id="exa",
            endpoint_class="/search",
            evidence="recorded",
            status=200,
            live_call=True,
            cost_event={
                "billable": True,
                "cost_usd": -1,
                "cost_source": "exa_cost_dollars",
            },
        )
