from __future__ import annotations

import base64
import json
from urllib.parse import parse_qsl, urlsplit

import pytest

from gateway.tee.supabase_source_v2 import (
    SUPABASE_READ_TIMEOUT_MS,
    SUPABASE_WEIGHT_SOURCE_ORIGIN,
    SupabaseSourceReaderV2,
    SupabaseSourceV2Error,
)
from leadpoet_canonical.attested_v2 import build_transport_attempt, sha256_bytes


HASH = "sha256:" + "a" * 64
NOW = "2026-07-10T20:00:00Z"


class FakeProvider:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request):
        self.requests.append(dict(request))
        outcome = self.outcomes.pop(0)
        body = json.dumps(outcome.get("rows", []), separators=(",", ":")).encode()
        terminal_status = outcome.get("terminal_status", "authenticated_response")
        authenticated = terminal_status == "authenticated_response"
        attempt = build_transport_attempt(
            request_id=("%032x" % len(self.requests)),
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id="supabase",
            attempt_number=request["attempt_number"],
            method="GET",
            destination_host="qplwoislplkcegvdmbim.supabase.co",
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=sha256_bytes(b""),
            credential_ref_hash=HASH,
            retry_policy_hash=HASH,
            timeout_ms=request["timeout_ms"],
            started_at=NOW,
            terminal_status=terminal_status,
            http_status=(outcome.get("http_status", 200) if authenticated else None),
            response_hash=(sha256_bytes(body) if authenticated else None),
            request_artifact_hash=("sha256:" + "%064x" % (100 + len(self.requests))),
            response_artifact_hash=(sha256_bytes(body) if authenticated else None),
            tls_peer_chain_hash=(HASH if authenticated else None),
            tls_protocol=("TLSv1.3" if authenticated else None),
            failure_code=(None if authenticated else outcome.get("failure_code", "timeout")),
            completed_at=NOW,
        )
        if not authenticated:
            return {
                "terminal_status": terminal_status,
                "failure_code": attempt["failure_code"],
                "transport_attempt": attempt,
            }
        return {
            "terminal_status": terminal_status,
            "http_status": attempt["http_status"],
            "body_b64": base64.b64encode(body).decode(),
            "transport_attempt": attempt,
        }


def _read(provider, *, policy_id="banned_hotkeys", parameters=None, sleeps=None):
    attempts = []
    artifacts = []
    observed_sleeps = sleeps if sleeps is not None else []
    reader = SupabaseSourceReaderV2(
        execute_provider=provider,
        retry_policy_hash=HASH,
        sleep=observed_sleeps.append,
    )
    rows = reader.read(
        policy_id=policy_id,
        parameters=parameters or {},
        job_id="weight-input:bans:23858",
        purpose="research_lab.ban_input.v2",
        record_transport=lambda attempt: attempts.append(dict(attempt)),
        record_artifact=artifacts.append,
    )
    return rows, attempts, artifacts, observed_sleeps


def test_measured_ban_query_cannot_change_project_table_columns_or_page_size():
    provider = FakeProvider([{"rows": [{"hotkey": "5A"}]}])
    rows, attempts, artifacts, sleeps = _read(provider)
    assert rows == [{"hotkey": "5A"}]
    assert len(attempts) == 1
    assert len(artifacts) == 2
    assert sleeps == []
    request = provider.requests[0]
    assert request["provider_id"] == "supabase"
    assert request["url"].startswith(
        SUPABASE_WEIGHT_SOURCE_ORIGIN
        + "/rest/v1/banned_hotkeys?select=hotkey&order=hotkey.asc"
    )
    assert request["headers"] == {
        "accept": "application/json",
        "range": "0-999",
        "range-unit": "items",
    }
    assert request["timeout_ms"] == SUPABASE_READ_TIMEOUT_MS


def test_champion_allocation_query_matches_live_reward_view_contract():
    provider = FakeProvider([{"rows": []}])
    rows, attempts, _artifacts, _sleeps = _read(
        provider,
        policy_id="allocation_champion_rewards",
        parameters={"epoch_id": 23991},
    )

    assert rows == []
    assert len(attempts) == 1
    url = urlsplit(provider.requests[0]["url"])
    assert url.path.endswith("/rest/v1/research_lab_champion_reward_current")
    query = parse_qsl(url.query, keep_blank_values=True)
    assert (
        "select",
        (
            "champion_reward_id,score_bundle_id,candidate_id,run_id,miner_hotkey,"
            "miner_uid,island,evaluation_epoch,current_reward_status,start_epoch,"
            "epoch_count,improvement_points,threshold_points,"
            "desired_alpha_percent,input_hash,anchored_hash"
        ),
    ) in query
    assert ("current_reward_status", "in.(active,queued,partially_paid)") in query
    assert ("start_epoch", "lte.23991") in query
    assert "reward_status" not in dict(query)["select"].split(",")
    assert "reward_kind" not in dict(query)["select"].split(",")


def test_transient_failures_are_terminally_recorded_before_existing_retries():
    provider = FakeProvider(
        [
            {"terminal_status": "transport_failure", "failure_code": "timeout"},
            {"http_status": 503, "rows": [{"error": "busy"}]},
            {"http_status": 200, "rows": [{"hotkey": "5B"}]},
        ]
    )
    rows, attempts, artifacts, sleeps = _read(provider)
    assert rows == [{"hotkey": "5B"}]
    assert [item["attempt_number"] for item in attempts] == [0, 1, 2]
    assert [item["terminal_status"] for item in attempts] == [
        "transport_failure",
        "authenticated_response",
        "authenticated_response",
    ]
    assert sleeps == [1.0, 3.0]
    assert len(artifacts) == 5
    assert len({request["logical_operation_id"] for request in provider.requests}) == 1


def test_repeated_policy_reads_scope_operations_by_typed_filters():
    first_receipt = "sha256:" + "1" * 64
    second_receipt = "sha256:" + "2" * 64
    provider = FakeProvider(
        [
            {"rows": [{"receipt_hash": first_receipt}]},
            {"rows": [{"receipt_hash": second_receipt}]},
            {"rows": [{"receipt_hash": first_receipt}]},
        ]
    )

    _read(
        provider,
        policy_id="attested_receipt_by_hash",
        parameters={"receipt_hash": first_receipt},
    )
    _read(
        provider,
        policy_id="attested_receipt_by_hash",
        parameters={"receipt_hash": second_receipt},
    )
    _read(
        provider,
        policy_id="attested_receipt_by_hash",
        parameters={"receipt_hash": first_receipt},
    )

    operation_ids = [
        request["logical_operation_id"] for request in provider.requests
    ]
    assert operation_ids[0] != operation_ids[1]
    assert operation_ids[0] == operation_ids[2]
    assert all(request["attempt_number"] == 0 for request in provider.requests)


def test_typed_query_parameters_cannot_inject_postgrest_syntax():
    provider = FakeProvider([{"rows": []}])
    with pytest.raises(SupabaseSourceV2Error, match="integer"):
        _read(
            provider,
            policy_id="fulfillment_active_rewards",
            parameters={"epoch_id": "1&select=secret"},
        )
    assert provider.requests == []


def test_source_add_migration_reads_one_exact_measured_reward_reference():
    reward_ref = "source_add_reward:201a08f0d2b503bf"
    provider = FakeProvider([{"rows": [{"reward_ref": reward_ref}]}])
    rows, attempts, _artifacts, _sleeps = _read(
        provider,
        policy_id="source_add_reward_by_ref",
        parameters={"reward_ref": reward_ref},
    )

    assert rows == [{"reward_ref": reward_ref}]
    assert len(attempts) == 1
    url = provider.requests[0]["url"]
    assert "research_lab_source_add_reward_current" in url
    assert "reward_ref=eq.source_add_reward%3A201a08f0d2b503bf" in url
    assert "limit=2" in url

    with pytest.raises(SupabaseSourceV2Error, match="reward_ref"):
        _read(
            FakeProvider([{"rows": []}]),
            policy_id="source_add_reward_by_ref",
            parameters={"reward_ref": reward_ref + "&select=secret"},
        )


def test_unmeasured_policy_and_inverted_epoch_range_fail_before_network():
    provider = FakeProvider([{"rows": []}])
    with pytest.raises(SupabaseSourceV2Error, match="not measured"):
        _read(provider, policy_id="host_selected_table")
    with pytest.raises(SupabaseSourceV2Error, match="inverted"):
        _read(
            provider,
            policy_id="sourcing_epoch_inputs",
            parameters={"start_epoch": 20, "end_epoch": 19},
        )
    assert provider.requests == []


def test_sourcing_query_reads_only_signed_epoch_documents_and_caps_window():
    provider = FakeProvider([{"rows": []}])
    rows, attempts, _artifacts, _sleeps = _read(
        provider,
        policy_id="sourcing_epoch_inputs",
        parameters={"start_epoch": 70, "end_epoch": 99},
    )
    assert rows == []
    assert len(attempts) == 1
    url = provider.requests[0]["url"]
    assert "select=epoch_id%2Cepoch_hash%2Creceipt_hash%2Csource_doc%2Creceipt_doc" in url
    assert "order=epoch_id.asc" in url
    assert "limit=30" in url
    assert "epoch_id=gte.70" in url
    assert "epoch_id=lte.99" in url


def test_leaderboard_query_is_bound_to_the_exact_observed_window():
    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id="fulfillment_leaderboard_winners",
        parameters={
            "window_start": "2026-07-03T20:00:00Z",
            "window_end": "2026-07-10T20:00:00Z",
        },
    )
    url = provider.requests[0]["url"]
    assert "computed_at=gte.2026-07-03T20%3A00%3A00Z" in url
    assert "computed_at=lte.2026-07-10T20%3A00%3A00Z" in url

    provider = FakeProvider([{"rows": []}])
    with pytest.raises(SupabaseSourceV2Error, match="inverted"):
        _read(
            provider,
            policy_id="fulfillment_leaderboard_winners",
            parameters={
                "window_start": "2026-07-10T20:00:00Z",
                "window_end": "2026-07-03T20:00:00Z",
            },
        )
    assert provider.requests == []


def test_qualification_sources_use_fixed_tables_and_typed_uuid_chunks():
    provider = FakeProvider([{"rows": [{"payload": {"epoch_id": 100}}]}])
    rows, _attempts, _artifacts, _sleeps = _read(
        provider,
        policy_id="qualification_epoch_assignment",
        parameters={"epoch_id": 100},
    )
    assert rows == [{"payload": {"epoch_id": 100}}]
    assignment_url = provider.requests[0]["url"]
    assert "/rest/v1/transparency_log?" in assignment_url
    assert "event_type=eq.EPOCH_INITIALIZATION" in assignment_url
    assert "payload-%3E%3Eepoch_id=eq.100" in assignment_url

    lead_id = "11111111-1111-4111-8111-111111111111"
    provider = FakeProvider([{"rows": [{"lead_id": lead_id}]}])
    rows, _attempts, _artifacts, _sleeps = _read(
        provider,
        policy_id="qualification_leads_by_ids",
        parameters={"lead_ids": [lead_id]},
    )
    assert rows == [{"lead_id": lead_id}]
    assert "/rest/v1/leads_private?" in provider.requests[0]["url"]
    assert "lead_id=in.%28" in provider.requests[0]["url"]

    provider = FakeProvider([{"rows": []}])
    with pytest.raises(SupabaseSourceV2Error, match="UUID"):
        _read(
            provider,
            policy_id="qualification_leads_by_ids",
            parameters={"lead_ids": ["x)&select=secret"]},
        )
    assert provider.requests == []


def test_historical_settlement_queries_are_fixed_and_epoch_bound():
    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id="legacy_allocation_by_hash",
        parameters={
            "allocation_hash": "sha256:" + "1" * 64,
            "netuid": 71,
            "epoch_id": 100,
        },
    )
    url = provider.requests[0]["url"]
    assert "/rest/v1/research_lab_emission_allocation_snapshots?" in url
    assert "allocation_hash=eq.sha256%3A" + "1" * 64 in url
    assert "netuid=eq.71" in url
    assert "epoch=eq.100" in url
    assert "limit=2" in url

    provider = FakeProvider([{"rows": []}])
    with pytest.raises(SupabaseSourceV2Error, match="allocation_hash"):
        _read(
            provider,
            policy_id="legacy_allocation_by_hash",
            parameters={
                "allocation_hash": "sha256:" + "1" * 64 + "&select=secret",
                "netuid": 71,
                "epoch_id": 100,
            },
        )
    assert provider.requests == []

    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id="legacy_finalized_allocation_migrations",
        parameters={"netuid": 71, "start_epoch": 90, "end_epoch": 100},
    )
    url = provider.requests[0]["url"]
    assert (
        "/rest/v1/research_lab_legacy_finalized_allocation_migrations_v2?"
        in url
    )
    assert "netuid=eq.71" in url
    assert "epoch_id=gte.90" in url
    assert "epoch_id=lte.100" in url
    assert "order=epoch_id.asc" in url

    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id="legacy_weight_bundles_by_epoch",
        parameters={"netuid": 71, "epoch_id": 100},
    )
    url = provider.requests[0]["url"]
    assert "/rest/v1/published_weight_bundles?" in url
    assert "netuid=eq.71" in url
    assert "epoch_id=eq.100" in url
    assert "limit=100" in url

    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id="legacy_audit_anchor_by_epoch",
        parameters={"netuid": 71, "epoch_id": 100},
    )
    url = provider.requests[0]["url"]
    assert "/rest/v1/research_lab_arweave_epoch_audit_anchor_current?" in url
    assert "epoch=eq.100" in url
    assert "audit_kind=eq.active" in url
    assert "current_anchor_status=eq.checkpointed" in url

    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id="legacy_transparency_event_by_hash",
        parameters={"event_hash": "sha256:" + "a" * 64},
    )
    assert "event_hash=eq." + "a" * 64 in provider.requests[0]["url"]

    provider = FakeProvider([{"rows": []}])
    with pytest.raises(SupabaseSourceV2Error, match="event_hash"):
        _read(
            provider,
            policy_id="legacy_transparency_event_by_hash",
            parameters={"event_hash": "a" * 64 + "&select=secret"},
        )
    assert provider.requests == []


@pytest.mark.parametrize(
    ("policy_id", "parameter_name"),
    (
        ("reimbursement_ticket_by_id", "ticket_id"),
        ("reimbursement_queue_by_ticket", "ticket_id"),
        ("reimbursement_receipt_by_id", "receipt_id"),
        ("reimbursement_payment_by_id", "payment_id"),
        ("reimbursement_queue_events_by_run", "run_id"),
    ),
)
def test_reimbursement_uuid_sources_reject_query_injection_before_network(
    policy_id,
    parameter_name,
):
    provider = FakeProvider([{"rows": []}])
    with pytest.raises(SupabaseSourceV2Error, match="UUID"):
        _read(
            provider,
            policy_id=policy_id,
            parameters={parameter_name: "x)&select=secret"},
        )
    assert provider.requests == []


def test_exhausted_authenticated_errors_fail_with_all_attempts_visible():
    provider = FakeProvider([{"http_status": 500}, {"http_status": 500}, {"http_status": 500}])
    attempts = []
    reader = SupabaseSourceReaderV2(
        execute_provider=provider,
        retry_policy_hash=HASH,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(SupabaseSourceV2Error, match="http_500"):
        reader.read(
            policy_id="banned_hotkeys",
            parameters={},
            job_id="weight-input:bans:23858",
            purpose="research_lab.ban_input.v2",
            record_transport=lambda attempt: attempts.append(dict(attempt)),
            record_artifact=lambda _digest: None,
        )
    assert len(attempts) == 3
    assert all(item["http_status"] == 500 for item in attempts)
