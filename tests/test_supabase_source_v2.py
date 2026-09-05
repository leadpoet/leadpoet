from __future__ import annotations

import base64
import json
from urllib.parse import parse_qsl, urlsplit

import pytest

from gateway.tee.supabase_source_v2 import (
    QUERY_POLICIES,
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
        parameters={"epoch_id": 23991, "include_paid": False},
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


def test_allocation_frontier_queries_are_exact_and_single_page():
    activation_provider = FakeProvider([{"rows": []}])
    _read(
        activation_provider,
        policy_id="allocation_settlement_frontier_activation",
        parameters={"netuid": 71},
    )
    activation_url = urlsplit(activation_provider.requests[0]["url"])
    activation_query = dict(
        parse_qsl(activation_url.query, keep_blank_values=True)
    )
    assert activation_url.path.endswith(
        "/rest/v1/research_lab_allocation_settlement_frontier_activation_v2"
    )
    assert activation_query["netuid"] == "eq.71"
    assert activation_query["limit"] == "1"
    assert QUERY_POLICIES[
        "allocation_settlement_frontier_activation"
    ].max_pages == 1

    frontier_provider = FakeProvider([{"rows": []}])
    _read(
        frontier_provider,
        policy_id="allocation_settlement_frontiers",
        parameters={"netuid": 71, "before_epoch": 24199},
    )
    frontier_url = urlsplit(frontier_provider.requests[0]["url"])
    frontier_query = dict(parse_qsl(frontier_url.query, keep_blank_values=True))
    assert frontier_url.path.endswith(
        "/rest/v1/research_lab_allocation_settlement_frontiers_v2"
    )
    assert frontier_query["netuid"] == "eq.71"
    assert frontier_query["allocation_epoch"] == "lt.24199"
    assert frontier_query["order"] == "allocation_epoch.desc"
    assert frontier_query["limit"] == "1"
    assert QUERY_POLICIES["allocation_settlement_frontiers"].max_pages == 1

    exact_provider = FakeProvider([{"rows": []}])
    _read(
        exact_provider,
        policy_id="allocation_settlement_frontier_by_epoch",
        parameters={"netuid": 71, "allocation_epoch": 24001},
    )
    exact_url = urlsplit(exact_provider.requests[0]["url"])
    exact_query = dict(parse_qsl(exact_url.query, keep_blank_values=True))
    assert exact_url.path.endswith(
        "/rest/v1/research_lab_allocation_settlement_frontiers_v2"
    )
    assert exact_query["netuid"] == "eq.71"
    assert exact_query["allocation_epoch"] == "eq.24001"
    assert exact_query["limit"] == "1"
    assert QUERY_POLICIES[
        "allocation_settlement_frontier_by_epoch"
    ].max_pages == 1


def test_allocation_frontier_receipt_query_rejects_hash_injection():
    provider = FakeProvider([{"rows": []}])
    with pytest.raises(SupabaseSourceV2Error, match="receipt_hash"):
        _read(
            provider,
            policy_id="attested_execution_result_by_receipt",
            parameters={"receipt_hash": HASH + "&select=result_doc"},
        )
    assert provider.requests == []


@pytest.mark.parametrize(
    ("policy_id", "parameters", "expected"),
    [
        (
            "compact_finalized_authority_cutover",
            {"netuid": 71},
            {
                "netuid": "eq.71",
                "authority_stage": "eq.finalized",
                "order": "epoch_id.asc",
                "limit": "1",
            },
        ),
        (
            "latest_compact_finalized_authority_summaries",
            {"netuid": 71},
            {
                "netuid": "eq.71",
                "authority_stage": "eq.finalized",
                "order": "epoch_id.desc,bundle_hash.asc",
                "limit": "2",
            },
        ),
        (
            "compact_finalized_authority_by_bundle_hash",
            {"netuid": 71, "bundle_hash": HASH},
            {
                "netuid": "eq.71",
                "bundle_hash": "eq." + HASH,
                "authority_stage": "eq.finalized",
                "limit": "1",
            },
        ),
        (
            "compact_finalized_authority_by_identity",
            {
                "netuid": 71,
                "source_epoch_id": 24558,
                "validator_hotkey": "validator-hotkey",
            },
            {
                "netuid": "eq.71",
                "epoch_id": "eq.24558",
                "validator_hotkey": "eq.validator-hotkey",
                "authority_stage": "eq.finalized",
                "order": "bundle_hash.asc",
                "limit": "11",
            },
        ),
    ],
)
def test_compact_settlement_queries_are_exact_bounded_and_finalized(
    policy_id,
    parameters,
    expected,
):
    provider = FakeProvider([{"rows": []}])

    _read(provider, policy_id=policy_id, parameters=parameters)

    url = urlsplit(provider.requests[0]["url"])
    query = dict(parse_qsl(url.query, keep_blank_values=True))
    assert url.path.endswith(
        "/rest/v1/research_lab_compact_weight_authorities_v2"
    )
    assert expected.items() <= query.items()
    assert QUERY_POLICIES[policy_id].max_pages == 1
    if policy_id.endswith("by_bundle_hash") or policy_id.endswith(
        "by_identity"
    ):
        assert "authority_doc" in query["select"].split(",")
    else:
        assert "authority_doc" not in query["select"].split(",")


def test_compact_settlement_identity_url_encodes_hotkey_filter_syntax():
    provider = FakeProvider([{"rows": []}])

    _read(
        provider,
        policy_id="compact_finalized_authority_by_identity",
        parameters={
            "netuid": 71,
            "source_epoch_id": 24558,
            "validator_hotkey": "validator&authority_stage=eq.published",
        },
    )

    query = parse_qsl(
        urlsplit(provider.requests[0]["url"]).query,
        keep_blank_values=True,
    )
    assert ("validator_hotkey", "eq.validator&authority_stage=eq.published") in query
    assert query.count(("authority_stage", "eq.finalized")) == 1
    assert ("authority_stage", "eq.published") not in query


def test_uncapped_champion_query_includes_paid_projection():
    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id="allocation_champion_rewards",
        parameters={"epoch_id": 23991, "include_paid": True},
    )

    query = dict(
        parse_qsl(urlsplit(provider.requests[0]["url"]).query)
    )
    assert query["current_reward_status"] == (
        "in.(active,queued,partially_paid,paid)"
    )


@pytest.mark.parametrize(
    ("policy_id", "table", "epoch_field"),
    [
        (
            "latest_native_compute_allocation_authority",
            "research_lab_finalized_allocation_epochs_v2",
            "epoch_id",
        ),
        (
            "latest_legacy_compute_allocation_authority",
            "research_lab_legacy_finalized_allocation_migrations_v2",
            "epoch_id",
        ),
    ],
)
def test_historical_compute_queries_are_strictly_before_target_epoch(
    policy_id,
    table,
    epoch_field,
):
    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id=policy_id,
        parameters={"epoch_id": 24030, "netuid": 71},
    )

    url = urlsplit(provider.requests[0]["url"])
    query = parse_qsl(url.query, keep_blank_values=True)
    assert url.path.endswith("/rest/v1/" + table)
    assert (epoch_field, "lt.24030") in query
    assert ("netuid", "eq.71") in query
    assert any(
        key.endswith("reimbursement_allocations") and value == "not.eq.[]"
        for key, value in query
    )
    assert any(
        key.endswith("historical_compute_fallback_source_epoch")
        and value == "is.null"
        for key, value in query
    )
    assert ("order", "epoch_id.desc") in query
    assert ("limit", "1") in query


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


def test_business_artifact_lookup_uses_the_exact_authenticated_hash():
    artifact_hash = "sha256:" + "3" * 64
    provider = FakeProvider([{"rows": []}])

    _read(
        provider,
        policy_id="attested_business_artifact_by_ref",
        parameters={
            "artifact_kind": "allocation",
            "artifact_ref": "epoch:24093",
            "artifact_hash": artifact_hash,
        },
    )

    query = parse_qsl(urlsplit(provider.requests[0]["url"]).query)
    assert ("artifact_kind", "eq.allocation") in query
    assert ("artifact_ref", "eq.epoch:24093") in query
    assert ("artifact_hash", "eq.%s" % artifact_hash) in query
    assert ("limit", "1") in query

    with pytest.raises(SupabaseSourceV2Error, match="artifact_hash"):
        _read(
            FakeProvider([{"rows": []}]),
            policy_id="attested_business_artifact_by_ref",
            parameters={
                "artifact_kind": "allocation",
                "artifact_ref": "epoch:24093",
                "artifact_hash": "not-a-hash",
            },
        )


def test_typed_query_parameters_cannot_inject_postgrest_syntax():
    provider = FakeProvider([{"rows": []}])
    with pytest.raises(SupabaseSourceV2Error, match="integer"):
        _read(
            provider,
            policy_id="fulfillment_active_rewards",
            parameters={"epoch_id": "1&select=secret"},
        )
    assert provider.requests == []


def test_fulfillment_rewards_use_stable_consensus_order_across_pages():
    provider = FakeProvider([{"rows": []}])

    _read(
        provider,
        policy_id="fulfillment_active_rewards",
        parameters={"epoch_id": 100},
    )

    query = parse_qsl(urlsplit(provider.requests[0]["url"]).query)
    assert (
        "select",
        "consensus_id,miner_hotkey,reward_pct,reward_expires_epoch",
    ) in query
    assert ("order", "consensus_id.asc") in query


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


def test_allocation_source_add_query_binds_fifo_creation_order():
    provider = FakeProvider([{"rows": []}])

    _read(
        provider,
        policy_id="allocation_source_add_rewards",
        parameters={"epoch_id": 100},
    )

    url = urlsplit(provider.requests[0]["url"])
    query = dict(parse_qsl(url.query, keep_blank_values=True))
    assert "created_at" in query["select"].split(",")
    assert query["order"] == "created_at.asc,reward_ref.asc"


def test_source_add_functional_probe_query_binds_approval_config():
    provider = FakeProvider([{"rows": []}])

    _read(
        provider,
        policy_id="source_add_functional_probe_by_submission",
        parameters={"submission_id": "source_add_submission:1234567890abcdef"},
    )

    url = urlsplit(provider.requests[0]["url"])
    query = dict(parse_qsl(url.query, keep_blank_values=True))
    assert url.path.endswith(
        "/rest/v1/research_lab_source_add_functional_probe_current"
    )
    assert query["submission_id"] == (
        "eq.source_add_submission:1234567890abcdef"
    )
    selected = query["select"].split(",")
    assert "evaluation_mode" in selected
    assert "config_ref" in selected
    assert query["limit"] == "2"


def test_unmeasured_policy_and_inverted_epoch_range_fail_before_network():
    provider = FakeProvider([{"rows": []}])
    assert "active_private_model_current" not in QUERY_POLICIES
    with pytest.raises(SupabaseSourceV2Error, match="not measured"):
        _read(provider, policy_id="active_private_model_current")
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
        policy_id="chain_realized_epoch_settlements",
        parameters={"netuid": 71, "start_epoch": 90, "end_epoch": 100},
    )
    url = provider.requests[0]["url"]
    assert "/rest/v1/research_lab_chain_realized_epoch_settlements_v1?" in url
    assert "netuid=eq.71" in url
    assert "epoch_id=gte.90" in url
    assert "epoch_id=lte.100" in url
    assert "order=epoch_id.asc" in url

    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id="chain_realized_obligation_credits",
        parameters={"netuid": 71, "start_epoch": 90, "end_epoch": 100},
    )
    url = provider.requests[0]["url"]
    assert "/rest/v1/research_lab_chain_realized_obligation_credits_v1?" in url
    assert "netuid=eq.71" in url
    assert "epoch_id=gte.90" in url
    assert "epoch_id=lte.100" in url
    assert "order=epoch_id.asc%2Cobligation_kind.asc%2Cobligation_source_id.asc" in url

    provider = FakeProvider([{"rows": []}])
    block_hash = "a" * 64
    hotkey = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
    _read(
        provider,
        policy_id="finalized_authority_by_chain_vector",
        parameters={
            "netuid": 71,
            "uids": [0, 17, 255],
            "weights_u16": [32768, 16384, 16383],
            "source_epoch_id": 24197,
            "validator_hotkey": hotkey,
            "finalized_block": 8715584,
            "finalized_block_hash": block_hash,
        },
    )
    parsed_url = urlsplit(provider.requests[0]["url"])
    assert parsed_url.path.endswith(
        "/rest/v1/research_lab_finalized_weight_vector_candidates_v1"
    )
    query = dict(parse_qsl(parsed_url.query, keep_blank_values=True))
    assert query["netuid"] == "eq.71"
    assert query["epoch_id"] == "eq.24197"
    assert query["validator_hotkey"] == "eq." + hotkey
    assert query["finalized_block"] == "eq.8715584"
    assert query["finalized_block_hash"] == "eq." + block_hash
    assert query["uids"] == "eq.[0,17,255]"
    assert query["weights_u16"] == "eq.[32768,16384,16383]"
    assert query["order"] == "finalized_block.desc,bundle_hash.asc"
    assert query["limit"] == "100"

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
    ("field", "value"),
    (
        ("uids", [17, 0]),
        ("uids", [0, 0]),
        ("uids", [False]),
        ("weights_u16", [0, 1, 2]),
        ("weights_u16", [65536, 1, 2]),
        ("weights_u16", [False]),
        ("finalized_block_hash", "0x" + "a" * 64),
        ("finalized_block_hash", "a" * 63 + "&"),
        ("validator_hotkey", "5FNV\ninjected"),
    ),
)
def test_finalized_chain_vector_query_rejects_noncanonical_parameters(
    field,
    value,
):
    parameters = {
        "netuid": 71,
        "uids": [0, 17, 255],
        "weights_u16": [32768, 16384, 16383],
        "source_epoch_id": 24197,
        "validator_hotkey": "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
        "finalized_block": 8715584,
        "finalized_block_hash": "a" * 64,
    }
    parameters[field] = value
    provider = FakeProvider([{"rows": []}])

    with pytest.raises(SupabaseSourceV2Error):
        _read(
            provider,
            policy_id="finalized_authority_by_chain_vector",
            parameters=parameters,
        )
    assert provider.requests == []


def test_latest_finalized_authority_reads_bounded_summaries_then_one_exact_row():
    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id="latest_finalized_allocation_authority_summaries",
        parameters={"netuid": 71},
    )
    summary_query = dict(
        parse_qsl(urlsplit(provider.requests[0]["url"]).query)
    )
    assert summary_query == {
        "select": (
            "bundle_hash,netuid,epoch_id,validator_hotkey,finalized_block,"
            "finalized_block_hash,finalization_receipt_hash"
        ),
        "netuid": "eq.71",
        "order": "finalized_block.desc,bundle_hash.asc",
        "limit": "2",
    }
    assert "bundle_doc" not in summary_query["select"]
    assert "finalization_doc" not in summary_query["select"]

    provider = FakeProvider([{"rows": []}])
    _read(
        provider,
        policy_id="finalized_allocation_authority_by_bundle_hash",
        parameters={"netuid": 71, "bundle_hash": HASH},
    )
    full_query = dict(parse_qsl(urlsplit(provider.requests[0]["url"]).query))
    assert full_query["netuid"] == "eq.71"
    assert full_query["bundle_hash"] == "eq." + HASH
    assert full_query["limit"] == "1"
    assert "bundle_doc" in full_query["select"].split(",")
    assert "finalization_doc" in full_query["select"].split(",")


def test_exact_finalized_authority_query_rejects_bundle_hash_injection():
    provider = FakeProvider([{"rows": []}])
    with pytest.raises(SupabaseSourceV2Error, match="bundle_hash"):
        _read(
            provider,
            policy_id="finalized_allocation_authority_by_bundle_hash",
            parameters={
                "netuid": 71,
                "bundle_hash": HASH + "&select=secret",
            },
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


def test_finalized_allocation_history_uses_bounded_complete_pages():
    rows = [{"epoch_id": epoch} for epoch in range(16)]
    provider = FakeProvider(
        [
            {"rows": rows[offset : offset + 2]}
            for offset in range(0, len(rows), 2)
        ]
        + [{"rows": []}]
    )

    observed, attempts, _artifacts, sleeps = _read(
        provider,
        policy_id="finalized_allocation_authorities",
        parameters={"netuid": 71, "start_epoch": 0, "end_epoch": 100},
    )

    assert observed == rows
    assert len(attempts) == 9
    assert sleeps == []
    assert QUERY_POLICIES["finalized_allocation_authorities"].page_size == 2
    assert [
        request["headers"]["range"] for request in provider.requests
    ] == [
        "0-1",
        "2-3",
        "4-5",
        "6-7",
        "8-9",
        "10-11",
        "12-13",
        "14-15",
        "16-17",
    ]
