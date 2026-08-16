from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_physical_v2_staging.py"
WORKFLOW = ROOT / ".github" / "workflows" / "physical-v2-staging.yml"


def _module():
    spec = importlib.util.spec_from_file_location("physical_v2_staging", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> dict:
    key = tmp_path / "staging.pem"
    key.write_text("test-only\n", encoding="utf-8")
    common = {"ssh_key": str(key)}
    prefix = "leadpoet/staging/production-parity/run-123456"
    return {
        "schema_version": "leadpoet.physical_v2_staging_config.v2",
        "environment": "production-parity-ephemeral",
        "ephemeral_stack_id": "run-123456",
        "network": "test",
        "netuid": 1,
        "chain_endpoint": "wss://test.finney.opentensor.ai:443",
        "network_genesis_hash": "0x" + "7" * 64,
        "gateway_public_url": "https://staging-gateway.example",
        "dashboard_report_url": (
            "https://staging-dashboard.example/api/research-lab"
        ),
        "dashboard_source_sha": "e" * 40,
        "timeout_seconds": 300,
        "rebenchmark_timeout_seconds": 300,
        "poll_seconds": 2,
        "required_consecutive_epochs": 3,
        "gateway": {
            **common,
            "ssh_host": "ec2-user@192.0.2.10",
            "restart_path": "/home/ec2-user/gw_restart.sh",
            "secret_id": prefix + "/gateway",
            "repo_root": "/home/ec2-user/leadpoet_repo",
            "python_bin": "/home/ec2-user/venv311/bin/python3",
        },
        "primary_validator": {
            **common,
            "ssh_host": "ec2-user@192.0.2.11",
            "restart_path": "/home/ec2-user/validator_restart.sh",
            "secret_id": prefix + "/validator",
            "repo_root": "/home/ec2-user/leadpoet/leadpoet",
            "container_name": "leadpoet-validator-main",
            "expected_hotkey": "4" * 48,
        },
        "audit_validators": [
            {
                **common,
                "ssh_host": "ec2-user@192.0.2.12",
                "repo_root": "/home/ec2-user/leadpoet/leadpoet",
                "unit_name": "leadpoet-auditor-a.service",
                "expected_hotkey": "5" * 48,
                "secret_id": prefix + "/auditor-a",
            },
            {
                **common,
                "ssh_host": "ec2-user@192.0.2.13",
                "repo_root": "/home/ec2-user/leadpoet/leadpoet",
                "unit_name": "leadpoet-auditor-b.service",
                "expected_hotkey": "6" * 48,
                "secret_id": prefix + "/auditor-b",
            },
        ],
        "dashboard": {
            **common,
            "ssh_host": "ec2-user@192.0.2.14",
            "repo_root": "/home/ec2-user/subnet_dashboard",
            "unit_name": "leadpoet-parity-dashboard.service",
            "source_sha": "e" * 40,
        },
    }


def _write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _provider_snapshot(
    *,
    direct_requests: int = 0,
    assigned_requests: int = 0,
    exa_assigned_success: int = 0,
    scrapingdog_assigned_success: int = 0,
    supabase_direct_success: int = 0,
    chain_weight_success: int = 0,
    cache_lookup_success: int = 0,
    cache_write_success: int = 0,
    outcome_append_success: int = 0,
    provider_transport_success: int = 0,
    request_failures: int = 0,
    accepted_tunnels: int = 0,
    completed_tunnels: int = 0,
    open_fds: int = 20,
    tcp_time_wait: int = 0,
) -> dict:
    routes = ("direct", "assigned_proxy")
    providers = ("exa", "scrapingdog", "supabase")
    request_counts = {
        "direct": {
            "started": direct_requests + request_failures,
            "succeeded": direct_requests,
            "failed": request_failures,
        },
        "assigned_proxy": {
            "started": assigned_requests,
            "succeeded": assigned_requests,
            "failed": 0,
        },
    }
    cleanup_counts = {
        route: {
            "attempted": values["started"],
            "succeeded": values["started"],
            "client_close_failed": 0,
            "transport_close_failed": 0,
        }
        for route, values in request_counts.items()
    }
    terminals = {
        provider: {
            route: {
                "authenticated_response": 0,
                "transport_failure": 0,
            }
            for route in routes
        }
        for provider in providers
    }
    terminals["exa"]["assigned_proxy"]["authenticated_response"] = (
        exa_assigned_success
    )
    terminals["scrapingdog"]["assigned_proxy"][
        "authenticated_response"
    ] = scrapingdog_assigned_success
    terminals["supabase"]["direct"]["authenticated_response"] = (
        supabase_direct_success
    )
    terminals["supabase"]["direct"]["transport_failure"] = request_failures
    success_2xx = {
        provider: {route: 0 for route in routes}
        for provider in providers
    }
    success_2xx["exa"]["assigned_proxy"] = exa_assigned_success
    success_2xx["scrapingdog"]["assigned_proxy"] = (
        scrapingdog_assigned_success
    )
    success_2xx["supabase"]["direct"] = supabase_direct_success
    stages = {
        stage: {"started": 0, "succeeded": 0, "failed": 0}
        for stage in (
            "provider_transport",
            "provider_cache_lookup",
            "provider_cache_write",
            "provider_outcome_restore",
            "provider_outcome_append",
        )
    }
    for stage, succeeded in {
        "provider_transport": provider_transport_success,
        "provider_cache_lookup": cache_lookup_success,
        "provider_cache_write": cache_write_success,
        "provider_outcome_append": outcome_append_success,
    }.items():
        stages[stage]["started"] = succeeded
        stages[stage]["succeeded"] = succeeded
    tcp_states = {
        state: 0
        for state in (
            "established",
            "syn_sent",
            "syn_received",
            "fin_wait_1",
            "fin_wait_2",
            "time_wait",
            "close",
            "close_wait",
            "last_ack",
            "listen",
            "closing",
            "new_syn_received",
            "other",
        )
    }
    tcp_states["time_wait"] = tcp_time_wait
    return {
        "request_counters": request_counts,
        "cleanup_counters": cleanup_counts,
        "provider_terminal_counts": terminals,
        "provider_2xx_success_counts": success_2xx,
        "chain_weight_observation_success_count": chain_weight_success,
        "semantics_stage_counters": stages,
        "scope_counts": {
            "direct_request_slot_active_count": 0,
            "direct_active_scope_count": 0,
            "direct_retired_scope_count": 0,
            "direct_active_lease_count": 0,
            "direct_retired_lease_count": 0,
            "assigned_active_scope_count": 0,
            "assigned_retired_scope_count": 0,
            "assigned_active_lease_count": 0,
            "assigned_retired_lease_count": 0,
        },
        "egress_counters": {
            "accepted_tunnel_count": accepted_tunnels,
            "active_tunnel_count": 0,
            "completed_tunnel_count": completed_tunnels,
            "failed_tunnel_count": 0,
            "socket_cleanup_failure_count": 0,
        },
        "resources": {
            "process_open_fd_count": open_fds,
            "process_nofile_soft_limit": 1024,
            "process_nofile_hard_limit": 4096,
            "ip_local_port_range_lower": 32768,
            "ip_local_port_range_upper": 60999,
            "ip_local_port_range_size": 28232,
            "loopback_tcp_total_count": tcp_time_wait,
            "loopback_tcp_scanned_row_count": tcp_time_wait + 1,
            "loopback_tcp_scan_truncated": 0,
            "loopback_tcp_state_counts": tcp_states,
        },
    }


def test_physical_staging_config_requires_disposable_real_boundaries(
    tmp_path: Path,
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))

    assert config.network == "test"
    assert config.required_consecutive_epochs == 3
    assert len(config.auditors) == 2
    assert len(
        {
            config.gateway.ssh_host,
            config.primary_validator.ssh_host,
            *(item.ssh_host for item in config.auditors),
            config.dashboard.ssh_host,
        }
    ) == 5


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda value: value.update(
                {"schema_version": "leadpoet.physical_v2_staging_config.v1"}
            ),
            "schema differs",
        ),
        (
            lambda value: value["gateway"].update(
                {"ssh_host": "ec2-user@52.91.135.79"}
            ),
            "production host",
        ),
        (
            lambda value: value["gateway"].update(
                {"secret_id": "leadpoet/prod/gateway/env"}
            ),
            "staging secret",
        ),
        (lambda value: value.update({"network": "finney"}), "testnet"),
        (
            lambda value: value.update(
                {"chain_endpoint": "wss://finney.opentensor.ai:443"}
            ),
            "chain authority",
        ),
        (
            lambda value: value.update(
                {"gateway_public_url": "http://52.91.135.79:8000"}
            ),
            "isolated URL",
        ),
        (
            lambda value: value.update(
                {"audit_validators": value["audit_validators"][:1]}
            ),
            "at least two",
        ),
        (
            lambda value: value["audit_validators"][0].update(
                {"ssh_host": value["gateway"]["ssh_host"]}
            ),
            "distinct hosts",
        ),
        (
            lambda value: value["audit_validators"][0].update(
                {"secret_id": "leadpoet/staging/shared-auditor"}
            ),
            "secret_id is invalid",
        ),
    ],
)
def test_physical_staging_config_rejects_parity_shortcuts(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    module = _module()
    value = _config(tmp_path)
    mutate(value)

    with pytest.raises(module.PhysicalStagingError, match=message):
        module.load_config(_write_config(tmp_path, value))


def test_provider_idle_window_requires_901_continuous_counter_stable_seconds(
    tmp_path: Path,
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))
    now = [0.0]
    calls = [0]
    initial = _provider_snapshot()
    after_prewarm = _provider_snapshot(
        direct_requests=1,
        accepted_tunnels=1,
        completed_tunnels=1,
    )

    def snapshotter():
        calls[0] += 1
        return initial if calls[0] == 1 else after_prewarm

    def sleeper(seconds):
        now[0] += float(seconds)

    baseline, evidence = module._wait_for_provider_idle_window(
        config,
        snapshotter=snapshotter,
        monotonic=lambda: now[0],
        sleeper=sleeper,
        idle_seconds=901.0,
        timeout_seconds=1000.0,
    )

    assert baseline == after_prewarm
    assert evidence["idle_seconds"] >= 901.0
    assert evidence["idle_reset_count"] == 1
    assert evidence["baseline_hash"] == evidence["final_hash"]
    with pytest.raises(module.PhysicalStagingError, match="shorter than 901"):
        module._wait_for_provider_idle_window(
            config,
            snapshotter=lambda: initial,
            monotonic=lambda: now[0],
            sleeper=sleeper,
            idle_seconds=900.0,
        )


def test_provider_idle_window_fails_when_activity_never_quiesces(
    tmp_path: Path,
) -> None:
    module = _module()
    config_doc = _config(tmp_path)
    config_doc["poll_seconds"] = 60
    config = module.load_config(_write_config(tmp_path, config_doc))
    now = [0.0]
    counter = [0]

    def snapshotter():
        counter[0] += 1
        return _provider_snapshot(
            direct_requests=counter[0],
            accepted_tunnels=counter[0],
            completed_tunnels=counter[0],
        )

    def sleeper(seconds):
        now[0] += float(seconds)

    with pytest.raises(module.PhysicalStagingError, match="never reached"):
        module._wait_for_provider_idle_window(
            config,
            snapshotter=snapshotter,
            monotonic=lambda: now[0],
            sleeper=sleeper,
            idle_seconds=901.0,
            timeout_seconds=902.0,
        )


def test_provider_idle_window_rejects_low_fd_or_ephemeral_headroom(
    tmp_path: Path,
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))
    low_fd = _provider_snapshot(open_fds=800)
    with pytest.raises(module.PhysicalStagingError, match="NOFILE headroom"):
        module._wait_for_provider_idle_window(
            config,
            snapshotter=lambda: low_fd,
            monotonic=lambda: 0.0,
            sleeper=lambda _seconds: None,
            idle_seconds=901.0,
            timeout_seconds=902.0,
        )

    low_ports = _provider_snapshot(tcp_time_wait=27_000)
    with pytest.raises(module.PhysicalStagingError, match="ephemeral-port"):
        module._wait_for_provider_idle_window(
            config,
            snapshotter=lambda: low_ports,
            monotonic=lambda: 0.0,
            sleeper=lambda _seconds: None,
            idle_seconds=901.0,
            timeout_seconds=902.0,
        )


def test_provider_idle_window_rechecks_counters_at_threshold(
    tmp_path: Path,
) -> None:
    module = _module()
    config_doc = _config(tmp_path)
    config_doc["poll_seconds"] = 60
    config = module.load_config(_write_config(tmp_path, config_doc))
    now = [0.0]
    threshold_reads = [0]
    initial = _provider_snapshot()
    changed = _provider_snapshot(
        direct_requests=1,
        accepted_tunnels=1,
        completed_tunnels=1,
    )

    def snapshotter():
        if now[0] >= 901.0:
            threshold_reads[0] += 1
            if threshold_reads[0] >= 2:
                return changed
        return initial

    def sleeper(seconds):
        now[0] += float(seconds)

    baseline, evidence = module._wait_for_provider_idle_window(
        config,
        snapshotter=snapshotter,
        monotonic=lambda: now[0],
        sleeper=sleeper,
        idle_seconds=901.0,
        timeout_seconds=1900.0,
    )

    assert baseline == changed
    assert evidence["idle_reset_count"] == 1
    assert now[0] >= 1802.0


def test_post_idle_provider_proof_joins_real_routes_cleanup_and_resources(
    tmp_path: Path,
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))
    baseline = _provider_snapshot()
    current = _provider_snapshot(
        direct_requests=4,
        assigned_requests=2,
        exa_assigned_success=1,
        scrapingdog_assigned_success=1,
        supabase_direct_success=4,
        chain_weight_success=1,
        provider_transport_success=2,
        cache_lookup_success=1,
        cache_write_success=1,
        outcome_append_success=1,
        accepted_tunnels=6,
        completed_tunnels=6,
        open_fds=21,
        tcp_time_wait=6,
    )

    evidence = module._wait_for_post_idle_provider_proof(
        config,
        baseline=baseline,
        snapshotter=lambda: current,
        monotonic=lambda: 0.0,
        sleeper=lambda _seconds: None,
        timeout_seconds=1.0,
    )

    assert evidence["provider_2xx_success_deltas"]["exa"][
        "assigned_proxy"
    ] == 1
    assert evidence["provider_2xx_success_deltas"]["scrapingdog"][
        "assigned_proxy"
    ] == 1
    assert evidence["chain_weight_observation_success_delta"] == 1
    assert evidence["cleanup_deltas"]["direct"]["attempted"] == 4
    assert evidence["egress_accepted_delta"] == 6
    assert evidence["open_fd_delta"] == 1

    failed = _provider_snapshot(
        direct_requests=4,
        assigned_requests=2,
        exa_assigned_success=1,
        scrapingdog_assigned_success=1,
        supabase_direct_success=4,
        chain_weight_success=1,
        provider_transport_success=2,
        cache_lookup_success=1,
        cache_write_success=1,
        outcome_append_success=1,
        request_failures=1,
        accepted_tunnels=7,
        completed_tunnels=7,
    )
    with pytest.raises(module.PhysicalStagingError, match="request failed"):
        module._wait_for_post_idle_provider_proof(
            config,
            baseline=baseline,
            snapshotter=lambda: failed,
            monotonic=lambda: 0.0,
            sleeper=lambda _seconds: None,
            timeout_seconds=1.0,
        )

    authenticated_non_2xx = _provider_snapshot(
        direct_requests=4,
        assigned_requests=2,
        supabase_direct_success=4,
        chain_weight_success=1,
        provider_transport_success=2,
        cache_lookup_success=1,
        cache_write_success=1,
        outcome_append_success=1,
        accepted_tunnels=6,
        completed_tunnels=6,
    )
    authenticated_non_2xx["provider_terminal_counts"]["exa"][
        "assigned_proxy"
    ]["authenticated_response"] = 1
    authenticated_non_2xx["provider_terminal_counts"]["scrapingdog"][
        "assigned_proxy"
    ]["authenticated_response"] = 1
    clock = [0.0]
    with pytest.raises(module.PhysicalStagingError, match="timed out"):
        module._wait_for_post_idle_provider_proof(
            config,
            baseline=baseline,
            snapshotter=lambda: authenticated_non_2xx,
            monotonic=lambda: clock[0],
            sleeper=lambda seconds: clock.__setitem__(
                0, clock[0] + float(seconds)
            ),
            timeout_seconds=1.0,
        )


def test_full_lane_rejects_snapshot_from_another_database_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))
    contract = {"candidate_sha": "a" * 40}
    snapshot = {
        "source_host_hash": module.production_database_host_hash(
            "other-production.example"
        )
    }
    monkeypatch.setattr(
        module, "verify_contract_checkout", lambda root, value: contract
    )
    monkeypatch.setattr(
        module, "validate_snapshot_manifest", lambda value: snapshot
    )

    with pytest.raises(
        module.PhysicalStagingError, match="source differs from production"
    ):
        module.run(
            config,
            commit="a" * 40,
            contract=contract,
            snapshot=snapshot,
            production_db_host="production-db.example",
        )


def test_physical_staging_requires_identical_finalized_authority() -> None:
    module = _module()
    bundle_hash = "sha256:" + "a" * 64
    weights_hash = "sha256:" + "b" * 64
    authority = {
        "authority_stage": "finalized",
        "bundle_hash": bundle_hash,
        "compact_submission": {
            "weight_result": {"epoch_id": 42, "weights_hash": weights_hash}
        },
    }

    assert module._authority_identity(authority) == (42, bundle_hash, weights_hash)
    assert module._matching_auditor_success(
        [
            {
                "event": "submission_success",
                "netuid": 1,
                "epoch": 42,
                "bundle_hash": bundle_hash,
                "weights_hash": weights_hash,
                "confirmation_stage": "timelocked_commit_finalized",
            }
        ],
        netuid=1,
        epoch_id=42,
        bundle_hash=bundle_hash,
        weights_hash=weights_hash,
    )
    assert module._matching_auditor_startup(
        [
            {
                "event": "startup_ready",
                "commit": "d" * 40,
                "netuid": 1,
                "hotkey": "5" * 48,
                "gateway_endpoint": "https://staging-gateway.example/",
                "weight_protocol": "authoritative_v2",
            }
        ],
        commit="d" * 40,
        netuid=1,
        hotkey="5" * 48,
        gateway_public_url="https://staging-gateway.example",
    )


def test_physical_staging_requires_independent_chain_confirmation(
    tmp_path: Path,
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))
    primary = config.primary_validator.expected_hotkey
    auditor_a, auditor_b = [item.expected_hotkey for item in config.auditors]
    vector = [[0, 32768], [7, 32767]]
    accepted = [
        {
            "epoch_id": epoch,
            "weights": vector,
            "primary_finalized": {"finalized_block": 100 + epoch},
            "auditors": {
                auditor_a: {"observed_last_update": 100 + epoch},
                auditor_b: {"observed_last_update": 100 + epoch},
            },
        }
        for epoch in (42, 43, 44)
    ]
    body = {
        "schema_version": "leadpoet.production_parity_chain_readback.v1",
        "network": "test",
        "chain_endpoint_host": "test.finney.opentensor.ai",
        "network_genesis_hash": config.network_genesis_hash,
        "netuid": 1,
        "finalized_block": 200,
        "finalized_block_hash": "0x" + "8" * 64,
        "validators": [
            {
                "hotkey": hotkey,
                "uid": uid,
                "last_update": 144,
                "weights": vector,
            }
            for uid, hotkey in enumerate((primary, auditor_a, auditor_b), 1)
        ],
    }
    readback = {**body, "readback_hash": module.sha256_json(body)}

    evidence = module._verify_independent_chain_acceptance(
        config, accepted, readback
    )

    assert evidence["finalized_block"] == 200
    assert set(evidence["visible_vector_epoch_by_hotkey"]) == {
        primary,
        auditor_a,
        auditor_b,
    }
    stale = json.loads(json.dumps(readback))
    stale["validators"][0]["last_update"] = 1
    with pytest.raises(module.PhysicalStagingError, match="LastUpdate"):
        module._verify_independent_chain_acceptance(config, accepted, stale)


def test_auditor_chain_confirmation_must_be_finalized() -> None:
    module = _module()
    event = {
        "event": "submission_chain_confirmation",
        "netuid": 1,
        "epoch": 42,
        "observed_last_update": 1234,
        "finalized_block_hash": "0x" + "a" * 64,
        "confirmation_stage": "timelocked_commit_finalized",
    }

    assert module._matching_auditor_chain_confirmation(
        [event], netuid=1, epoch_id=42
    ) == {
        "observed_last_update": 1234,
        "finalized_block_hash": "0x" + "a" * 64,
    }
    assert module._matching_auditor_chain_confirmation(
        [{**event, "confirmation_stage": "awaiting_finalized_commit"}],
        netuid=1,
        epoch_id=42,
    ) is None


def test_dashboard_readback_matches_real_public_api_shape() -> None:
    module = _module()
    identity = {
        "report_id": "report-1",
        "benchmark_date": "2026-08-15",
        "rolling_window_hash": "sha256:" + "a" * 64,
        "aggregate_score": 61.25,
        "item_count": 40,
    }
    value = {
        "success": True,
        "data": {
            "benchmark": {
                "reportId": "report-1",
                "benchmarkDate": "2026-08-15",
                "rollingWindowHash": "sha256:" + "a" * 64,
                "aggregateScore": 61.25,
                "itemCount": 40,
            }
        },
    }

    assert module._contains_dashboard_identity(value, identity)
    assert not module._contains_dashboard_identity(
        {**value, "data": {"benchmark": {**value["data"]["benchmark"], "itemCount": 20}}},
        identity,
    )


def test_rebenchmark_identity_requires_candidate_policy_and_full_distribution() -> None:
    module = _module()
    policy = {
        "public_total_icps": 4,
        "public_weak_total": 3,
        "public_strong_total": 1,
        "private_total_icps": 4,
        "private_weak_total": 1,
        "private_strong_total": 3,
        "conditional_total_icps": 8,
        "selection_policy": "centered_conditional_hash_rotated_tails:v1",
        "policy_hash": "sha256:" + "9" * 64,
    }
    report_doc = {
        "report_type": "research_lab_public_daily_benchmark",
        "aggregate_score": 51.25,
        "item_count": 16,
        "rolling_window_hash": "sha256:" + "a" * 64,
        "visibility_split": {
            "split_policy": policy["selection_policy"],
            "rolling_window_hash": "sha256:" + "a" * 64,
            "public_count": 4,
            "private_count": 4,
            "conditional_count": 8,
            "public_strength_counts": {"weak": 3, "strong": 1},
            "private_strength_counts": {"weak": 1, "strong": 3},
        },
    }
    report_doc["report_public_hash"] = module.sha256_json(report_doc)
    value = {
        "current_report_status": "published",
        "benchmark_quality": "passed",
        "report_id": "report-1",
        "benchmark_bundle_id": "bundle-1",
        "benchmark_date": "2026-08-15",
        "rolling_window_hash": "sha256:" + "a" * 64,
        "private_model_artifact_hash": "sha256:" + "b" * 64,
        "private_model_manifest_hash": "sha256:" + "c" * 64,
        "report_doc": report_doc,
    }

    identity = module._rebenchmark_identity(value, policy=policy)
    assert identity["item_count"] == 16
    assert identity["category_counts"] == {
        "public": 4,
        "private": 4,
        "conditional": 8,
    }

    value["report_doc"]["visibility_split"]["split_policy"] = "stale-policy"
    value["report_doc"]["report_public_hash"] = module.sha256_json(
        {
            key: item
            for key, item in value["report_doc"].items()
            if key != "report_public_hash"
        }
    )
    with pytest.raises(module.PhysicalStagingError, match="candidate policy"):
        module._rebenchmark_identity(value, policy=policy)


def test_candidate_readiness_probe_is_exact_secret_and_checkout_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))
    captured: list[str] = []
    monkeypatch.setattr(
        module,
        "_ssh",
        lambda host, key, command, timeout: captured.append(command)
        or json.dumps(
            {
                "available": True,
                "reason": "daily_baseline_published",
                "benchmark_date": "2026-08-15",
                "report_id": "report-1",
                "benchmark_bundle_id": "bundle-1",
                "rolling_window_hash": "sha256:" + "a" * 64,
            }
        ),
    )

    result = module._candidate_rebenchmark_readiness(
        config,
        candidate_sha="d" * 40,
    )

    assert result["available"] is True
    assert config.gateway.repo_root in captured[0]
    assert config.gateway.python_bin in captured[0]
    assert config.gateway.secret_id in captured[0]
    assert "d" * 40 in captured[0]


def test_rebenchmark_acceptance_is_bound_to_snapshot_utc_date(tmp_path: Path) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))

    with pytest.raises(module.PhysicalStagingError, match="date is invalid"):
        module._wait_for_rebenchmark_acceptance(
            config,
            candidate_sha="d" * 40,
            policy={},
            expected_benchmark_date="not-a-date",
            started_at=0,
        )


def test_primary_acceptance_requires_signed_exact_bundle_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))
    bundle_hash = "sha256:" + "a" * 64
    weights_hash = "sha256:" + "b" * 64
    body = {
        "state": "signed",
        "publication": {
            "epoch_id": 42,
            "success": True,
            "weights_hash": weights_hash,
            "weight_submission_event_hash": "sha256:" + "c" * 64,
        },
        "compact_submission": {
            "bundle_hash": bundle_hash,
            "weight_result": {"epoch_id": 42, "weights_hash": weights_hash},
        },
        "extrinsic_signature_results": [
            {
                "extrinsic_hash": "0x" + "d" * 64,
                "validator_hotkey": "4" * 48,
            }
        ],
    }
    journal = {**body, "journal_hash": module.sha256_json(body)}
    outputs = iter(
        [
            json.dumps(journal),
            (
                "Authoritative V2 gateway bundle persisted:\n"
                "Authoritative V2 finalized chain state persisted:\n"
            ),
        ]
    )
    monkeypatch.setattr(module, "_ssh", lambda *args, **kwargs: next(outputs))

    accepted = module._primary_finalized(
        config,
        since_epoch_seconds=1,
        epoch_id=42,
        bundle_hash=bundle_hash,
        weights_hash=weights_hash,
        finalized_block=1234,
    )

    assert accepted is not None
    assert accepted["finalized_block"] == 1234
    wrong_body = {
        **body,
        "extrinsic_signature_results": [
            {
                "extrinsic_hash": "0x" + "d" * 64,
                "validator_hotkey": "9" * 48,
            }
        ],
    }
    wrong_journal = {
        **wrong_body,
        "journal_hash": module.sha256_json(wrong_body),
    }
    monkeypatch.setattr(
        module,
        "_ssh",
        lambda *args, **kwargs: json.dumps(wrong_journal),
    )
    assert (
        module._primary_finalized(
            config,
            since_epoch_seconds=1,
            epoch_id=42,
            bundle_hash=bundle_hash,
            weights_hash=weights_hash,
            finalized_block=1234,
        )
        is None
    )
    assert accepted["bundle_hash"] == bundle_hash
    assert accepted["extrinsic_hashes"] == ["0x" + "d" * 64]


def test_auditor_restart_installs_candidate_dependencies_only_when_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))
    commands: list[str] = []
    monkeypatch.setattr(
        module,
        "_ssh",
        lambda host, key, command, timeout: commands.append(command) or "",
    )

    module._restart_auditor(config.auditors[0], "d" * 40)

    command = commands[0]
    assert "previous_commit" in command
    assert "diff --name-only" in command
    assert "pip install -r" in command
    assert config.auditors[0].secret_id in command
    assert "systemctl restart" in command


def test_independent_weight_acceptance_survives_rebenchmark_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    commit = "d" * 40
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))
    oracle = module.validate_historical_oracle(
        json.loads(
            (ROOT / "tests/restart_rehearsal/fixtures/august_9_known_good_v2.json").read_text(
                encoding="utf-8"
            )
        )
    )
    contract = {
        "candidate_sha": commit,
        "historical_oracle_hash": module.sha256_json(oracle),
        "contract_hash": "sha256:" + "1" * 64,
        "policy_commitments": {
            "conditional_icp": {
                "public_total_icps": 10,
                "private_total_icps": 10,
                "conditional_total_icps": 20,
                "public_weak_total": 7,
                "private_weak_total": 3,
            }
        },
    }
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    snapshot = {
        "source_sha": commit,
        "capture_sha": commit,
        "source_host_hash": module.production_database_host_hash(
            "production-db.example"
        ),
        "manifest_hash": "sha256:" + "2" * 64,
        "database": {
            "capture_utc_date": datetime.now(timezone.utc).date().isoformat(),
            "target_rebenchmark_date": datetime.now(timezone.utc).date().isoformat(),
            "latest_completed_benchmark_date": yesterday,
            "current_day_rebenchmark_run_count": 0,
            "current_day_benchmark_bundle_count": 0,
        },
    }
    provider_control_order: list[str] = []
    monkeypatch.setattr(module, "verify_contract_checkout", lambda root, value: contract)
    monkeypatch.setattr(module, "validate_snapshot_manifest", lambda value: snapshot)
    monkeypatch.setattr(
        module,
        "_restart_exact_release",
        lambda *args, **kwargs: provider_control_order.append("restart"),
    )
    monkeypatch.setattr(module, "_restart_auditor", lambda *args: None)
    monkeypatch.setattr(
        module,
        "_dashboard_release_evidence",
        lambda config: {"source_sha": "e" * 40, "status": "active"},
    )
    monkeypatch.setattr(
        module,
        "_pause_staging_provider_work",
        lambda config: provider_control_order.append("pause")
        or {"autoresearch_paused": True, "scoring_paused": True},
    )
    monkeypatch.setattr(
        module,
        "_wait_for_provider_idle_window",
        lambda config: provider_control_order.append("idle")
        or ({}, {"idle_seconds": 901.0}),
    )
    monkeypatch.setattr(
        module,
        "_configure_staging_controls",
        lambda config: provider_control_order.append("resume")
        or {"ok": True},
    )
    monkeypatch.setattr(
        module,
        "_wait_for_post_idle_provider_proof",
        lambda config, *, baseline: provider_control_order.append("proof")
        or {"proof_seconds": 1.0},
    )
    monkeypatch.setattr(
        module,
        "_gateway_json",
        lambda config, path, **kwargs: (
            {"status": "ready", "commit_sha": commit}
            if path == "/health/v2-authority"
            else {"git_commit": commit}
        ),
    )
    monkeypatch.setattr(
        module,
        "_wait_for_rebenchmark_acceptance",
        lambda *args, **kwargs: provider_control_order.append("rebenchmark")
        or (_ for _ in ()).throw(
            module.PhysicalStagingError("rebenchmark failed independently")
        ),
    )
    weights = [
        {
            "epoch_id": epoch,
            "bundle_hash": "sha256:" + str(epoch)[-1] * 64,
            "weights_hash": "sha256:" + "a" * 64,
            "primary_finalized": {"epoch_id": epoch},
            "chain_readback": {
                "readback_hash": "sha256:" + "f" * 64,
            },
        }
        for epoch in (40, 41, 42)
    ]
    monkeypatch.setattr(
        module,
        "_wait_for_canonical_acceptance",
        lambda *args, **kwargs: provider_control_order.append("weights")
        or weights,
    )

    def fake_run(command, **kwargs):
        output = commit + "\n" if "rev-parse" in command else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(module, "_run", fake_run)

    ledger = module.run(
        config,
        commit=commit,
        contract=contract,
        snapshot=snapshot,
        production_db_host="production-db.example",
    )
    stages = {item["stage_id"]: item for item in ledger["stages"]}

    assert ledger["status"] == "failed"
    assert stages["full-rebenchmark-and-assignment"]["status"] == "failed"
    assert stages["canonical-weight-bundles"]["status"] == "passed"
    assert stages["primary-finalization"]["status"] == "passed"
    assert stages["audit-finalization"]["status"] == "passed"
    assert stages["candidate-not-superseded"]["status"] == "passed"
    assert provider_control_order[:5] == [
        "restart",
        "pause",
        "idle",
        "resume",
        "proof",
    ]
    assert set(provider_control_order[5:]) == {"rebenchmark", "weights"}
    ledger_stage_order = [item["stage_id"] for item in ledger["stages"]]
    assert ledger_stage_order.index("exact-paired-restart") < (
        ledger_stage_order.index("staging-control-boundary")
    )
    assert ledger_stage_order.index("staging-control-boundary") < (
        ledger_stage_order.index("provider-transport-post-idle")
    )
    assert ledger_stage_order.index("provider-transport-post-idle") < (
        ledger_stage_order.index("full-rebenchmark-and-assignment")
    )


def test_physical_staging_workflow_is_attestation_and_cleanup_bound() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_run:" in source
    assert "Attested V2 Release" in source
    assert "vars.LEADPOET_PARITY_INFRA_READY == 'true' &&" in source
    assert "LEADPOET_PARITY_ENFORCEMENT_ENABLED == 'true'" in source
    assert "github.event_name == 'workflow_dispatch' ||" in source
    assert "RELEASE_SOURCE_PREFIX:" in source
    assert "PRODUCTION_DB_HOST:" in source
    assert '--production-db-host "$PRODUCTION_DB_HOST"' in source
    assert '--release-prefix "$RELEASE_SOURCE_PREFIX"' in source
    assert "run_physical_v2_staging.py" in source
    assert "--snapshot-archive-version-id" in source
    assert "ObjectLockRetainUntilDate" in source
    assert "cleaned_secret_count\") != 6" in source
    assert "stack_deleted\") is not True" in source
    assert "attested-v2/candidates/$CANDIDATE_SHA" in source
    assert "--prefix attested-v2/releases" in source


def test_physical_staging_release_channel_is_explicit_and_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))
    observed: dict[str, str] = {}

    def fake_run(command, *, env=None, timeout):
        observed.update(env or {})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "_run", fake_run)
    module._restart_exact_release(
        config,
        "a" * 40,
        release_prefix="attested-v2/releases",
    )
    assert observed["LEADPOET_RELEASE_PREFIX"] == "attested-v2/releases"

    with pytest.raises(module.PhysicalStagingError, match="prefix is invalid"):
        module._restart_exact_release(
            config,
            "a" * 40,
            release_prefix="attested-v2/other",
        )


def test_paired_restart_uses_candidate_channel_without_changing_default() -> None:
    source = (ROOT / "scripts" / "restart_attested_release_local.sh").read_text(
        encoding="utf-8"
    )

    assert 'RELEASE_PREFIX="${LEADPOET_RELEASE_PREFIX:-attested-v2/releases}"' in source
    assert "attested-v2/releases|attested-v2/candidates" in source
    assert "GATEWAY_V2_RELEASE_PREFIX='$RELEASE_PREFIX'" in source
    assert "VALIDATOR_V2_RELEASE_PREFIX='$RELEASE_PREFIX'" in source
