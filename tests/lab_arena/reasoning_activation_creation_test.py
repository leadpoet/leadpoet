"""Mandatory reasoning activation must retry before any round is persisted."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

import pytest

from lab_arena import contracts, reasoning_capture
from lab_arena.driver import drive_once
from lab_arena.service import ArenaService, ServiceError
from tests.lab_arena.full_thinking_round_test import approved_registry, opted_in


CUTOFF = datetime(2026, 10, 29, tzinfo=timezone.utc)


class Store:
    def __init__(self):
        self.rows = {}
        self.writes = []

    def get_round(self, round_id):
        return deepcopy(self.rows.get(round_id))

    def create_round(self, round_id, configuration):
        self.writes.append(round_id)
        if round_id in self.rows:
            return {"status": "existing"}
        self.rows[round_id] = {"round_id": round_id, "status": "open",
                              "configuration_doc": deepcopy(configuration)}
        return {"status": "created"}


def service_and_store():
    service = opted_in()
    store = service._store = Store()
    service._require_round_ownership = ArenaService._require_round_ownership.__get__(service)
    return service, store


def broken_registry():
    raise RuntimeError("registry unavailable")


@pytest.mark.parametrize("failure,reason", [
    ("registry_missing", "registry_unavailable"),
    ("registry_invalid", "registry_unavailable"),
    ("registry_empty", "registry_unavailable"),
    ("registry_exception", "registry_unavailable"),
    ("baseline_missing", "baseline_unready"),
    ("baseline_invalid", "baseline_unready"),
    ("baseline_url", "baseline_unready"),
    ("flat_policy", "flat_policy_required"),
])
def test_activation_prerequisite_failure_is_retryable_without_round_or_provider_write(monkeypatch, failure, reason):
    service, store = service_and_store()
    if failure == "registry_missing":
        service._config.reasoning_registry_source = None
        monkeypatch.setattr(reasoning_capture, "load_registry", lambda: None)
    elif failure.startswith("registry_"):
        value = {} if failure == "registry_invalid" else {"schema_version": 1, "approved_routes": [], "baseline_route": None}
        service._config.reasoning_registry_source = broken_registry if failure == "registry_exception" else lambda: value
    elif failure == "baseline_missing":
        service._config.defaults.reasoning_baseline_source_sha256 = None
    elif failure == "baseline_invalid":
        service._config.defaults.reasoning_baseline_source_sha256 = "malformed"
    elif failure == "baseline_url":
        service._config.defaults.reasoning_baseline_source_url = "https://example.org/unpinned.tar.gz"
    else:
        service._config.defaults.flat_per_icp_cost_policy = False
    # Catalog discovery and provider/account creation must not run on rejection.
    service._config.deepline_catalog_source = lambda **_: pytest.fail("catalog read after failed activation")
    service._config.broker_factory = lambda *_: pytest.fail("provider dispatch")
    with pytest.raises(ServiceError, match=reason) as error:
        service.create_round(CUTOFF)
    assert error.value.status == 503
    assert store.rows == {} and store.writes == []


@pytest.mark.parametrize("strict", [False, True])
def test_idempotent_creation_returns_frozen_legacy_or_strict_without_live_prerequisites(strict):
    service, store = service_and_store()
    cutoff = CUTOFF if strict else datetime(2026, 10, 27, tzinfo=timezone.utc)
    if not strict:
        service._config.defaults.reasoning_capture_from = "2026-10-28T00:00:00Z"
    original = service.create_round(cutoff)
    frozen_hash = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    service._config.reasoning_registry_source = broken_registry
    service._config.defaults.reasoning_baseline_source_sha256 = None
    service._config.defaults.flat_per_icp_cost_policy = False
    service.runner_settings = lambda: pytest.fail("must reuse frozen settings")
    service._config.deepline_catalog_source = lambda **_: pytest.fail("must reuse frozen catalog")
    result = service.create_round(cutoff)
    assert hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest() == frozen_hash
    assert len(store.writes) == 1
    assert (result.get("reasoning_capture_policy") == reasoning_capture.POLICY) is strict


@pytest.mark.parametrize("race", [False, True])
@pytest.mark.parametrize("invalid,code", [
    ("not_ready", "reasoning_capture_activation_not_ready"),
    ("mode", "round_mode_mismatch"),
    ("network", "round_network_mismatch"),
])
def test_existing_and_create_race_validate_frozen_scope_and_reject_explicit_not_ready(race, invalid, code):
    service, store = service_and_store()
    config = service.create_round(CUTOFF)
    row = store.rows[config["round_id"]]
    if invalid == "not_ready":
        for key in ("reasoning_capture_policy", "reasoning_route_registry", "reasoning_baseline_source_sha256",
                    "training_minimum_icp_coverage_percent", "training_maximum_missing_reasoning_percent"):
            row["configuration_doc"].pop(key, None)
        row["configuration_doc"].update(reasoning_activation_status="not_ready", reasoning_activation_reason="registry_unavailable")
    elif invalid == "mode":
        row["configuration_doc"]["mode"] = "shadow"
    else:
        row["configuration_doc"]["netuid"] = 999
    before = deepcopy(store.rows)
    store.writes.clear()
    if race:
        real_get = store.get_round
        reads = []
        def race_get(round_id):
            reads.append(round_id)
            return None if len(reads) == 1 else real_get(round_id)
        store.get_round = race_get
    with pytest.raises(ServiceError, match=code) as error:
        service.create_round(CUTOFF)
    assert error.value.status == (503 if invalid == "not_ready" else 409)
    assert store.rows == before
    assert store.writes == ([config["round_id"]] if race else [])


def test_idempotent_create_checks_ownership_before_read():
    service, store = service_and_store()
    service._config.pinned_round_id = "different-round"
    store.get_round = lambda _: pytest.fail("ownership must precede read")
    with pytest.raises(ServiceError, match="round_scope_mismatch"):
        service.create_round(CUTOFF)


def test_actual_daily_driver_retries_same_due_round_after_registry_recovers():
    service, store = service_and_store()
    service._config.defaults.daily_cutoff_hour_utc = 0
    service._config.defaults.min_submission_hours = 6
    service.now = lambda: datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
    service.open_round = lambda **_: next(iter(store.rows.values()), None)
    events = []
    service.promote_pending_baselines = lambda: events.append("promotion") or {"promoted": 0}
    service.active_rounds = lambda: []
    service.activate_pending_rewards = lambda: events.append("rewards") or {"activated": 0}
    service.reconcile_closed_provider_costs = lambda: events.append("costs") or {"status": "none"}
    service._config.reasoning_registry_source = broken_registry
    assert "failed ensure_daily_round: ServiceError" in drive_once(service)
    assert store.rows == {} and store.writes == []
    assert events == ["promotion", "rewards", "costs"]
    service._config.reasoning_registry_source = approved_registry
    assert "created arena-2026-10-29" in drive_once(service)
    config = store.rows["arena-2026-10-29"]["configuration_doc"]
    assert config["reasoning_capture_policy"] == reasoning_capture.POLICY
    assert contracts.validate_round_configuration(json.loads(json.dumps(config))) == config
    assert "reasoning_activation_status" not in config
    assert store.writes == ["arena-2026-10-29"]
