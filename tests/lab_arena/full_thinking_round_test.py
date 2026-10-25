"""Future full-thinking rounds freeze approved routes and cost limits atomically."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts, reasoning_capture, source_bundle
from lab_arena.service import ArenaService, RoundDefaults, ServiceConfig, ServiceError
from lab_arena.promotion import GitPromoter
from tests.lab_arena.test_lab_arena_promotion_git import repository, source, git
from tests.lab_arena.deepline_round_runtime_test import _service
from tests.lab_arena.test_deepline_catalog import frozen


OPEN = datetime(2026, 10, 23, tzinfo=timezone.utc)
DIGEST = "sha256:" + "a" * 64
BASELINE_URL = "https://github.com/leadpoet/leadpoet-sales-agent/archive/" + "b" * 40 + ".tar.gz"


def approved_registry():
    model = "openai/gpt-6-luna"
    return {
        "schema_version": 1,
        "baseline_route": {"model_id": model, "api": "openrouter.chat"},
        "approved_routes": [{
            "model_id": model,
            "model_version": model,
            "provider": "openai",
            "provider_name": "OpenAI",
            "api": "openrouter.chat",
            "mode": {"effort": "high"},
            "parser_version": reasoning_capture.PARSER_VERSION,
            "semantic_class": "full_generated_thinking",
            "full_text_fields": ["reasoning_content"],
            "max_output_tokens": 8192,
            "answer_headroom_tokens": 1024,
            "context_tokens": 128000,
            "price_ceiling": {
                "prompt": "0.000001", "completion": "0.000002",
                "internal_reasoning": "0.000002", "request": "0",
            },
            "evidence": {
                "documentation_url": "https://example.org/model/route",
                "probe_id": "probe-2026-10-23",
                "verified_at": "2026-10-23T00:00:00Z",
                "probe_manifest_sha256": "b" * 64,
            },
            "allowed_fallback_routes": [],
            "status": "approved",
        }],
    }


def opted_in(registry=None):
    service = _service(lambda **_: frozen())
    service._config.defaults.flat_per_icp_cost_policy = True
    service._config.defaults.per_icp_cost_policy = False
    service._config.defaults.reasoning_capture_from = "2026-10-01T00:00:00Z"
    service._config.defaults.reasoning_baseline_source_sha256 = DIGEST
    service._config.defaults.reasoning_baseline_source_url = BASELINE_URL
    service._config.reasoning_registry_source = lambda: registry if registry is not None else approved_registry()
    return service


def test_future_round_freezes_approved_routes_and_six_dollar_limits():
    registry = approved_registry()
    service = opted_in(registry)
    config = service.create_round(OPEN)
    assert config["reasoning_capture_policy"] == reasoning_capture.POLICY
    assert config["reasoning_route_registry"] == registry
    assert config["reasoning_baseline_source_sha256"] == DIGEST
    assert config["baseline_source_url"] == BASELINE_URL
    assert config["execution_icp_cap_microusd"] == 6_000_000
    assert config["cost_per_company_microusd"] == 6_000_000
    assert config["execution_cap_microusd"] == 6_000_000 * contracts.benchmark_icp_count(config)
    assert contracts.validate_round_configuration(json.loads(json.dumps(config))) == config
    registry["approved_routes"][0]["mode"]["effort"] = "low"
    assert config["reasoning_route_registry"]["approved_routes"][0]["mode"]["effort"] == "high"
    assert ArenaService._public_round_output_policy(config)["reasoning_capture_policy"] == reasoning_capture.POLICY


def test_round_before_activation_keeps_four_dollar_limits():
    service = opted_in()
    service._config.defaults.reasoning_capture_from = "2026-11-01T00:00:00Z"
    service._config.reasoning_registry_source = lambda: (_ for _ in ()).throw(AssertionError("must not load"))
    config = service.create_round(OPEN)
    assert "reasoning_capture_policy" not in config
    assert "reasoning_route_registry" not in config
    assert config["execution_icp_cap_microusd"] == 4_000_000
    assert config["cost_per_company_microusd"] == 4_000_000


@pytest.mark.parametrize("registry,digest", [
    ({"schema_version": 1, "approved_routes": [], "baseline_route": None}, DIGEST),
    ({"bad": "catalog"}, DIGEST),
    (approved_registry(), None),
    (approved_registry(), "malformed-digest"),
])
def test_unready_activation_keeps_round_creation_and_legacy_costs(registry, digest, caplog):
    service = opted_in(registry)
    service._config.defaults.reasoning_baseline_source_sha256 = digest
    config = service.create_round(OPEN)
    assert "reasoning_capture_policy" not in config
    assert config["execution_icp_cap_microusd"] == 4_000_000
    assert config["cost_per_company_microusd"] == 4_000_000
    assert "arena_full_thinking_not_ready" in caplog.text
    assert ArenaService._public_round_output_policy(config) == {
        **ArenaService._public_round_output_policy({
            key: value for key, value in config.items()
            if key not in ("reasoning_activation_status", "reasoning_activation_reason")
        }),
        "reasoning_activation_status": "not_ready",
        "reasoning_activation_reason": config["reasoning_activation_reason"],
    }


def test_capture_registry_exception_cannot_stop_round_creation():
    service = opted_in()
    service._config.reasoning_registry_source = lambda: (_ for _ in ()).throw(RuntimeError("capture metadata down"))
    config = service.create_round(OPEN)
    assert config["reasoning_activation_status"] == "not_ready"
    assert config["reasoning_activation_reason"] == "registry_unavailable"
    assert "reasoning_capture_policy" not in config


@pytest.mark.parametrize("source_url", [None, "https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz",
                                        "https://example.org/archive/" + "b" * 40 + ".tar.gz"])
def test_unpinned_baseline_source_marks_future_activation_not_ready(source_url):
    service = opted_in()
    service._config.defaults.reasoning_baseline_source_url = source_url
    config = service.create_round(OPEN)
    assert config["reasoning_activation_status"] == "not_ready"
    assert config["reasoning_activation_reason"] == "baseline_unready"
    assert config["execution_icp_cap_microusd"] == 4_000_000
    assert "reasoning_capture_policy" not in config


def test_malformed_baseline_digest_cannot_stop_service_validation_or_round_creation():
    ServiceConfig(
        mode="shadow", store=None, object_store=None, signer=None, chain=None,
        verify_signature=lambda *_args: True, daily_icp_source=lambda **_kwargs: {},
        banned_hotkeys_source=lambda: (), broker_factory=lambda *_args: None,
        defaults=RoundDefaults(reasoning_capture_from="2026-10-01T00:00:00Z",
                               reasoning_baseline_source_sha256="malformed-digest"),
    )
    service = opted_in()
    service._config.defaults.reasoning_baseline_source_sha256 = "malformed-digest"
    config = service.create_round(OPEN)
    assert config["reasoning_activation_status"] == "not_ready"
    assert config["reasoning_activation_reason"] == "baseline_unready"


@pytest.mark.parametrize("field,value", [
    ("execution_icp_cap_microusd", 4_000_000),
    ("cost_per_company_microusd", 4_000_000),
    ("execution_cap_microusd", 80_000_000),
    ("reasoning_baseline_source_sha256", None),
    ("baseline_source_url", "https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz"),
    ("reasoning_route_registry", {"schema_version": 1, "approved_routes": [], "baseline_route": None}),
])
def test_full_round_rejects_partial_or_changed_snapshot(field, value):
    config = opted_in().create_round(OPEN)
    changed = deepcopy(config)
    changed[field] = value
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(changed)


def test_run_context_uses_frozen_route_snapshot():
    config = opted_in().create_round(OPEN)
    service = object.__new__(ArenaService)
    run = {"run_id": "r1", "round_id": config["round_id"], "assignment_id": "a1", "attempt": 1,
           "icp_position": 0, "miner_hotkey": "5" * 48, "submission_id": "s1", "stage": 1, "kind": "execute"}
    service._store = SimpleNamespace(get_run=lambda _run_id: run)
    service._require_round_ownership = lambda _round_id: None
    service._hot_round = lambda _round_id: {"round_id": config["round_id"], "configuration_doc": config}
    _run, context = service._run_context("r1", "lease-token")
    assert context.reasoning_capture_policy == reasoning_capture.POLICY
    assert context.reasoning_route_registry == config["reasoning_route_registry"]


def test_baseline_archive_mismatch_blocks_registration(monkeypatch):
    config = opted_in().create_round(OPEN)
    payload = b"different archive"
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(mode="shadow", defaults=SimpleNamespace(
        baseline_source_url=config["baseline_source_url"]),
        baseline_source_fetcher=lambda *_args: payload)
    service._objects = SimpleNamespace(get_bounded=lambda *_args: payload)
    service._store = SimpleNamespace(get_submission=lambda _sid: None,
                                     register_submission=lambda *_args: pytest.fail("source registered"))
    monkeypatch.setattr(source_bundle, "validate_source_archive", lambda _payload: {"source_size_bytes": len(payload)})
    with pytest.raises(ServiceError, match="reasoning_baseline_source_mismatch"):
        service._initial_baseline({"round_id": config["round_id"], "configuration_doc": config})


def test_full_round_fetches_pinned_baseline_url_and_rejects_wrong_digest_before_storage(monkeypatch):
    config = opted_in().create_round(OPEN)
    config["mode"] = "live"
    seen = []
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(mode="live", defaults=SimpleNamespace(
        baseline_source_url="https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz"),
        baseline_source_fetcher=lambda url, _limit: seen.append(url) or b"different archive")
    service._objects = SimpleNamespace(
        get_bounded=lambda *_args: (_ for _ in ()).throw(FileNotFoundError()),
        put=lambda *_args: pytest.fail("wrong archive stored"),
    )
    service._store = SimpleNamespace(get_submission=lambda _sid: None,
                                     register_submission=lambda *_args: pytest.fail("source registered"))
    service._pending_promotion_blocks = lambda: False
    monkeypatch.setattr(source_bundle, "validate_source_archive", lambda payload: {"source_size_bytes": len(payload)})
    with pytest.raises(ServiceError, match="reasoning_baseline_source_mismatch"):
        service._initial_baseline({"round_id": config["round_id"], "configuration_doc": config})
    assert seen == [BASELINE_URL]


@pytest.mark.parametrize("status", ["accepted", "frozen"])
@pytest.mark.parametrize("stored", ["matching", "mismatch", "unavailable"])
def test_reused_full_thinking_baseline_is_verified_before_freeze(status, stored):
    config = opted_in().create_round(OPEN)
    payload = b"tested immutable baseline archive"
    config["reasoning_baseline_source_sha256"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    round_id = config["round_id"]
    source_ref = "arena/%s/sources/baseline-%s.tar.gz" % (
        round_id, round_id.removeprefix("arena-"))
    row = {
        "round_id": round_id, "submission_id": "baseline-" + round_id.removeprefix("arena-"),
        "miner_hotkey": config["baseline_hotkey"], "is_king": True,
        "source_ref": source_ref, "source_size_bytes": len(payload), "status": status,
    }
    round_row = {"round_id": round_id, "configuration_doc": config,
                 "champion_funding_frozen": True}
    reads = []

    def get_bounded(ref, _limit):
        reads.append(ref)
        if stored == "unavailable":
            raise FileNotFoundError(ref)
        return payload if stored == "matching" else b"different archive"

    service = object.__new__(ArenaService)
    service._objects = SimpleNamespace(get_bounded=get_bounded)
    service._config = SimpleNamespace(defaults=SimpleNamespace(
        baseline_source_url=config["baseline_source_url"]))
    service._round = lambda _round_id: round_row
    service._store = SimpleNamespace(
        get_submission=lambda _sid: row,
        freeze_champion_funding=lambda _rid: {"status": "existing"},
        list_submissions=lambda _rid, *, status: [row] if status == row["status"] else [],
        update_submission=lambda *_args: pytest.fail("unexpected participant mutation"),
    )
    if stored == "matching":
        result = (service._initial_baseline(round_row) if status == "accepted"
                  else service.freeze_participants(round_id))
        assert result == (row if status == "accepted" else [ArenaService._participant(row, is_king=True)])
    else:
        code = ("baseline_source_not_ready" if stored == "unavailable"
                else "reasoning_baseline_source_mismatch")
        with pytest.raises(ServiceError, match=code):
            if status == "accepted":
                service._initial_baseline(round_row)
            else:
                service.freeze_participants(round_id)
    assert reads == [source_ref]


def test_open_next_round_adopts_champion_promoted_before_baseline_freeze(tmp_path):
    remote, _, old_commit = repository(tmp_path)
    original, champion = source("original"), source("champion")
    service = opted_in()
    old_url = "https://github.com/leadpoet/leadpoet-sales-agent/archive/" + old_commit + ".tar.gz"
    service._config.defaults.reasoning_baseline_source_url = old_url
    service._config.defaults.reasoning_baseline_source_sha256 = (
        "sha256:" + hashlib.sha256(original).hexdigest()
    )
    next_config = service.create_round(datetime(2026, 10, 24, tzinfo=timezone.utc))
    round_row = {"round_id": next_config["round_id"], "configuration_doc": next_config,
                 "champion_funding_frozen": False}
    assert next_config["baseline_source_url"] == old_url

    promoter = GitPromoter(str(remote), tmp_path / "promotion-cache")
    plan = promoter.prepare(champion, round_id="arena-2026-10-23", submission_id="winner",
                            timestamp="2026-10-23T00:00:00Z")
    assert promoter.publish(champion, plan=plan, round_id="arena-2026-10-23",
                            submission_id="winner") == plan["commit"]
    assert git(remote, "rev-parse", "lab") == plan["commit"]
    champion_url = "https://github.com/leadpoet/leadpoet-sales-agent/archive/" + plan["commit"] + ".tar.gz"
    reads, submissions, objects = [], {}, {}

    class Store:
        def latest_promoted_champion(self, **_scope):
            return {"round_id": "arena-2026-10-23", "winner_submission_id": "winner",
                    "commit": plan["commit"],
                    "reasoning_capture_policy": reasoning_capture.POLICY}

        def freeze_full_thinking_baseline(self, round_id, **selection):
            assert round_id == next_config["round_id"]
            assert selection["expected_promotion_round_id"] == "arena-2026-10-23"
            assert selection["expected_winner_submission_id"] == "winner"
            assert selection["source_url"] == champion_url
            assert selection["source_sha256"] == "sha256:" + hashlib.sha256(champion).hexdigest()
            round_row["configuration_doc"] = {
                **round_row["configuration_doc"],
                "baseline_source_url": selection["source_url"],
                "reasoning_baseline_source_sha256": selection["source_sha256"],
            }
            round_row["champion_funding_frozen"] = True
            round_row["champion_submission_id"] = "winner"
            return {"status": "frozen"}

        def list_submissions(self, _round_id, *, status):
            return [row for row in submissions.values() if row["status"] == status]

        def get_submission(self, submission_id):
            return submissions.get(submission_id)

        def register_submission(self, round_id, submission_id, hotkey, fields):
            submissions[submission_id] = {
                "round_id": round_id, "submission_id": submission_id,
                "miner_hotkey": hotkey, "status": "accepted", **fields,
            }
            return {"status": "registered"}

        def update_submission(self, _round_id, submission_id, old, new, *_args):
            assert submissions[submission_id]["status"] == old
            submissions[submission_id]["status"] = new
            return {"status": "ok"}

    service._store = Store()
    service._round = lambda _round_id: round_row
    service._objects = SimpleNamespace(
        get_bounded=lambda ref, _limit: objects[ref],
        put=lambda ref, payload: objects.__setitem__(ref, payload),
    )
    service._config.baseline_source_fetcher = lambda url, _limit: (
        reads.append(url) or (champion if url == champion_url else original)
    )
    service._pending_promotion_blocks = lambda: False

    participants = service.freeze_participants(next_config["round_id"])
    assert len(participants) == 1 and participants[0]["is_king"] is True
    assert round_row["champion_submission_id"] == "winner"
    assert reads == [champion_url, champion_url]
    assert objects[participants[0]["source_ref"]] == champion
    assert service.freeze_participants(next_config["round_id"]) == participants
    assert reads == [champion_url, champion_url]


def test_new_round_freezes_eighty_percent_coverage_and_publishes_it():
    config = opted_in().create_round(OPEN)
    assert config["training_minimum_icp_coverage_percent"] == 80
    assert ArenaService._public_round_output_policy(config)["training_minimum_icp_coverage_percent"] == 80
    assert contracts.validate_round_configuration(json.loads(json.dumps(config))) == config
    assert config["training_maximum_missing_reasoning_percent"] == 1
    assert ArenaService._public_round_output_policy(config)["training_maximum_missing_reasoning_percent"] == 1


def test_missing_reasoning_allowance_is_frozen_and_legacy_absence_is_zero():
    service = opted_in()
    service._config.defaults.training_maximum_missing_reasoning_percent = 2
    config = service.create_round(OPEN)
    assert config["training_maximum_missing_reasoning_percent"] == 2
    service._config.defaults.training_maximum_missing_reasoning_percent = 1
    service._store.create_round = lambda *_args: {"status": "existing"}
    service._round = lambda _round_id: {"configuration_doc": deepcopy(config)}
    assert service.create_round(OPEN)["training_maximum_missing_reasoning_percent"] == 2
    legacy = dict(config)
    del legacy["training_maximum_missing_reasoning_percent"]
    assert "training_maximum_missing_reasoning_percent" not in contracts.validate_round_configuration(legacy)
    assert ArenaService._public_round_output_policy(legacy)["training_maximum_missing_reasoning_percent"] == 0


@pytest.mark.parametrize("value", [None, True, False, "1", 1.0, -1, 101])
def test_missing_reasoning_allowance_rejects_invalid_values(value):
    config = opted_in().create_round(OPEN)
    config["training_maximum_missing_reasoning_percent"] = value
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(config)


def test_missing_reasoning_allowance_cannot_be_attached_to_policy_off_round():
    service = opted_in()
    service._config.defaults.reasoning_capture_from = "2026-11-01T00:00:00Z"
    config = service.create_round(OPEN)
    assert "training_maximum_missing_reasoning_percent" not in config
    config["training_maximum_missing_reasoning_percent"] = 1
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(config)


def test_coverage_override_is_frozen_and_legacy_absence_stays_absent():
    service = opted_in()
    service._config.defaults.training_minimum_icp_coverage_percent = 90
    config = service.create_round(OPEN)
    assert config["training_minimum_icp_coverage_percent"] == 90
    service._config.defaults.training_minimum_icp_coverage_percent = 80
    service._store.create_round = lambda *_args: {"status": "existing"}
    service._round = lambda _round_id: {"configuration_doc": deepcopy(config)}
    assert service.create_round(OPEN)["training_minimum_icp_coverage_percent"] == 90
    legacy = dict(config)
    del legacy["training_minimum_icp_coverage_percent"]
    assert "training_minimum_icp_coverage_percent" not in contracts.validate_round_configuration(legacy)
    assert ArenaService._public_round_output_policy(legacy)["training_minimum_icp_coverage_percent"] == 100


@pytest.mark.parametrize("value", [None, True, False, "80", 80.0, 0, -1, 101])
def test_frozen_coverage_rejects_invalid_values(value):
    config = opted_in().create_round(OPEN)
    config["training_minimum_icp_coverage_percent"] = value
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(config)


def test_coverage_setting_cannot_be_attached_to_policy_off_round():
    service = opted_in()
    service._config.defaults.reasoning_capture_from = "2026-11-01T00:00:00Z"
    config = service.create_round(OPEN)
    assert "training_minimum_icp_coverage_percent" not in config
    config["training_minimum_icp_coverage_percent"] = 80
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(config)
