"""Frozen research catalog and cost-only quota cross the actual runtime seams."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts, deepline_catalog, lab_arena_checkpoint, scoring, shim
from lab_arena.service import ArenaService, RoundDefaults
from tests.lab_arena.test_deepline_catalog import frozen, row


def _service(source=None):
    service = object.__new__(ArenaService)
    digest = "sha256:" + "a" * 64
    defaults = RoundDefaults(
        runner_hotkeys=("5" * 48,), baseline_hotkey="5" * 48,
        runner_capacity_slots={"5" * 48: 251},
        scorer_image_digest=digest,
        scorer_image_reference="registry.example/scorer@" + digest,
        per_icp_cost_policy=True, integrity_from="2000-01-01T00:00:00Z",
    )
    service._config = SimpleNamespace(
        defaults=defaults, mode="live", network_name="finney", netuid=71,
        deepline_catalog_source=source,
    )
    service._scorer_policy = scoring.build_scorer_policy()
    service.runner_settings = lambda: (["5" * 48], [])
    service._require_round_ownership = lambda _round_id: None
    service._require_integrity_schema = lambda: None
    service._store = SimpleNamespace(create_round=lambda *_args: {"status": "created"})
    return service


def _round(source=None):
    return _service(source).create_round(datetime(2026, 10, 7, tzinfo=timezone.utc))


def test_new_round_freezes_catalog_prices_and_cost_only_deepline():
    snapshot = frozen()
    calls = []
    def source(**kwargs):
        calls.append(kwargs)
        return snapshot
    config = _round(source)
    assert config["deepline_catalog"] == snapshot
    assert config["call_quotas"] == {"scrapingdog": 200, "deepline": 0, "openrouter": 2000}
    assert config["scoring_call_quotas"]["deepline"] == 0
    assert config["execution_icp_cap_microusd"] == 4_000_000
    assert config["cost_per_company_microusd"] == 800_000
    assert calls == [{"allow_people": False}]
    assert contracts.validate_round_configuration(json.loads(json.dumps(config))) == config
    # A restarted gateway cannot replace the frozen snapshot from newer discovery.
    _service(lambda **_: (_ for _ in ()).throw(AssertionError("must not refetch")))._freeze_deepline_catalog(config)
    assert config["deepline_catalog"] == snapshot


def test_legacy_round_retains_its_quota_and_new_zero_requires_catalog():
    config = _round()
    assert config["call_quotas"]["deepline"] == 200
    config["call_quotas"]["deepline"] = 0
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(config)


def test_catalog_cannot_enable_people_for_company_round_or_change_hashed_price():
    config = _round(lambda **_: frozen())
    config["deepline_catalog"] = frozen(allow_people=True)
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(config)
    config["deepline_catalog"] = frozen()
    config["deepline_catalog"]["tools"][0]["pricing"]["usd_per_unit"] = 0
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(config)


def test_unlimited_snapshot_is_explicit_only_for_deepline():
    document = {"schema_version": lab_arena_checkpoint.QUOTA_SNAPSHOT_SCHEMA_VERSION,
                "providers": {p: {"limit": 200, "used": 1, "remaining": 199, "inflight": 0}
                              for p in contracts.PROVIDERS}}
    document["providers"]["deepline"] = {"limit": 0, "used": 500, "remaining": None, "inflight": 2}
    assert lab_arena_checkpoint.validate_quota_snapshot(document) == document
    for provider, values in (("openrouter", {"limit": 0, "remaining": None}),
                             ("deepline", {"remaining": 0}),
                             ("deepline", {"inflight": 501})):
        broken = deepcopy(document)
        broken["providers"][provider].update(values)
        with pytest.raises(lab_arena_checkpoint.QuotaUnavailable):
            lab_arena_checkpoint.validate_quota_snapshot(broken)


def test_shim_worker_uses_frozen_catalog_and_rejects_unknown_tool():
    snapshot = frozen(row("new_vendor_company_search"))
    frame = {"schema_version": shim.OPERATION_FRAME_SCHEMA_VERSION,
             "operation_id": "deepline.execute", "timeout_ms": 5000,
             "parameters": {"tool": "new_vendor_company_search", "payload": {"query": "company"}}}
    operation, parameters, timeout = shim.decode_operation_frame(
        json.dumps(frame).encode(), deepline_catalog=snapshot)
    assert (operation, parameters["tool"], timeout) == ("deepline.execute", "new_vendor_company_search", 5000)
    with pytest.raises(Exception):
        shim.decode_operation_frame(json.dumps(frame).encode())
    frame["parameters"]["tool"] = "unknown_company_search"
    with pytest.raises(Exception):
        shim.decode_operation_frame(json.dumps(frame).encode(), deepline_catalog=snapshot)


def test_gateway_context_uses_its_round_catalog_not_model_input():
    snapshot = frozen()
    service = _service()
    service._store = SimpleNamespace(get_run=lambda _: {
        "run_id": "run", "round_id": "round", "assignment_id": "assignment",
        "attempt": 1, "icp_position": 2, "miner_hotkey": "miner",
        "submission_id": "submission", "stage": 1, "kind": "execute"})
    service._hot_round = lambda _: {"round_id": "round", "configuration_doc": {"deepline_catalog": snapshot}}
    _, context = service._run_context("run", "lease-secret")
    assert context.deepline_catalog == snapshot


def test_provider_observations_survive_more_than_former_call_limit():
    from lab_arena import provider_observations
    from tests.lab_arena.provider_observations_test import _rows, _company, RUN_ID
    rows = [{"entry_id": i, "call_identity": "other-%s" % i} for i in range(1000)]
    rows += [dict(r, entry_id=1000 + i) for i, r in enumerate(_rows())]
    reads = []
    class Store:
        def list_ledger(self, **kwargs):
            reads.append(kwargs)
            cursor = kwargs.get("after_entry_id")
            return [r for r in rows if cursor is None or r["entry_id"] > cursor][:kwargs["limit"]]
    result = provider_observations.resolve_observations(
        Store(), {"run_id": RUN_ID, "status": "accepted"}, [_company()], "2026-09-25")
    assert len(result) == 1
    assert len(reads) == 2
    assert reads[1]["after_entry_id"] == 999
