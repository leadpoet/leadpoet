"""A fresh two-ICP shadow proof cannot reuse outputs or private ICPs."""

import copy
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts
from scripts import verify_arena_parallel_round as verification
from tests.test_verify_arena_parallel_round import ROUND_ID, Service

SOURCE = "arena-2026-10-15"
TARGET = "arena-2026-10-16-trainingproof"


def source_fixture():
    row = {
        "round_id": SOURCE, "status": "published", "benchmark_ref": "source-benchmark",
        "configuration_doc": {"network_name": "finney", "netuid": 71, "stage_1_icp_count": 2, "stage_2_icp_count": 1},
    }
    document = {"schema_version": "leadpoet.lab_arena.benchmark.v1", "round_id": SOURCE,
                "icps": [{"icp_id": "original-%d" % index, "prompt": "complete ICP %d" % index} for index in range(3)]}
    public = {"round_id": SOURCE, "icps": [{"icp_position": 0}, {"icp_position": 2}]}
    reads = []
    def read(ref, maximum):
        reads.append(ref)
        assert ref == "source-benchmark"
        raw = contracts.canonical_json(document).encode()
        assert len(raw) < maximum
        return raw
    built = SimpleNamespace(
        config=SimpleNamespace(network_name="finney", netuid=71),
        store=SimpleNamespace(get_round=lambda name: row if name == SOURCE else None),
        _objects=SimpleNamespace(get_bounded=read), public_benchmark=lambda _name: public,
    )
    return built, row, document, public, reads


def test_fresh_selection_reads_only_public_benchmark_and_reports_hashes():
    built, _row, document, _public, reads = source_fixture()
    result = verification._fresh_source_icps(built, SOURCE, TARGET, [2, 0])
    assert reads == ["source-benchmark"]
    assert [item["prompt"] for item in result["icps"]] == [document["icps"][index]["prompt"] for index in (2, 0)]
    assert [item["icp_id"] for item in result["icps"]] == [TARGET + ":fresh:0", TARGET + ":fresh:1"]
    assert result["evidence"]["saved_outputs_reused"] is False
    assert result["evidence"]["mode"] == "fresh_sourcing"
    assert "prompt" not in json.dumps(result["evidence"])


@pytest.mark.parametrize("positions", [[0], [0, 0], [0, 1], [0, 3], [-1, 0], [True, 0], [0, 2, 1]])
def test_fresh_selection_refuses_private_or_invalid_positions(positions):
    built, *_ = source_fixture()
    with pytest.raises(verification.VerificationError):
        verification._fresh_source_icps(built, SOURCE, TARGET, positions)


@pytest.mark.parametrize("change", ["unpublished", "network", "netuid", "benchmark_round", "benchmark_schema", "benchmark_count", "public_round"])
def test_fresh_selection_checks_published_network_and_benchmark_identity(change):
    built, row, document, public, _reads = source_fixture()
    if change == "unpublished": row["status"] = "stage1_execution"
    elif change == "network": row["configuration_doc"]["network_name"] = "test"
    elif change == "netuid": row["configuration_doc"]["netuid"] = 401
    elif change == "benchmark_round": document["round_id"] = TARGET
    elif change == "benchmark_schema": document["schema_version"] = "other"
    elif change == "benchmark_count": document["icps"].pop()
    elif change == "public_round": public["round_id"] = TARGET
    with pytest.raises(verification.VerificationError):
        verification._fresh_source_icps(built, SOURCE, TARGET, [0, 2])


@pytest.mark.parametrize("flags", [
    {"replay_published_round": SOURCE}, {"replay_active_frozen_round": SOURCE},
    {"replay_assignments": ["source:0"]}, {"replay_historical": True},
    {"replay_miner_score_route": True},
])
def test_fresh_flags_refuse_replay_before_service_construction(flags):
    with pytest.raises(verification.VerificationError, match="cannot be combined"):
        verification._build_pinned_service(TARGET, published_icp_round=SOURCE, icp_positions=[0, 2], **flags)


def test_fresh_builder_preserves_runtime_dependencies_and_has_no_replay(monkeypatch):
    from lab_arena import api, service, wiring
    from lab_arena.service import RoundDefaults, ServiceConfig

    built, *_ = source_fixture()
    original_source = object()
    built.config = ServiceConfig(
        mode="live", store=built.store, object_store=built._objects, signer=None,
        chain=object(), verify_signature=lambda *_: True,
        daily_icp_source=lambda **_: None, banned_hotkeys_source=lambda: (),
        broker_factory=lambda *_: None,
        defaults=RoundDefaults(runner_hotkeys=("existing-runner",)),
        baseline_source_fetcher=original_source,
    )
    modes = []
    def build(mode):
        modes.append(mode)
        return built, None
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(wiring, "build_service_from_environment", build)
    monkeypatch.setattr(service, "ArenaService", lambda config: SimpleNamespace(config=config))
    monkeypatch.setattr(api, "create_app", lambda _service: "app")
    result, app = verification._build_pinned_service(TARGET, published_icp_round=SOURCE, icp_positions=[0, 2])
    assert app == "app" and modes == ["live"]
    assert result.config.defaults.benchmark_icp_count == 2
    assert result.config.defaults.runner_slot_ceiling == 1
    assert result.config.defaults.runner_hotkeys == ("existing-runner",)
    assert result.config.defaults.rewards_enabled is False
    assert result.config.mode == "shadow" and result.config.pinned_round_id == TARGET
    assert result.config.reward_signer_factory is None and result.config.baseline_promoter_factory is None
    assert result.config.baseline_source_fetcher is original_source
    assert not hasattr(result, "_saved_output_replay")
    source = result.config.daily_icp_source(set_id=7, active_at=None)
    assert source["set_id"] == 7 and len(source["icps"]) == 2
    source["icps"][0]["prompt"] = "mutated"
    assert result.config.daily_icp_source(set_id=7, active_at=None)["icps"][0]["prompt"] != "mutated"


@pytest.mark.parametrize("change", [None, "changed_icp", "wrong_round", "missing_benchmark", "challenger", "reward_enabled", "reward_activated"])
def test_fresh_resume_checks_frozen_inputs_membership_and_rewards(change):
    service = Service(count=2)
    row = service.store.row
    service.config.defaults.runner_slot_ceiling = 1
    row["configuration_doc"]["runner_slot_ceiling"] = 1
    service._fresh_sourcing = {"icps": [{"icp_id": "first", "prompt": "a"}, {"icp_id": "second", "prompt": "b"}]}
    document = {"schema_version": "leadpoet.lab_arena.benchmark.v1", "round_id": ROUND_ID,
                "icps": copy.deepcopy(service._fresh_sourcing["icps"])}
    service.object_documents[row["benchmark_ref"]] = contracts.canonical_json(document).encode()
    if change == "changed_icp": document["icps"][0]["prompt"] = "different"
    elif change == "wrong_round": document["round_id"] = TARGET
    elif change == "missing_benchmark": row["benchmark_ref"] = None
    elif change == "challenger": row["participants"].append({"submission_id": "challenger"})
    elif change == "reward_enabled": row["configuration_doc"]["rewards_enabled"] = True
    elif change == "reward_activated": row["reward_activated_at"] = "2026-10-16T00:00:00Z"
    if row.get("benchmark_ref"):
        service.object_documents[row["benchmark_ref"]] = contracts.canonical_json(document).encode()
    if change:
        with pytest.raises(verification.VerificationError):
            verification._validate_frozen_round(service, row)
    else:
        verification._validate_frozen_round(service, row)


def test_parser_accepts_explicit_fresh_source_selection():
    args = verification.build_parser().parse_args([
        "serve", "--round-id", TARGET, "--environment-file", "/tmp/gateway.env",
        "--published-icp-round", SOURCE, "--icp-position", "0", "--icp-position", "2",
    ])
    assert args.published_icp_round == SOURCE and args.icp_position == [0, 2]
