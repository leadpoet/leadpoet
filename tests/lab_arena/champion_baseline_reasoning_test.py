"""The promoted champion remains the baseline across entrant-policy changes."""

import hashlib
from types import SimpleNamespace

import pytest

from lab_arena import baseline_reasoning, contracts, reasoning_capture
from lab_arena import service as arena_service
from tests.lab_arena.full_thinking_round_test import OPEN, opted_in
from tests.lab_arena.test_lab_arena_promotion_git import source


def test_legacy_promoted_source_replaces_bootstrap_at_freeze():
    service = opted_in()
    config = service.create_round(OPEN)
    champion = source("legacy-champion")
    commit = "b" * 40
    expected_url = "https://github.com/leadpoet/leadpoet-sales-agent/archive/" + commit + ".tar.gz"
    selected = []
    service._store = SimpleNamespace(
        latest_promoted_champion=lambda **_scope: {
            "round_id": "arena-prior", "winner_submission_id": "winner",
            "commit": commit, "reasoning_capture_policy": None,
        },
        freeze_full_thinking_baseline=lambda _rid, **selection: (
            selected.append(selection) or {"status": "frozen"}),
    )
    fetched = []
    service._config.baseline_source_fetcher = lambda url, _limit: fetched.append(url) or champion
    service._freeze_full_thinking_baseline(config["round_id"], {
        "round_id": config["round_id"], "configuration_doc": config,
    })
    assert selected[0]["source_url"] == expected_url
    assert selected[0]["source_sha256"] == "sha256:" + hashlib.sha256(champion).hexdigest()
    assert fetched == [expected_url]


def baseline_round(contract):
    config = opted_in().create_round(OPEN)
    config[baseline_reasoning.FIELD] = contract
    rid = config["round_id"]
    baseline = {"submission_id": "baseline-" + rid.removeprefix("arena-"),
                "miner_hotkey": config["baseline_hotkey"], "is_king": True}
    return {"round_id": rid, "configuration_doc": config, "participants": [baseline]}, baseline


def test_baseline_capture_contract_is_separate_from_entrant_contract():
    row, baseline = baseline_round({})
    assert baseline_reasoning.for_run(row, baseline) == {}
    entrant = dict(baseline, submission_id="miner", is_king=True)
    assert baseline_reasoning.for_run(row, entrant)["reasoning_capture_policy"] == reasoning_capture.POLICY
    row["participants"] = []
    with pytest.raises(ValueError, match="not frozen"):
        baseline_reasoning.for_run(row, baseline)


def test_historical_round_retains_round_capture_contract():
    row, baseline = baseline_round({})
    del row["configuration_doc"][baseline_reasoning.FIELD]
    assert baseline_reasoning.for_run(row, baseline)["reasoning_capture_policy"] == reasoning_capture.POLICY


def test_frozen_baseline_keeps_original_registry():
    original = baseline_reasoning.from_configuration(opted_in().create_round(OPEN))
    row, baseline = baseline_round(original)
    row["configuration_doc"]["reasoning_route_registry"] = {"different": "future registry"}
    assert baseline_reasoning.for_run(row, baseline) == original


def test_baseline_export_contract_cannot_be_replaced_after_rehash():
    row, _ = baseline_round({})
    example = {"role": "baseline", "configuration": row["configuration_doc"],
               baseline_reasoning.EXPORT_FIELD: {}}
    assert baseline_reasoning.for_example(example) == {}
    example[baseline_reasoning.EXPORT_FIELD] = baseline_reasoning.from_configuration(row["configuration_doc"])
    with pytest.raises(ValueError, match="differs"):
        baseline_reasoning.for_example(example)


@pytest.mark.parametrize("contract", [{}, None, {"reasoning_capture_policy": "wrong"}])
def test_baseline_capture_contract_json_roundtrip_and_validation(contract):
    import json
    row, _ = baseline_round(contract)
    config = json.loads(json.dumps(row["configuration_doc"]))
    if contract == {}:
        assert contracts.validate_round_configuration(config) == config
    else:
        with pytest.raises(contracts.ArenaContractError):
            contracts.validate_round_configuration(config)


def test_run_context_resolves_baseline_contract_but_preserves_challenger_gate():
    row, baseline = baseline_round({})
    run = dict(baseline, run_id="run", round_id=row["round_id"], assignment_id="assignment",
               attempt=1, icp_position=0, stage=1, kind="execute")
    service = object.__new__(arena_service.ArenaService)
    service._store = SimpleNamespace(get_run=lambda _id: run)
    service._require_round_ownership = lambda _id: None
    service._hot_round = lambda _id: row
    _, context = service._run_context("run", "token")
    assert context.reasoning_capture_policy is None
    assert context.reasoning_route_registry is None
    run["submission_id"] = "challenger"
    _, context = service._run_context("run", "token")
    assert context.reasoning_capture_policy == reasoning_capture.POLICY


def test_legacy_baseline_is_truthfully_ineligible_for_strict_training():
    from lab_arena.public_dashboard import _training_eligibility_projection
    row, baseline = baseline_round({})
    service = object.__new__(arena_service.ArenaService)
    service._store = SimpleNamespace(get_submission=lambda _id: baseline)
    result = service._submission_training_eligibility(row, baseline["submission_id"], [])
    assert result == {"eligible": False, "reasons": ["baseline_reasoning_not_required"],
                      "exemption": "champion_baseline"}
    public = service._public_round_output_policy(row["configuration_doc"])
    assert public["reasoning_capture_policy"] == reasoning_capture.POLICY
    assert public["baseline_reasoning_capture_policy"] is None
    assert _training_eligibility_projection({"training_data_eligibility": result}) == {
        "training_data_eligibility": {"eligible": False, "reasons": ["baseline_reasoning_not_required"]}}


def test_training_export_keeps_round_cost_policy_and_reports_actual_baseline_capture():
    from lab_arena.training_export import assemble_example
    from tests.lab_arena.training_export_test import fixture
    values = fixture()
    row = values["round_row"]
    config = row["configuration_doc"]
    capture = baseline_reasoning.from_configuration(opted_in().create_round(OPEN))
    config.update(capture)
    config[baseline_reasoning.FIELD] = {}
    config["baseline_hotkey"] = "baseline-owner"
    baseline_id = "baseline-" + row["round_id"].removeprefix("arena-")
    for participant in row["participants"]:
        participant.update(submission_id=baseline_id, miner_hotkey="baseline-owner", is_king=True)
    values["submission_row"].update(submission_id=baseline_id, miner_hotkey="baseline-owner", is_king=True)
    for run in values["runs"]:
        run.update(submission_id=baseline_id, miner_hotkey="baseline-owner")
    exported = assemble_example(**values)
    assert exported["role"] == "baseline"
    assert exported["configuration"] == config
    assert exported[baseline_reasoning.EXPORT_FIELD] == {}
    assert baseline_reasoning.for_example(exported) == {}


@pytest.mark.parametrize("is_baseline", [True, False])
def test_handle_claim_lease_uses_source_capture_contract(is_baseline):
    from tests.lab_arena.test_lab_arena_service_rules import _runner_snapshot

    runner = "5" * 48
    rid = "arena-2026-10-29"
    digest = "sha256:" + "a" * 64
    baseline = {
        "submission_id": "baseline-2026-10-29", "miner_hotkey": "5" + "A" * 47,
        "is_king": True, "source_ref": "arena/" + rid + "/sources/baseline.tar.gz",
        "source_size_bytes": 123,
    }
    challenger = dict(baseline, submission_id="sub-new-entrant", miner_hotkey="5" + "B" * 47,
                      is_king=False, source_ref="arena/" + rid + "/sources/challenger.tar.gz")
    participant = baseline if is_baseline else challenger
    capture = baseline_reasoning.from_configuration(opted_in().create_round(OPEN))
    configuration = {
        "runner_hotkeys": [runner], "runner_slot_ceiling": 8, "lease_ttl_seconds": 420,
        "scorer_image_digest": digest, "scorer_image_reference": "registry.example/scorer@" + digest,
        "baseline_hotkey": baseline["miner_hotkey"], baseline_reasoning.FIELD: {}, **capture,
    }
    row = {"round_id": rid, "status": "stage1", "participants": [baseline, challenger],
           "evaluation_date": "2026-10-29", "configuration_doc": configuration}
    response = {
        "status": "leased", "kind": "execute", "round_id": rid,
        "run_id": "run-1", "assignment_id": "assignment-1", "stage": 1,
        "submission_id": participant["submission_id"], "miner_hotkey": participant["miner_hotkey"],
        "icp_position": 0, "attempt": 1, "lease_generation": 1, "stage_generation": 1,
        "lease_expires_at": "2026-10-29T00:07:00Z",
    }
    service = object.__new__(arena_service.ArenaService)
    service._store = SimpleNamespace(
        get_submission=lambda _id: dict(participant, status="frozen", code_review_status="passed"),
        get_submission_credential=lambda *_args: None,
        claim_assignment=lambda **_kwargs: dict(response),
    )
    service._config = SimpleNamespace(
        chain=SimpleNamespace(metagraph=lambda *, finalized: _runner_snapshot(runner)),
        validator_authorizer=lambda _hotkey: (True, "validator"),
    )
    service._request_round = lambda *_args, **_kwargs: ({
        "hotkey": runner, "body": {"declared_parallelism": 1},
        "request_id": "0" * 32, "signature": "sig",
    }, row)
    service._lease_token = lambda _request: "lease-token"
    service.benchmark_icps = lambda _rid: [{}]

    lease = service.handle_claim({})
    assert lease["submission_id"] == participant["submission_id"]
    assert lease["miner_hotkey"] == participant["miner_hotkey"]
    assert lease["source_ref"] == participant["source_ref"]
    if is_baseline:
        assert "reasoning_capture_policy" not in lease
    else:
        assert lease["reasoning_capture_policy"] == reasoning_capture.POLICY


@pytest.mark.parametrize("model", [
    "openai/gpt-5.6-sol", "google/gemini-3-flash-preview",
    "anthropic/claude-sonnet-5.5", "anthropic/claude-sonnet-4.5",
    "openai/gpt-5-mini", "perplexity/sonar", "perplexity/sonar-pro",
])
def test_original_champion_model_requests_keep_controls_but_new_entrants_require_review(model):
    from lab_arena import broker

    row, baseline = baseline_round({})
    parameters = {"model": model, "messages": [{"role": "user", "content": "Find companies"}],
                  "max_tokens": 1024}
    baseline_capture = baseline_reasoning.for_run(row, baseline)
    admitted, annotation = broker.agent_reasoning_parameters(
        "openrouter.chat", parameters, policy=baseline_capture.get("reasoning_capture_policy"),
        registry=baseline_capture.get("reasoning_route_registry"))
    assert admitted == parameters
    assert annotation is None
    entrant_capture = baseline_reasoning.for_run(row, dict(baseline, submission_id="new-entrant"))
    with pytest.raises(broker.BrokerError, match="invalid_request"):
        broker.agent_reasoning_parameters(
            "openrouter.chat", parameters, policy=entrant_capture["reasoning_capture_policy"],
            registry=entrant_capture["reasoning_route_registry"])
