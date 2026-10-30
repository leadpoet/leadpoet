"""A source episode must have one split across native target families."""

from __future__ import annotations

import copy

from lab_arena import contracts, training_dataset, training_steps
from tests.lab_arena import training_dataset_test, training_steps_test


def hybrid(run_id: str, position: int, *, distinct_icp: bool = False):
    company = training_dataset_test.claim(
        "Company %d" % position, "company-%d.example" % position,
        "company-%d" % position)
    source = training_dataset_test.example(
        run_id, [training_dataset_test.row(0, company, "qualified")],
        submission_id="shared-submission", icp_position=position,
        assignment_id="assignment-%d" % position,
        **({"icp": {**training_dataset_test.ICP, "industry": "Industry %d" % position}}
           if distinct_icp else {}))
    action = copy.deepcopy(training_steps_test.fixture())
    call = action["provider_calls"][0]
    call["run_id"] = run_id
    call["call_identity"] = "call-%d" % position
    call["response_provenance"]["worker_receipt_ref"]["run_id"] = run_id
    for ref in call["response_provenance"]["agent_receipt_refs"]:
        ref["run_id"] = run_id
    payload = action["worker_observation_artifacts"]["run"]["payload"]
    for receipt in payload["response_receipts"]:
        receipt["call_identity"] = call["call_identity"]
    for observation in payload["agent_observations"]:
        observation["data"]["call_identity"] = call["call_identity"]
    body = call["request"]["outbound"]["body"]
    body["messages"][0]["content"] = "Find company %d" % position
    call["request"]["outbound"]["body_sha256"] = contracts.document_hash(body)
    artifact = action["worker_observation_artifacts"]["run"]
    artifact["payload_sha256"] = contracts.document_hash(payload)
    for ref in call["response_provenance"]["agent_receipt_refs"] + [call["response_provenance"]["worker_receipt_ref"]]:
        ref["payload_sha256"] = artifact["payload_sha256"]
    source["provider_calls"] = [call]
    source["worker_observation_artifacts"] = {run_id: artifact}
    return training_dataset_test.seal(source)


def split_runs(splits, *, company: bool):
    return {
        split: {record["meta"]["run_id"] if company else ref["run_id"]
                for record in splits[split]
                for ref in ([None] if company else record["source_refs"])}
        for split in ("train", "eval")
    }


def test_original_episode_has_same_split_across_company_and_steps():
    examples = [hybrid("run-zero", 0), hybrid("run-one", 1)]
    company, _ = training_dataset.build(examples)
    steps, _ = training_steps.build(examples)
    assert sum(map(len, (company["train"], company["eval"]))) == 2
    assert sum(map(len, (steps["train"], steps["eval"]))) == 2
    labels = split_runs(company, company=True)
    actions = split_runs(steps, company=False)
    assert not (labels["train"] & actions["eval"])
    assert not (labels["eval"] & actions["train"])


def test_shared_custom_rules_and_input_order_keep_all_episode_splits_aligned():
    examples = [hybrid("run-%d" % position, position, distinct_icp=True)
                for position in range(8)]
    for mode in ("company", "company_and_icp"):
        company_rules = training_dataset.Rules(seed="same-custom-seed", eval_percent=40,
                                               eval_mode=mode)
        step_rules = training_steps.Rules(seed="same-custom-seed", eval_percent=40,
                                         eval_mode=mode)
        first_company, _ = training_dataset.build(examples, company_rules)
        first_steps, _ = training_steps.build(examples, step_rules)
        assert first_company == training_dataset.build(list(reversed(examples)), company_rules)[0]
        assert first_steps == training_steps.build(list(reversed(examples)), step_rules)[0]
        labels = split_runs(first_company, company=True)
        actions = split_runs(first_steps, company=False)
        assert labels == actions
        assert labels["train"] and labels["eval"]


def test_aliases_and_identical_inputs_bridge_before_family_selection():
    first = hybrid("run-first", 0, distinct_icp=True)
    second = hybrid("run-second", 1, distinct_icp=True)
    for source in (first, second):
        source["companies"][0]["judgment"]["company_identity_alias_keys"] = ["verified:shared-alias"]
        training_dataset_test.seal(source)
    graph = training_dataset._SourceSplitGroups()
    keys = [graph.add(source, production_rounds_only=True, eval_mode="company")
            for source in (first, second)]
    assert graph.group(keys[0]) == graph.group(keys[1])

    # A duplicate model input is a shared source edge even when one company's
    # label is excluded. Target extraction must not create a family-only edge.
    second["companies"][0]["judgment"].pop("company_identity_alias_keys")
    first["companies"][0]["judgment"].pop("company_identity_alias_keys")
    second["completeness"]["dimensions"]["input_output_label"] = False
    first_body = first["provider_calls"][0]["request"]["outbound"]["body"]
    second_outbound = second["provider_calls"][0]["request"]["outbound"]
    second_outbound["body"] = copy.deepcopy(first_body)
    second_outbound["body_sha256"] = contracts.document_hash(first_body)
    for source in (first, second):
        training_dataset_test.seal(source)
    labels, _ = training_dataset.build([first, second], training_dataset.Rules(eval_mode="company"))
    steps, _ = training_steps.build([first, second], training_steps.Rules(eval_mode="company"))
    assert len(labels["train"] + labels["eval"]) == 1
    admitted = steps["train"] + steps["eval"]
    assert len(admitted) == 1 and len(admitted[0]["source_refs"]) == 2
    assert split_runs(labels, company=True)["train"] <= split_runs(steps, company=False)["train"]
    assert split_runs(labels, company=True)["eval"] <= split_runs(steps, company=False)["eval"]


def test_redacted_alias_does_not_bridge_and_malformed_company_does_not_drop_action():
    first = hybrid("run-first", 0, distinct_icp=True)
    second = hybrid("run-second", 1, distinct_icp=True)
    for source in (first, second):
        source["companies"][0]["judgment"]["company_identity_alias_keys"] = ["[REDACTED]"]
        training_dataset_test.seal(source)
    graph = training_dataset._SourceSplitGroups()
    keys = [graph.add(source, production_rounds_only=True, eval_mode="company_and_icp")
            for source in (first, second)]
    assert graph.group(keys[0]) != graph.group(keys[1])
    first["companies"].append({"bad": "malformed company row"})
    training_dataset_test.seal(first)
    actions, _ = training_steps.build([first])
    assert len(actions["train"] + actions["eval"]) == 1


def test_retries_share_original_episode_and_only_exact_main_requests_bridge():
    original = hybrid("run-original", 0, distinct_icp=True)
    retry = hybrid("run-retry", 0, distinct_icp=True)
    retry["assignment_id"] = "different-retry-assignment"
    training_dataset_test.seal(retry)
    graph = training_dataset._SourceSplitGroups()
    first = graph.add(original, production_rounds_only=True, eval_mode="company_and_icp")
    second = graph.add(retry, production_rounds_only=True, eval_mode="company_and_icp")
    assert first == second

    unrelated = hybrid("run-unrelated", 1, distinct_icp=True)
    outbound = unrelated["provider_calls"][0]["request"]["outbound"]
    outbound["body"] = copy.deepcopy(original["provider_calls"][0]["request"]["outbound"]["body"])
    outbound["body_sha256"] = contracts.document_hash(outbound["body"])
    outbound["host"] = "other.example"
    training_dataset_test.seal(unrelated)
    third = graph.add(unrelated, production_rounds_only=True, eval_mode="company_and_icp")
    assert graph.group(first) != graph.group(third)

    outbound["host"] = "openrouter.ai"
    unrelated["provider_calls"][0]["request_capture"] = {"redacted": "truthy malformed flag"}
    training_dataset_test.seal(unrelated)
    assert training_dataset._split_model_request_links(unrelated) == []


def test_missing_episode_identity_fails_closed_and_temporal_split_matches():
    invalid = hybrid("run-invalid", 0)
    invalid["stage"] = None
    training_dataset_test.seal(invalid)
    labels, _ = training_dataset.build([invalid])
    steps, _ = training_steps.build([invalid])
    assert not labels["train"] and not labels["eval"]
    assert not steps["train"] and not steps["eval"]
    assert any(row["reason"] == "episode_identity_unverified" for row in steps["analysis"])

    past = hybrid("run-past", 0)
    future = hybrid("run-future", 1)
    past["evaluation_date"] = "2026-10-17"
    future["evaluation_date"] = "2026-10-19"
    for source in (past, future):
        training_dataset_test.seal(source)
    labels, _ = training_dataset.build([past, future],
                                       training_dataset.Rules(eval_after="2026-10-18"))
    steps, _ = training_steps.build([past, future],
                                    training_steps.Rules(eval_after="2026-10-18"))
    assert not labels["train"] and not steps["train"]
    assert split_runs(labels, company=True)["eval"] == {"run-future"}
    assert split_runs(steps, company=False)["eval"] == {"run-future"}
