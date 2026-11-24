from dataclasses import replace
from copy import deepcopy
import json

import pytest

from lab_arena import collection_judge as judge, collection_judge_review as review
from tests.lab_arena.collection_judge_test import Caller, POLICY, PROMPT, packet
from tests.lab_arena.collection_judge_review_test import batch, keys, complete_marks


def candidate_policy():
    return replace(
        POLICY,
        pilot=True,
        fable_version="moonshotai/kimi-k3",
        astra_version="mistralai/mistral-large-2512",
    )


@pytest.mark.parametrize(
    ("generator", "independent"),
    [
        ("moonshotai/kimi-k3.1-pro", "mistral"),
        ("mistralai/mistral-large-4-0", "kimi"),
    ],
)
def test_sibling_never_scores_its_own_generator_even_on_low_confidence(
    generator, independent
):
    policy = candidate_policy()
    task = packet()
    schedule = judge.assign_judges(
        policy,
        [
            dict(
                run_id=task["run_id"],
                generator=generator,
                harness="h",
                miner="m",
                slot=3,
                cohort=3,
            )
        ],
    )
    assignment = schedule[task["run_id"]]
    caller = Caller(lambda v, _: v.update(confidence="low"))
    got = judge.judge_run(policy, assignment, task, caller, prompt=PROMPT)
    assert [r["judge"] for r in got["records"]] == [independent]
    assert [r["model"] for r in caller.requests] == [policy.version(independent)]


def test_missing_or_unsupported_claim_stops_before_sending_private_judge_requests():
    task = packet()
    unit = {"id": "answer", "text": "It renewed.", "sources": {"ledger": "It expired."}}
    task["claim_reader"] = {"units": [unit], "results": []}
    caller = Caller()
    assignment = {"run_id": task["run_id"], "primary": "astra", "overlap": True}
    result = judge.judge_run(POLICY, assignment, task, caller, prompt=PROMPT)
    assert (
        not caller.requests
        and not result["records"]
        and not judge.training_selection(result)
    )
    task["claim_reader"]["results"] = [
        {
            "unit_sha256": judge.digest(unit),
            "record": {
                "id": "answer",
                "strict": True,
                "sentences": [{"text": unit["text"], "level": "invented", "unread": 0}],
            },
        }
    ]
    result = judge.judge_run(POLICY, assignment, task, caller, prompt=PROMPT)
    assert (
        not caller.requests
        and not result["records"]
        and not judge.training_selection(result, approved_claim_runs=[task["run_id"]])
    )


def test_signed_human_disagreement_holds_even_with_unanimous_judges(batch):
    form = complete_marks(batch)
    for scores in form["marks"].values():
        scores.update(truth=0)
    key, pub = keys()
    signed = review.sign_review(batch, form, "fixture independent human", key)
    assert "human_agreement" in signed["report"]["holds"]
    with pytest.raises(ValueError, match="pass review"):
        review.verify_review(batch, signed, pub)


def test_frozen_split_requires_independent_template_rules_and_tool_variant():
    from lab_arena.collection_quality import admit_split, verify_split

    case = {
        "template_id": "C01",
        "rule_pool": "train.rules",
        "tool_variant": "functions",
    }
    split = {
        "schema": "collection.split.v1",
        "training": {k: [v] for k, v in case.items()},
        "heldout": {k: [v + ".hidden"] for k, v in case.items()},
    }
    proof = admit_split(case, split)
    assert verify_split(proof, split) == proof
    for field, value in case.items():
        invalid = deepcopy(split)
        invalid["heldout"][field] = [value]
        with pytest.raises(ValueError, match="held-out"):
            admit_split(case, invalid)
        with pytest.raises(ValueError):
            admit_split({**case, field: None}, split)
    with pytest.raises(ValueError):
        verify_split(dict(proof, manifest_sha256="0" * 64), split)


def test_fresh_reader_accounts_for_final_answer_and_all_three_original_votes():
    from lab_arena.collection_claims import ensure_claim_units, read_fresh

    p = ensure_claim_units(
        packet(),
        task={"files": {"crm": "Expired."}},
        out={"answer": "Renewed.", "files": {"crm": "Expired."}},
    )

    class Reader:
        calls = []

        def judge(self, **req):
            self.calls.append(req)
            return '{"labels":["invented"]}'

    reader = Reader()
    got = read_fresh(
        p,
        policy=candidate_policy(),
        generator="moonshotai/kimi-k3.1-pro",
        broker=reader,
    )
    assert judge.claim_verdict(got) == "failed"
    assert len({x["request_id"] for x in reader.calls}) == 3
    assert {x["model"] for x in reader.calls} == {"mistralai/mistral-large-2512"}
    model_call = Caller()
    assert (
        judge.judge_run(
            POLICY,
            {"run_id": p["run_id"], "primary": "astra", "overlap": False},
            got,
            model_call,
            prompt=PROMPT,
        )["records"]
        == []
    )
    assert not model_call.requests


def test_position_sensitive_rank_is_held_and_sibling_cannot_rank_its_own_work():
    class PositionCaller(Caller):
        def __call__(self, **req):
            self.requests.append(req)
            doc = json.loads(req["messages"][-1]["content"])
            order = list(doc["runs"])
            return json.dumps(
                dict(
                    doc["identity"],
                    order=order,
                    margin="clear",
                    why_first="First.",
                    loser_tags={k: "stale_source_used" for k in order[1:]},
                )
            )

    policy = candidate_policy()
    docs = []
    for i, model in enumerate(("moonshotai/kimi-k3.1-pro", "moonshotai/kimi-k3.1-pro")):
        p = packet(str(i))
        p["steps"][0]["action"] = str(i)
        assign = judge.assign_judges(
            policy, [dict(run_id=str(i), generator=model, miner="m", harness="h")]
        )[str(i)]
        docs.append((p, judge.judge_run(policy, assign, p, Caller(), prompt=PROMPT)))
    call = PositionCaller()
    rank = judge.rank_scenario(policy, docs, call, prompt=PROMPT)
    assert rank["status"] == "order_disagreement" and rank["record"] is None
    assert [x["model"] for x in call.requests] == [policy.version("mistral")] * 2
    assert len({x["request_id"] for x in call.requests}) == 2


def test_candidate_panel_requires_distinct_approval_of_the_actual_blind_receipt(batch):
    from dataclasses import asdict
    from tests.lab_arena.collection_judge_test import fail_truth

    policy = candidate_policy()
    b = deepcopy(batch)
    b["policy"] = asdict(policy)
    assigned = judge.assign_judges(policy, b["assignments"])
    for run in b["runs"]:
        rid = run["packet"]["run_id"]
        run["assessment"] = judge.judge_run(
            policy,
            assigned[rid],
            run["packet"],
            Caller(
                (lambda v, n: fail_truth(v)) if b["planted"].get(rid) is False else None
            ),
            prompt=PROMPT,
        )
    for rid in b["drift"]:
        b["drift"][rid] = {role: dict.fromkeys(judge.PARTS, 3) for role in policy.roles}
    private, public = keys()
    approver, trust = keys()
    signed = review.sign_review(b, complete_marks(b), "fixture reviewer", private)
    assert signed["report"]["holds"] == ["judge_qualification"]
    with pytest.raises(ValueError, match="separate approval key"):
        review.approve_qualification(b, signed, public, "fixture operator", private)
    cert = review.approve_qualification(b, signed, public, "fixture operator", approver)
    assert not cert["allow_live"]
    with pytest.raises(ValueError, match="signature"):
        review.verify_qualification(
            b, signed, public, dict(cert, allow_live=True), trust
        )
    live = review.approve_qualification(
        b, signed, public, "fixture operator", approver, allow_live=True
    )
    assert review.verify_qualification(b, signed, public, live, trust)["allow_live"]
    assert review.verify_qualification(b, signed, public, cert, trust)[
        "review_sha256"
    ] == judge.digest(signed)
    with pytest.raises(ValueError, match="signature"):
        review.verify_qualification(b, signed, public, cert, keys()[1])
    with pytest.raises(ValueError, match="differs"):
        review.verify_qualification(
            b, signed, public, dict(cert, prompt_sha256="x"), trust
        )
    with pytest.raises(ValueError, match="pass review"):
        review.verify_review(b, signed, public)


def test_live_panel_requires_explicit_approved_prices_and_matching_scope():
    from scripts.run_locked_collection import load_batch, verify_live_panel
    from tests.lab_arena.collection_qualification_judge_test import manifest
    from tests.lab_arena.collection_judge_test import SPLIT

    raw = {
        **manifest(),
        "mode": "live",
        "split_manifest": SPLIT,
        "panel_qualification_sha256": "c" * 64,
    }
    policy = load_batch(raw, PROMPT)
    assert policy.qualification and not policy.pilot
    approved = {
        "panel_qualification_sha256": "c" * 64,
        "split_manifest_sha256": judge.digest(SPLIT),
    }
    assert verify_live_panel(raw, policy, approved) == "c" * 64
    for bad in (
        {},
        {**approved, "split_manifest_sha256": "d" * 64},
        {**approved, "panel_qualification_sha256": "e" * 64},
    ):
        with pytest.raises(ValueError, match="differs"):
            verify_live_panel(raw, policy, bad)
    with pytest.raises(ValueError, match="qualification"):
        load_batch(
            {k: v for k, v in raw.items() if k != "panel_qualification_sha256"}, PROMPT
        )


def test_reader_declines_sensitive_source_before_any_dispatch():
    from lab_arena.collection_claims import ensure_claim_units, read_fresh

    p = ensure_claim_units(
        packet(),
        task={"files": {"example": "Authorization: Bearer " + "a" * 40}},
        out={"answer": "Approved", "files": {}},
    )

    class Reader:
        def judge(self, **req):
            raise AssertionError("sensitive text should not be sent")

    out = read_fresh(
        p, policy=candidate_policy(), generator="qwen/qwen3.8-27b", broker=Reader()
    )
    assert out["checks"]["privacy"] is False and "a" * 40 not in json.dumps(out)


def test_negative_claim_outcome_is_valid_miner_statement_metadata():
    from lab_arena.collection_intake import _validate_quality

    _validate_quality({"training_status": "claim_rejected"})


def test_qualification_control_provenance_excludes_source_siblings():
    from lab_arena.collection_controls import verify_control_family

    panel = list(judge.QUALIFICATION_JUDGES)
    for source in (None, "mistralai/mistral-large-4-0", "moonshotai/kimi-k3.1-pro"):
        with pytest.raises(ValueError, match="independent source family"):
            verify_control_family({"provenance": {"source_generator": source}}, panel)
    verify_control_family({"provenance": {"source_generator": "human"}}, panel)


def test_long_claim_unit_checks_every_named_sentence_even_when_old_array_drifts():
    from lab_arena.collection_claims import read_fresh

    task = packet()
    sentences = [f"Account {i} renewed." for i in range(22)]
    unit = {
        "id": "file",
        "text": " ".join(sentences),
        "sources": {"signed-ledger": "renewal ledger"},
    }
    task["claim_reader"] = {"units": [unit], "results": []}

    class Reader:
        def __init__(self):
            self.seen = []

        def judge(self, **request):
            data = json.loads(request["messages"][-1]["content"])
            group = data["sentences"]
            if isinstance(group, list):
                return json.dumps({"labels": ["supported"] * (len(group) + 1)})
            assert 1 <= len(group) <= 8 and data["context"] == unit["text"]
            self.seen.extend(group)
            return json.dumps(
                {
                    "labels": {
                        i: ("invented" if i == "s18" else "supported")
                        for i in reversed(group)
                    }
                }
            )

    reader = Reader()
    checked = read_fresh(
        task,
        policy=candidate_policy(),
        generator="mistralai/mistral-large-4-0",
        broker=reader,
        protocol="collection.fresh-claims.v2",
    )
    assert judge.claim_verdict(checked) == "failed"
    from collections import Counter

    assert Counter(reader.seen) == {f"s{i}": 3 for i in range(1, 23)}
    result = checked["claim_reader"]["results"][0]["record"]["sentences"]
    assert [i["text"] for i in result] == sentences
    assert [i["text"] for i in result if i["level"]] == [sentences[17]]
    from lab_arena.collection_claims import verify_keyed_unit
    proof = checked["fresh_claim_proof"][judge.digest(unit)]
    record = checked["claim_reader"]["results"][0]["record"]
    verify_keyed_unit(unit, proof, record)
    shifted = deepcopy(proof)
    shifted["sentence_ids"][0][0] = "s9"
    with pytest.raises(ValueError, match="coverage"):
        verify_keyed_unit(unit, shifted, record)
    with pytest.raises(ValueError, match="majority"):
        verify_keyed_unit(unit, proof, dict(record, strict=False))


def test_keyed_claims_reject_displaced_or_missing_sentence_ids_and_mixed_provenance():
    from lab_arena.collection_claims import KEYED_V2, read_fresh

    unit = {"id": "answers", "text": "First. Second.", "sources": {"file": "Original."}}
    p = packet()
    p["claim_reader"] = {"units": [unit], "results": []}

    class WrongKeys:
        def judge(self, **req):
            return '{"labels":{"s1":"supported","s3":"supported"}}'

    out = read_fresh(
        p,
        policy=candidate_policy(),
        generator="mistralai/mistral-large-4-0",
        broker=WrongKeys(),
        protocol=KEYED_V2,
    )
    assert not out["claim_reader"]["results"] and judge.claim_verdict(out) == "missing"
    digest = judge.digest(unit)
    p["claim_reader"]["results"] = [{"unit_sha256": digest, "record": {}}]
    p["fresh_claim_proof"] = {digest: {"schema": "collection.fresh-claims.v1"}}
    with pytest.raises(ValueError, match="different protocol"):
        read_fresh(
            p,
            policy=candidate_policy(),
            generator="mistralai/mistral-large-4-0",
            broker=WrongKeys(),
            protocol=KEYED_V2,
        )


def test_historical_replay_cannot_masquerade_as_a_training_release():
    from lab_arena.collection_quality import require_release_partition
    from lab_arena.collection_workflow import TRAIN_ONLY_PARTITION_SHA256

    settings = {
        "split_sha256": "sha256:" + TRAIN_ONLY_PARTITION_SHA256,
        "cases": [{"scenario_id": "C07-s3013"}],
    }
    with pytest.raises(ValueError, match="private audit only"):
        require_release_partition(settings)
    settings["cases"] = [{"scenario_id": "P7-s3013"}]
    require_release_partition(settings)
    with pytest.raises(ValueError, match="private audit only"):
        require_release_partition(dict(settings, split_sha256="sha256:" + "a" * 64))
