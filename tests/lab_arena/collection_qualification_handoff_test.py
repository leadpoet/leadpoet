"""The staged qualification handoff never guesses a human mark or a reference verdict."""

import json

import pytest


def test_missing_qualification_binding_or_incomplete_reference_keeps_original(tmp_path):
    from scripts.collection_qualification_followup import qualification_for_poll

    raw = {"name": "original"}
    assert qualification_for_poll(raw, {}, tmp_path, None, None) == (raw, None)


def test_review_sample_templates_are_not_feedback():
    from scripts.collection_review import blind_files
    from lab_arena import collection_judge as judge

    jobs = {"a": {"case": "private"}}
    marks = blind_files(jobs, "1" * 64)
    assert marks["batch_sha256"] == "1" * 64 and marks["sample_sha256"] == judge.digest(
        jobs
    )
    assert all(v is None for v in marks["marks"]["a"].values())
    assert set(marks["marks"]["a"]) == set(judge.PARTS)


def test_independent_planted_labels_and_blind_sample_safety(tmp_path):
    from scripts.collection_qualification_followup import reviewed_plants, store_once
    from tests.lab_arena.collection_judge_test import packet
    from lab_arena import collection_judge as judge
    from copy import deepcopy
    from scripts.collection_review import read_file
    import json

    cases, bindings = {}, {}
    for idx in (0, 1):
        p = packet(f"run-{idx}")
        p["task"] += f" {idx}"
        unit = {
            "id": "test-fact",
            "text": "The ledger has a claim.",
            "sources": {"ledger": "The ledger has a claim."},
        }
        p["claim_reader"] = {
            "units": [unit],
            "results": [
                {
                    "unit_sha256": judge.digest(unit),
                    "record": {
                        "id": unit["id"],
                        "strict": False,
                        "sentences": [{"text": unit["text"], "level": None}],
                    },
                }
            ],
        }
        a, b = judge._packet(dict(p, run_id="blind-projection"))
        src = {"source_generator": "human", "packet_sha256": judge.digest(p)}
        alias = f"private-{idx}"
        cases[alias] = {"scenario": a, "run": b}
        bindings[alias] = {
            "source": src,
            "packet": p,
            "source_sha256": "sha256:" + judge.digest(src),
        }
    decisions = {
        "sample_sha256": judge.digest(cases),
        "reviewer": "test-independent",
        "reviews": {
            "private-0": {"accept": True, "rationale": "Matches the returned record."},
            "private-1": {
                "accept": False,
                "rationale": "Contradicts the returned record.",
            },
        },
    }
    labels = reviewed_plants(cases, decisions, bindings)
    assert len(labels) == 2 and {p["expected"] for p in labels} == {True, False}
    blank = deepcopy(decisions)
    blank["reviews"]["private-0"]["accept"] = None
    with pytest.raises(ValueError, match="genuine"):
        reviewed_plants(cases, blank, bindings)
    wrong = deepcopy(bindings)
    wrong["private-1"]["source_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="original source"):
        reviewed_plants(cases, decisions, wrong)
    target = tmp_path / "packet"
    files = {"jobs.json": cases, "marks.blank.json": {}}
    store_once(target, files)
    store_once(target, files)
    assert json.loads(read_file(target / "jobs.json")) == cases
    with pytest.raises(ValueError, match="frozen qualification"):
        store_once(target, dict(files, **{"marks.blank.json": {"invented": 1}}))
    assert json.loads(read_file(target / "marks.blank.json")) == {}


def test_full_materialized_handoff_at_the_business_boundary(tmp_path, monkeypatch):
    """Exercise the normal selector, exact frontier, genuine-input holds and atomic dispatch files.

    Signatures, frozen anchor provenance and typed signed blind receipts have their own
    integration coverage. Stub only those sources to drive the completed state in this test.
    """
    from scripts import collection_qualification_followup as handoff
    from scripts.collection_review import write_new
    from lab_arena import collection_judge as judge
    from types import SimpleNamespace
    from hashlib import sha256

    ref = {"scenario_ids": [f"P7-s{n}" for n in range(1, 11)]}
    draft = {
        "scenario_ids": [f"P7-s{n}" for n in range(1, 25)],
        "budgets_microusd": {str(k): 300000 for k in range(6)},
    }
    reference, qual = "rq-" + "a" * 64, "rq-" + "b" * 64

    def private(name, data):
        path = tmp_path / name
        write_new(path, data)
        return sha256(path.read_bytes()).hexdigest()

    cases = {"a": {"x": 1}, "b": {"x": 2}}
    bindings = {"a": {"source": 1}, "b": {"source": 2}}
    cfg = {
        "schema": "collection.p7-qualification-followup.v1",
        "reference_id": reference,
        "draft_id": qual,
        "reference_sha256": private("complete-reference.json", ref),
        "draft_sha256": private(
            "qualification-settings.pending.json", {"manifest": draft}
        ),
        "cases_sha256": private("planted-cases.json", cases),
        "bindings_sha256": private("planted-source-bindings.json", bindings),
        "total_budget_microusd": 43200000,
    }
    private("qualification-handoff.json", cfg)
    monkeypatch.setattr(
        handoff,
        "load_batch",
        lambda r, p: SimpleNamespace(
            batch_id=reference if r == ref else qual, pilot=True, qualification=True
        ),
    )

    class DB:
        rows = {}

        def rpc(self, n, p):
            assert n == "collection_pilot_costs_v1"
            return self.rows.get(p["p_batch"], [])

    prompt = {
        "rubric_version": "fixture",
        "rubric": "test dimensions",
        "examples": {},
        "run_instructions": "test",
    }
    db = DB()

    def finished(n):
        return [
            {
                "slot": i,
                "attempt": 1,
                "status": "accepted",
                "unresolved": 0,
                "artifact_sha256": "sha256:" + "a" * 64,
            }
            for i in range(n)
        ]

    assert handoff.qualification_for_poll(ref, prompt, tmp_path, db, None) == (
        ref,
        None,
    )
    db.rows[reference] = finished(60)
    assert handoff.qualification_for_poll(ref, prompt, tmp_path, db, None) == (
        ref,
        None,
    )
    private("returned-plant-decisions.json", {"original": True})
    monkeypatch.setattr(
        handoff,
        "reviewed_plants",
        lambda c, d, b: (
            ["two original human forms"]
            if c == cases and d == {"original": True} and b == bindings
            else pytest.fail("source changed")
        ),
    )
    sealed = dict(
        draft,
        control_slots={"0": True, "1": False},
        drift_slots={str(i): {} for i in range(2, 32)},
    )
    new_id = "rq-new"
    monkeypatch.setattr(handoff, "seal", lambda *args: (sealed, {"cases": {}}, new_id))
    result, control = handoff.qualification_for_poll(ref, prompt, tmp_path, db, None)
    assert (
        result == sealed
        and control.is_file()
        and not (tmp_path / "blind-ordinary-review").exists()
    )
    db.rows[new_id] = finished(144)
    monkeypatch.setattr(
        handoff, "source_bundle", lambda *a: {"fixture": "verified original"}
    )
    jobs = {str(i): {"task": str(i)} for i in range(50)}
    monkeypatch.setattr(
        handoff.review, "prepare_review", lambda b, for_qualification: (None, jobs)
    )
    result, _ = handoff.qualification_for_poll(ref, prompt, tmp_path, db, None)
    marks = json.loads(
        (tmp_path / "blind-ordinary-review/marks.blank.json").read_bytes()
    )
    assert len(marks["marks"]) == 50 and all(
        v is None for mark in marks["marks"].values() for v in mark.values()
    )
    assert marks["sample_sha256"] == judge.digest(jobs)
