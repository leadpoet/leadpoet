import json
from copy import deepcopy
from types import SimpleNamespace
from lab_arena import collection_claims as cc, collection_judge as judge
from tests.lab_arena.collection_judge_review_test import keys
from tests.lab_arena.collection_judge_test import packet, Caller, ASSIGN, POLICY, PROMPT
import pytest


def saved_unit():
    return {
        "id": "email",
        "text": "The quote is $35.",
        "sources": {"sheet": "The quote is $35."},
    }


def forbidden(*args):
    raise AssertionError("network path reached")


ask = forbidden


def reader_check(unit, secret, label):
    body = json.dumps(
        {
            "model": cc.MODEL,
            "questions": {"s1": {"criteria": "test"}},
            "state": unit["sources"],
        }
    ).encode()
    votes = [ask(secret, body, label)["answers"]["s1"]["choice"] for _ in range(3)]
    return {
        "id": unit["id"],
        "sentences": [
            {"text": unit["text"], "votes": votes, "n": 1, "level": None, "unread": 0}
        ],
        "strict": False,
        "lenient": False,
        "placeholders": [],
    }


def entries():
    u = saved_unit()
    record = {
        "id": "email",
        "sentences": [
            {
                "text": u["text"],
                "votes": ["supported"] * 3,
                "n": 1,
                "level": None,
                "unread": 0,
            }
        ],
        "strict": False,
        "lenient": False,
        "placeholders": [],
    }
    reader = SimpleNamespace(
        MODEL=cc.MODEL,
        CALLS=3,
        split=lambda x: [x],
        CRITERIA={"supported": "allowed"},
        check=reader_check,
    )
    b = SimpleNamespace(
        build=lambda *_: {"guard": {"token": "g"}},
        plain=lambda v: v,
        units=lambda *_: [u],
    )
    spec = {"task": "B1", "seed": 2011, "guard": {"token": "g"}}
    return cc.ingest_run(
        b,
        reader,
        "source-real-id",
        spec,
        {"answer": u["text"]},
        {"finish": "answered"},
        [record],
        judge.digest([record]),
    )


def test_verify_import_no_network_and_bind_original_and_target_run_separately():
    private, public = keys()
    signed = cc.sign_pack(entries(), private)
    pack = cc.verify_pack(signed, public, judge.digest(signed))
    p = packet()
    p["claim_reader"] = {"units": [saved_unit()], "results": []}
    q = cc.apply_pack(p, batch_id="batch", pack=pack)
    assert (
        p["claim_reader"]["results"] == []
        and q["claim_reader"]["results"][0]["record"]["strict"] is False
    )
    binding = next(iter(q["claim_source_proof"]["bindings"].values()))
    assert (
        binding["source_run_id"] == "source-real-id"
        and binding["run_id"] == p["run_id"]
        and binding["batch_id"] == "batch"
    )
    call = Caller()
    result = judge.judge_run(POLICY, ASSIGN, q, call, prompt=PROMPT)
    assert judge.training_selection(result) and "source-real-id" not in json.dumps(
        call.requests
    )
    for mod in ("sources", "text", "id"):
        changed = deepcopy(p)
        u = changed["claim_reader"]["units"][0]
        u[mod] = {"other": "different"} if mod == "sources" else "new"
        assert not cc.apply_pack(changed, batch_id="batch", pack=pack)["claim_reader"][
            "results"
        ]
    tampered = deepcopy(signed)
    tampered["entries"][0]["source_run_id"] = "injected"
    with pytest.raises(ValueError, match="signature"):
        cc.verify_pack(tampered, public, judge.digest(tampered))
    with pytest.raises(ValueError, match="signature"):
        cc.verify_pack(signed, keys()[1], judge.digest(signed))
    with pytest.raises(ValueError, match="changed"):
        cc.verify_pack(signed, public, "0" * 64)


def test_conflicting_origin_reports_stay_for_independent_review():
    private, public = keys()
    (first,) = entries()
    second = deepcopy(first)
    second["source_run_id"] = "other"
    second["record"]["sentences"][0]["votes"] = ["other"] * 3
    signed = cc.sign_pack([first, second], private)
    pack = cc.verify_pack(signed, public, judge.digest(signed))
    p = packet()
    p["claim_reader"] = {"units": [saved_unit()], "results": []}
    assert not cc.apply_pack(p, batch_id="batch", pack=pack)["claim_reader"]["results"]
