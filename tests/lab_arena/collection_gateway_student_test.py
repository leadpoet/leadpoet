"""Native gateway provenance uses only read-only billing lookups and write-once evidence."""

from copy import deepcopy
import json
import os
import pytest
from lab_arena.broker import ProviderResponse
from lab_arena.contracts import ArenaContractError
from lab_arena import collection_gateway_student as origin, collection_experiment as xp
from lab_arena.collection_checkpoint import FILES
from lab_arena.collection_workflow import (
    experiment_revisions,
    store_artifact,
    workflow_sources,
)
from lab_arena.service import LocalObjectStore
from scripts.collection_review import write_new
from tests.lab_arena.collection_experiment_test import settings


class OwnerAccount:
    def __init__(self):
        self.ids = []
        self.status = 200
        self.cost = 0.00012

    def send(self, *, method, url, body, headers, **unused):
        assert (
            method == "GET"
            and not body
            and headers["Authorization"] == "Bearer fixture-owner-key"
        )
        assert url.startswith("https://openrouter.ai/api/v1/generation?id=gen-")
        gen = url.split("id=")[1]
        self.ids.append(gen)
        d = {
            "id": gen,
            "model": "mistralai/ministral-3b-2512",
            "provider_name": "Mistral",
            "cancelled": False,
            "usage": self.cost,
            "total_cost": self.cost,
        }
        return ProviderResponse(self.status, {}, json.dumps({"data": d}).encode())


def test_original_gateway_student_replays_without_external_signature(
    tmp_path, monkeypatch
):
    agent, poet = [
        os.getenv(k)
        for k in ("COLLECTION_TEST_AGENT_REPO", "COLLECTION_TEST_POET_REPO")
    ]
    if not agent or not poet:
        pytest.skip("requires pinned public agent and private Poet builder")
    db = LocalObjectStore(tmp_path / "store")
    raw = settings()
    model = "mistralai/ministral-3b-2512"
    run = "p7-owner-fixture"
    monkeypatch.delenv("LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE", raising=False)
    with workflow_sources(
        agent,
        poet,
        corrections=True,
        builder_revision=experiment_revisions(["P7-s3013"])[1],
    ) as sources:
        b = sources[1]
        br = b.branch
        task = b.build("P7", 3013)
        split = b.training_partition("P7")
        folder = tmp_path / "native"
        br.fake_source(
            folder,
            ("P7", 3013, b.PLAIN),
            task,
            dict(b.perfect(task), answer="wrong final receipt"),
        )
        _, tools = br.fresh(task, tmp_path / "world")
        stream = [
            json.loads(s)
            for s in (folder / "transcript.jsonl").read_text().splitlines()
        ]
        stream[0]["content"] = br.run_any.SYSTEM.format(
            names=", ".join(t["function"]["name"] for t in tools),
            today=br.run_any.TODAY_TEXT,
        )
        (folder / "transcript.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in stream)
        )
        meta = json.loads((folder / "meta.json").read_bytes())
        meta.update(model=model, local=False)
        (folder / "meta.json").write_text(json.dumps(meta))
        hist = br.history_of(folder)
        calls = []
        for n, at in enumerate(hist["turns"]):
            turn = hist["messages"][at]
            calls.append(
                {
                    "request": {
                        "model": model,
                        "tools": tools,
                        "messages": hist["messages"][:at],
                        "provider": {
                            "only": ["mistral/zdr"],
                            "allow_fallbacks": False,
                            "zdr": True,
                            "data_collection": "deny",
                            "enforce_distillable_text": True,
                        },
                    },
                    "response": {
                        "id": f"gen-fixture-{n}",
                        "model": model,
                        "provider": "Mistral",
                        "usage": {"cost": 0.00012},
                        "openrouter_metadata": {
                            "endpoints": {
                                "available": [
                                    {
                                        "model": model,
                                        "provider": "Mistral",
                                        "selected": True,
                                    }
                                ]
                            }
                        },
                        "choices": [
                            {
                                "message": turn,
                                "finish_reason": (
                                    "tool_calls" if turn.get("tool_calls") else "stop"
                                ),
                            }
                        ],
                    },
                }
            )
        native = {
            "schema": origin.NATIVE,
            "original": {n: (folder / n).read_text() for n in FILES},
            "provenance": {
                "schema": "leadpoet.collection.owner-student.v1",
                "model": model,
                "revision": model,
                "route": "mistral/zdr",
                "provider": "Mistral",
                "calls": calls,
            },
        }
        lnk = store_artifact(db, origin.ORIGIN, run, "native", native)
        assert origin.native_scenario(db, run, lnk["sha256"]) == "P7-s3013"
        acct = OwnerAccount()
        src, proof = origin.stage_gateway_source(
            db,
            sources,
            run_id=run,
            native_sha256=lnk["sha256"],
            split=split,
            transport=acct,
            owner_key="fixture-owner-key",
        )
        assert len(acct.ids) == len(calls) and src["rejected_fatal"]
        raw.update(
            split_sha256=xp._hash(split),
            provenance_sha256=xp._hash(proof),
            cases=[{"scenario_id": src["scenario_id"], "source_sha256": xp._hash(src)}],
        )
        spec = xp.load_experiment(raw, {})
        pack = tmp_path / "pack"
        pack.mkdir()
        for doc in (src, proof, split):
            write_new(pack / (xp._hash(doc)[7:] + ".json"), doc)
        resolved = xp.read_inputs(spec, pack, sources, objects=db)
        assert resolved[0][xp._hash(src)][0] == src
        # Swapping untrusted files or a local generation receipt for the original is insufficient.
        altered = deepcopy(src)
        altered["provenance"]["calls"][0]["response"]["usage"]["cost"] = 0
        with pytest.raises(ValueError, match="privately verified"):
            xp.verify_provenance(spec, proof, {xp._hash(src): altered}, objects=db)
        for cost, status, pattern in [
            (0.45, 200, "cost differs"),
            (0.00012, 404, "HTTP 404"),
        ]:
            acct.cost, acct.status = cost, status
            with pytest.raises(ValueError, match=pattern):
                origin.stage_gateway_source(
                    db,
                    sources,
                    run_id=run,
                    native_sha256=lnk["sha256"],
                    split=split,
                    transport=acct,
                    owner_key="fixture-owner-key",
                )
        db._path(lnk["key"]).unlink()
        with pytest.raises(ArenaContractError, match="no object"):
            xp.verify_provenance(spec, proof, {xp._hash(src): src}, objects=db)


def test_lookup_cannot_follow_injected_ids_or_anonymous_origins():
    account = OwnerAccount()
    for rid, secret in (
        ("https://example.invalid/secret", "not-a-real-key"),
        ("gen-something", None),
    ):
        with pytest.raises(ValueError, match="credential unavailable"):
            origin.read_owner_charge(account, secret, {"id": rid}, {})
    assert not account.ids
