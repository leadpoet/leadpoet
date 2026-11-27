"""Exercise experiment admission, partial evidence and paired evaluation without paid calls."""

from copy import deepcopy
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZipFile

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from lab_arena import collection_experiment as xp
from lab_arena import collection_experiment_export as exports
from lab_arena import (
    collection_intake as intake,
    collection_judge as judge,
    contracts,
    store,
)
from lab_arena.collection_workflow import CORRECTION_CHAMPION_REV, store_artifact
from lab_arena.service import LocalObjectStore
from tests.lab_arena import collection_intake_test as ifx
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)


def settings():
    return {
        "schema": xp.SCHEMA,
        "id": "e4",
        "question": "Do corrections improve held-out sales success?",
        "pass_mark": "Positive paired lower bound against the untuned and own-run baselines",
        "mode": "pilot",
        "budget_microusd": 400000,
        "split_sha256": "sha256:" + "b" * 64,
        "terms_sha256": contracts.hash_bytes(intake.CORRECTION_TERMS.encode()),
        "provenance_sha256": "sha256:" + "c" * 64,
        "cases": [{"scenario_id": "C06-s3013", "source_sha256": "sha256:" + "f" * 64}],
        "arms": [
            {
                "id": "adjust",
                "model": intake.MODELS[1],
                "role": "continuation",
                "repeats": 2,
                "reasoning_effort": "low",
                "capture_reasoning": True,
                "max_steps": 10,
                "max_tokens": 4096,
                "budget_microusd": 120000,
            }
        ],
    }


def signed(key, body):
    return {
        **body,
        "signature": base64.b64encode(key.sign(judge._json(body).encode())).decode(),
    }


def pub(key):
    return key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def test_budget_settings_and_routing_bind_every_arm_and_repeat():
    raw = settings()
    alt = dict(raw["arms"][0], id="fresh", role="expert", repeats=1)
    raw["arms"].append(alt)
    orig = xp.load_experiment(raw, {})
    assert [a["id"] for _, a, _ in orig.slots] == ["adjust", "fresh", "adjust"]
    assert orig.policy.qualification
    revised = deepcopy(raw)
    revised["arms"][0]["max_tokens"] -= 1
    assert orig.batch_id != xp.load_experiment(revised, {}).batch_id
    for field, value in (("budget_microusd", 2), ("mode", "live")):
        with pytest.raises(ValueError):
            xp.load_experiment(dict(raw, **{field: value}), {})
    for field, value in (
        ("model", "gpt-closed"),
        ("repeats", 13),
        ("max_tokens", 8193),
    ):
        r = deepcopy(raw)
        r["arms"][0][field] = value
        with pytest.raises(ValueError):
            xp.load_experiment(r, {})


@pytest.fixture
def database():
    migrated = list(DEFAULT_MIGRATIONS)
    for prefix in (
        720,
        721,
        722,
        723,
        724,
        725,
        726,
        727,
        728,
        729,
        732,
        733,
        738,
        740,
        744,
    ):
        (script,) = Path("scripts").glob(str(prefix) + "-*.sql")
        migrated.append(script.name)
    yield from database_with_lab_arena_migration(tuple(migrated))


def test_variable_pilot_schedule_requires_new_consent_and_never_pays(
    database, monkeypatch
):
    pg, dsn = database

    def connect():
        c = pg.connect(**dsn)
        c.autocommit = True
        return c

    db = store.PsycopgTransport(connect, role="lab_arena_service")
    raw = settings()
    spec = xp.load_experiment(raw, {})
    miner = intake.CollectionIntake(db, ifx.manager(), "pilot")
    legacy, credentials = ifx.registration("A", "pilot")
    miner.register(legacy, legacy["hotkey"], credentials)
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", "off")
    with pytest.raises(
        intake.IntakeError, match="collection_correction_terms_unavailable"
    ):
        miner.claim_experiment(spec, CORRECTION_CHAMPION_REV)
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", raw["terms_sha256"])
    assert miner.claim_experiment(spec, CORRECTION_CHAMPION_REV) is None
    for char in "BC":
        auth, credentials = ifx.registration(char, "pilot")
        auth["body"]["task_scope"] = "synthetic_corrections"
        auth["body"]["terms_sha256"] = raw["terms_sha256"]
        miner.register(auth, auth["hotkey"], credentials)
    first = miner.claim_experiment(spec, CORRECTION_CHAMPION_REV)
    assert (
        first["miner_hotkey"] != legacy["hotkey"]
        and first["price_microusd"] == 0
        and first["slot"] == 0
    )
    # The first failed attempt leaves no unresolved provider exposure; replacement must keep its slot/model.
    miner.complete(first["job_id"], first["lease_token"], "failed", {}, None, 10)
    job = miner.claim_experiment(spec, CORRECTION_CHAMPION_REV)
    assert (job["attempt"], job["slot"], job["model"], job["scenario_id"]) == (
        2,
        0,
        intake.MODELS[1],
        "C06-s3013",
    )
    assert (
        job["job_id"] != first["job_id"]
        and job["miner_hotkey"] != first["miner_hotkey"]
    )
    miner.complete(
        job["job_id"], job["lease_token"], "rejected", {}, "sha256:" + "c" * 64, 10
    )
    assert miner.claim_experiment(spec, CORRECTION_CHAMPION_REV) is None
    for char in "DE":
        auth, credentials = ifx.registration(char, "pilot")
        auth["body"].update(
            task_scope="synthetic_corrections", terms_sha256=raw["terms_sha256"]
        )
        miner.register(auth, auth["hotkey"], credentials)
    second = miner.claim_experiment(spec, CORRECTION_CHAMPION_REV)
    assert (second["slot"], second["attempt"], second["model"]) == (
        1,
        1,
        intake.MODELS[1],
    )
    miner.complete(
        second["job_id"],
        second["lease_token"],
        "rejected",
        {},
        "sha256:" + "d" * 64,
        10,
    )
    assert miner.claim_experiment(spec, CORRECTION_CHAMPION_REV) is None
    assert db.rpc("collection_reward_start_v1", {})["first_epoch"] is None
    db.close()


def test_refuse_unsigned_student_sources_or_unqualified_training_export(
    tmp_path, monkeypatch
):
    raw = settings()
    key = Ed25519PrivateKey.generate()
    public = tmp_path / "key.pem"
    public.write_bytes(pub(key))
    monkeypatch.setenv("LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE", str(public))
    s = {"student_model": "student/4b"}
    h = xp._hash(s)
    raw["cases"][0]["source_sha256"] = h
    body = {
        "schema": "collection.poet-student-sources.v1",
        "producer": "training-team",
        "split_sha256": raw["split_sha256"],
        "sources": {
            h: {
                "role": "untuned_student",
                "model": "student/4b",
                "model_revision": "d" * 40,
                "source_run_id": "real-run-test-fixture",
                "rights": "owned_synthetic_training_and_review",
                "split": "training",
            }
        },
    }
    sig = signed(key, body)
    raw["provenance_sha256"] = xp._hash(sig)
    spec = xp.load_experiment(raw, {})
    assert xp.verify_provenance(spec, sig, {h: s}) == body
    with pytest.raises(ValueError):
        xp.verify_provenance(spec, signed(Ed25519PrivateKey.generate(), body), {h: s})
    with pytest.raises(ValueError):
        xp.verify_provenance(spec, sig, {h: {"student_model": "another/model"}})


def test_feedback_binds_trained_dataset_and_paired_heldout_tasks(tmp_path):
    key = Ed25519PrivateKey.generate()
    objects = LocalObjectStore(tmp_path / "s3")
    d = "sha256:" + "a" * 64
    split = {
        "schema": "collection.split.v1",
        "training": {
            k: ["train"] for k in ("template_id", "rule_pool", "tool_variant")
        },
        "heldout": {
            k: ["heldout"] for k in ("template_id", "rule_pool", "tool_variant")
        },
    }
    sh = xp._hash(split)
    partition = {
        k: "sha256:" + exports.sha256(b"heldout").hexdigest() for k in split["heldout"]
    }
    metadata = {
        "schema": exports.CORRECTIONS,
        "batch_id": "xp-" + "b" * 64,
        "dataset_sha256": d,
        "split_sha256": sh,
    }
    store_artifact(objects, metadata["batch_id"], "export", "release", metadata)
    store_artifact(objects, metadata["batch_id"], "export", "split", split)
    body = {
        "schema": exports.FEEDBACK,
        "dataset_sha256": d,
        "run_id": "train-1",
        "server_sha256": d,
        "heldout_sha256": d,
        "split_sha256": sh,
        "poet_revision": "d" * 40,
        "base_revision": "qwen-untuned",
        "tuned_revision": "qwen-tuned",
        "own_runs_revision": "qwen-own",
        "tasks": [
            {
                "id_sha256": d,
                "base": False,
                "tuned": True,
                "own_runs": True,
                "partition_sha256": partition,
            }
        ],
    }
    ref = exports.record_feedback(metadata, signed(key, body), objects, pub(key))
    value = json.loads(objects.get(ref["key"]))
    assert (
        value["comparisons"]["base"]["lift"] == 1
        and value["comparisons"]["own_runs"]["lift"] == 0
    )
    assert value["comparisons"]["base"]["lower_95"] < 0
    duplicate = dict(body, tasks=body["tasks"] * 2)
    with pytest.raises(ValueError, match="repeated"):
        exports.record_feedback(metadata, signed(key, duplicate), objects, pub(key))
    with pytest.raises(ValueError, match="dataset"):
        exports.record_feedback(
            metadata,
            signed(key, dict(body, dataset_sha256="sha256:" + "f" * 64)),
            objects,
            pub(key),
        )


def test_correction_first_polling_falls_through_only_when_idle_or_inactive(
    tmp_path, monkeypatch
):
    from scripts import run_locked_collection as runner

    baseline = tmp_path / "qual"
    baseline.write_text(json.dumps({"name": "legacy"}))
    candidate = tmp_path / "next"
    raw = settings()
    candidate.write_text(json.dumps(raw))
    candidate.chmod(0o600)
    argv = [
        "--pilot-only",
        "--batch",
        str(baseline),
        "--next-correction",
        str(candidate),
        "--experiment-inputs",
        str(tmp_path),
        "--prompt",
        str(tmp_path / "rubric"),
    ]
    seen = []
    monkeypatch.setattr(
        runner, "_run_options", lambda args, p, v, **kw: seen.append(v) or 0
    )
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", "off")
    assert runner.main(argv) == 0 and seen == [{"name": "legacy"}]
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", raw["terms_sha256"])
    seen.clear()
    assert runner.main(argv) == 0 and seen == [raw, {"name": "legacy"}]
    monkeypatch.setattr(
        runner, "_run_options", lambda args, p, v, **kw: seen.append(v) or 1
    )
    seen.clear()
    assert runner.main(argv) == 0 and seen == [raw]


def test_staged_correction_input_failure_does_not_starve_baseline(
    tmp_path, monkeypatch, capsys
):
    from scripts import run_locked_collection as runner

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"name": "ordinary-pack"}))
    candidate = tmp_path / "next.json"
    raw = settings()
    candidate.write_text(json.dumps(raw))
    candidate.chmod(0o600)
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", raw["terms_sha256"])
    monkeypatch.delenv("LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE", raising=False)
    argv = [
        "--pilot-only",
        "--batch",
        str(baseline),
        "--next-correction",
        str(candidate),
        "--experiment-inputs",
        str(tmp_path),
        "--prompt",
        str(tmp_path / "rubric"),
    ]
    seen = []

    def dispatch(args, p, candidate_raw, **options):
        seen.append(candidate_raw)
        if candidate_raw.get("schema") == xp.SCHEMA:
            raise ValueError("private input or trusted producer key missing")
        return 1

    monkeypatch.setattr(runner, "_run_options", dispatch)
    assert runner.main(argv) == 0 and seen == [raw, {"name": "ordinary-pack"}]
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "correction_held" and "private input" not in str(error)

    def unknown_admission(args, p, candidate_raw, *, before_claim):
        before_claim()
        raise OSError("lost claim acknowledgement")

    monkeypatch.setattr(runner, "_run_options", unknown_admission)
    with pytest.raises(OSError, match="lost claim acknowledgement"):
        runner.main(argv)


@pytest.mark.parametrize(
    ("keyed", "source_free"), [(False, False), (True, False), (True, True)]
)
def test_real_harness_checkpoint_and_archive_under_funded_job_contract(
    database, tmp_path, monkeypatch, keyed, source_free
):
    """Native PG worker path; only network, payer and training producer are controlled fixtures."""
    import os
    from lab_arena import broker
    from lab_arena.collection_workflow import workflow_sources, prepare_student_source
    from tests.lab_arena.collection_judge_test import PROMPT

    poet = os.getenv("COLLECTION_TEST_POET_REPO")
    agent = os.getenv("COLLECTION_TEST_AGENT_REPO")
    if not poet or not agent:
        pytest.skip("requires both pinned generator source checkouts")
    pg, dsn = database

    def conn():
        c = pg.connect(**dsn)
        c.autocommit = True
        return c

    db = store.PsycopgTransport(conn, role="lab_arena_service")
    objects = LocalObjectStore(tmp_path / "store")
    key = Ed25519PrivateKey.generate()
    pub_path = tmp_path / "test-custodian"
    pub_path.write_bytes(pub(key))
    monkeypatch.setenv("LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE", str(pub_path))
    raw = settings()
    raw["cases"][0]["scenario_id"] = "P7-s3013"
    if keyed:
        raw["claim_protocol"] = "collection.fresh-claims.v2"
    raw["arms"][0].update(repeats=1, max_steps=90)
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", raw["terms_sha256"])
    from bittensor_wallet import Keypair

    enrolled = intake.CollectionIntake(db, ifx.manager(), "pilot")
    signer = Keypair.create_from_uri("//TrajectoryPilot")
    auth, cred = ifx.registration("H", "pilot")
    auth = contracts.build_signed_request(
        scope=intake.SCOPE["register"],
        round_id=intake.scope_id("pilot"),
        hotkey=signer.ss58_address,
        body=dict(
            auth["body"],
            task_scope=intake.CORRECTION_SCOPE,
            terms_sha256=raw["terms_sha256"],
        ),
        timestamp=1,
        sign_message=lambda msg: signer.sign(msg.encode()).hex(),
    )
    enrolled.register(auth, auth["hotkey"], cred)
    with workflow_sources(
        agent, poet, corrections=True, builder_revision=xp.source_revisions(raw)[1]
    ) as src:
        harness, builders = src
        br = builders.branch
        case = builders.build("P7", 3013)
        broken = dict(builders.perfect(case), answer="wrong final receipt")
        path = tmp_path / "student"
        br.fake_source(path, ("P7", 3013, builders.PLAIN), case, broken)
        _, tools = br.fresh(case, tmp_path / "world")
        transcript = [
            json.loads(l) for l in (path / "transcript.jsonl").read_text().splitlines()
        ]
        transcript[0]["content"] = br.run_any.SYSTEM.format(
            names=", ".join(t["function"]["name"] for t in tools),
            today=br.run_any.TODAY_TEXT,
        )
        (path / "transcript.jsonl").write_text(
            "".join(json.dumps(l) + "\n" for l in transcript)
        )
        m = json.loads((path / "meta.json").read_bytes())
        m["local"] = True
        (path / "meta.json").write_text(json.dumps(m))
        student = prepare_student_source(harness, builders, path, model=m["model"])
        h = xp._hash(student)
        raw["cases"][0]["source_sha256"] = h

        def stage(value):
            d = xp._hash(value)
            p = tmp_path / (d[7:] + ".json")
            p.write_text(json.dumps(value))
            p.chmod(0o600)
            return d

        stage(student)
        split = builders.training_partition("P7")
        raw["split_sha256"] = stage(split)
        att = signed(
            key,
            {
                "schema": "collection.poet-student-sources.v1",
                "producer": "test-fixture",
                "split_sha256": raw["split_sha256"],
                "sources": {
                    h: {
                        "role": "untuned_student",
                        "model": student["student_model"],
                        "model_revision": "a" * 40,
                        "source_run_id": "fixture",
                        "rights": "owned_synthetic_training_and_review",
                        "split": "training",
                    }
                },
            },
        )
        raw["provenance_sha256"] = stage(att)
        if source_free:
            raw.pop("provenance_sha256")
            raw["cases"] = [{"scenario_id": "P7-s3013"}]
            raw["arms"][0]["role"] = "expert"
        spec = xp.load_experiment(raw, PROMPT)
        inputs = xp.read_inputs(spec, tmp_path, src)
        good = builders.perfect(case)
        todo = br.still_to_do(
            br.reference_plan(case, good, tmp_path / "reference"),
            [] if source_free else student["checkpoint"]["log"],
        )
        ask = br.scripted_expert(
            todo,
            (
                "incorrect final receipt"
                if source_free
                else (good.get("answer") or "Done.")
            ),
        )
        from lab_arena.collection_broker import batch_versions

        versions = batch_versions(spec.routes)
        handled = []

        class Transport:
            def send(self, **kwargs):
                b = json.loads(kwargs["body"])
                model = b["model"]
                assert (
                    b["provider"]["zdr"] and b["provider"]["enforce_distillable_text"]
                )
                if model == spec.slots[0][1]["model"]:
                    doc = ask(None, None, b, None, None)
                    for call in doc["choices"][0]["message"].get("tool_calls", []):
                        call["id"] = "new-" + str(len(handled))
                else:
                    unit = json.loads(b["messages"][-1]["content"])
                    if keyed:
                        assert set(unit) == {"sentences", "sources", "context"}
                        assert type(unit["sentences"]) is dict
                        labels = dict.fromkeys(unit["sentences"], "supported")
                    else:
                        assert set(unit) == {"sentences", "sources"}
                        labels = ["supported"] * len(unit["sentences"])
                    answer = json.dumps({"labels":labels})
                    doc = {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": answer},
                            }
                        ]
                    }
                handled.append(model)
                doc.update(
                    id="gen-" + str(len(handled)),
                    model=model,
                    provider="Fixture",
                    openrouter_metadata={
                        "endpoints": {
                            "available": [
                                {
                                    "selected": True,
                                    "model": versions[model],
                                    "provider": "Fixture",
                                }
                            ]
                        }
                    },
                    usage={
                        "cost": 0.00001,
                        "completion_tokens": 3,
                        "prompt_tokens": 45,
                    },
                )
                return broker.ProviderResponse(200, {}, json.dumps(doc).encode())

        pricing = broker.price_table_from_models_response(
            {
                "data": [
                    {
                        "id": i,
                        "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
                    }
                    for i in versions
                ]
            },
            fetched_at="2026-11-24T00:00:00Z",
        )
        pricing["endpoint_providers"] = dict.fromkeys(versions, "Fixture")
        assert (
            xp.run_experiment(
                spec=spec,
                sources=src,
                inputs=inputs,
                intake=enrolled,
                database=db,
                objects=objects,
                transport=Transport(),
                pricing=pricing,
                owner_lookup=lambda hk: SimpleNamespace(coldkey=hk),
                judge_key="test-owner-only",
                max_jobs=1,
                epoch=lambda: 1,
            )
            == 1
        )
    costs = db.rpc("collection_pilot_costs_v1", {"p_batch": spec.batch_id})
    (j,) = costs
    assert j["status"] == "accepted" and j["generator_calls"] == handled.count(
        spec.slots[0][1]["model"]
    )
    assert j["miner_cost_microusd"] == 10 * j["generator_calls"]
    rid = j["job_id"]
    if source_free:
        # The qualification cryptography is exercised by the signed six-model test.
        monkeypatch.setattr(
            exports, "verified_panel", lambda *a, **kw: {"test_panel": "verified"}
        )
        # A sibling's unresolved call can be retried later without suppressing this original.
        pending_id = "00000000-0000-4000-8000-000000000002"
        def with_pending(name, args):
            data = db.rpc(name, args)
            if name in {"collection_pilot_costs_v1", "collection_training_consents_v1"}:
                extra = dict(data[0], job_id=pending_id, unresolved=1)
                if name == "collection_pilot_costs_v1":
                    extra.update(attempt=2, status="failed", artifact_sha256=None)
                return [*data, extra]
            return data
        path = tmp_path / "full-negative.zip"
        metadata = exports.write_dataset(
            spec,
            SimpleNamespace(rpc=with_pending),
            objects,
            path,
            qualification={"test": True},
            full_trajectories=True,
        )
        assert metadata["pending_runs"] == [{"run_id": pending_id, "slot": j["slot"], "reason": "billing_pending"}]
        assert (
            metadata["trajectory_rows"] == metadata["verified_failures"] == 1
            and metadata.get("correction_pairs", 0) == 0
        )
        with ZipFile(path) as z:
            row = json.loads(z.read(f"trajectories/{rid}.jsonl"))
            assert (
                row["verified_task_pass"] is False and row["final_turn_present"] is True
            )
            assert row["native_grade"]["score"] < 1 and row["first_failed_check"]
            assert (
                row["experiment"]["role"] == "expert"
                and "student_source_sha256" not in row
            )
            assert row["messages"][-1]["content"] == "incorrect final receipt"
            assert row["miner_cost_microusd"] == j["miner_cost_microusd"]
            assert not any(n.startswith("pairs/") for n in z.namelist())
        db.close()
        return
    # The private wire reply is bound to the original billing receipt. Its bytes
    # are retained before reasoning alias normalization, grading, or claim review.
    wire_ids = []
    total_billed = 0
    for number in range(1, j["generator_calls"] + 1):
        import hashlib

        k = f"wire-generate-{number}"
        ref = json.loads(
            objects.get_bounded(
                f"collection/{spec.batch_id}/{rid}/saved/{k}.json", 4096
            )
        )
        raw = objects.get_bounded(ref["key"], 1024 * 1024)
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == ref["sha256"]
        wire = json.loads(raw)
        assert (
            hashlib.sha256(wire["body_utf8"].encode()).hexdigest()
            == wire["response_sha256"]
        )
        assert json.loads(wire["body_utf8"])["id"] == wire["provider_generation_id"]
        total_billed += wire["miner_cost_microusd"]
        wire_ids.append(wire["provider_generation_id"])
    assert (
        len(set(wire_ids)) == j["generator_calls"]
        and total_billed == j["miner_cost_microusd"]
    )
    out = tmp_path / "audit.zip"
    audit = exports.write_dataset(spec, db, objects, out)
    assert audit["schema"] == exports.AUDIT and audit["rows"] == 0
    with ZipFile(out) as f:
        row = json.loads(f.read(rid + "/generator.json"))
        pkt = json.loads(f.read(rid + "/packet.json"))
        assert any(n.startswith(rid + "/events/") for n in f.namelist())
        pair = exports._pair(row, pkt, student)
        assert pair and pair["rejected"] != pair["chosen"]
        assert pair["context"] and row["correction"]["prefix_messages"] == len(
            pair["context"]
        )
        assert pair["final_turn_retained"]

    def decline(*args, **kw):
        raise ValueError("expected independent review")

    monkeypatch.setattr(exports, "verified_panel", decline)
    with pytest.raises(ValueError, match="independent review"):
        exports.write_dataset(
            spec,
            db,
            objects,
            tmp_path / "unapproved",
            qualification={"approver_key": b"untrusted"},
        )
    assert not (tmp_path / "unapproved").exists()
    # Isolate the recipe after the independently tested panel authorization gate.
    monkeypatch.setattr(
        exports, "verified_panel", lambda *a, **kw: {"test_panel": "verified"}
    )
    released = exports.write_dataset(
        spec,
        db,
        objects,
        tmp_path / "fixture-release.zip",
        qualification={"test": True},
    )
    assert (
        released["schema"] == exports.CORRECTIONS
        and released["rows"] == released["endings_kept"] == 1
    )
    assert released["price_observations"] == [
        {
            "role": "continuation",
            "task_class": "P7",
            "model": j["model"],
            "measured_runs": 1,
            "miner_cost_microusd": j["miner_cost_microusd"],
            "proposed_fixed_microusd": (5 * j["miner_cost_microusd"] + 3) // 4,
        }
    ]
    with ZipFile(tmp_path / "fixture-release.zip") as output:
        assert sorted(output.namelist()) == ["metadata.json", f"pairs/{rid}.jsonl"]
        chosen = json.loads(output.read(f"pairs/{rid}.jsonl"))
        assert (
            chosen["chosen"] == pair["chosen"]
            and chosen["expert_final_outcome"]["final_turn"] == row["messages"][-1]
        )
        assert "evidence" not in chosen and "checkpoint" not in chosen

    full_path = tmp_path / "full-trajectories.zip"
    metadata = exports.write_dataset(
        spec,
        db,
        objects,
        full_path,
        qualification={"test": True},
        full_trajectories=True,
    )
    with ZipFile(full_path) as z:
        traj = json.loads(z.read(f"trajectories/{rid}.jsonl"))
        pref = json.loads(z.read(f"pairs/{rid}.jsonl"))
        assert (
            traj["messages"] == row["messages"]
            and traj["final_turn_present"]
            and traj["verified_task_pass"] is True
        )
        assert (
            traj["miner_cost_microusd"] == j["miner_cost_microusd"]
            and traj["builder_commit"]
        )
        assert traj["experiment"]["role"] == "continuation" and traj["step_checks"]
        assert pref["rejected"] == pair["rejected"] and pref["chosen"] == pair["chosen"]
        assert metadata["correction_pairs"] == metadata["trajectory_rows"] == 1
        assert metadata["dataset_sha256"] != released["dataset_sha256"]
        assert all("reference_answer" not in msg for msg in traj["messages"])
    # If an interruption lost an external attestation, retain only the hold.
    # A missing proof must never release the source transcript or abort the archive.
    partial_path = tmp_path / "partial-missing-attestation.zip"
    optional = objects.get_optional_bounded
    def without_attestation(path, limit):
        return None if path.endswith("/saved/producer-attestation.json") else optional(path, limit)
    def interrupted(name, args):
        records = db.rpc(name, args)
        if name == "collection_pilot_costs_v1":
            return [dict(record, status="failed", artifact_sha256=None) for record in records]
        return records
    with monkeypatch.context() as part:
        part.setattr(objects, "get_optional_bounded", without_attestation)
        exports.write_dataset(spec, SimpleNamespace(rpc=interrupted), objects, partial_path,
                              full_trajectories=True, qualification={"test": True}, publish=False)
    with ZipFile(partial_path) as z:
        held = json.loads(z.read(f"trajectories/{rid}.jsonl"))
        assert held["capture_hold"] == "missing_producer_attestation" and held["messages"] == []
        assert held["verified_task_pass"] is None and f"pairs/{rid}.jsonl" not in z.namelist()
    # Missing original events must also hold release, even if DB and summary survive.
    first_event = objects._path(f"collection/{spec.batch_id}/{rid}/events/00000.json")
    first_event.unlink()
    with pytest.raises(ValueError, match="journal"):
        exports.write_dataset(spec, db, objects, tmp_path / "missing-audit")
    db.close()
