"""Authenticate Poet node trajectories and replay the original local tools before training."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from lab_arena import collection_judge as judge
from lab_arena.collection_checkpoint import (
    FILES,
    MAX_SOURCE_BYTES,
    correction_scenario,
    require_private_evidence,
)
from lab_arena.collection_programs import (
    STUDENT,
    STUDENT_REVISION,
    _hash,
    identity,
    sandbox,
    producer_partition,
)
from lab_arena.collection_quality import admit_built_split
from scripts.collection_review import read_file

NODES = "collection.poet-native-runs.v3"
RESTATEMENTS = "collection.poet-restatements.v3"


def authenticated(document, public_key, schema):
    key = (
        load_pem_public_key(read_file(public_key, maximum=8192, private=False))
        if isinstance(public_key, (str, Path))
        else public_key
    )
    body = {k: v for k, v in document.items() if k != "signature"}
    if (
        not isinstance(key, Ed25519PublicKey)
        or set(body) != {"schema", "producer", "records"}
        or body["schema"] != schema
        or not isinstance(body["producer"], str)
        or not isinstance(body["records"], dict)
        or not 1 <= len(body["records"]) <= 10000
    ):
        raise ValueError("invalid signed Poet run index")
    try:
        key.verify(
            base64.b64decode(document["signature"], validate=True),
            judge._json(body).encode(),
        )
    except (KeyError, ValueError, TypeError, InvalidSignature) as exc:
        raise ValueError("invalid node index signature") from exc
    return body


def _read_record(folder, digest):
    import re

    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest)):
        raise ValueError("invalid node object identity")
    value = json.loads(
        read_file(folder / (digest[7:] + ".json"), maximum=MAX_SOURCE_BYTES * 3)
    )
    if _hash(value) != digest:
        raise ValueError("node dossier differs from signed input")
    return value


def node_trajectory(
    original,
    *,
    harness,
    builders,
    producer_run_id,
    student_model,
    model_revision,
    repeat,
):
    """Return a checked run, including an unfinished/rejected one. A node's grade is never authoritative."""
    from lab_arena.collection_checkpoint import _student_history

    if type(repeat) is not int or not 0 <= repeat < 8:
        raise ValueError("unexpected student repeat")
    if student_model.lower() != STUDENT or model_revision != STUDENT_REVISION:
        raise ValueError("unexpected student weights")
    if (
        set(original) != set(FILES)
        or sum(len(v.encode()) for v in original.values()) > MAX_SOURCE_BYTES
    ):
        raise ValueError("missing or oversized original node dossier")
    for name, data in original.items():
        value = (
            [json.loads(line) for line in data.splitlines() if line.strip()]
            if name.endswith(".jsonl")
            else json.loads(data)
        )
        require_private_evidence(value)
        if builders.KEYS.search(data):
            raise ValueError("private credential in synthetic node log")
    spec = json.loads(original["trace.json"])
    scenario = f"{spec['task']}-s{spec['seed']}"
    family, variant, ordinal = identity(scenario)
    entry = sandbox(family)
    if entry["kind"] != "train" or builders.source_revision != entry["builder"]:
        raise ValueError("wrong node producer or held-out sandbox")
    task = correction_scenario(builders, {"original": original})
    part = producer_partition(builders, family, spec["seed"])
    if _hash(part) != entry["partition"]:
        raise ValueError("node partition differs from registry")
    stamp = admit_built_split(builders, task, spec.get("shape"), part)
    meta = json.loads(original["meta.json"])
    end = json.loads(original["out.json"])
    if (
        meta.get("local") is not True
        or meta.get("model") != student_model
        or meta.get("repeat", meta.get("run")) != repeat
    ):
        raise ValueError("node run lacks exact local student identity")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        folder = root / "original"
        folder.mkdir()
        for name, data in original.items():
            (folder / name).write_text(data, encoding="utf8")
        br = builders.branch
        hist = br.history_of(folder)
        if len(hist["turns"]) > 256:
            raise ValueError("node trace too long for bounded replay")
        messages = hist["messages"]
        tools = br.replay(task, hist, root / "replay", grade=False)
        if tools["differs"] is not None:
            raise ValueError("native tool reply differs from deterministic replay")
        actual = br.snapshot(task, tools["world"], tools["ws"])
        if any(
            judge.digest(actual[k]) != judge.digest(end[k])
            for k in ("files", "log", "state")
        ):
            raise ValueError("node final state differs from replay")
        if messages[:2] != [
            {
                "role": "system",
                "content": br.run_any.SYSTEM.format(
                    names=", ".join(x["function"]["name"] for x in tools["tools"]),
                    today=br.run_any.TODAY_TEXT,
                ),
            },
            {"role": "user", "content": task["prompt"]},
        ]:
            raise ValueError("node input does not match its frozen harness")
        if builders.heldout.found_in(judge._json(messages)):
            raise ValueError("held-out data in node trace")
        complete = (
            meta.get("finish") == "answered"
            and not hist["nudges"]
            and not any(
                x
                for k, x in (meta.get("fixes") or {}).items()
                if k != "max_tokens_lowered"
            )
        )
        answer = messages[-1].get("content") or "" if messages else ""
        if "think>" in answer:
            answer = br.run_any.think_free(answer).strip()
        if complete and (
            not hist["turns"]
            or messages[-1]["role"] != "assistant"
            or messages[-1].get("tool_calls")
            or answer != end.get("answer")
        ):
            raise ValueError("student completion is not the observed final answer")
        grade = builders.grade(task, end)
        if judge.digest(grade) != judge.digest(json.loads(original["check.json"])):
            raise ValueError("submitted node grade differs from native grade")
        history = _student_history(builders, task, hist, root / "states")
        passed = (
            (grade["score"] == 1 and not grade.get("hard_fail")) if complete else None
        )
        mistake = None
        if passed is False:
            located = br.locate(task, hist, end, root / "branch", grade)
            turn = located.get("branch")
            mistake = {
                "turn": turn,
                "check": grade.get("hard_fail")
                or next(
                    (k for k, v in grade.get("credits", {}).items() if v != 1),
                    "native_grade",
                ),
                "kind": "observed_branch" if type(turn) is int else "terminal_check",
            }
        literal = [
            json.loads(line)
            for line in original["transcript.jsonl"].splitlines()
            if line.strip()
        ]
        assistant = [m for m in literal if m.get("role") == "assistant"]
        if len(assistant) == len(hist["turns"]):
            for idx, native in zip(hist["turns"], assistant, strict=True):
                for name in ("reasoning", "reasoning_content"):
                    if isinstance(native.get(name), str) and native[name]:
                        messages[idx][name] = native[name]
                if not messages[idx].get("reasoning") and not messages[idx].get(
                    "reasoning_content"
                ):
                    import re

                    exposed = re.match(
                        r"^\s*<think>(.*?)</think>",
                        native.get("raw_content") or "",
                        re.DOTALL,
                    )
                    if exposed:
                        messages[idx]["reasoning"] = exposed[1]
        return {
            "run_id": producer_run_id,
            "model": STUDENT,
            "repeat": repeat,
            "model_revision": model_revision,
            "role": "student",
            "scenario_id": scenario,
            "ordinal": ordinal,
            "variant": variant,
            "case_pattern": ordinal % entry["patterns"],
            "split": stamp,
            "messages": messages,
            "tools": tools["tools"],
            "label_turns": hist["turns"],
            "no_loss_turns": [] if complete else hist["turns"],
            "verified_complete": bool(complete),
            "verified_task_pass": passed,
            "first_failed_check": mistake,
            "native_grade": grade,
            "state_history": history,
            "harness_commit": harness.source_revision,
            "builder_commit": builders.source_revision,
        }


def verified_node_runs(bundle, *, harness, builders, public_key):
    bundle = Path(bundle)
    body = authenticated(
        json.loads(read_file(bundle / "nodes.json")), public_key, NODES
    )
    seen = set()
    draws = set()
    fields = {
        "producer_run_id",
        "scenario_id",
        "student_model",
        "model_revision",
        "repeat",
        "source_sha256",
    }
    for info in body["records"].values():
        if not isinstance(info, dict) or set(info) != fields:
            raise ValueError("invalid signed native run")
        draw = (info["scenario_id"], info["model_revision"], info["repeat"])
        if info["producer_run_id"] in seen or draw in draws:
            raise ValueError("duplicate authenticated student draw")
        seen.add(info["producer_run_id"])
        draws.add(draw)
    for digest, info in body["records"].items():
        row = node_trajectory(
            _read_record(bundle / "nodes", digest),
            harness=harness,
            builders=builders,
            producer_run_id=info["producer_run_id"],
            student_model=info["student_model"],
            model_revision=info["model_revision"],
            repeat=info["repeat"],
        )
        if row["scenario_id"] != info["scenario_id"]:
            raise ValueError("node scenario differs from signed identity")
        row.update(
            producer=body["producer"],
            native_sha256=digest,
            source_sha256=info["source_sha256"],
        )
        if row["source_sha256"] is not None:
            src = _read_record(bundle / "inputs", row["source_sha256"])
            from lab_arena.collection_checkpoint import validate_student_source

            validate_student_source(harness, builders, src)
            if _hash(src["original"]) != digest:
                raise ValueError("student branch belongs to another native run")
        yield row


def restatement_verifier(builders):
    """Use the separately reviewed node verifier with the original task builder on sys.path."""
    import subprocess, sys, types
    from hashlib import sha256

    pin = "f4f006eb556f839b42d9f90b823654230cb627a1"
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(builders.poet_repo),
            "show",
            pin + ":harness/gtm-tasks/student_v3.py",
        ]
    )
    if (
        sha256(source).hexdigest()
        != "d03ed81848aa2a9d1a10d0079de71ec734b589f8508af94770c8a8544e5e1a53"
    ):
        raise ValueError("restatement verifier digest changed")
    module = types.ModuleType("collection_student_verifier")
    module.__file__ = str(Path(builders.__file__).parent / "student_v3.py")
    saved = list(sys.path)
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    finally:
        sys.path[:] = saved
    return module


def verified_restatements(bundle, *, builders, public_key, teacher_by_id):
    bundle = Path(bundle)
    body = authenticated(
        json.loads(read_file(bundle / "restatements.json")), public_key, RESTATEMENTS
    )
    verifier = restatement_verifier(builders)
    seen = set()
    groups = {}
    for digest, run_id in body["records"].items():
        record = _read_record(bundle / "restatements", digest)
        if (
            record["producer_run_id"] != run_id
            or str(record["student_model"]).lower() != STUDENT
            or record["model_revision"] != STUDENT_REVISION
            or run_id in seen
        ):
            raise ValueError("restatement student identity or run ID differs")
        seen.add(run_id)
        key = (
            record["teacher_run_id"],
            record["teacher_trajectory_sha256"],
            record["checkpoint_sha256"],
            record["message_index"],
        )
        refs = groups.setdefault(key, {})
        rep = record["repeat"]
        if type(rep) is not int or not 0 <= rep < 8 or rep in refs:
            raise ValueError("duplicate or invalid restatement draw")
        refs[rep] = digest
    if any(set(refs) != set(range(8)) for refs in groups.values()):
        raise ValueError("incomplete eight-draw restatement group")
    for state, refs in groups.items():
        teacher = teacher_by_id(state[0])
        family, _, _ = identity(teacher["scenario_id"])
        if (
            not teacher["verified_complete"]
            or not teacher["verified_task_pass"]
            or teacher.get("builder_commit") != builders.source_revision
            or builders.source_revision != sandbox(family)["builder"]
        ):
            raise ValueError("teacher lacks verified registered training evidence")
        accepted = []
        for digest in refs.values():
            record = _read_record(bundle / "restatements", digest)
            if verifier.verify_restatement(teacher, record):
                accepted.append(
                    {**record, "producer": body["producer"], "record_sha256": digest}
                )
        yield from accepted
