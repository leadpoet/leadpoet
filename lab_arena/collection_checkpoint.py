"""Import a guarded Poet student run and prove its decision boundary offline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from decimal import Decimal
from pathlib import Path

FILES = ("trace.json", "meta.json", "transcript.jsonl", "out.json", "check.json")
MAX_SOURCE_BYTES = 16 * 1024 * 1024
SCHEMA = "leadpoet.collection.student-source.v1"


def require_private_evidence(value):
    from lab_arena.training_capture import sanitize_training_value

    _, changes = sanitize_training_value(value, aggregate_parts=True)
    if changes:
        raise ValueError("sensitive or opaque material in collection evidence")


def digest(value):
    data = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(data.encode()).hexdigest()


def correction_scenario(builders, source):
    spec = json.loads(source["original"]["trace.json"])
    if not re.fullmatch(r"C\d{2}|K\d{2}|B[1-8]|E[1-7]|P[1-7]", spec["task"]):
        raise ValueError("unsupported student task")
    if type(spec["seed"]) is not int or not 0 < spec["seed"] <= 99999999:
        raise ValueError("invalid student seed")
    task = builders.build(
        spec["task"], spec["seed"], spec.get("shape"), spec.get("product")
    )
    plain = builders.plain(task)
    if not builders.guard.stamped(plain) or spec["guard"] != plain["guard"]:
        raise ValueError("student source differs from the guarded scenario")
    return task


def scenario_split(builders, task, shape=None):
    native = getattr(builders, "training_identity", None)
    if native is not None:
        identity = native(task, shape=shape)
        if identity is not None:
            return identity
    plain = builders.plain(task)
    kind = builders.kind_of(plain["id"])
    rule_pool = None
    if (
        kind in builders.train_rules.PAIRS
        and plain.get("rule_text") == "train"
        and not builders.train_rules.shared(kind, plain)
    ):
        rule_pool = "poet:train_rules:" + kind
    return {
        "template_id": plain["id"],
        "rule_pool": rule_pool,
        "tool_variant": ",".join(
            str(x) if x is not None else "plain" for x in builders.shape_of(shape)
        ),
    }


def _native_files(folder):
    data = {}
    remaining = MAX_SOURCE_BYTES
    for name in FILES:
        fd = os.open(Path(folder) / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > remaining:
                raise ValueError("student source is not a bounded regular file")
            raw = stream.read(remaining + 1)
            if len(raw) > remaining:
                raise ValueError("student source exceeds its size limit")
            remaining -= len(raw)
            data[name] = raw.decode("utf-8")
    return data


def _ordered_calls(raw):
    calls = []
    for ordinal, call in enumerate(raw.get("tool_calls") or []):
        if "index" in call and (
            type(call["index"]) is not int or call["index"] != ordinal
        ):
            raise ValueError("invalid provider tool order")
        calls.append({k: v for k, v in call.items() if k != "index"})
    return calls


def _owner_evidence(provenance, model, hist, tools):
    # A separate, owner-funded student diagnostic. The caller must pin the private
    # artifact in the operator batch; an uploaded miner document is not a receipt.
    allowed = {
        "qwen/qwen3.5-9b": (
            "qwen/qwen3.5-9b",
            "qwen/qwen3.5-9b-20260310",
            "deepinfra/bf16",
            "DeepInfra",
        ),
        "mistralai/ministral-3b-2512": (
            "mistralai/ministral-3b-2512",
            "mistralai/ministral-3b-2512",
            "mistral/zdr",
            "Mistral",
        ),
    }.get(model)
    if (
        allowed is None
        or not isinstance(provenance, dict)
        or provenance.get("schema") != "leadpoet.collection.owner-student.v1"
        or tuple(provenance.get(k) for k in ("model", "revision", "route", "provider"))
        != allowed
        or model != allowed[0]
        or len(provenance.get("calls", ())) != len(hist["turns"])
    ):
        raise ValueError("unverified owner-funded student route")
    seen = set()
    for at, call in zip(hist["turns"], provenance["calls"], strict=True):
        request, response = call["request"], call["response"]
        policy = request["provider"]
        if (
            request.get("model") != model
            or request.get("tools") != tools
            or request.get("messages") != hist["messages"][:at]
            or policy.get("only") != [allowed[2]]
            or policy.get("allow_fallbacks") is not False
            or policy.get("zdr") is not True
            or policy.get("data_collection") != "deny"
            or policy.get("enforce_distillable_text") is not True
        ):
            raise ValueError("student request differs from its authorized replay")
        selected = [
            r
            for r in response["openrouter_metadata"]["endpoints"]["available"]
            if r.get("selected") is True
        ]
        rid = response["id"]
        cost = Decimal(str(response["usage"]["cost"]))
        if (
            len(selected) != 1
            or selected[0].get("model") != allowed[1]
            or selected[0].get("provider") != allowed[3]
            or response.get("provider") != allowed[3]
            or response.get("model") != model
            or not isinstance(rid, str)
            or not rid
            or rid in seen
            or not cost.is_finite()
            or cost < 0
        ):
            raise ValueError("invalid student provider receipt")
        seen.add(rid)
        (choice,) = response["choices"]
        raw = choice["message"]
        if (
            raw.get("role") != "assistant"
            or choice.get("finish_reason") not in {"stop", "tool_calls"}
            or (raw.get("content") or "") != hist["messages"][at]["content"]
            or _ordered_calls(raw) != hist["messages"][at].get("tool_calls", [])
        ):
            raise ValueError("student provider decision differs from its source")


def _student_history(builders, task, hist, scratch):
    """Reconstruct every pre-turn world and full deterministic local tool result."""
    br = builders.branch
    world, _ = br.fresh(task, scratch)
    messages = hist["messages"]
    states, steps = {}, []
    for n, at in enumerate(hist["turns"]):
        state = br.snapshot(task, world, scratch)
        state["files"] = {**task["files"], **state["files"]}
        identity = digest(state)
        states.setdefault(identity, state)
        # The native grader reads modified deliverables only.
        graded = br.snapshot(task, world, scratch)
        before = builders.grade(task, graded)
        turn = messages[at]
        outputs = []
        for ordinal, c in enumerate(turn.get("tool_calls", [])):
            result = br.dispatch(
                world, scratch, c["function"]["name"], c["function"]["arguments"]
            )
            observed = messages[at + 1 + ordinal]
            if (
                observed.get("role") != "tool"
                or observed["tool_call_id"] != c["id"]
                or result[:20000] != observed["content"]
            ):
                raise ValueError("student tool output differs during recording")
            outputs.append({"id": c["id"], "output": result})
        answer = turn.get("content") or ""
        if "think>" in answer:
            answer = br.run_any.think_free(answer).strip()
        after = br.snapshot(
            task, world, scratch, answer="" if turn.get("tool_calls") else answer
        )
        steps.append(
            {
                "turn": n,
                "message": at,
                "state_sha256": identity,
                "pre_checks": before,
                "post_checks": builders.grade(task, after),
                "tool_outputs": outputs,
            }
        )
    return {"states": states, "turns": steps}


def _derive(builders, original, model, scratch, provenance=None):
    # The CLI reads the producer's original files. Downstream users rebuild from
    # these same bytes; the stored student checkpoint itself is never trusted.

    br = builders.branch
    if set(original) != set(FILES) or not all(
        isinstance(x, str) for x in original.values()
    ):
        raise ValueError("missing original student evidence")
    if sum(len(s.encode()) for s in original.values()) > MAX_SOURCE_BYTES:
        raise ValueError("student source exceeds its size limit")
    if any(builders.KEYS.search(s) for s in original.values()):
        raise ValueError("student record contains credentials")
    for name, value in original.items():
        parsed = (
            [json.loads(line) for line in value.splitlines() if line.strip()]
            if name.endswith(".jsonl")
            else json.loads(value)
        )
        require_private_evidence(parsed)
    src = scratch / "source"
    src.mkdir()
    for name in FILES:
        (src / name).write_text(original[name], encoding="utf-8")
    meta = json.loads(original["meta.json"])
    if not isinstance(model, str) or not model.strip() or meta.get("model") != model:
        raise ValueError("student source needs the assigned model identity")
    if type(meta.get("local")) is not bool or meta["local"] != (provenance is None):
        raise ValueError("student funding origin differs from its producer record")
    if provenance is not None:
        require_private_evidence(provenance)
        encoded = json.dumps(provenance, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode()) > 32 * 1024 * 1024 or builders.KEYS.search(encoded):
            raise ValueError("invalid owner-funded student provenance")
    if meta.get("finish") != "answered" or any(
        v for k, v in (meta.get("fixes") or {}).items() if k != "max_tokens_lowered"
    ):
        raise ValueError("student source is incomplete or contains repaired decisions")

    task = correction_scenario(builders, {"original": original})
    spec = json.loads(original["trace.json"])
    hist = br.history_of(src)
    if hist["nudges"] or not hist["turns"] or len(hist["turns"]) > 256:
        raise ValueError("student history cannot be restored exactly")
    full = json.loads(original["out.json"])
    check = json.loads(original["check.json"])
    grade = builders.grade(task, full)
    if digest(grade) != digest(check):
        raise ValueError("student grade differs from its fresh code check")
    messages = hist["messages"]
    steps = hist["turns"]
    final = messages[steps[-1]]
    answer = final.get("content", "")
    if "think>" in answer:
        answer = br.run_any.think_free(answer).strip()
    if (
        final.get("tool_calls")
        or steps[-1] != len(messages) - 1
        or answer != full.get("answer")
        or messages[1] != {"role": "user", "content": task["prompt"]}
    ):
        raise ValueError("student final answer or task differs")
    if builders.heldout.found_in(json.dumps(messages, ensure_ascii=False)):
        raise ValueError("hidden evaluation material in student history")
    found = br.locate(task, hist, full, scratch / "locate", check)
    branch = found.get("branch")
    if type(branch) is not int or not 0 <= branch < len(steps):
        raise ValueError("student source has no replayable failed decision")
    rep = br.replay(task, hist, scratch / "prefix", upto=branch, grade=False)
    if rep["differs"] is not None:
        raise ValueError("student tool results differ during replay")
    if provenance is not None:
        _owner_evidence(provenance, model, hist, rep["tools"])
    expected_system = br.run_any.SYSTEM.format(
        names=", ".join(t["function"]["name"] for t in rep["tools"]),
        today=br.run_any.TODAY_TEXT,
    )
    if messages[0] != {"role": "system", "content": expected_system}:
        raise ValueError("student system prompt differs from the recorded harness")
    cut = steps[branch]
    world = rep["world"]
    checkpoint = {
        "messages": messages[:cut],
        "files": br.base.collect(rep["ws"]),
        "state": world.s if world else {},
        "log": world.log if world else [],
        "tools": rep["tools"],
    }
    history = _student_history(builders, task, hist, scratch / "history")
    branch_state = history["states"][history["turns"][branch]["state_sha256"]]
    if any(branch_state[key] != checkpoint[key] for key in ("files", "state", "log")):
        raise ValueError("branch state differs from original pre-turn state")
    return {
        "student_history": history,
        "schema": SCHEMA,
        "poet_revision": builders.source_revision,
        "original": original,
        "source_sha256": digest(original),
        "student_model": model,
        **({"provenance": provenance} if provenance is not None else {}),
        "scenario_id": f"{spec['task']}-s{spec['seed']}",
        "split": scenario_split(builders, task, spec.get("shape")),
        "branch_turn": branch,
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest(checkpoint),
        "rejected": messages[cut],
        "rejected_fatal": found["rejected_fatal"],
        "failure": found,
    }


def prepare_student_source(harness, builders, folder, *, model, provenance=None):
    with tempfile.TemporaryDirectory(prefix="collection-student-") as temp:
        return _derive(builders, _native_files(folder), model, Path(temp), provenance)


def validate_student_source(harness, builders, source):
    with tempfile.TemporaryDirectory(prefix="collection-student-") as temp:
        proof = _derive(
            builders,
            source["original"],
            source["student_model"],
            Path(temp),
            source.get("provenance"),
        )
    if digest(proof) != digest(source):
        raise ValueError("student branch does not match its replay")
    return proof


def correction_checks(builders, task, source, messages, terminal_grade):
    br = builders.branch
    cut = len(source["checkpoint"]["messages"])
    chosen = messages[cut] if len(messages) > cut else {}
    prefix = messages[:cut]
    projection = [
        {k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}
        for m in prefix
    ]
    checks = {
        "same_pre_state": projection == source["checkpoint"]["messages"],
        "different_action": chosen.get("role") == "assistant"
        and not br.same_turn(chosen, source["rejected"]),
        "student_action_fatal": source["rejected_fatal"] is True,
        "expert_step_recoverable": False,
    }
    if not (
        checks["same_pre_state"]
        and checks["different_action"]
        and checks["student_action_fatal"]
    ):
        return checks
    if not chosen.get("tool_calls"):
        checks["expert_step_recoverable"] = terminal_grade[
            "score"
        ] == 1 and not terminal_grade.get("hard_fail")
        return checks
    with tempfile.TemporaryDirectory(prefix="collection-branch-check-") as root:
        scratch = Path(root)
        plan = br.reference_plan(task, builders.perfect(task), scratch / "reference")
        hist = {
            "messages": messages,
            "turns": [i for i, m in enumerate(messages) if m["role"] == "assistant"],
        }
        answer = builders.perfect(task).get("answer") or "Done."
        checks["expert_step_recoverable"] = br.completable(
            task, hist, source["branch_turn"] + 1, plan, answer, scratch / "after"
        )
    return checks
