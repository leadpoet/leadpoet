"""Host-side workflow integration. Build only synthetic training cases from pinned code."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path

from lab_arena import collection_judge as judge

CHAMPION_REV = "1f3b2039bdd1234b69ecfa5d6ef8c61c549948ff"
POET_REV = "2c9a9ab870f2e8f15a4c81615d91876a4c80c874"


def _materialize(repo, commit, destination, *paths):
    # Ignore mutable working-tree files, node caches and .git hooks. Only regular
    # files actually recorded by the selected git tree can become modules.
    archive = subprocess.check_output(
        ["git", "-C", str(repo), "archive", "--format=tar", commit, *paths]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for item in tar:
            if item.isdir():
                continue
            target = destination / item.name
            if not target.resolve().is_relative_to(destination) or not item.isfile():
                raise ValueError("unsafe pinned source entry")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(item).read())


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def workflow_sources(champion_repo, poet_repo):
    """Enter once in a dedicated worker process: Poet9B uses flat sibling imports."""
    with tempfile.TemporaryDirectory(prefix="collection-pinned-") as temp:
        root = Path(temp).resolve()
        agent, poet = root / "agent", root / "poet"
        _materialize(champion_repo, CHAMPION_REV, agent)
        _materialize(poet_repo, POET_REV, poet, "harness/gtm-tasks")
        tasks = poet / "harness/gtm-tasks"
        before = list(sys.path)
        try:
            sys.path[:0] = [str(agent), str(tasks)]
            yield _load("collection_harness", agent / "harness.py"), _load(
                "collection_tasks", tasks / "trace_rows.py"
            )
        finally:
            sys.path[:] = before
            for name, mod in list(sys.modules.items()):
                file = getattr(mod, "__file__", None)
                if file and Path(file).resolve().is_relative_to(root):
                    sys.modules.pop(name, None)


def build_scenario(builders, scenario_id):
    match = re.fullmatch(
        r"(C\d{2}|K\d{2}|B[1-8]|E[1-7]|P[1-6])-s([1-9]\d{0,7})", scenario_id
    )
    if not match:
        raise ValueError("unknown workflow scenario id")
    task_id, seed = match[1], int(match[2])
    task = builders.build(task_id, seed)
    if not builders.guard.stamped(builders.plain(task)):
        raise ValueError("unverified scenario build")
    return task


def _encode(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class IncompleteEvidence(ValueError):
    pass


# Anchor excerpts on citations, identifiers, named entities, and quantities present in
# the reference decision, submitted output, or subsequent tool call. Include every
# occurrence, including contradictory values adjacent to the cited name. The model's
# full tool response is retained in the original generator row.
_TOKEN = re.compile(r"[\w@./:#$%+-]+", re.UNICODE)
_EVIDENCE_SIGNALS = frozenset(
    {
        "suppressed",
        "unsubscribe",
        "opt-out",
        "denied",
        "failed",
        "error",
        "confidential",
        "override",
        "ignore",
        "previous",
        "conflicting",
        "superseded",
        "expired",
    }
)


def _anchors(raw):
    anchors = set(_EVIDENCE_SIGNALS)
    for match in _TOKEN.finditer(raw):
        word = match.group().strip(".:/-")
        if word and (
            any(c.isdigit() for c in word)
            or any(c in word for c in "@/_-#$%")
            or word[:1].isupper()
        ):
            anchors.add(word.casefold())
    return anchors


def _excerpts(raw, anchors, limit):
    if len(raw) <= limit:
        return raw
    spans = []
    for match in _TOKEN.finditer(raw):
        if match.group().strip(".:/-").casefold() not in anchors:
            continue
        start = max(0, match.start() - 160)
        stop = min(len(raw), match.end() + 160)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], stop)
        else:
            spans.append([start, stop])
        # Every selected span must fit. Dropping a matching, possibly conflicting
        # record would make the packet appear more complete than its evidence.
        if sum(b - a for a, b in spans) > limit:
            raise IncompleteEvidence("selected_evidence_exceeds_limit")
    if not spans:
        raise IncompleteEvidence("oversized_evidence_without_anchor")
    rendered = "\n".join(
        f"[excerpt {start}:{stop}/{len(raw)}] " + raw[start:stop]
        for start, stop in spans
    )
    if len(rendered) > limit:
        raise IncompleteEvidence("selected_evidence_exceeds_limit")
    return rendered


def _steps(messages, reference, final_output):
    turns = []
    for m in messages:
        if m["role"] == "assistant":
            action = _encode(m.get("tool_calls") or m["content"])
            if len(action) > 8192:
                raise IncompleteEvidence("action_exceeds_limit")
            reasoning = m.get("reasoning") or m.get("reasoning_content")
            if reasoning and len(reasoning) > judge.MAX_VISIBLE_REASONING_CHARS:
                raise IncompleteEvidence("visible_reasoning_exceeds_limit")
            turns.append(
                (
                    {
                        "step": len(turns) + 1,
                        "action": action,
                        **({"reasoning": reasoning} if reasoning else {}),
                    },
                    [],
                )
            )
        elif m["role"] == "tool":
            if not turns:
                raise IncompleteEvidence("unpaired_tool_evidence")
            turns[-1][1].append(m["content"])
    relevant = _anchors(reference + "\n" + final_output)
    for i, (turn, results) in enumerate(turns):
        following = turns[i + 1][0]["action"] if i + 1 < len(turns) else ""
        turn["evidence"] = _excerpts(
            "\n".join(results), relevant | _anchors(following), 8192
        )
    return [turn for turn, _ in turns]


def play_workflow(harness, builders, job, broker):
    task = build_scenario(builders, job["scenario_id"])
    sequence = 0

    def complete(body):
        nonlocal sequence
        sequence += 1
        return broker.model_call(f"generate-{sequence}", body)

    world = builders.world(task)
    run = harness.run_task(
        {"type": "workflow", "scenario": task},
        model=job["model"],
        complete=complete,
        world=world,
    )
    out = run["out"]
    grade = builders.grade(task, out)
    msgs = run["messages"]
    labels = [n for n, m in enumerate(msgs) if m["role"] == "assistant"]
    structure = builders.shape_problems(msgs, labels, run["tools"])
    checks = {
        "answered": run["finish"] == "answered",
        "shape": not structure,
        "code_grade": grade["score"] == 1,
        "no_hard_failure": not grade.get("hard_fail"),
        "privacy": not builders.KEYS.search(_encode([msgs, run["tools"]])),
        "training_split": not builders.heldout.found_in(_encode(msgs)),
    }
    units = [u for u in builders.units(task, out) if u.get("text")]
    facts = {u["id"]: u["sources"] for u in units}
    reference = _encode(builders.perfect(task))
    deliverable = _encode({"answer": out["answer"], "files": out["files"]})
    packet = {
        "run_id": job["job_id"],
        "scenario_id": job["scenario_id"],
        "task": task["prompt"],
        "reference_answer": reference,
        "claim_reader": {"units": units, "results": []},
        "grading_notes": _encode(task["rubric"]),
        "final_output": deliverable,
        "checks": checks,
    }
    try:
        for field in ("task", "reference_answer", "grading_notes", "final_output"):
            if len(packet[field]) > 65536:
                raise IncompleteEvidence(field + "_exceeds_limit")
        packet["facts"] = _excerpts(
            _encode(facts or task["files"]),
            _anchors(reference + "\n" + deliverable),
            65536,
        )
        packet["steps"] = _steps(msgs, reference, deliverable)
    except IncompleteEvidence as exc:
        # The generator row still contains the full original run, and the job retains
        # its private source identities. No partial packet is dispatched to a judge.
        packet["projection_error"] = str(exc)
        packet["steps"] = []
    row = {
        "run": job["job_id"],
        "source": "locked_workflow",
        "task": job["scenario_id"],
        "teacher": job["model"],
        "guard": builders.plain(task)["guard"]["token"],
        "tools": run["tools"],
        "messages": msgs,
        "label_turns": labels,
        "no_loss_turns": builders.no_loss_turns(
            msgs, labels, builders.planted_faults(task)
        ),
        "steps": [{"seq": n + 1, "message": at} for n, at in enumerate(labels)],
    }
    return row, packet


def export_row(row, assessment, mode):
    if (
        mode not in {"decisions", "reasoning", "hard_steps"}
        or row["run"] != assessment["run_id"]
    ):
        raise ValueError("invalid export mode or assessment binding")
    keep = assessment["keep_steps"]
    if assessment["status"] not in {"accepted", "salvaged"} or not 1 <= keep <= len(
        row["label_turns"]
    ):
        return None
    last = (
        row["label_turns"][keep]
        if keep < len(row["label_turns"])
        else len(row["messages"])
    )
    labels = row["label_turns"][:keep]
    hard = {labels[n - 1] for n in assessment["hard_steps"] if 1 <= n <= keep}
    messages = []
    for at, msg in enumerate(row["messages"][:last]):
        m = dict(msg)
        thought = m.pop("reasoning_content", None) or m.get("reasoning")
        m.pop("reasoning", None)
        if thought and (mode == "reasoning" or mode == "hard_steps" and at in hard):
            m["reasoning_content"] = thought
        messages.append(m)
    return dict(
        row,
        messages=messages,
        label_turns=labels,
        steps=row["steps"][:keep],
        no_loss_turns=[i for i in row["no_loss_turns"] if i in labels],
    )


def store_artifact(objects, batch_id, run_id, kind, value):
    if not all(
        re.fullmatch(r"[A-Za-z0-9_-]{1,128}", s) for s in [batch_id, run_id, kind]
    ):
        raise ValueError("invalid collection artifact identifier")
    data = _encode(value).encode()
    identity = hashlib.sha256(data).hexdigest()
    key = f"collection/{batch_id}/{run_id}/{kind}-{identity}.json"
    objects.put(key, data)
    if objects.get_bounded(key, len(data)) != data:
        raise RuntimeError("stored collection artifact did not verify")
    return {"key": key, "sha256": "sha256:" + identity}
