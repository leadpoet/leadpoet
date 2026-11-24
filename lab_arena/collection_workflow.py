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
from lab_arena.collection_generation import (
    CAPTURE_V2,
    GenerationCapture,
    OutputLimitReached,
    QuarantinedWorkflow,
)
from lab_arena.collection_checkpoint import (
    correction_checks,
    correction_scenario,
    prepare_student_source,
    require_private_evidence,
    scenario_split,
    validate_student_source,
)

CHAMPION_REV = "1f3b2039bdd1234b69ecfa5d6ef8c61c549948ff"
POET_REV = "2c9a9ab870f2e8f15a4c81615d91876a4c80c874"
# Existing comparison batches retain both pinned revisions in their batch identity.
CORRECTION_CHAMPION_REV = "a9be241e0060b5ae8eefe55b79779eb6e73bd863"
CORRECTION_POET_REV = "d37173aca32ea59c7c5ffea46b0941726c55f30f"


COMPARISON_CAPTURE = CAPTURE_V2
LEGACY_CAPTURE = "collection.workflow-events.v1"
# The P7 producer pins its train-only task, rule pool and tool definitions.
TRAIN_ONLY_POET_REV = "1ff55087fac4cee0748247d47321f94daca7e18b"
TRAIN_ONLY_PARTITION_SHA256 = (
    "ff43fc006e3777d21a90d91b021ca3c177b5ebf2068afae555140daffadaf101"
)


def experiment_revisions(scenarios=()):
    builder = (
        TRAIN_ONLY_POET_REV
        if any(s.startswith("P7-s") for s in scenarios)
        else CORRECTION_POET_REV
    )
    return CORRECTION_CHAMPION_REV, builder


def comparison_revisions(settings):
    if "capture_version" not in settings:
        return CHAMPION_REV, POET_REV
    if settings["capture_version"] not in (LEGACY_CAPTURE, COMPARISON_CAPTURE):
        raise ValueError("unknown comparison capture version")
    return CORRECTION_CHAMPION_REV, TRAIN_ONLY_POET_REV


def _materialize(repo, commit, destination, *paths):
    # Ignore mutable working-tree files, node caches and .git hooks. Only regular
    # files actually recorded by the selected git tree can become modules.
    selected = [commit]
    if paths:
        # The gateway holds only the approved subtree. Asking Git to filter a
        # root archive can still traverse unavailable, unrelated private trees.
        subtree, = paths
        selected = ["--prefix=" + subtree + "/", commit + ":" + subtree]
    archive = subprocess.check_output(
        ["git", "-C", str(repo), "archive", "--format=tar", *selected]
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
def workflow_sources(
    champion_repo, poet_repo, *, corrections=False, builder_revision=None
):
    """Enter once in a dedicated worker process: Poet9B uses flat sibling imports."""
    if builder_revision is None:
        builder_revision = CORRECTION_POET_REV if corrections else POET_REV
    allowed = {CORRECTION_POET_REV, TRAIN_ONLY_POET_REV} if corrections else {POET_REV}
    if builder_revision not in allowed:
        raise ValueError("unapproved workflow builder revision")
    with tempfile.TemporaryDirectory(prefix="collection-pinned-") as temp:
        root = Path(temp).resolve()
        agent, poet = root / "agent", root / "poet"
        _materialize(
            champion_repo,
            CORRECTION_CHAMPION_REV if corrections else CHAMPION_REV,
            agent,
        )
        _materialize(
            poet_repo,
            builder_revision,
            poet,
            "harness/gtm-tasks",
        )
        tasks = poet / "harness/gtm-tasks"
        before = list(sys.path)
        try:
            sys.path[:0] = [str(agent), str(tasks)]
            builders = _load("collection_tasks", tasks / "trace_rows.py")
            if corrections:
                builders.branch = _load("collection_branches", tasks / "branch_runs.py")
            harness = _load("collection_harness", agent / "harness.py")
            harness.source_revision = (
                CORRECTION_CHAMPION_REV if corrections else CHAMPION_REV
            )
            builders.source_revision = builder_revision
            # These are the producer's committed frozen evaluation identities.
            # A batch creator cannot reclassify an evaluation template as training.
            frozen = json.loads((tasks / "FROZEN_V2.json").read_bytes())
            published = {key.split(":")[1] for key in frozen["builds"]}
            if not published or not builders.kmodule().TEMPLATES:
                raise ValueError("missing pinned evaluation template registry")
            builders.evaluation_templates = published | set(
                builders.kmodule().TEMPLATES
            )
            yield harness, builders
        finally:
            sys.path[:] = before
            for name, mod in list(sys.modules.items()):
                file = getattr(mod, "__file__", None)
                if file and Path(file).resolve().is_relative_to(root):
                    sys.modules.pop(name, None)


def build_scenario(builders, scenario_id):
    match = re.fullmatch(
        r"(C\d{2}|K\d{2}|B[1-8]|E[1-7]|P[1-7])-s([1-9]\d{0,7})", scenario_id
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


def _steps(messages, reference, final_output, *, full=False):
    turns = []
    for m in messages:
        if m["role"] == "assistant":
            action = (
                _encode({k: m[k] for k in ("content", "tool_calls") if k in m})
                if full
                else _encode(m.get("tool_calls") or m["content"])
            )
            if not full and len(action) > 8192:
                raise IncompleteEvidence("action_exceeds_limit")
            reasoning = m.get("reasoning") or m.get("reasoning_content")
            if (
                full
                and m.get("reasoning")
                and m.get("reasoning_content")
                and m["reasoning"] != m["reasoning_content"]
            ):
                reasoning = _encode(
                    {k: m[k] for k in ("reasoning", "reasoning_content")}
                )
            if (
                not full
                and reasoning
                and len(reasoning) > judge.MAX_VISIBLE_REASONING_CHARS
            ):
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
        evidence = "\n".join(results)
        turn["evidence"] = (
            evidence
            if full
            else _excerpts(evidence, relevant | _anchors(following), 8192)
        )
    return [turn for turn, _ in turns]


def play_workflow(
    harness,
    builders,
    job,
    broker,
    *,
    continuation=None,
    student_source=None,
    max_steps=90,
    max_tokens=8192,
    record=None,
    save=None,
    capture_version=None,
):
    if continuation is not None and student_source is not None:
        raise ValueError("a run has one starting state")
    source = continuation if continuation is not None else student_source
    extended = hasattr(builders, "branch")
    if not extended and (
        source is not None
        or max_steps != 90
        or max_tokens != 8192
    ):
        raise ValueError("a correction needs the pinned checkpoint harness")
    shape = None
    if source is not None:
        validate_student_source(harness, builders, source)
        if source["scenario_id"] != job["scenario_id"]:
            raise ValueError("source and assigned scenario differ")
        task = correction_scenario(builders, source)
        shape = json.loads(source["original"]["trace.json"]).get("shape")
    else:
        task = build_scenario(builders, job["scenario_id"])
    capture = GenerationCapture(job["job_id"], builders, save)
    capture.check(
        {k: task.get(k) for k in ("prompt", "state", "files", "tools")}, "scenario"
    )
    events = []
    experiment = job.get("experiment", {})
    sequence = 0

    def observe_legacy_tools(messages):
        # Old pinned harnesses expose only model-visible tool messages. Preserve
        # those exact bytes and identify them as such in the private journal.
        for m in messages[len(capture.messages) :]:
            if m.get("role") != "tool":
                raise ValueError(
                    "legacy tool history differs from recorded provider calls"
                )
            callers = (
                c
                for msg in reversed(capture.messages)
                for c in msg.get("tool_calls", [])
            )
            call = next(c for c in callers if c["id"] == m["tool_call_id"])
            observe(
                {
                    "index": capture.index,
                    "kind": "tool",
                    "turn": sum(m["role"] == "assistant" for m in capture.messages) - 1,
                    "call_id": m["tool_call_id"],
                    "name": call["function"]["name"],
                    "output": m["content"],
                    "representation": "model_visible",
                }
            )

    def complete(body):
        nonlocal sequence
        sequence += 1
        if capture_version == CAPTURE_V2 and not experiment:
            spent = sum(n if n is not None else cap for n, cap, _ in capture.calls)
            remaining = 65536 - spent
            if remaining < 1:
                raise OutputLimitReached("comparison generation cap reached")
            body = dict(body, max_tokens=min(body["max_tokens"], remaining))
        if experiment and "reasoning_effort" in experiment:
            body["reasoning"] = {
                "effort": experiment["reasoning_effort"],
                "exclude": not experiment["capture_reasoning"],
            }
        if not extended and record is not None:
            if sequence == 1:
                observe(
                    {
                        "index": 0,
                        "kind": "start",
                        "model": job["model"],
                        "messages": body["messages"],
                        "tools": body["tools"],
                        "prefix_turns": 0,
                        "representation": "model_visible",
                    }
                )
            else:
                observe_legacy_tools(body["messages"])
        request_id = f"generate-{sequence}"
        result = broker.model_call(request_id, body)
        allowance = body["max_tokens"]
        if hasattr(broker, "arm") and getattr(broker, "calls", None):
            allowance = broker.calls[-1]["output_allowance"]
        capture.returned(
            request_id, dict(body, max_tokens=allowance), result, allowance
        )
        if not extended and record is not None:
            choices = result.get("choices")
            if (
                isinstance(choices, list)
                and len(choices) == 1
                and isinstance(choices[0].get("message"), dict)
            ):
                choice = choices[0]
                observe(
                    {
                        "index": capture.index,
                        "kind": "reply",
                        "turn": sequence - 1,
                        "model": result.get("model"),
                        "receipt": result.get("id"),
                        "usage": result.get("usage"),
                        "finish_reason": choice.get("finish_reason"),
                        "message": choice["message"],
                    }
                )
        return result

    def observe(event):
        capture.check(event, "event-" + event["kind"])
        capture.observe(event)
        visible = {
            "start": ("messages", "tools"),
            "reply": ("message",),
            "tool": ("name", "output"),
        }.get(event["kind"])
        if visible:
            event["visible_training_safe"] = not builders.heldout.found_in(
                _encode([event[k] for k in visible])
            )
            capture.visible_training_safe &= event["visible_training_safe"]
        if event["kind"] in {"before", "finish"}:
            state = {
                "state": event["state"],
                "log": event["log"],
                "answer": event.get("answer", ""),
                "files": {
                    p: v for p, v in event["files"].items() if task["files"].get(p) != v
                },
            }
            if event["kind"] == "finish":
                capture.archive_run(
                    {
                        "model": job["model"],
                        "finish": event["finish"],
                        "messages": capture.messages,
                        "tools": capture.tools,
                        "out": state,
                    },
                    task,
                    job,
                )
            try:
                event["code_checks"] = builders.grade(task, state)
            except Exception:
                if record is not None:
                    record(event)
                raise
        if record is not None:
            record(event)
        # Original steps are written once to the private recorder. Only short
        # labels are duplicated in the generator row for offline slicing.
        if event["kind"] in {"before", "finish"}:
            events.append(
                {
                    "index": event["index"],
                    "kind": event["kind"],
                    "turn": event.get("turn"),
                    "checks": event["code_checks"],
                }
            )
        if event["kind"] == "reply":
            events.append(
                {
                    k: event[k]
                    for k in (
                        "index",
                        "kind",
                        "turn",
                        "receipt",
                        "usage",
                        "model",
                        "finish_reason",
                    )
                }
            )

    world = builders.world(task)
    adapter = getattr(broker, "broker", broker)
    use_wire = save is not None and hasattr(adapter, "record_generation")
    if use_wire:
        prior_hook = adapter.record_generation
        adapter.record_generation = capture.wire_response
    try:
        run = harness.run_task(
            {"type": "workflow", "scenario": task},
            model=job["model"],
            complete=complete,
            world=world,
            **(
                {
                    "resume": continuation["checkpoint"] if continuation else None,
                    "record": observe,
                    "max_steps": max_steps,
                    "max_tokens": max_tokens,
                }
                if extended
                else {}
            ),
        )
    except QuarantinedWorkflow:
        raise
    except Exception as exc:
        latest = capture.calls[-1][2] if capture.calls else None
        limited = isinstance(exc, OutputLimitReached) or (
            latest == "length" and str(exc) == "incomplete model reply"
        )
        run = capture.partial(
            task, world, job["model"], "output_limit" if limited else "interrupted"
        )
        if not limited:
            capture.archive_run(run, task, job)
            raise
        if extended:
            out = run["out"]
            files = {**task["files"], **out["files"]}
            observe(
                {
                    "kind": "finish",
                    "index": capture.index,
                    "finish": "output_limit",
                    "answer": "",
                    "files": files,
                    "state": out["state"],
                    "log": out["log"],
                }
            )
    finally:
        if use_wire:
            adapter.record_generation = prior_hook
    if not extended and record is not None:
        observe_legacy_tools(run["messages"])
        out = run["out"]
        observe(
            {
                "index": capture.index,
                "kind": "finish",
                "finish": run["finish"],
                "answer": out["answer"],
                "state": out["state"],
                "files": {**task["files"], **out["files"]},
                "log": out["log"],
            }
        )
    capture.archive_run(run, task, job)
    out = run["out"]
    grade = builders.grade(task, out)
    msgs = run["messages"]
    labels = [n for n, m in enumerate(msgs) if m["role"] == "assistant"]
    # A partial tool call can be syntactically unfinished. Never validate it as
    # an executed action or dispatch it. Its original bytes are already private.
    structure = (
        ["output_limit"]
        if run["finish"] == "output_limit"
        else builders.shape_problems(msgs, labels, run["tools"])
    )
    checks = {
        "answered": run["finish"] == "answered",
        "shape": not structure,
        "code_grade": grade["score"] == 1,
        "no_hard_failure": not grade.get("hard_fail"),
        "privacy": not builders.KEYS.search(_encode([msgs, run["tools"]])),
        "training_split": capture.visible_training_safe
        and not builders.heldout.found_in(_encode(msgs)),
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
    if capture_version == CAPTURE_V2:
        from lab_arena.collection_judge_segments import annotate_packet

        packet["facts"] = _encode(
            {"cited_sources": facts, "original_files": task["files"]}
        )
        full_results = []
        for m in msgs:
            original = capture.tool_outputs.get(m.get("tool_call_id"))
            if (
                m["role"] == "tool"
                and original is not None
                and original != m["content"]
            ):
                # State what the model actually saw when a public tool clipped its
                # return. The rest is evidence for the verifier, not for the model.
                result = {
                    "original_result": original,
                    "model_visible_prefix_chars": len(m["content"]),
                }
                if not original.startswith(m["content"]):
                    result["model_visible_content"] = m["content"]
                m = dict(m, content=_encode(result))
            full_results.append(m)
        packet["steps"] = _steps(full_results, reference, deliverable, full=True)
        annotate_packet(packet)
        if packet.get("projection_error"):
            # A bounded review cannot accept this run. The original judge material
            # is still available for a future authorized review with larger limits.
            capture.check(packet, "full-judge-evidence")
            full = save("full-judge-evidence", packet) if save else None
            packet = {
                k: packet[k]
                for k in (
                    "run_id",
                    "scenario_id",
                    "checks",
                    "judge_segments",
                    "projection_error",
                )
            }
            packet.update(
                steps=[], claim_reader={"units": [], "results": []}, full_evidence=full
            )
    else:
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
    row["split"] = packet["split"] = scenario_split(builders, task, shape)
    if extended:
        row["experiment"] = experiment
        before = [e for e in events if e["kind"] in ("before", "finish")]
        row["evidence"] = {
            "generation": capture.summary(run),
            "step_checks": [
                {"turn": a["turn"], "before": a["checks"], "after": b["checks"]}
                for a, b in zip(before, before[1:])
                if a["kind"] == "before"
            ],
            "out": out,
            "grade": grade,
            "finish": run["finish"],
            "events": events,
            "initial_files": task["files"],
            "initial_state": task.get("state"),
        }
        if continuation is not None:
            checkpoint = source["checkpoint"]
            cut = len(checkpoint["messages"])
            row["correction"] = {
                k: source[k]
                for k in (
                    "source_sha256",
                    "student_model",
                    "branch_turn",
                    "rejected",
                    "rejected_fatal",
                    "checkpoint",
                    "checkpoint_sha256",
                    "failure",
                )
            }
            row["correction"].update(
                prefix_messages=cut,
                correction_step_checks=correction_checks(
                    builders, task, source, msgs, grade
                ),
            )
            row["no_loss_turns"] = sorted(
                set(row["no_loss_turns"]) | {i for i in labels if i < cut}
            )
    from lab_arena.collection_claims import ensure_claim_units

    if not extended:
        row["evidence"] = {
            "out": out,
            "grade": grade,
            "finish": run["finish"],
            "generation": capture.summary(run),
        }
    capture.check(row, "generator")
    if save is not None:
        save("generator", row)
    if not (capture_version == CAPTURE_V2 and packet.get("projection_error")):
        packet = ensure_claim_units(packet, task=task, out=out, messages=msgs)
    capture.check(packet, "judge-input")
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
    # World snapshots and negative/student rows contain private grader inputs.
    # Each training recipe projects only the public interaction and its targets.
    public = {k: v for k, v in row.items() if k not in {"evidence", "correction"}}
    return dict(
        public,
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
