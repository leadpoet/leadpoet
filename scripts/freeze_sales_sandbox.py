"""Freeze a private, reproducible Sales Sandbox dataset for Poet training and evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from lab_arena.collection_judge import digest
from scripts.collection_review import read_file

STUDENT = "Qwen/Qwen3.5-4B"
STUDENT_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CLOSE = "qwen/qwen3.8-27b"
FAR = "deepseek/deepseek-v4-pro-0813"
ARM_NAMES = (
    "own passes",
    "own passes on mid-difficulty cases",
    "own pass versus own failure",
    "teacher pass versus student failure",
    "close teacher without reasoning",
    "same close teacher with short reasoning",
    "far teacher without reasoning",
    "student-state corrections",
    "expert-state corrections",
    "own passes and failure labels",
    "student restatements",
    "three training variants",
    "two training sandboxes",
    "groups of eight scored student runs",
)
MAX_ROW = 128 << 20
INPUT_SHA = re.compile(r"[0-9a-f]{64}")
POOL_ID = re.compile(r"(?:pool|dv)-[0-9a-f]{64}")
ARM_ORDER = tuple(f"A{n}" for n in range(1, 15))


def encoded(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def archive_entry(z, name, data):
    info = ZipInfo(name)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    z.writestr(info, data)


def read_member(z, name):
    if z.getinfo(name).file_size > MAX_ROW:
        raise ValueError("oversized private training row")
    return json.loads(z.read(name))


def reasoning(message):
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    details = [
        entry.get("text", "")
        for entry in message.get("reasoning_details") or []
        if isinstance(entry, dict) and entry.get("type") == "reasoning.text"
    ]
    if details:
        return "\n".join(details).strip()
    content = message.get("content")
    if (
        isinstance(content, str)
        and content.lstrip().startswith("<think>")
        and "</think>" in content
    ):
        return content.lstrip()[len("<think>") : content.index("</think>")].strip()
    return ""


def initial_context(row):
    messages = row["messages"]
    first = next(
        (i for i, msg in enumerate(messages) if msg["role"] == "assistant"),
        len(messages),
    )
    return digest({"messages": messages[:first], "tools": row["tools"]})


@dataclass(frozen=True)
class Observation:
    run: str
    model: str
    case: str
    family: str
    variant: str
    pattern: int | None
    collection: str
    origin: str
    role: str
    repeat: int
    group: str
    context: str
    builder: str
    split: str
    passed: bool | None
    first_failure: bool
    short: bool
    score: float | None
    effort: str
    route: str | None
    revision: str
    checkpoint: str | None

    @property
    def matched_case(self):
        # Different tools, initial evidence or pinned builders cannot be direct preferences.
        return self.case, self.builder, self.split, self.variant, self.context


def observation(row, short_words, *, node=False):
    run = row["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run) or row.get("capture_hold"):
        raise ValueError("invalid or quarantined training source")
    family, sep, seed = row["scenario_id"].rpartition("-s")
    split = row["split"]
    if not (
        sep
        and seed.isdigit()
        and int(seed) > 0
        and not 101 <= int(seed) <= 299
        and family == split["template_id"]
        and INPUT_SHA.fullmatch(split["manifest_sha256"])
    ):
        raise ValueError("unverified scenario partition")
    experiment = row.get("experiment") or {}
    program = experiment.get("program") or {}
    variant = program.get("variant", "base")
    collection = program.get(
        "collection", "C4" if experiment.get("role") == "continuation" else "C1"
    )
    if variant not in {"base", "brief", "review"} or collection not in {
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
    }:
        raise ValueError("unexpected experimental partition")
    if program and (
        program.get("sandbox") != family
        or program.get("use")
        != ("pass_rate_only" if collection == "C5" else "training")
    ):
        raise ValueError("program provenance differs from training purpose")
    if program:
        from lab_arena import collection_programs as programs

        entry = programs.sandbox(family)
        identified_family, identified_variant, ordinal = programs.identity(
            row["scenario_id"]
        )
        if (
            program.get("schema") != programs.SCHEMA
            or program.get("registry_sha256") != "sha256:" + digest(entry)
            or entry["kind"] != "train"
            or row["builder_commit"] != entry["builder"]
            or "sha256:" + split["manifest_sha256"] != entry["partition"]
            or (identified_family, identified_variant) != (family, variant)
            or type(experiment.get("ordinal")) is not int
            or experiment["ordinal"] != ordinal
            or experiment.get("case_pattern") != ordinal % entry["patterns"]
        ):
            raise ValueError("scenario differs from its training-only program identity")
    if collection != "C5" and experiment.get("reasoning_effort", "low") == "high":
        raise ValueError(
            "unclassified high-effort record cannot become a training target"
        )
    is_student = (
        row["model"].lower() == STUDENT.lower() and row["model_revision"] == STUDENT_REV
    )
    if (
        node or collection in ("C3", "C6") or experiment.get("role") == "student"
    ) and not is_student:
        raise ValueError(
            "student training source differs from the approved untuned weights"
        )
    passed = (
        row.get("verified_task_pass") if row.get("verified_complete") is True else None
    )
    if passed is not None and type(passed) is not bool:
        raise ValueError("invalid native task label")
    turns = set(row.get("label_turns", ())) - set(row.get("no_loss_turns", ()))
    short = any(
        m["role"] == "assistant"
        and i in turns
        and 0 < len(reasoning(m).split()) <= short_words
        for i, m in enumerate(row["messages"])
    )
    if row.get("verified_complete") is True and (
        row.get("partial") or not row.get("final_turn_present", True)
    ):
        raise ValueError("inconsistent native completion")
    score = (row.get("native_grade") or {}).get("score")
    if score is not None and (type(score) not in (int, float) or not 0 <= score <= 1):
        raise ValueError("invalid native score")
    return Observation(
        run,
        row["model"],
        row["scenario_id"],
        family,
        variant,
        experiment.get("case_pattern"),
        collection,
        "verified_node" if node else ("v3" if program else "earlier_training"),
        experiment.get("role", "expert"),
        experiment.get("repeat", 1),
        row["repeat_group_id"],
        initial_context(row),
        row["builder_commit"],
        split["manifest_sha256"],
        passed,
        bool(row.get("first_failed_check")),
        short,
        score,
        experiment.get("reasoning_effort", "low"),
        row.get("provider_route"),
        row["model_revision"],
        row.get("student_source_sha256"),
    )


def packs(root, snapshot):
    if not INPUT_SHA.fullmatch(snapshot):
        raise ValueError("invalid snapshot digest")
    raw = read_file(root / "snapshots" / (snapshot + ".json"), maximum=128 << 20)
    if sha256(raw).hexdigest() != snapshot:
        raise ValueError("private snapshot hash mismatch")
    index = json.loads(raw)
    if index["schema"] != "collection.private-export-index.v1":
        raise ValueError("unknown capture format")
    for batch, cases in sorted(index["offers"].items()):
        if not POOL_ID.fullmatch(batch):
            raise ValueError("unexpected batch identity")
        for case, group in sorted(cases.items()):
            ready = sorted(group.get("released", ()))
            if not ready:
                continue
            if set(ready) & group.get("held", {}).keys():
                raise ValueError("quarantined run listed as released")
            name = group["file"]
            if not re.fullmatch(r"[0-9a-f]{64}\.zip", name):
                raise ValueError("invalid private archive path")
            path = root / batch / name
            with path.open("rb") as stream:
                if file_digest(stream, "sha256").hexdigest() != group["archive_sha256"]:
                    raise ValueError("immutable archive hash mismatch")
            with ZipFile(path) as z:
                if (
                    read_member(z, "metadata.json")["dataset_sha256"]
                    != group["dataset_sha256"]
                ):
                    raise ValueError("dataset hash differs from selected index")
                for rid in ready:
                    row = read_member(z, f"trajectories/{rid}.jsonl")
                    if (row["run_id"], row["batch_id"], row["scenario_id"]) != (
                        rid,
                        batch,
                        case,
                    ):
                        raise ValueError("trajectory identity mismatch")
                    pair_path = f"pairs/{rid}.jsonl"
                    pair = (
                        read_member(z, pair_path) if pair_path in z.namelist() else None
                    )
                    yield row, pair, group["archive_sha256"]


def make_arms(observations, corrections, restatements, variant_family, second_family):
    arms = {a: [] for a in ARM_ORDER}
    groups = defaultdict(list)
    teachers = defaultdict(list)
    obs = sorted(observations, key=lambda o: o.run)
    for o in obs:
        if o.passed is None or o.collection == "C5":
            continue
        if o.role == "student" and o.collection == "C3":
            groups[o.group].append(o)
            if o.passed and o.variant == "base":
                arms["A1"].append({"run": o.run})
            if not o.passed and o.first_failure and o.variant == "base":
                arms["A10"].append({"run": o.run, "task": "diagnose_failure"})
        if (
            o.role == "expert"
            and o.collection == "C1"
            and o.passed
            and o.variant == "base"
        ):
            teachers[o.matched_case].append(o)
            if o.model == CLOSE and o.short:
                arms["A5"].append({"run": o.run})
                arms["A6"].append({"run": o.run})
            if o.model == FAR:
                arms["A7"].append({"run": o.run})
        if o.run in corrections:
            arms["A8" if corrections[o.run] == "student" else "A9"].append(
                {"pair": o.run}
            )
    for members in groups.values():
        # Frozen student groups are complete only with all eight original repetitions.
        repeats = [o.repeat for o in members]
        if (
            len(set(repeats)) != len(repeats)
            or len({o.matched_case for o in members}) != 1
            or len({o.model for o in members}) != 1
        ):
            raise ValueError(
                "student repeat group crosses cases or duplicates a repetition"
            )
        good = [o for o in members if o.passed]
        bad = [o for o in members if not o.passed]
        if members[0].variant != "base":
            continue
        if len(members) == 8 and set(repeats) in (set(range(8)), set(range(1, 9))):
            if 2 <= len(good) <= 6:
                arms["A2"].extend({"run": o.run} for o in good)
            if all(o.score is not None for o in members):
                arms["A14"].append(
                    {
                        "group": [
                            [o.run, o.score]
                            for o in sorted(members, key=lambda x: x.repeat)
                        ]
                    }
                )
        for g, b in zip(good, bad):
            arms["A3"].append({"chosen": g.run, "rejected": b.run})
        candidates = teachers.get(members[0].matched_case, [])
        for i, b in enumerate(bad):
            if candidates:
                arms["A4"].append(
                    {"chosen": candidates[i % len(candidates)].run, "rejected": b.run}
                )
    arms["A10"] += arms["A1"]
    arms["A11"].extend({"restatement": key} for key in sorted(restatements))

    # Use distinct runs in both the treatment and the same-size base control.
    # Match variants by the producer ordinal rather than cosmetic IDs.
    candidates = [
        o for o in obs if o.role == "student" and o.passed and o.collection == "C3"
    ]
    baseline = [
        o for o in candidates if o.family == variant_family and o.variant == "base"
    ]
    variants = defaultdict(lambda: defaultdict(list))
    families = defaultdict(list)
    for o in candidates:
        if o.family == variant_family:
            ordinal = int(o.case.rpartition("-s")[2]) % 10_000_000
            variants[(o.builder, ordinal)][o.variant].append(o)
        if o.variant == "base":
            families[o.family].append(o)
    for _, same_case in sorted(variants.items()):
        draws = zip(*(same_case[name] for name in ("base", "brief", "review")))
        for drawn in draws:
            if len(arms["A12"]) + 3 > len(baseline):
                break
            arms["A12"].extend({"run": o.run} for o in drawn)
    for first, second in zip(families[variant_family], families[second_family]):
        if len(arms["A13"]) + 2 > len(baseline):
            break
        arms["A13"].extend([{"run": first.run}, {"run": second.run}])
    return arms


def trainer_row(row, *, short_words=0):
    """The trainer sees only the observed messages, tools and verified target positions."""
    messages = deepcopy(row["messages"])
    labels = list(row["label_turns"])
    if any(
        type(i) is not int
        or not 0 <= i < len(messages)
        or messages[i]["role"] != "assistant"
        for i in labels
    ):
        raise ValueError("invalid loss positions")
    for m in messages:
        if m["role"] != "assistant":
            continue
        exposed = reasoning(m)
        content = m.get("content")
        if isinstance(content, str) and content.lstrip().startswith("<think>"):
            m["content"] = (
                content.split("</think>", 1)[1].lstrip()
                if "</think>" in content
                else ""
            )
        for k in ("reasoning", "reasoning_content", "reasoning_details"):
            m.pop(k, None)
        m["reasoning_content"] = (
            exposed if 0 < len(exposed.split()) <= short_words else ""
        )
    return dict(
        messages=messages,
        tools=row["tools"],
        label_turns=labels,
        no_loss_turns=row.get("no_loss_turns", []),
        steps=[],
        run_id=row["run_id"],
    )


def validate_pair(row, pair):
    if (
        pair["run_id"] != row["run_id"]
        or pair["scenario_id"] != row["scenario_id"]
        or pair["split"] != row["split"]
    ):
        raise ValueError("correction pair differs from the native trajectory")
    pre = pair["context"]
    at = len(pre)
    if (
        row.get("verified_task_pass") is not True
        or pre != row["messages"][:at]
        or pair["chosen"] != row["messages"][at]
        or pair["chosen"] == pair["rejected"]
        or pair["step_checks"]["same_pre_state"] is not True
    ):
        raise ValueError("correction requires the original verified counterfactual")
    if pair["student_model"].lower() == STUDENT.lower():
        if (
            row.get("experiment", {}).get("program", {}).get("source_kind")
            != "node_student"
        ):
            raise ValueError("student correction lacks verified node provenance")
        return "student"
    return "expert"


def freeze(
    root,
    dest,
    *,
    snapshot=None,
    short_words=256,
    variant_family="P9",
    second_family="P10",
    node_rows=(),
    restated=(),
):
    if dest.exists() or not 1 <= short_words <= 4096 or variant_family == second_family:
        raise ValueError("invalid or existing training output")
    if snapshot is None:
        snapshot = json.loads(read_file(root / "current.json"))["snapshot"]
    observations = []
    corrections = {}
    seen = set()
    provenance = {}
    restatements = set()
    lineage = sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as temp:
        bundle = Path(temp) / "bundle"
        bundle.mkdir(mode=0o700)
        with ZipFile(bundle / "sources.zip", "x", compression=ZIP_DEFLATED) as z:

            def add(row, pair, source, node):
                o = observation(row, short_words, node=node)
                if o.run in seen:
                    raise ValueError("duplicate source ID")
                seen.add(o.run)
                observations.append(o)
                blob = encoded(row)
                row_hash = sha256(blob).hexdigest()
                archive_entry(z, "runs/" + o.run + ".jsonl", blob)
                if pair is not None:
                    corrections[o.run] = validate_pair(row, pair)
                    archive_entry(z, "pairs/" + o.run + ".jsonl", encoded(pair))
                lineage.update(encoded([o.run, source, row_hash]))
                provenance[o.run] = {"archive": source, "trajectory": row_hash}

            for row, pair, source in packs(root, snapshot):
                add(row, pair, source, False)
            for row, source in node_rows:
                add(row, None, source, True)

            def teacher_by_id(run):
                if run not in seen:
                    raise ValueError("unknown teacher in frozen snapshot")
                teacher = read_member(z, f"runs/{run}.jsonl")
                o = observation(teacher, short_words)
                if (
                    o.collection not in ("C1", "C2", "C4")
                    or not o.passed
                    or o.role == "student"
                ):
                    raise ValueError(
                        "teacher is outside the released training collection"
                    )
                return teacher

            records = restated(teacher_by_id) if callable(restated) else restated
            for rec in records:
                key = rec["run_id"]
                if (
                    key in seen
                    or key in restatements
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key)
                ):
                    raise ValueError("duplicate or invalid restatement ID")
                restatements.add(key)
                archive_entry(z, "restatements/" + key + ".jsonl", encoded(rec))
                lineage.update(encoded([key, digest(rec)]))
        arms = make_arms(
            observations, corrections, restatements, variant_family, second_family
        )
        (bundle / "arms").mkdir(mode=0o700)
        counts = {}
        for arm, rows in arms.items():
            content = b"".join(encoded(r) for r in rows)
            (bundle / "arms" / f"{arm}.jsonl").write_bytes(content)
            counts[arm] = {
                "recipe": ARM_NAMES[int(arm[1:]) - 1],
                "examples": len(rows),
                "sha256": sha256(content).hexdigest(),
            }
        controls = {}
        for arm in ("A12", "A13"):
            base_ids = {
                o.run
                for o in observations
                if o.family == variant_family and o.variant == "base"
            }
            matches = [r for r in arms["A1"] if r["run"] in base_ids][: len(arms[arm])]
            if len(matches) != len(arms[arm]):
                raise ValueError("unmatched base control")
            content = b"".join(encoded(r) for r in matches)
            (bundle / "arms" / (arm + "-control.jsonl")).write_bytes(content)
            controls[arm] = {
                "examples": len(matches),
                "sha256": sha256(content).hexdigest(),
            }
        maps = defaultdict(list)
        for o in observations:
            maps[
                (
                    o.origin,
                    o.collection,
                    o.family,
                    o.variant,
                    o.case,
                    o.model,
                    o.role,
                    o.group,
                )
            ].append(o)
        with (bundle / "case-map.jsonl").open("xb") as f:
            for key, g in sorted(maps.items()):
                origin, coll, fam, v, case, model, role, group_id = key
                f.write(
                    encoded(
                        dict(
                            origin=origin,
                            collection=coll,
                            sandbox=fam,
                            variant=v,
                            scenario=case,
                            model=model,
                            role=role,
                            pattern=g[0].pattern,
                            repeat_group_id=group_id,
                            repeats=sorted(o.repeat for o in g),
                            reasoning_effort=g[0].effort,
                            route=g[0].route,
                            model_revision=g[0].revision,
                            source_state=g[0].checkpoint,
                            attempts=len(g),
                            passes=sum(o.passed is True for o in g),
                            failures=sum(o.passed is False for o in g),
                            incomplete=sum(o.passed is None for o in g),
                        )
                    )
                )
        (bundle / "provenance.json").write_bytes(encoded(provenance))
        with (bundle / "sources.zip").open("rb") as f:
            payload = file_digest(f, "sha256").hexdigest()
        manifest = {
            "schema": "collection.sales-training-freeze.v3",
            "snapshot": snapshot,
            "source_sha256": payload,
            "lineage_sha256": lineage.hexdigest(),
            "provenance_sha256": sha256(
                (bundle / "provenance.json").read_bytes()
            ).hexdigest(),
            "case_map_sha256": sha256(
                (bundle / "case-map.jsonl").read_bytes()
            ).hexdigest(),
            "short_reasoning_words": short_words,
            "runs": len(observations),
            "arm_counts": counts,
            "controls": controls,
            "case_patterns_by_sandbox": {
                family: sorted(
                    {
                        o.pattern
                        for o in observations
                        if o.family == family and o.pattern is not None
                    }
                )
                for family in sorted({o.family for o in observations})
            },
            "verified_outcomes": {
                "pass": sum(o.passed is True for o in observations),
                "fail": sum(o.passed is False for o in observations),
                "incomplete": sum(o.passed is None for o in observations),
            },
            "positive_corrections": dict(Counter(corrections.values())),
            "verified_restatements": len(restatements),
            "sandbox_counts": dict(Counter(o.family for o in observations)),
            "collection_counts": dict(Counter(o.collection for o in observations)),
            "source_counts": dict(Counter(o.origin for o in observations)),
            "matched_controls": {
                "A12": [variant_family, "base", "brief", "review"],
                "A13": [variant_family, second_family],
            },
        }
        (bundle / "manifest.json").write_bytes(encoded(manifest))
        for file in bundle.rglob("*"):
            if file.is_file():
                file.chmod(0o600)
        bundle.rename(dest)
    return manifest


def materialize(bundle, arm, output, *, control=False):
    if (
        arm not in ARM_ORDER
        or output.exists()
        or (control and arm not in {"A12", "A13"})
    ):
        raise ValueError("unknown training arm, control, or existing destination")
    manifest = json.loads(read_file(bundle / "manifest.json"))
    if manifest["schema"] != "collection.sales-training-freeze.v3":
        raise ValueError("unexpected dataset")
    source = bundle / "sources.zip"
    with source.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != manifest["source_sha256"]:
            raise ValueError("shared training data changed")
    name = arm + ("-control" if control else "")
    recipe = read_file(bundle / "arms" / (name + ".jsonl"))
    expected = (manifest["controls"] if control else manifest["arm_counts"])[arm]
    if sha256(recipe).hexdigest() != expected["sha256"]:
        raise ValueError("training recipe hash differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(source) as z, tempfile.NamedTemporaryFile(dir=output.parent) as temp:

        def get(run):
            return read_member(z, f"runs/{run}.jsonl")

        def shape(r):
            return trainer_row(
                r, short_words=manifest["short_reasoning_words"] if arm == "A6" else 0
            )

        for raw in recipe.splitlines():
            item = json.loads(raw)
            if "run" in item:
                orig = get(item["run"])
                example = shape(orig)
                if item.get("task") == "diagnose_failure":
                    first = orig["first_failed_check"]
                    answer = {
                        k: first[k] for k in ("check", "kind", "turn") if k in first
                    }
                    if not answer.get("check"):
                        raise ValueError("no verified failure to diagnose")
                    example["messages"].extend(
                        [
                            {
                                "role": "user",
                                "content": "Which task check first failed in this attempt?",
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps(answer, sort_keys=True),
                            },
                        ]
                    )
                    example["label_turns"] = [len(example["messages"]) - 1]
                    example["no_loss_turns"] = []
            elif "pair" in item:
                pair = read_member(z, f"pairs/{item['pair']}.jsonl")

                def branch(action):
                    row = {
                        "messages": [*pair["context"], action],
                        "tools": pair["tools"],
                        "label_turns": [len(pair["context"])],
                        "run_id": pair["run_id"],
                    }
                    return shape(row)

                example = {
                    "chosen": branch(pair["chosen"]),
                    "rejected": branch(pair["rejected"]),
                    "checkpoint_sha256": pair["checkpoint_sha256"],
                    "source_sha256": pair["source_sha256"],
                }
            elif "chosen" in item:
                better, worse = get(item["chosen"]), get(item["rejected"])
                if initial_context(better) != initial_context(worse):
                    raise ValueError(
                        "paired trajectories have different initial contexts"
                    )
                example = {"chosen": shape(better), "rejected": shape(worse)}
            elif "group" in item:
                example = {
                    "members": [
                        {"run_id": rid, "reward": score, "row": shape(get(rid))}
                        for rid, score in item["group"]
                    ]
                }
            else:
                example = shape(
                    read_member(z, f"restatements/{item['restatement']}.jsonl")
                )
            temp.write(encoded(example))
        temp.flush()
        os.fsync(temp.fileno())
        os.link(temp.name, output)
    return expected["examples"]


def signed_students(bundles, harness, builders, key):
    from lab_arena.collection_node_runs import verified_node_runs

    seen_repeats = set()
    for bundle in bundles:
        for row in verified_node_runs(
            bundle, harness=harness, builders=builders, public_key=key
        ):
            family = row["split"]["template_id"]
            group = [
                row["model_revision"],
                row["builder_commit"],
                row["scenario_id"],
                initial_context(row),
            ]
            rep = row["repeat"]
            if (
                type(rep) is not int
                or not 0 <= rep < 8
                or (tuple(group), rep) in seen_repeats
            ):
                raise ValueError("invalid or duplicate node repetition")
            seen_repeats.add((tuple(group), rep))
            from lab_arena.collection_programs import program as make_program

            program = make_program("C3", family, row["variant"])
            exact = {
                **row,
                "repeat_group_id": "sha256:" + digest(group),
                "partial": not row["verified_complete"],
                "final_turn_present": row["verified_complete"],
                "no_loss_turns": [],
                "experiment": {
                    "program": program,
                    "ordinal": row["ordinal"],
                    "case_pattern": row["case_pattern"],
                    "role": "student",
                    "reasoning_effort": "low",
                    "repeat": rep,
                },
            }
            yield exact, row["native_sha256"]


def signed_restatements(bundles, builders, key, teacher_by_id):
    from lab_arena.collection_node_runs import verified_restatements

    seen = set()
    for bundle in bundles:
        for row in verified_restatements(
            bundle, builders=builders, public_key=key, teacher_by_id=teacher_by_id
        ):
            turn = row["message_index"]
            teacher = teacher_by_id(row["teacher_run_id"])
            repeat = (row["teacher_run_id"], row["checkpoint_sha256"], row["repeat"])
            if repeat in seen:
                raise ValueError("duplicate restatement repetition")
            seen.add(repeat)
            yield {
                **row,
                "run_id": row["producer_run_id"],
                "messages": [*teacher["messages"][:turn], row["student_action"]],
                "tools": teacher["tools"],
                "label_turns": [turn],
                "no_loss_turns": [],
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    snap = sub.add_parser("freeze")
    snap.add_argument("--export-root", required=True, type=Path)
    snap.add_argument(
        "--snapshot",
        help="Immutable index SHA-256. Defaults to the current operator index.",
    )
    snap.add_argument("--output", required=True, type=Path)
    snap.add_argument(
        "--short-words",
        type=int,
        default=256,
        help="Maximum exposed words per complete reasoning turn in A6",
    )
    snap.add_argument("--variant-family", default="P9")
    snap.add_argument("--second-family", default="P10")
    snap.add_argument(
        "--node-bundle",
        type=Path,
        action="append",
        default=[],
        help="Private signed C3 input from the registered Poet producer",
    )
    snap.add_argument(
        "--restatement-bundle",
        type=Path,
        action="append",
        default=[],
        help="Private signed C6 input referring to teacher rows in this snapshot",
    )
    snap.add_argument("--champion-repo", type=Path)
    snap.add_argument("--poet-repo", type=Path)
    train = sub.add_parser("materialize")
    train.add_argument("--bundle", required=True, type=Path)
    train.add_argument("--arm", choices=ARM_ORDER, required=True)
    train.add_argument("--control", action="store_true")
    train.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.action == "freeze":
        from contextlib import ExitStack

        with ExitStack() as stack:
            nodes = ()
            restatements = ()
            if args.node_bundle or args.restatement_bundle:
                from lab_arena.collection_workflow import workflow_sources
                from lab_arena.collection_programs import sandbox

                key = os.environ.get("LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE")
                if not key or not args.champion_repo or not args.poet_repo:
                    parser.error(
                        "signed node input requires the pinned producer public key and both source checkouts"
                    )
                h, b = stack.enter_context(
                    workflow_sources(
                        args.champion_repo,
                        args.poet_repo,
                        corrections=True,
                        builder_revision=sandbox("P9")["builder"],
                    )
                )
                nodes = signed_students(args.node_bundle, h, b, Path(key))
                restatements = lambda lookup: signed_restatements(
                    args.restatement_bundle, b, Path(key), lookup
                )
            m = freeze(
                args.export_root,
                args.output,
                snapshot=args.snapshot,
                short_words=args.short_words,
                variant_family=args.variant_family,
                second_family=args.second_family,
                node_rows=nodes,
                restated=restatements,
            )
        print(
            json.dumps(
                {
                    "snapshot": m["snapshot"],
                    "runs": m["runs"],
                    "examples": {k: v["examples"] for k, v in m["arm_counts"].items()},
                }
            )
        )
    else:
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "examples": materialize(
                        args.bundle, args.arm, args.output, control=args.control
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
