"""Private, content-addressed known-answer and repeatability cases for a collection batch."""

from __future__ import annotations

from copy import deepcopy
import re

from lab_arena import collection_judge as judge

PACK_SCHEMA = "collection.controls.v1"
CASE_SCHEMA = "collection.control-case.v1"
PACKET_FIELDS = (
    "scenario_id",
    "task",
    "facts",
    "reference_answer",
    "grading_notes",
    "steps",
    "final_output",
    "checks",
    "claim_reader",
    "split",
)


def template(packet):
    # Never put generator identity, cost, held-out labels or previous judge judgments in the packet.
    projected = {k: deepcopy(packet[k]) for k in PACKET_FIELDS if k in packet}
    judge._packet(dict(projected, run_id="control-check"))
    if not all(projected["checks"].values()):
        raise ValueError("control must pass deterministic checks to reach judges")
    return projected


def planted_case(
    packet, *, expected, reviewer, rationale, source_sha256, source_generator=None
):
    """Record a previously reviewed private known-good/bad example. No label is inferred."""
    case = {
        "schema": CASE_SCHEMA,
        "kind": "planted",
        "packet": template(packet),
        "expected": expected,
        "provenance": {
            "reviewer": reviewer,
            "rationale": rationale,
            "source_sha256": source_sha256,
            **({"source_generator": source_generator} if source_generator else {}),
        },
    }
    validate_case(case)
    return case


def drift_case(packet, assessment, manifest, *, source_manifest_sha256):
    """Freeze the five observed scores of each judge from a prior private measured run."""
    if source_manifest_sha256 != "sha256:" + judge.digest(manifest):
        raise ValueError("reference manifest differs")
    if manifest["packet"]["sha256"] != "sha256:" + judge.digest(packet) or manifest[
        "assessment"
    ]["sha256"] != "sha256:" + judge.digest(assessment):
        raise ValueError("reference judge inputs or outputs differ from manifest")
    if not (
        manifest["job_id"] == packet["run_id"] == assessment["run_id"]
        and manifest["scenario_id"]
        == packet["scenario_id"]
        == assessment["scenario_id"]
    ):
        raise ValueError("reference run identity differs")
    source_policy = judge.BatchPolicy(**manifest["judge_policy"])
    scenario, run = judge._packet(packet)
    if assessment["policy_sha256"] != judge.digest(
        manifest["judge_policy"]
    ) or assessment["packet_sha256"] != judge.digest([scenario, run]):
        raise ValueError("reference scores are not bound to the reviewed packet")
    records = assessment["records"]
    if len(records) != 2 or {v["judge"] for v in records} != set(source_policy.roles):
        raise ValueError("both recorded reference judgments required")
    for v in records:
        identity = {
            "spec_version": judge.SPEC,
            "run_id": packet["run_id"],
            "scenario_id": packet["scenario_id"],
            "judge": v["judge"],
            "judge_version": source_policy.version(v["judge"]),
        }
        if judge.validate_verdict(v, identity, len(run["steps"]))["cannot_judge"]:
            raise ValueError("incomplete drift reference")
    case = {
        "schema": CASE_SCHEMA,
        "kind": "drift",
        "packet": template(packet),
        "expected": {v["judge"]: dict(v["scores"]) for v in records},
        "provenance": {
            "source_manifest_sha256": source_manifest_sha256,
            **(
                {"source_generator": manifest["model"]} if manifest.get("model") else {}
            ),
            "source_batch_id": source_policy.batch_id,
            "source_run_id": packet["run_id"],
            "judge_versions": {
                j: source_policy.version(j) for j in source_policy.roles
            },
        },
    }
    validate_case(case)
    return case


def validate_case(case):
    judge._fields(case, ("schema", "kind", "packet", "expected", "provenance"))
    judge._require(
        case["schema"] == CASE_SCHEMA
        and type(case["packet"]) is dict
        and template(case["packet"]) == case["packet"],
        "control case content",
    )
    expected, provenance = case["expected"], case["provenance"]
    provenance = dict(provenance)
    if "source_generator" in provenance:
        source = provenance.pop("source_generator")
        judge._require(judge._text(source, 256), "control generator identity")
    if case["kind"] == "planted":
        judge._fields(provenance, ("reviewer", "rationale", "source_sha256"))
        judge._require(
            type(expected) is bool
            and judge._text(provenance["reviewer"], 128)
            and judge._text(provenance["rationale"], 2048)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", provenance["source_sha256"]),
            "private control review provenance",
        )
    elif case["kind"] == "drift":
        judge._fields(
            provenance,
            (
                "source_manifest_sha256",
                "source_batch_id",
                "source_run_id",
                "judge_versions",
            ),
        )
        judge._require(
            re.fullmatch(r"sha256:[0-9a-f]{64}", provenance["source_manifest_sha256"])
            and all(
                re.fullmatch(r"[A-Za-z0-9_-]{1,128}", provenance[k])
                for k in ("source_batch_id", "source_run_id")
            ),
            "drift source identity",
        )
        versions = provenance["judge_versions"]
        roles = judge.JUDGES
        if versions == {j.role: m for m, j in judge.QUALIFICATION_JUDGES.items()}:
            roles = judge.QUALIFICATION_ROLES
        judge._fields(versions, roles)
        judge._fields(expected, roles)
        for version in provenance["judge_versions"].values():
            judge._require(judge._text(version, 128), "drift source version")
        for scores in expected.values():
            judge._fields(scores, judge.PARTS)
            judge._require(
                all(judge._integer(v, 0, 3) for v in scores.values()),
                "drift reference scores",
            )
    else:
        raise ValueError("unknown control case kind")
    return judge.digest(case)


def verify_control_family(case, judge_models):
    source = case["provenance"].get("source_generator")
    if not source or any(
        judge.model_family(source) == judge.model_family(model)
        for model in judge_models
    ):
        raise ValueError("calibration controls require an independent source family")


class ControlPack:
    def __init__(self, manifest, private_pack):
        judge._fields(private_pack, ("schema", "cases"))
        hashes = manifest.get("control_artifacts", {})
        planted, drift = manifest["control_slots"], manifest["drift_slots"]
        if not (
            isinstance(hashes, dict)
            and private_pack["schema"] == PACK_SCHEMA
            and isinstance(private_pack["cases"], dict)
            and not set(planted) & set(drift)
            and set(hashes) == set(planted) | set(drift)
            and set(hashes.values()) == set(private_pack["cases"])
        ):
            raise ValueError("control slot/pack manifest differs")
        self.cases = {}
        self.hashes = {}
        templates = set()
        for slot, sha in hashes.items():
            if (
                not isinstance(slot, str)
                or not re.fullmatch(r"0|[1-9][0-9]*", slot)
                or int(slot) >= 6 * len(manifest["scenario_ids"])
            ):
                raise ValueError("control slot outside batch")
            case = private_pack["cases"][sha]
            if validate_case(case) != sha or (case["kind"] == "planted") != (
                slot in planted
            ):
                raise ValueError("control artifact differs")
            if "split_manifest" in manifest:
                from lab_arena.collection_quality import verify_split

                verify_split(case["packet"].get("split"), manifest["split_manifest"])
                panel = [
                    key
                    for key in manifest.get("routes", {})
                    if key in judge.QUALIFICATION_JUDGES
                    or key in ("openai/gpt-6.2-astra", "anthropic/claude-fable-5.2")
                ]
                verify_control_family(case, panel)
            expected = (planted if slot in planted else drift)[slot]
            if (
                judge.digest(expected) != judge.digest(case["expected"])
                or case["packet"]["scenario_id"]
                != manifest["scenario_ids"][int(slot) // 6]
            ):
                raise ValueError("control packet/label does not match scheduled case")
            td = judge.digest(case["packet"])
            if td in templates:
                raise ValueError("duplicate control case")
            templates.add(td)
            self.cases[int(slot)] = deepcopy(case)
            self.hashes[int(slot)] = sha

    def bind(self, job):
        case = self.cases.get(job["slot"])
        if case is None:
            return None
        if job["scenario_id"] != case["packet"]["scenario_id"]:
            raise ValueError("control job scenario differs")
        return {
            "kind": case["kind"],
            "case_sha256": self.hashes[job["slot"]],
            "case": deepcopy(case),
            "packet": dict(deepcopy(case["packet"]), run_id=job["job_id"]),
        }
