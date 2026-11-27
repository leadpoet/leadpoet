"""Check independently frozen training and hidden partitions before using a scenario."""

from __future__ import annotations

import hashlib
import json

PARTITIONS = ("template_id", "rule_pool", "tool_variant")


def _ids(values):
    if not isinstance(values, list) or not values or len(values) > 10000:
        raise ValueError("split requires an explicit nonempty pool")
    if any(type(v) is not str or not v.strip() or len(v) > 256 for v in values) or len(
        set(values)
    ) != len(values):
        raise ValueError("invalid or duplicate split identity")
    return set(values)


def admit_split(case, split_manifest):
    """Return a content binding using identities supplied by the trusted task builder."""
    if (
        set(split_manifest) != {"schema", "training", "heldout"}
        or split_manifest["schema"] != "collection.split.v1"
    ):
        raise ValueError("invalid frozen split manifest")
    if any(
        set(split_manifest[side]) != set(PARTITIONS) for side in ("training", "heldout")
    ):
        raise ValueError("incomplete split partition")
    for k in PARTITIONS:
        train, test = (
            _ids(split_manifest[side][k]) for side in ("training", "heldout")
        )
        if train & test or case.get(k) not in train:
            raise ValueError("unknown or held-out " + k)
    encoded = json.dumps(
        split_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return {
        **{k: case[k] for k in PARTITIONS},
        "manifest_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def verify_split(stamp, split_manifest):
    if not isinstance(stamp, dict) or stamp != admit_split(stamp, split_manifest):
        raise ValueError("training split binding differs")
    return stamp


def admit_built_split(builders, task, shape, split):
    from lab_arena.collection_checkpoint import scenario_split

    identity = scenario_split(builders, task, shape)
    if identity["template_id"] in builders.evaluation_templates:
        raise ValueError("scenario uses a frozen evaluation template")
    if identity["template_id"] in {"P7", "P8"}:
        if not hasattr(
            builders, "training_partition"
        ) or split != builders.training_partition(identity["template_id"]):
            raise ValueError(
                f"split differs from the producer-approved {identity['template_id']} partition"
            )
    return admit_split(identity, split)


def require_release_partition(settings):
    """Historical C/K/B/E/P1–P6 are known evaluation templates at their original pins."""
    from lab_arena.collection_workflow import (
        TRAIN_ONLY_PARTITION_SHA256,
        SALES_EVIDENCE_PARTITION_SHA256,
    )

    pins = {"P7": TRAIN_ONLY_PARTITION_SHA256, "P8": SALES_EVIDENCE_PARTITION_SHA256}
    families = {case["scenario_id"].split("-s")[0] for case in settings["cases"]}
    if (
        len(families) != 1
        or not families <= pins.keys()
        or settings["split_sha256"] != "sha256:" + pins[next(iter(families))]
    ):
        raise ValueError(
            "historical or unverified evaluation partition is private audit only"
        )
