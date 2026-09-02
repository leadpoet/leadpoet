"""Submission admission (labarena.md section 6.3 steps 1-7).

Turns an uploaded package into a frozen, immutable image: inspection and
the raise-mode secret scan (already done at upload), the offline build in
the builder container (no network, no credentials, no miner code executed),
the screening pass on a floor runner (one fixture ICP with providers live,
three synthetic ICPs with providers refused), and the accept/reject record.
Every rejection maps to a published rule id.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from lab_arena import build, contracts
from lab_arena.service import ArenaService, ServiceError

FIXTURE_ICP = {
    "icp_id": "arena:screening:fixture",
    "prompt": "Series A SaaS companies in the United States that announced a funding round in the last 12 months",
    "industry": "Software",
    "sub_industry": "SaaS",
    "employee_count": ["11-50", "51-200", "201-500"],
    "company_stage": "Series A",
    "geography": "United States",
    "country": "United States",
    "product_service": "A subscription platform revenue teams use to manage pipeline",
    "intent_signals": ["Announced a Series A or later funding round in the last 12 months, per a press release"],
    "intent_signal": "Announced a Series A or later funding round in the last 12 months, per a press release",
    "max_companies": 5,
    "excluded_companies": ["excluded.example.com"],
}


def synthetic_icps() -> List[Dict[str, Any]]:
    industries = ("Energy", "Education", "Transportation")
    return [
        dict(FIXTURE_ICP, icp_id="arena:screening:synthetic:%d" % index, industry=industry, sub_industry=industry + " services", prompt="Synthetic screening profile %d for %s" % (index, industry))
        for index, industry in enumerate(industries)
    ]


@dataclass(frozen=True)
class AdmissionOutcome:
    submission_id: str
    status: str
    rule: Optional[str]
    image_digest: Optional[str]


# An immutable image identity: a bare content digest or a registry reference pinned by digest.
IMAGE_DIGEST_RE = re.compile(r"^(?:[a-z0-9][a-z0-9._/-]{0,200}@)?sha256:[0-9a-f]{64}$")

RunModel = Callable[[str, Mapping[str, Any], bool], Sequence[Mapping[str, Any]]]
ImageBuilder = Callable[[build.PackageInspection, str], str]


def admit_submission(
    service: ArenaService,
    *,
    round_id: str,
    submission_id: str,
    image_builder: ImageBuilder,
    run_model: RunModel,
) -> AdmissionOutcome:
    """Build, screen, and record one uploaded submission.

    ``image_builder(inspection, submission_id) -> image_digest`` runs inside the
    builder container (``build.build_image``); ``run_model(image_digest, icp,
    providers_enabled) -> companies`` executes the frozen image in a fresh
    sandbox on a floor runner. Neither receives credentials from here.
    """

    submission = service.store.get_submission(submission_id)
    if submission is None or submission.get("round_id") != round_id:
        raise ServiceError("submission_missing", 404)
    if submission["status"] != "uploaded":
        return AdmissionOutcome(submission_id, submission["status"], submission.get("rejection_rule"), submission.get("image_digest"))
    archive = service.config.object_store.get(submission["package_ref"])
    try:
        inspection = build.inspect_package(archive)
        build.scan_source_archive_raise(inspection.files)
    except (build.PackageRejected, build.SecretMaterialFound) as exc:
        rule = getattr(exc, "rule_id", "package.rejected")
        service.store.update_submission(round_id, submission_id, "uploaded", "rejected", {"rejection_rule": rule})
        return AdmissionOutcome(submission_id, "rejected", rule, None)
    try:
        image_digest = image_builder(inspection, submission_id)
    except build.PackageRejected as exc:
        service.store.update_submission(round_id, submission_id, "uploaded", "rejected", {"rejection_rule": exc.rule_id})
        return AdmissionOutcome(submission_id, "rejected", exc.rule_id, None)
    if not isinstance(image_digest, str) or not IMAGE_DIGEST_RE.match(image_digest):
        service.store.update_submission(round_id, submission_id, "uploaded", "rejected", {"rejection_rule": build.RULE_BUILD_IMAGE_ID_INVALID})
        return AdmissionOutcome(submission_id, "rejected", build.RULE_BUILD_IMAGE_ID_INVALID, None)
    screening = build.screen_model(lambda icp, providers_enabled: run_model(image_digest, icp, providers_enabled), fixture_icp=FIXTURE_ICP, synthetic_icps=synthetic_icps())
    screening_doc = {"accepted": bool(screening.accepted), "rule": screening.rule_id, "detail": getattr(screening, "detail", None), "fixture_companies": getattr(screening, "fixture_company_count", None)}
    result = service.accept_built_submission(round_id, submission_id, image_digest=image_digest, source_tree_hash=inspection.source_tree_hash, scan_result={"mode": "raise", "findings": 0}, screening_result=screening_doc)
    if result.get("status") == "duplicate_artifact":
        return AdmissionOutcome(submission_id, "rejected", "package.duplicate_artifact", image_digest)
    if not screening.accepted:
        return AdmissionOutcome(submission_id, "rejected", screening.rule_id, image_digest)
    return AdmissionOutcome(submission_id, "accepted", None, image_digest)


def admit_uploaded_submissions(service: ArenaService, *, round_id: str, image_builder: ImageBuilder, run_model: RunModel) -> List[AdmissionOutcome]:
    """Admit every uploaded submission of the open round in upload order."""

    outcomes = []
    for submission in service.store.list_submissions(round_id, status="uploaded"):
        outcomes.append(admit_submission(service, round_id=round_id, submission_id=submission["submission_id"], image_builder=image_builder, run_model=run_model))
    return outcomes


def docker_image_builder(*, base_image: str, base_image_digest: str, wheelhouse_dir: Path, docker_runner: Callable[..., Any], work_dir: Path) -> ImageBuilder:
    """An ``ImageBuilder`` over ``build.build_image`` for the builder container.

    The build context holds only the package's files and the offline
    wheelhouse; the environment passed to the build is empty, so no credential
    can reach it, and the resulting digest is immutable.
    """

    def builder(inspection: build.PackageInspection, submission_id: str) -> str:
        spec = build.BuildSpec(
            base_image=base_image, base_image_digest=base_image_digest, wheelhouse_dir=Path(wheelhouse_dir),
            entry_point=inspection.entry_point, source_files=dict(inspection.files), dependency_lock=tuple(inspection.dependency_lock),
        )
        result = build.build_image(spec, docker_runner=docker_runner, context_dir=Path(work_dir) / submission_id, environment={})
        return str(result.image_digest)

    return builder
