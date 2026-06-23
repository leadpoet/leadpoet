#!/usr/bin/env python3
"""Verify Research Lab private scoring cannot run on validators."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "neurons" / "validator.py"
WORK_API = ROOT / "gateway" / "qualification" / "api" / "work.py"


def main() -> int:
    errors: list[str] = []
    validator_text = VALIDATOR.read_text(encoding="utf-8")
    work_api_text = WORK_API.read_text(encoding="utf-8")

    forbidden_validator_markers = (
        "PrivateModelArtifactManifest",
        "CandidatePatchManifest",
        "DockerPrivateModelRunner",
        "DockerPrivateModelSpec",
        "evaluate_private_model_pair(",
        "sign_digest_with_kms",
        "private_model_manifest",
        "candidate_patch_manifest",
    )
    for marker in forbidden_validator_markers:
        if marker in validator_text:
            errors.append(f"validator still contains private Research Lab scoring marker: {marker}")

    if "Research Lab private scoring is gateway-owned" not in validator_text:
        errors.append("validator must explicitly document gateway-owned Research Lab scoring")
    if re.search(r"if\s+not\s+legacy_model_competition_enabled:[\s\S]{0,600}request-batch-evaluation", validator_text):
        errors.append("validator appears to request batch evaluation while legacy model competition is disabled")

    forbidden_work_api_markers = (
        '"work_kind": "research_lab_candidate"',
        '"private_model_manifest"',
        '"candidate_patch_manifest"',
        '"private_model_manifest_doc"',
    )
    for marker in forbidden_work_api_markers:
        if marker in work_api_text:
            errors.append(f"validator work API still exposes Research Lab private work marker: {marker}")
    if "gateway-owned scoring is authoritative; returning no private work" not in work_api_text:
        errors.append("validator work API must explicitly return no Research Lab private work")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Research Lab gateway scoring boundary verified: validators cannot receive or run private candidate work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
