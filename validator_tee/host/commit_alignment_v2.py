"""Exact-commit alignment between validator host code and its measured EIF."""

from __future__ import annotations

import re
from typing import Any, Dict


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ValidatorCommitAlignmentV2Error(RuntimeError):
    """The approved, host, or enclave commit identity differs."""


def verify_validator_v2_commit_alignment(
    client: Any,
    *,
    expected_commit: str,
    host_commit: str,
    required: bool,
) -> Dict[str, Any]:
    expected = str(expected_commit or "").lower()
    if not _COMMIT_RE.fullmatch(expected):
        if required:
            raise ValidatorCommitAlignmentV2Error(
                "VALIDATOR_V2_DEPLOY_COMMIT is required in production"
            )
        return {}
    observed_host = str(host_commit or "").lower()
    boot = dict(client.get_authoritative_v2_boot_identity())
    if observed_host != expected:
        raise ValidatorCommitAlignmentV2Error(
            "validator host commit differs from approved V2 commit"
        )
    if str(boot.get("commit_sha") or "").lower() != expected:
        raise ValidatorCommitAlignmentV2Error(
            "validator enclave commit differs from approved V2 commit"
        )
    return {
        "schema_version": "leadpoet.validator_commit_alignment.v2",
        "commit_sha": expected,
        "boot_identity_hash": boot.get("boot_identity_hash"),
        "pcr0": boot.get("pcr0"),
    }
