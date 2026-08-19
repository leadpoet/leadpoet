#!/usr/bin/env python3
"""Write one bounded, secret-free Full bootstrap failure document."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Sequence


SCHEMA_VERSION = "leadpoet.production_parity_full.v3"
RUN_RE = re.compile(r"^pp-[0-9]{1,20}-[0-9]{1,6}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
BOOTSTRAP_STAGES = frozenset(
    {
        "bootstrap-environment",
        "bootstrap-workspace",
        "candidate-bundle-download",
        "candidate-bundle-metadata",
        "candidate-bundle-file-integrity",
        "candidate-bundle-head",
        "candidate-bundle-verify",
        "candidate-repository-init",
        "candidate-bundle-fetch",
        "candidate-checkout",
        "candidate-remote-rebind",
        "canonical-origin-fetch",
        "host-python-import",
        "host-entrypoint",
        "evidence-upload",
        "ssm-command",
    }
)
ERROR_CATEGORIES = frozenset(
    {
        "CommandFailed",
        "HostImportFailed",
        "HostEntrypointFailed",
        "EvidenceUploadFailed",
        "SsmFailed",
        "SsmCancelled",
        "SsmTimedOut",
        "SsmCancelling",
        "SsmDeliveryTimedOut",
    }
)
BOOTSTRAP_SSM_FAILURE_CODES = (
    (40, "bootstrap-environment", "CommandFailed"),
    (41, "bootstrap-workspace", "CommandFailed"),
    (42, "candidate-bundle-download", "CommandFailed"),
    (43, "candidate-bundle-metadata", "CommandFailed"),
    (44, "candidate-bundle-file-integrity", "CommandFailed"),
    (45, "candidate-bundle-head", "CommandFailed"),
    (46, "candidate-bundle-verify", "CommandFailed"),
    (47, "candidate-repository-init", "CommandFailed"),
    (48, "candidate-bundle-fetch", "CommandFailed"),
    (49, "candidate-checkout", "CommandFailed"),
    (50, "candidate-remote-rebind", "CommandFailed"),
    (51, "canonical-origin-fetch", "CommandFailed"),
    (52, "host-python-import", "HostImportFailed"),
    (53, "host-entrypoint", "HostEntrypointFailed"),
    (54, "evidence-upload", "EvidenceUploadFailed"),
)
MAX_EVIDENCE_BYTES = 1_024


class BootstrapEvidenceError(RuntimeError):
    """A bounded bootstrap document could not be validated or retained."""


def bootstrap_failure_identity_from_response_code(
    response_code: object,
) -> tuple[str, str] | None:
    """Project only an exact, allowlisted shell failure response code."""

    if type(response_code) is not int:
        return None
    for code, stage, category in BOOTSTRAP_SSM_FAILURE_CODES:
        if response_code == code:
            return stage, category
    return None


def failure_payload(
    *,
    run_id: str,
    base_sha: str,
    candidate_sha: str,
    stage: str,
    error_category: str,
) -> bytes:
    if (
        RUN_RE.fullmatch(run_id) is None
        or SHA_RE.fullmatch(base_sha) is None
        or SHA_RE.fullmatch(candidate_sha) is None
        or base_sha == candidate_sha
        or stage not in BOOTSTRAP_STAGES
        or error_category not in ERROR_CATEGORIES
    ):
        raise BootstrapEvidenceError("bootstrap evidence identity is invalid")
    value = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "base_sha": base_sha,
        "status": "failed",
        "failure_stage": stage,
        "error_type": error_category,
        "cleanup": {},
    }
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise BootstrapEvidenceError("bootstrap evidence exceeds its bound")
    return payload


def write_failure_no_replace(*, output: Path, payload: bytes) -> bool:
    """Atomically create evidence without following links or replacing success."""

    if (
        not output.is_absolute()
        or output.name != "full-evidence.json"
        or not payload
        or len(payload) > MAX_EVIDENCE_BYTES
    ):
        raise BootstrapEvidenceError("bootstrap evidence path is invalid")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(output.parent, directory_flags)
    temporary_name = ""
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        temporary_name = f".full-evidence.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory)
        except FileExistsError as exc:
            raise BootstrapEvidenceError(
                "bootstrap evidence staging collision"
            ) from exc
        try:
            identity = os.fstat(descriptor)
            if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                raise BootstrapEvidenceError("bootstrap evidence target is invalid")
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise BootstrapEvidenceError(
                        "bootstrap evidence write made no progress"
                    )
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                output.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError:
            return False
        os.fsync(directory)
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)
    return True


def retain_failure(
    *,
    output: Path,
    run_id: str,
    base_sha: str,
    candidate_sha: str,
    stage: str,
    error_category: str,
) -> tuple[bytes, bool]:
    payload = failure_payload(
        run_id=run_id,
        base_sha=base_sha,
        candidate_sha=candidate_sha,
        stage=stage,
        error_category=error_category,
    )
    return payload, write_failure_no_replace(output=output, payload=payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--error-category", required=True)
    args = parser.parse_args(argv)
    try:
        retain_failure(
            output=args.output,
            run_id=args.run_id,
            base_sha=args.base_sha,
            candidate_sha=args.candidate_sha,
            stage=args.stage,
            error_category=args.error_category,
        )
    except (BootstrapEvidenceError, OSError, ValueError):
        print("ERROR: bounded bootstrap evidence retention failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
