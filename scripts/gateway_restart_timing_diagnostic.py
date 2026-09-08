#!/usr/bin/env python3
"""Project one bounded, candidate-bound gateway restart timing record."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


GATEWAY_RESTART_TIMING_STAGES = frozenset(
    {
        "active_release_lineage_selection",
        "ancestry_frontier_recovery",
        "ancestry_postcheckpoint",
        "ancestry_precheckpoint",
        "attested_runtime_and_enclave_build",
        "build_provenance",
        "bootstrap",
        "completed",
        "dependency_import_preflight",
        "dependency_install",
        "dependency_preflight",
        "docker_disk_cleanup",
        "gateway_health_check",
        "gateway_process_launch",
        "git_activate",
        "git_prepare",
        "git_prepared_tree_verification",
        "git_tree_verification",
        "historical_release_acquisition",
        "host_restart_script_install",
        "lab_arena_claim_drain",
        "lab_arena_destructive_authorization",
        "lab_arena_service_start",
        "local_release_build",
        "miner_maintenance_pre_hydration",
        "miner_maintenance_runtime_verify",
        "python_cache_cleanup",
        "restart_reexec",
        "runtime_env_and_ecr",
        "source_add_shutdown_quiescence",
        "stateful_epoch_cutover",
        "stateful_epoch_cutover_preflight",
        "v2_credential_envelope_preparation",
        "v2_kms_provision",
        "v2_offline_artifact_prepare",
        "v2_pre_shutdown_preflight",
        "v2_release_lineage_revalidation",
        "v2_runtime_bootstrap",
        "v2_runtime_readiness",
        "validator_weight_input_http_check",
        "validator_weight_input_repair",
        "validator_weight_input_storage_preflight",
    }
)
GATEWAY_RESTART_TIMING_STATUSES = frozenset({"failed", "passed", "reached"})
GATEWAY_RESTART_TIMING_MAX_BYTES = 128 * 1024
GATEWAY_RESTART_TIMING_MAX_ELAPSED_SECONDS = 72_000
GATEWAY_RESTART_TIMING_PRE_CANDIDATE_STAGES = frozenset(
    {"bootstrap", "git_prepare"}
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def gateway_restart_timing_diagnostic(
    timing_dir: Path,
    *,
    expected_candidate_sha: str,
) -> dict[str, Any] | None:
    """Return fixed fields from one exact-candidate, run-owned timing ledger."""

    if SHA_RE.fullmatch(expected_candidate_sha) is None:
        return None
    try:
        ledgers = list(timing_dir.glob("gateway-*.jsonl"))
    except OSError:
        return None
    if len(ledgers) != 1:
        return None
    descriptor = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NONBLOCK
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(ledgers[0], flags)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not 0 < metadata.st_size <= GATEWAY_RESTART_TIMING_MAX_BYTES
        ):
            return None
        raw = os.read(descriptor, GATEWAY_RESTART_TIMING_MAX_BYTES + 1)
        if len(raw) != metadata.st_size:
            return None
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not lines or len(lines) > 512:
        return None
    try:
        final = json.loads(lines[-1])
    except (TypeError, ValueError):
        return None
    if not isinstance(final, Mapping):
        return None
    stage = final.get("stage")
    status = final.get("status")
    elapsed = final.get("elapsed_seconds")
    commit_sha = final.get("commit_sha")
    if (
        final.get("schema_version") != "leadpoet.gateway_restart_timing.v1"
        or not isinstance(stage, str)
        or stage not in GATEWAY_RESTART_TIMING_STAGES
        or not isinstance(status, str)
        or status not in GATEWAY_RESTART_TIMING_STATUSES
        or (
            commit_sha != expected_candidate_sha
            and not (
                commit_sha is None
                and stage in GATEWAY_RESTART_TIMING_PRE_CANDIDATE_STAGES
            )
        )
        or not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or not 0 <= elapsed <= GATEWAY_RESTART_TIMING_MAX_ELAPSED_SECONDS
    ):
        return None
    return {
        "final_stage": stage,
        "final_status": status,
        "elapsed_seconds": round(float(elapsed), 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-dir", required=True, type=Path)
    parser.add_argument("--candidate-sha", required=True)
    args = parser.parse_args(argv)
    diagnostic = gateway_restart_timing_diagnostic(
        args.timing_dir,
        expected_candidate_sha=args.candidate_sha,
    )
    if diagnostic is None:
        return 1
    if diagnostic["final_status"] != "failed":
        return 1
    print(
        "REHEARSAL_GATEWAY_RESTART_TIMING "
        f"candidate={args.candidate_sha} "
        f"stage={diagnostic['final_stage']} "
        f"status={diagnostic['final_status']} "
        f"elapsed_seconds={diagnostic['elapsed_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
