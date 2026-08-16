"""Public entrypoints for the exact dynamic rebenchmark transition."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Iterator, Mapping


_REBENCHMARK_LAUNCH_ENV_NAMES = frozenset(
    {
        "BITTENSOR_NETWORK",
        "BITTENSOR_NETUID",
        "RESEARCH_LAB_BENCHMARK_CONCURRENCY",
        "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS",
        "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY",
        "RESEARCH_LAB_CHAMPION_EVAL_DAYS",
        "RESEARCH_LAB_CHAMPION_FRESH_ICP_COUNT",
        "RESEARCH_LAB_CHAMPION_ICPS_PER_DAY",
        "RESEARCH_LAB_CHAMPION_RETAINED_ICP_COUNT",
        "RESEARCH_LAB_CHAMPION_WINDOW_MODE",
        "RESEARCH_LAB_CONDITIONAL_FRESH_ICP_COUNT",
        "RESEARCH_LAB_CONDITIONAL_HOLDOUT_TOTAL_ICPS",
        "RESEARCH_LAB_CONDITIONAL_VALIDATION_MODE",
        "RESEARCH_LAB_PRIVATE_HOLDOUT_TOTAL_ICPS",
        "RESEARCH_LAB_PRIVATE_HOLDOUT_WEAK_TOTAL",
        "RESEARCH_LAB_PROVIDER_PREFLIGHT_TTL_SECONDS",
        "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_ICPS_PER_DAY",
        "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_TOTAL_ICPS",
        "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_WEAK_PER_DAY",
        "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_WEAK_TOTAL",
        "RESEARCH_LAB_SCORING_WORKER_MAX_LOAD_PER_CPU",
        "RESEARCH_LAB_SCORING_WORKER_MIN_AVAILABLE_MEMORY_MB",
        "RESEARCH_LAB_SCORING_WORKER_MODEL_TIMEOUT_SECONDS",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT",
        "RESEARCH_LAB_SCORING_WORKER_RECYCLE_RSS_MB",
        "RESEARCH_LAB_SCORING_WORKER_TOTAL_WORKERS",
    }
)
_REBENCHMARK_LAUNCH_ENV_PREFIXES = (
    "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_",
    "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_",
    "QUALIFICATION_WEBSHARE_PROXY_",
    "RESEARCH_LAB_SCORING_WORKER_PROXY_",
)
TRANSITION_SOURCE_PATHS = (
    "gw_restart.sh",
    "gateway/research_lab/code_build.py",
    "gateway/research_lab/config.py",
    "gateway/research_lab/dev_eval_runner.py",
    "gateway/research_lab/model_authority_v2.py",
    "gateway/research_lab/provider_preflight.py",
    "gateway/research_lab/scoring_worker.py",
    "gateway/research_lab/source_add_trial_runner.py",
    "gateway/research_lab/worker_autostart.py",
    "gateway/research_lab/worker_process.py",
    "gateway/tee/code_hash.py",
    "gateway/tee/prepare_gateway_envelopes_v2.py",
    "gateway/tee/protected_workflows.json",
    "gateway/tee/protected_workflows.py",
    "gateway/tee/scoring_executor.py",
    "gateway/tee/stage_attested_runtime.sh",
    "gateway/tee/topology.json",
    "research_lab/eval/private_runtime.py",
    "scripts/record_research_lab_dev_snapshots.py",
    "scripts/run_research_lab_scoring_worker.py",
    "scripts/run_research_lab_scoring_worker_fleet.py",
    "validator_tee/host/docker_operation_guard_v2.py",
    "validator_tee/scripts/docker_operation_lock_v2.sh",
    "validator_tee/scripts/reclaim_docker_storage_v2.sh",
)


def transition_source_paths_by_commit(
    *, from_sha: str, candidate_sha: str
) -> dict[str, tuple[str, ...]]:
    """Return the release-specific exact source inventory for this transition."""

    from research_lab.docker_operation_lock_v2 import (
        shared_docker_operation_source_paths,
    )

    candidate_paths = tuple(
        dict.fromkeys(
            (*TRANSITION_SOURCE_PATHS, *shared_docker_operation_source_paths())
        )
    )
    return {
        str(from_sha): tuple(TRANSITION_SOURCE_PATHS),
        str(candidate_sha): candidate_paths,
    }


def rebenchmark_launch_environment() -> dict[str, str]:
    """Return the launch-derived, non-production rebenchmark environment."""

    from contract_adapter import _gateway_secret

    source = _gateway_secret()
    selected = {
        str(name): str(value)
        for name, value in source.items()
        if name in _REBENCHMARK_LAUNCH_ENV_NAMES
        or any(name.startswith(prefix) for prefix in _REBENCHMARK_LAUNCH_ENV_PREFIXES)
    }
    indexed_profiles = {
        name.rsplit("_", 1)[-1]
        for name, value in selected.items()
        if name.startswith("RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_")
        and value
        and name.rsplit("_", 1)[-1].isdigit()
    }
    explicit_count = int(
        selected.get("RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT", "0") or 0
    )
    # The restart adapter's one-process acceleration must not collapse the
    # sealed two-profile production topology. Real explicit counts that are
    # wider remain authoritative and future indexed profiles expand this
    # automatically.
    derived_count = max(explicit_count, len(indexed_profiles))
    if derived_count:
        selected["RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT"] = str(derived_count)
        selected["RESEARCH_LAB_SCORING_WORKER_TOTAL_WORKERS"] = str(derived_count)
    return selected


@contextmanager
def patched_rebenchmark_launch_environment(
    values: Mapping[str, str],
) -> Iterator[None]:
    controlled = {
        name
        for name in os.environ
        if name in _REBENCHMARK_LAUNCH_ENV_NAMES
        or any(name.startswith(prefix) for prefix in _REBENCHMARK_LAUNCH_ENV_PREFIXES)
    } | set(values)
    previous = {name: (name in os.environ, os.environ.get(name)) for name in controlled}
    try:
        for name in controlled:
            os.environ.pop(name, None)
        os.environ.update({str(name): str(value) for name, value in values.items()})
        yield
    finally:
        for name, (present, value) in previous.items():
            if present and value is not None:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)


@lru_cache(maxsize=1024)
def _git_blob_sha256(source_root: str, commit_sha: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit_sha}:{path}"],
        cwd=source_root,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def git_blob_identity(source_root: Path, commit_sha: str, path: str) -> dict[str, str]:
    return {
        "path": path,
        "commit_sha": commit_sha,
        "sha256": _git_blob_sha256(
            str(source_root.resolve()),
            str(commit_sha),
            str(path),
        ),
    }


def exercise_dynamic_rebenchmark_restart_recovery(
    *,
    source_root: Path,
    from_sha: str,
    candidate_sha: str,
    publication_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from dynamic_rebenchmark_workflow_v2 import exercise_transition

    return exercise_transition(
        source_root=source_root,
        from_sha=from_sha,
        candidate_sha=candidate_sha,
        publication_context=publication_context,
    )
