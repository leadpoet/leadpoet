#!/usr/bin/env python3
"""Exact configured-bank dev-snapshot publication rehearsal.

The scenario runs the candidate's production refresh controller and all three
production CLIs in child processes.  Only privileged external boundaries are
adapted by ``dev_snapshot_boundary/sitecustomize.py``.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping


SCENARIO_NAME = "dev-snapshot-downstream-publication"
SCENARIO_INVARIANT = "dev_snapshot_downstream_publication_verified"
BOUNDARY_SCHEMA = "leadpoet.rehearsal_dev_snapshot_boundary.v1"
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_BUCKET = "rehearsal-dev-snapshots"
_BASE_PREFIX = "research-lab/dev-snapshots/"
_KMS_KEY_ID = "alias/rehearsal-dev-snapshot"
_POINTER_URI = f"s3://{_BUCKET}/{_BASE_PREFIX}current.json"
_PROVIDER_MODEL_ID = "openai/rehearsal-model"
_SELECTION_SEED = "exact-rehearsal-snapshot"
_ARGV_CONTRACT_SCHEMA = "leadpoet.rehearsal_dev_snapshot_argv.v1"
_NEGATIVE_PROBE_ENV = "REHEARSAL_DEV_SNAPSHOT_NEGATIVE_PROBE"


class DevSnapshotWorkflowTimeout(RuntimeError):
    """The isolated snapshot process group exceeded its bounded deadline."""

    def __init__(
        self,
        label: str,
        *,
        timeout_seconds: float,
        term_sent: bool,
        kill_sent: bool,
        returncode: int | None,
    ) -> None:
        super().__init__(f"{label} timed out after {timeout_seconds:g}s")
        self.label = str(label)
        self.timeout_seconds = float(timeout_seconds)
        self.term_sent = bool(term_sent)
        self.kill_sent = bool(kill_sent)
        self.returncode = returncode


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _redacted_cli_argv_shapes() -> dict[str, list[str]]:
    provider_args = ["--provider-model-id", "<provider-model-id:0>"]
    return {
        "export": [
            "<python>",
            "<candidate>/scripts/export_research_lab_dev_icp_inputs.py",
            "--out-dir",
            "<refresh>/inputs",
            "--seed",
            "<selection-seed>",
            "--expected-private-model-manifest-hash",
            "<active-manifest-hash>",
        ],
        "record": [
            "<python>",
            "<candidate>/scripts/record_research_lab_dev_snapshots.py",
            "--source-icps",
            "<refresh>/inputs/source_icps.json",
            "--snapshot-dir",
            "<refresh>/snapshot",
            "--champion-image",
            "<active-image-digest>",
            "--source-commit",
            "<active-source-commit>",
            "--model-config-hash",
            "<active-config-hash>",
            "--private-model-manifest-hash",
            "<active-manifest-hash>",
            "--timeout-seconds",
            "<snapshot-icp-timeout-seconds>",
            "--cancel-file",
            "<refresh>/cancel-recording",
            "--record",
            *provider_args,
        ],
        "publish_immutable": [
            "<python>",
            "<candidate>/scripts/publish_research_lab_dev_snapshot.py",
            "--source-dir",
            "<refresh>/snapshot",
            "--s3-base-uri",
            "<snapshot-base-uri>",
            "--kms-key-id",
            "<snapshot-kms-key-id>",
            "--skip-current-pointer",
        ],
        "publish_pointer": [
            "<python>",
            "<candidate>/scripts/publish_research_lab_dev_snapshot.py",
            "--source-dir",
            "<refresh>/snapshot",
            "--s3-base-uri",
            "<snapshot-base-uri>",
            "--kms-key-id",
            "<snapshot-kms-key-id>",
        ],
    }


def expected_cli_argv_contract_hashes() -> dict[str, str]:
    return {
        phase: _canonical_hash(
            {
                "schema_version": _ARGV_CONTRACT_SCHEMA,
                "phase": phase,
                "redacted_argv": argv,
            }
        )
        for phase, argv in _redacted_cli_argv_shapes().items()
    }


def expected_docker_bootstrap_hashes() -> dict[str, str]:
    from research_lab.eval import private_runtime
    from research_lab.eval.snapshot_store import (
        dev_record_bootstrap,
        dev_replay_bootstrap,
    )

    docker_bootstrap = getattr(private_runtime, "_DOCKER_ADAPTER_BOOTSTRAP", "")
    if not docker_bootstrap:
        raise RuntimeError("candidate Docker adapter bootstrap is unavailable")
    return {
        "record": "sha256:"
        + hashlib.sha256((dev_record_bootstrap() + docker_bootstrap).encode()).hexdigest(),
        "replay": "sha256:"
        + hashlib.sha256((dev_replay_bootstrap() + docker_bootstrap).encode()).hexdigest(),
    }


def _signal_process_group(process: subprocess.Popen[str], signum: int) -> bool:
    try:
        os.killpg(process.pid, signum)
        return True
    except ProcessLookupError:
        return False


def _run_in_new_process_group(
    command: list[str],
    *,
    env: Mapping[str, str],
    timeout_seconds: float,
    label: str,
    term_grace_seconds: float = 2.0,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded process tree and reap it after TERM/KILL on timeout."""

    process = subprocess.Popen(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired as exc:
        term_sent = _signal_process_group(process, signal.SIGTERM)
        kill_sent = False
        try:
            stdout, stderr = process.communicate(
                timeout=max(0.05, float(term_grace_seconds))
            )
        except subprocess.TimeoutExpired:
            kill_sent = _signal_process_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()
        else:
            # A descendant can outlive a reaped leader after closing inherited
            # pipes. Kill any remaining member of the isolated group.
            kill_sent = _signal_process_group(process, signal.SIGKILL)
        process.wait()
        raise DevSnapshotWorkflowTimeout(
            label,
            timeout_seconds=float(timeout_seconds),
            term_sent=term_sent,
            kill_sent=kill_sent,
            returncode=process.returncode,
        ) from exc
    except BaseException:
        _signal_process_group(process, signal.SIGTERM)
        try:
            process.communicate(timeout=max(0.05, float(term_grace_seconds)))
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
            process.communicate()
        process.wait()
        raise
    return subprocess.CompletedProcess(
        list(command), int(process.returncode or 0), stdout, stderr
    )


def _source_root() -> Path:
    configured = str(os.getenv("REHEARSAL_SOURCE_ROOT") or "/source").strip()
    resolved = Path(configured).resolve()
    if not (
        resolved.is_dir()
        and (resolved / "gateway").is_dir()
        and (resolved / "scripts").is_dir()
    ):
        raise RuntimeError("REHEARSAL_SOURCE_ROOT is not a candidate source tree")
    return resolved


def _candidate_sha(source_root: Path) -> str:
    configured = str(os.getenv("REHEARSAL_CANDIDATE_SHA") or "").strip().lower()
    if configured:
        if _COMMIT_RE.fullmatch(configured) is None:
            raise RuntimeError("rehearsal candidate SHA is invalid")
        return configured
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    resolved = str(completed.stdout or "").strip().lower()
    if completed.returncode or _COMMIT_RE.fullmatch(resolved) is None:
        raise RuntimeError("could not resolve the exact rehearsal candidate SHA")
    return resolved


def _completed_baseline_fixture(
    *, now: datetime, manifest_hash: str
) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
    """Build a complete configured bank through production policy functions."""

    from gateway.research_lab.config import ResearchLabGatewayConfig
    from gateway.research_lab.git_tree_models import TreePolicy
    from gateway.research_lab.icp_window import (
        WINDOW_MODE_HYBRID_FRESH_RETAINED,
        select_rolling_icp_window_from_sets,
    )
    from research_lab.canonical import sha256_json
    from research_lab.eval.conditional_validation import (
        build_conditional_category_assignment,
    )

    gateway_config = ResearchLabGatewayConfig()
    policy = gateway_config.conditional_validation_policy()
    tree_policy = TreePolicy.from_env()
    if not policy.enabled:
        raise RuntimeError("configured conditional validation policy is not enabled")
    total_icps = int(policy.total_icps)
    if total_icps != int(policy.fresh_icp_count + policy.retained_icp_count):
        raise RuntimeError("configured conditional policy does not partition its bank")
    if total_icps < int(tree_policy.live_max_icps_per_node):
        raise RuntimeError("configured baseline bank is smaller than the tree cohort")

    def _icp(index: int) -> dict[str, Any]:
        ordinal = index + 1
        return {
            "icp_id": f"dev-snapshot-{ordinal:03d}",
            "industry": f"Configured Industry {ordinal:03d}",
            "sub_industry": f"Configured Subindustry {ordinal:03d}",
            "product_service": f"Configured Service {ordinal:03d}",
            "intent_signals": [f"configured intent signal {ordinal:03d}"],
            "employee_count": "51-200",
            "geography": "United States",
            "company_count": 5,
        }

    fresh_set_id = int(now.strftime("%Y%m%d"))
    fresh_items = [_icp(index) for index in range(int(policy.fresh_icp_count))]
    retained_items = [
        _icp(index)
        for index in range(
            int(policy.fresh_icp_count),
            int(policy.fresh_icp_count + policy.retained_icp_count),
        )
    ]
    current_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    private_sets = [
        {
            "set_id": fresh_set_id,
            "icps": fresh_items,
            "icp_set_hash": sha256_json({"icps": fresh_items}),
            "active_from": current_start.isoformat().replace("+00:00", "Z"),
            "active_until": (current_start + timedelta(days=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "is_active": True,
        },
        {
            "set_id": fresh_set_id - 1,
            "icps": retained_items,
            "icp_set_hash": sha256_json({"icps": retained_items}),
            "active_from": (current_start - timedelta(days=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "active_until": current_start.isoformat().replace("+00:00", "Z"),
            "is_active": False,
        },
    ]
    window = select_rolling_icp_window_from_sets(
        private_sets,
        days=int(gateway_config.lab_champion_eval_days),
        icps_per_day=int(gateway_config.lab_champion_icps_per_day),
        window_mode=WINDOW_MODE_HYBRID_FRESH_RETAINED,
        fresh_icp_count=int(policy.fresh_icp_count),
        retained_icp_count=int(policy.retained_icp_count),
        min_new_icp_count=int(policy.fresh_icp_count),
        required_total_icps=total_icps,
        require_unique_icps=True,
        required_fresh_set_id=fresh_set_id,
        require_fresh_set_active_at=now,
    )
    benchmark_items = [dict(item) for item in window.benchmark_items]
    if len(benchmark_items) != total_icps:
        raise RuntimeError("production window selector did not build the configured bank")
    summaries = [
        {
            "icp_ref": str(item["icp_ref"]),
            "icp_hash": str(item["icp_hash"]),
            "score": round(10.0 + (80.0 * index / max(1, total_icps - 1)), 6),
        }
        for index, item in enumerate(benchmark_items)
    ]
    serving_hash = sha256_json(
        {"purpose": "rehearsal-baseline-serving-model", "manifest_hash": manifest_hash}
    )
    assignment = build_conditional_category_assignment(
        rolling_window_hash=window.window_hash,
        benchmark_items=benchmark_items,
        per_icp_summaries=summaries,
        policy=policy,
        baseline_serving_model_version_hash=serving_hash,
    )
    category_counts = dict(assignment.get("category_counts") or {})
    if sum(int(value) for value in category_counts.values()) != total_icps:
        raise RuntimeError("production conditional assignment is incomplete")

    benchmark_bundle_id = "rehearsal-completed-configured-baseline"
    benchmark_bundle_hash = sha256_json(
        {
            "benchmark_bundle_id": benchmark_bundle_id,
            "rolling_window_hash": window.window_hash,
            "private_model_manifest_hash": manifest_hash,
            "per_icp_summaries": summaries,
            "category_assignment_hash": assignment["assignment_hash"],
        }
    )
    created_at = now.isoformat().replace("+00:00", "Z")
    tables = {
        "research_lab_private_model_benchmark_current": [
            {
                "benchmark_bundle_id": benchmark_bundle_id,
                "benchmark_bundle_hash": benchmark_bundle_hash,
                "benchmark_date": now.date().isoformat(),
                "private_model_manifest_hash": manifest_hash,
                "rolling_window_hash": window.window_hash,
                "evaluation_epoch": 24_600,
                "benchmark_quality": "passed",
                "score_summary_doc": {
                    "aggregate_score": float(assignment["aggregate_score"]),
                    "per_icp_summaries": summaries,
                    "category_assignment": assignment,
                },
                "current_benchmark_status": "completed",
                "created_at": created_at,
            }
        ],
        "research_lab_rolling_icp_windows": [
            {
                "rolling_window_hash": window.window_hash,
                "window_doc": dict(window.public_doc),
                "created_at": created_at,
            }
        ],
        "qualification_private_icp_sets": private_sets,
    }
    return tables, total_icps, int(tree_policy.live_max_icps_per_node)


_CHAMPION_ADAPTER = r'''import json
import os
import sys

import requests


def run_icp(icp, context):
    icp_id = str(icp["icp_id"])
    exa = requests.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": os.environ["EXA_API_KEY"]},
        json={"query": icp_id, "numResults": 1},
        timeout=10,
    )
    exa.raise_for_status()
    scrape = requests.get(
        "https://api.scrapingdog.com/scrape",
        params={
            "api_key": os.environ["SCRAPINGDOG_API_KEY"],
            "url": "https://" + icp_id + ".example",
        },
        timeout=10,
    )
    scrape.raise_for_status()
    routed = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
        json={
            "model": "openai/rehearsal-model",
            "messages": [{"role": "user", "content": icp_id}],
        },
        timeout=10,
    )
    routed.raise_for_status()
    receipt = {
        "runtime_cap_seconds": context["runtime_options"]["runtime_cap_seconds"],
        "capability_contract": {
            "host_registered": ["deadline", "emit", "probe_origin", "resolve_host"]
        },
        "industry_taxonomy": {"taxonomy_content_hash": "sha256:" + "a" * 64},
        "firmographic_discovery": {"plan": {"target": 5}},
        "branches": [
            {
                "source": "news",
                "compiled_source": "news",
                "source_override": False,
                "route_tool_ids": ["intent.news", "intent.company_site"],
                "route_sources": ["news", "company_site"],
                "route_plan_sha256": "b" * 64,
                "route_policy_sha256": "c" * 64,
                "route_catalog_sha256": "d" * 64,
                "route_context_sha256": "e" * 64,
            }
        ],
    }
    sys.stderr.write("sourcing_branch_receipt " + json.dumps(receipt) + "\n")
    return [
        {
            "company_name": "Configured " + icp_id,
            "company_website": "https://" + icp_id + ".example",
            "industry": str(icp.get("industry") or ""),
            "sub_industry": str(icp.get("sub_industry") or ""),
            "employee_count": "51-200",
            "country": "United States",
            "description": exa.json()["results"][0]["title"],
            "intent_details": routed.json()["choices"][0]["message"]["content"],
            "intent_evidence": scrape.json()["html"],
            "intent_source_url": "https://" + icp_id + ".example/evidence",
            "intent_date": "2026-08-15",
        }
    ]
'''


def _child_environment(
    *,
    root: Path,
    source_root: Path,
    state_path: Path,
    tree_width: int,
    candidate_sha: str,
) -> dict[str, str]:
    from gateway.research_lab.config import RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD
    from research_lab.docker_operation_lock_v2 import (
        DOCKER_OPERATION_LOCK_FILE_ENV,
    )
    from research_lab.eval.private_runtime import private_model_env_passthrough

    boundary_root = Path(__file__).resolve().with_name("dev_snapshot_boundary")
    if not (boundary_root / "sitecustomize.py").is_file():
        raise RuntimeError("dev-snapshot strict boundary module is missing")
    env = dict(os.environ)
    env.pop(_NEGATIVE_PROBE_ENV, None)
    env.pop("GATEWAY_BUILD_INFO_FILE", None)
    env.pop("GATEWAY_BUILD_INFO_GIT_ROOT", None)
    for name in private_model_env_passthrough():
        env.pop(name, None)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(boundary_root), str(source_root))),
            "REHEARSAL_SOURCE_ROOT": str(source_root),
            "REHEARSAL_CANDIDATE_SHA": candidate_sha,
            "REHEARSAL_DEV_SNAPSHOT_BOUNDARY_STATE": str(state_path),
            "RESEARCH_LAB_RUNTIME_SOURCE_ROOT": str(source_root),
            "GITHUB_SHA": candidate_sha,
            "RESEARCH_LAB_DEV_SNAPSHOT_AUTO_REFRESH_ENABLED": "true",
            "RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED": "true",
            "RESEARCH_LAB_DEV_SNAPSHOT_KMS_KEY_ID": _KMS_KEY_ID,
            "RESEARCH_LAB_DEV_SNAPSHOT_PROVIDER_MODEL_IDS": json.dumps(
                [_PROVIDER_MODEL_ID], separators=(",", ":")
            ),
            "RESEARCH_LAB_DEV_SNAPSHOT_URI": _POINTER_URI,
            "RESEARCH_LAB_DEV_SNAPSHOT_MISS_POLICY": "strict",
            "RESEARCH_LAB_DEV_SNAPSHOT_REFRESH_STATE_PATH": str(
                root / "refresh-state.json"
            ),
            "RESEARCH_LAB_DEV_SNAPSHOT_REFRESH_WORK_ROOT": str(root / "work"),
            "RESEARCH_LAB_DEV_SNAPSHOT_CACHE_DIR": str(root / "cache"),
            "RESEARCH_LAB_DEV_SNAPSHOT_REFRESH_COMMAND_TIMEOUT_SECONDS": "300",
            "RESEARCH_LAB_DEV_SNAPSHOT_SELECTION_SEED": _SELECTION_SEED,
            DOCKER_OPERATION_LOCK_FILE_ENV: str(
                root / "locks" / "docker-operation-v2.lock"
            ),
            "SUPABASE_URL": "https://rehearsal.supabase.invalid",
            "SUPABASE_SERVICE_ROLE_KEY": "rehearsal-service-role",
            "EXA_API_KEY": "rehearsal-exa-key",
            "SCRAPINGDOG_API_KEY": "rehearsal-scrapingdog-key",
            "OPENROUTER_API_KEY": "rehearsal-openrouter-key",
            "TMPDIR": str(root / "tmp"),
            RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD["live_max_icps_per_node"]: str(
                tree_width
            ),
        }
    )
    return env


def _run_negative_boundary_probes(env: Mapping[str, str]) -> None:
    probes = {
        "requests": (
            "import requests; requests.get('https://undeclared.invalid/path')",
            "dev-snapshot HTTP seam requests is not allowlisted",
        ),
        "urllib": (
            "import urllib.request; urllib.request.urlopen('https://api.exa.ai/search')",
            "dev-snapshot HTTP seam urllib is not allowlisted",
        ),
        "httpx_sync": (
            "import httpx; httpx.get('https://api.exa.ai/search')",
            "dev-snapshot HTTP seam httpx_sync is not allowlisted",
        ),
        "httpx_async": (
            "import asyncio, httpx\n"
            "async def probe():\n"
            "    async with httpx.AsyncClient() as client:\n"
            "        await client.get('https://api.exa.ai/search')\n"
            "asyncio.run(probe())\n",
            "dev-snapshot HTTP seam httpx_async is not allowlisted",
        ),
        "aiohttp": (
            "import aiohttp, asyncio\n"
            "async def probe():\n"
            "    async with aiohttp.ClientSession() as client:\n"
            "        await client.get('https://api.exa.ai/search')\n"
            "asyncio.run(probe())\n",
            "dev-snapshot HTTP seam aiohttp is not allowlisted",
        ),
        "subprocess": (
            "import subprocess; subprocess.run(['/dev-snapshot-undeclared'])",
            "dev-snapshot subprocess operation is not allowlisted",
        ),
        "docker_argv": (
            "import os, subprocess; "
            "subprocess.run(['docker', 'info', '--format', 'bad'], "
            "text=True, capture_output=True, timeout=1, env=dict(os.environ), "
            "check=False)",
            "dev-snapshot Docker info contract differs",
        ),
        "aws_service": (
            "import boto3; boto3.client('dynamodb')",
            "dev-snapshot AWS service is not allowlisted",
        ),
    }
    for name, (source, expected_error) in probes.items():
        completed = _run_in_new_process_group(
            [sys.executable, "-c", source],
            env={**dict(env), _NEGATIVE_PROBE_ENV: name},
            timeout_seconds=10,
            label=f"dev-snapshot-negative-{name}",
            term_grace_seconds=0.5,
        )
        if completed.returncode == 0 or expected_error not in completed.stderr:
            raise RuntimeError(f"dev-snapshot negative {name} probe did not fail closed")


async def _run_production_refresh(state: Mapping[str, Any]) -> dict[str, Any]:
    from gateway.research_lab.dev_eval_runner import snapshot_readiness
    from gateway.research_lab.git_tree_models import TreePolicy
    from gateway.research_lab.snapshot_refresh import maybe_refresh_dev_snapshot

    identity = dict(state["active_artifact"])
    active_load_count = 0

    async def _active_loader(_config: Any, *, register_bootstrap: bool) -> Any:
        nonlocal active_load_count
        if register_bootstrap:
            raise RuntimeError("snapshot refresh requested an unexpected bootstrap")
        active_load_count += 1
        return SimpleNamespace(artifact=SimpleNamespace(**identity))

    tree_policy = TreePolicy.from_env()
    result = await maybe_refresh_dev_snapshot(
        SimpleNamespace(),
        worker_index=0,
        tree_policy=tree_policy,
        active_loader=_active_loader,
    )
    if str(result.get("status") or "") != "refreshed":
        raise RuntimeError(
            "production snapshot refresh did not complete: "
            + str(result.get("last_error") or result.get("reason") or "unknown")
        )
    readiness = snapshot_readiness(
        _POINTER_URI,
        expected_dev_icp_count=tree_policy.live_max_icps_per_node,
        require_current_day=True,
    )
    if not readiness.get("ready"):
        raise RuntimeError(
            "published current pointer is not ready: "
            + str(readiness.get("reason") or "unknown")
        )
    return {
        "status": str(result["status"]),
        "refresh_reason": str(result.get("refresh_reason") or ""),
        "snapshot_manifest_hash": str(result.get("snapshot_manifest_hash") or ""),
        "active_load_count": active_load_count,
        "readiness": {
            key: readiness.get(key)
            for key in (
                "ready",
                "reason",
                "manifest_hash",
                "ready_hash",
                "pointer_hash",
                "configured_snapshot_uri",
                "resolved_snapshot_uri",
                "snapshot_bank_size",
                "dev_set_size",
                "expected_dev_set_size",
                "benchmark_date",
                "private_model_manifest_hash",
                "champion_image_digest",
                "source_commit",
                "model_config_hash",
                "provider_model_ids",
            )
        },
    }


def _execute_child(state_path: Path) -> int:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    result = asyncio.run(_run_production_refresh(state))
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    result_path = Path(str(state["result_path"])).resolve()
    result_path.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError("dev-snapshot boundary emitted no events")
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        decoded = json.loads(line)
        if not isinstance(decoded, Mapping):
            raise RuntimeError("dev-snapshot boundary event is invalid")
        events.append(dict(decoded))
    return events


def _declared_boundary_operations_exact(
    events: list[dict[str, Any]], *, source_root: Path
) -> bool:
    """Account for every emitted operation against the frozen contract."""

    try:
        contract = json.loads(
            (
                source_root
                / "tests"
                / "restart_rehearsal"
                / "boundary_contract.json"
            ).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(contract, Mapping):
        return False
    boundaries = contract.get("boundaries")
    if (
        contract.get("schema_version")
        != "leadpoet.local_restart_boundary_contract.v1"
        or not isinstance(boundaries, Mapping)
    ):
        return False

    bindings = {
        ("supabase", "select"): ("supabase_postgrest", "select"),
        ("s3", "get_object"): ("aws_s3_object_lock", "get_object"),
        ("s3", "list_objects_v2"): (
            "aws_s3_object_lock",
            "list_objects_v2",
        ),
        ("s3", "put_object"): ("aws_s3_object_lock", "put_object"),
        ("kms", "sign"): ("aws_kms", "sign"),
        ("kms", "verify"): ("aws_kms", "verify"),
        ("provider", "request"): ("http_service", "provider_request"),
        ("provider_container", "record"): ("docker_daemon", "run"),
        ("provider_container", "replay"): ("docker_daemon", "run"),
        ("provider_container", "remove"): ("docker_daemon", "remove"),
        ("docker_daemon", "state"): ("docker_daemon", "state"),
        ("http_seam", "rejected"): (
            "http_service",
            "provider_request",
        ),
    }
    mapped_count = 0
    for event in events:
        key = (
            str(event.get("kind") or ""),
            str(event.get("operation") or ""),
        )
        binding = bindings.get(key)
        if binding is not None:
            definition = boundaries.get(binding[0])
            if (
                not isinstance(definition, Mapping)
                or definition.get("reject_unknown") is not True
                or binding[1]
                not in tuple(definition.get("allowed_operations") or ())
            ):
                return False
            mapped_count += 1
            continue
        if key[0] == "production_command":
            # These are repository-owned internal commands, whose exact argv
            # commitments are checked separately.
            continue
        if key in {("subprocess", "rejected"), ("aws", "rejected")}:
            # Undeclared operations are accounted here only as denied calls;
            # their exact negative evidence is checked by the scenario.
            if int(event.get("negative_probe") or 0) not in {0, 1}:
                return False
            continue
        return False
    return mapped_count > 0


def exercise_dev_snapshot_downstream_publication() -> dict[str, Any]:
    """Exercise export -> capture/replay -> immutable publish -> pointer/readiness."""

    from research_lab.canonical import sha256_json

    source_root = _source_root()
    candidate_sha = _candidate_sha(source_root)
    now = datetime.now(timezone.utc)
    active_artifact = {
        "image_digest": (
            "rehearsal.invalid/leadpoet/champion@sha256:"
            + sha256_json({"candidate": candidate_sha}).split(":", 1)[1]
        ),
        "git_commit_sha": candidate_sha,
        "config_hash": sha256_json(
            {"purpose": "dev-snapshot-rehearsal-config", "candidate": candidate_sha}
        ),
        "manifest_hash": sha256_json(
            {"purpose": "dev-snapshot-rehearsal-manifest", "candidate": candidate_sha}
        ),
    }
    tables, total_icps, tree_width = _completed_baseline_fixture(
        now=now,
        manifest_hash=active_artifact["manifest_hash"],
    )

    with tempfile.TemporaryDirectory(prefix="leadpoet-dev-snapshot-rehearsal-") as raw:
        root = Path(raw).resolve()
        champion_root = root / "champion"
        champion_root.mkdir(parents=True)
        (root / "tmp").mkdir()
        (champion_root / "research_lab_adapter.py").write_text(
            _CHAMPION_ADAPTER, encoding="utf-8"
        )
        state = {
            "schema_version": BOUNDARY_SCHEMA,
            "root": str(root),
            "source_root": str(source_root),
            "champion_root": str(champion_root),
            "bucket": _BUCKET,
            "base_prefix": _BASE_PREFIX,
            "kms_key_id": _KMS_KEY_ID,
            "active_artifact": active_artifact,
            "supabase_tables": tables,
            "result_path": str(root / "result.json"),
            "selection_seed": _SELECTION_SEED,
            "provider_model_ids": [_PROVIDER_MODEL_ID],
            "expected_cli_argv_contract_hashes": (
                expected_cli_argv_contract_hashes()
            ),
            "expected_docker_bootstrap_hashes": expected_docker_bootstrap_hashes(),
        }
        state_path = root / "boundary-state.json"
        state_path.write_text(
            json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--execute",
            str(state_path),
        ]
        child_env = _child_environment(
            root=root,
            source_root=source_root,
            state_path=state_path,
            tree_width=tree_width,
            candidate_sha=candidate_sha,
        )
        completed = _run_in_new_process_group(
            command,
            env=child_env,
            timeout_seconds=180,
            label="exact dev-snapshot workflow",
        )
        if completed.returncode:
            detail = str(completed.stderr or completed.stdout or "")[-2400:]
            raise RuntimeError(
                f"exact dev-snapshot child failed ({completed.returncode}): {detail}"
            )
        try:
            child = json.loads((root / "result.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("exact dev-snapshot child returned invalid evidence") from exc
        except OSError as exc:
            raise RuntimeError("exact dev-snapshot child emitted no evidence") from exc
        _run_negative_boundary_probes(child_env)
        events = _load_events(root / "events.jsonl")
        readiness = dict(child.get("readiness") or {})

        command_events = [
            event
            for event in events
            if event.get("kind") == "production_command"
        ]
        commands = [str(event.get("operation") or "") for event in command_events]
        command_phases = [str(event.get("phase") or "") for event in command_events]
        command_hashes = [
            str(event.get("argv_contract_hash") or "")
            for event in command_events
        ]
        expected_argv_hashes = expected_cli_argv_contract_hashes()
        expected_phases = [
            "export",
            "record",
            "publish_immutable",
            "publish_pointer",
        ]
        production_cli_argv_contracts_exact = (
            command_phases == expected_phases
            and command_hashes
            == [expected_argv_hashes[phase] for phase in expected_phases]
            and all(event.get("argv_redacted") is True for event in command_events)
        )
        production_commands_exact = production_cli_argv_contracts_exact and commands == [
            "export_research_lab_dev_icp_inputs.py",
            "record_research_lab_dev_snapshots.py",
            "publish_research_lab_dev_snapshot.py",
            "publish_research_lab_dev_snapshot.py",
        ] and all(
            int(event.get("returncode") or 0) == 0
            for event in events
            if event.get("kind") == "production_command"
        )
        selected_tables = [
            str(event.get("table") or "")
            for event in events
            if event.get("kind") == "supabase"
            and event.get("operation") == "select"
        ]
        expected_tables = {
            "research_lab_private_model_benchmark_current",
            "research_lab_rolling_icp_windows",
            "qualification_private_icp_sets",
        }
        supabase_export_exact = (
            set(selected_tables) == expected_tables
            and len(selected_tables) == len(expected_tables)
        )
        provider_events = [
            event
            for event in events
            if event.get("kind") == "provider"
            and event.get("operation") == "request"
        ]
        provider_counts = Counter(
            str(event.get("provider") or "") for event in provider_events
        )
        all_container_events = [
            event
            for event in events
            if event.get("kind") == "provider_container"
        ]
        container_events = [
            event
            for event in all_container_events
            if event.get("operation") in {"record", "replay"}
        ]
        container_counts = Counter(
            str(event.get("operation") or "") for event in all_container_events
        )
        expected_bootstrap_hashes = expected_docker_bootstrap_hashes()
        record_events = [
            event for event in container_events if event.get("operation") == "record"
        ]
        replay_events = [
            event for event in container_events if event.get("operation") == "replay"
        ]
        docker_bootstrap_contracts_exact = (
            len(record_events) == total_icps * 2
            and len(replay_events) == total_icps
            and sum(
                event.get("reuse_existing") is True for event in record_events
            )
            == total_icps
            and sum(
                event.get("reuse_existing") is False for event in record_events
            )
            == total_icps
            and all(
                event.get("argv_exact") is True
                and event.get("timeout_bounded") is True
                and event.get("network_disabled") is False
                and event.get("bootstrap_hash")
                == expected_bootstrap_hashes["record"]
                for event in record_events
            )
            and all(
                event.get("argv_exact") is True
                and event.get("timeout_bounded") is True
                and event.get("network_disabled") is True
                and event.get("reuse_existing") is False
                and event.get("bootstrap_hash")
                == expected_bootstrap_hashes["replay"]
                for event in replay_events
            )
        )
        provider_record_replay_exact = (
            provider_counts
            == Counter(
                {
                    "exa": total_icps,
                    "scrapingdog": total_icps,
                    "openrouter": total_icps,
                }
            )
            and container_counts["record"] == total_icps * 2
            and container_counts["replay"] == total_icps
            and sum(container_counts.values()) == total_icps * 3
            and all(event.get("client") == "requests" for event in provider_events)
            and all(
                int(event.get("returncode") or 0) == 0
                for event in container_events
            )
            and docker_bootstrap_contracts_exact
        )
        docker_daemon_events = [
            event
            for event in events
            if event.get("kind") == "docker_daemon"
            and event.get("operation") == "state"
        ]
        docker_daemon_readiness_exact = (
            len(docker_daemon_events) == total_icps * 3
            and all(
                int(event.get("returncode") or 0) == 0
                and event.get("ready") is True
                and event.get("argv_exact") is True
                and event.get("timeout_bounded") is True
                for event in docker_daemon_events
            )
        )
        s3_puts = [
            event
            for event in events
            if event.get("kind") == "s3" and event.get("operation") == "put_object"
        ]
        ready_indexes = [
            index
            for index, event in enumerate(s3_puts)
            if str(event.get("key") or "").endswith("/READY.json")
        ]
        pointer_indexes = [
            index
            for index, event in enumerate(s3_puts)
            if bool(event.get("current_pointer"))
        ]
        immutable_ready_before_pointer = (
            len(ready_indexes) == 1
            and pointer_indexes == [len(s3_puts) - 1]
            and ready_indexes[0] < pointer_indexes[0]
        )
        kms_signs = [
            event
            for event in events
            if event.get("kind") == "kms" and event.get("operation") == "sign"
        ]
        kms_verifies = [
            event
            for event in events
            if event.get("kind") == "kms" and event.get("operation") == "verify"
        ]
        signed_readiness_exact = (
            len(kms_signs) == 2
            and len(kms_verifies) >= 4
            and all(bool(event.get("signature_valid")) for event in kms_verifies)
            and bool(readiness.get("ready"))
            and readiness.get("reason") == "ready"
        )
        configured_baseline_complete = (
            int(readiness.get("snapshot_bank_size") or 0) == total_icps
            and int(readiness.get("dev_set_size") or 0) == tree_width
            and int(readiness.get("expected_dev_set_size") or 0) == tree_width
            and readiness.get("benchmark_date") == now.date().isoformat()
        )
        active_identity_rechecked = (
            int(child.get("active_load_count") or 0) >= 4
            and readiness.get("private_model_manifest_hash")
            == active_artifact["manifest_hash"]
            and readiness.get("champion_image_digest")
            == active_artifact["image_digest"]
            and readiness.get("source_commit") == active_artifact["git_commit_sha"]
            and readiness.get("model_config_hash") == active_artifact["config_hash"]
            and readiness.get("provider_model_ids") == [_PROVIDER_MODEL_ID]
        )
        resolved_uri = str(readiness.get("resolved_snapshot_uri") or "")
        manifest_hash = str(readiness.get("manifest_hash") or "")
        immutable_target_exact = (
            _SHA256_RE.fullmatch(manifest_hash) is not None
            and resolved_uri
            == f"s3://{_BUCKET}/{_BASE_PREFIX}{manifest_hash.split(':', 1)[-1]}"
            and str(child.get("snapshot_manifest_hash") or "") == manifest_hash
        )
        work_root = root / "work"
        cleanup_complete = not work_root.exists() or not any(work_root.iterdir())
        rejected_events = [
            event
            for event in events
            if event.get("operation") == "rejected"
        ]
        http_rejection_events = [
            event
            for event in rejected_events
            if event.get("kind") == "http_seam"
        ]
        http_rejections = Counter(
            str(event.get("client") or "") for event in http_rejection_events
        )
        expected_http_rejections = Counter(
            {
                "requests": 1,
                "urllib": 1,
                "httpx_sync": 1,
                "httpx_async": 1,
                "aiohttp": 1,
            }
        )
        alternate_http_seams_fail_closed = (
            http_rejections == expected_http_rejections
            and all(
                int(event.get("negative_probe") or 0) == 1
                and event.get("probe_id") == event.get("client")
                for event in http_rejection_events
            )
        )
        subprocess_rejections = [
            event
            for event in rejected_events
            if event.get("kind") == "subprocess"
        ]
        explicit_subprocess_rejections = [
            event
            for event in subprocess_rejections
            if int(event.get("negative_probe") or 0) == 1
        ]
        production_git_rejections = [
            event
            for event in subprocess_rejections
            if int(event.get("negative_probe") or 0) == 0
        ]
        production_git_discovery_fail_closed = (
            len(production_git_rejections) == 2
            and all(
                event.get("command_class") == "other"
                and event.get("command_name") == "git"
                and not event.get("probe_id")
                for event in production_git_rejections
            )
        )
        explicit_subprocess_probe_ids = Counter(
            str(event.get("probe_id") or "")
            for event in explicit_subprocess_rejections
        )
        unknown_subprocess_rejected = (
            explicit_subprocess_probe_ids
            == Counter({"subprocess": 1, "docker_argv": 1})
            and any(
                event.get("probe_id") == "subprocess"
                and event.get("command_class") == "other"
                and event.get("command_name") == "dev-snapshot-undeclared"
                for event in explicit_subprocess_rejections
            )
            and any(
                event.get("probe_id") == "docker_argv"
                and event.get("command_class") == "docker"
                and event.get("command_name") == "docker"
                for event in explicit_subprocess_rejections
            )
            and production_git_discovery_fail_closed
            and len(subprocess_rejections) == 4
        )
        aws_rejections = [
            event
            for event in rejected_events
            if event.get("kind") == "aws"
        ]
        unknown_aws_service_rejected = (
            len(aws_rejections) == 1
            and int(aws_rejections[0].get("negative_probe") or 0) == 1
            and aws_rejections[0].get("probe_id") == "aws_service"
            and aws_rejections[0].get("service_class") == "unknown"
        )
        expected_negative_probe_ids = Counter(
            {
                "requests": 1,
                "urllib": 1,
                "httpx_sync": 1,
                "httpx_async": 1,
                "aiohttp": 1,
                "subprocess": 1,
                "docker_argv": 1,
                "aws_service": 1,
            }
        )
        observed_negative_probe_ids = Counter(
            str(event.get("probe_id") or "")
            for event in rejected_events
            if int(event.get("negative_probe") or 0) == 1
        )
        negative_boundary_evidence_complete = (
            alternate_http_seams_fail_closed
            and unknown_subprocess_rejected
            and unknown_aws_service_rejected
            and observed_negative_probe_ids == expected_negative_probe_ids
            and len(rejected_events)
            == sum(expected_negative_probe_ids.values()) + 2
        )
        declared_boundary_operations_exact = _declared_boundary_operations_exact(
            events, source_root=source_root
        )
        unknown_boundaries_rejected = (
            declared_boundary_operations_exact
            and negative_boundary_evidence_complete
        )
        predicates = {
            "production_commands_exact": production_commands_exact,
            "production_cli_argv_contracts_exact": (
                production_cli_argv_contracts_exact
            ),
            "docker_bootstrap_contracts_exact": (
                docker_bootstrap_contracts_exact
            ),
            "configured_baseline_complete": configured_baseline_complete,
            "supabase_export_exact": supabase_export_exact,
            "provider_record_replay_exact": provider_record_replay_exact,
            "docker_daemon_readiness_exact": docker_daemon_readiness_exact,
            "immutable_ready_before_pointer": immutable_ready_before_pointer,
            "signed_readiness_exact": signed_readiness_exact,
            "active_identity_rechecked": active_identity_rechecked,
            "immutable_target_exact": immutable_target_exact,
            "cleanup_complete": cleanup_complete,
            "unknown_boundaries_rejected": unknown_boundaries_rejected,
            "alternate_http_seams_fail_closed": alternate_http_seams_fail_closed,
            "unknown_subprocess_rejected": unknown_subprocess_rejected,
            "production_git_discovery_fail_closed": (
                production_git_discovery_fail_closed
            ),
            "unknown_aws_service_rejected": unknown_aws_service_rejected,
            "declared_boundary_operations_exact": (
                declared_boundary_operations_exact
            ),
            "negative_boundary_evidence_complete": (
                negative_boundary_evidence_complete
            ),
        }
        failed = sorted(name for name, passed in predicates.items() if not passed)
        if failed:
            rejection_summary = [
                {
                    key: event.get(key)
                    for key in (
                        "kind",
                        "operation",
                        "client",
                        "command_class",
                        "command_name",
                        "service_class",
                        "negative_probe",
                        "probe_id",
                    )
                    if event.get(key) not in (None, "")
                }
                for event in rejected_events
            ]
            raise RuntimeError(
                "dev-snapshot downstream rehearsal invariant failed: "
                + ",".join(failed)
                + "; rejected="
                + json.dumps(rejection_summary, sort_keys=True, separators=(",", ":"))
            )
        return {
            "scenario": SCENARIO_NAME,
            "invariant": SCENARIO_INVARIANT,
            **predicates,
            "configured_bank_size": total_icps,
            "configured_tree_width": tree_width,
            "provider_request_count": sum(provider_counts.values()),
            "container_execution_count": (
                container_counts["record"] + container_counts["replay"]
            ),
            "docker_daemon_readiness_count": len(docker_daemon_events),
            "s3_object_put_count": len(s3_puts),
            "kms_verification_count": len(kms_verifies),
            "production_cli_argv_contract_hashes": command_hashes,
            "docker_bootstrap_contract_hashes": expected_bootstrap_hashes,
            "negative_probe_ids": dict(observed_negative_probe_ids),
            "production_git_rejection_count": len(production_git_rejections),
            "manifest_hash": manifest_hash,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.execute is None:
        parser.error("--execute is required for direct invocation")
    return _execute_child(args.execute.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
