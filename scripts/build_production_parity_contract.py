#!/usr/bin/env python3
"""Build the exact candidate contract consumed by both parity lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.tee.rehearsal_behavior_contract_v2 import (  # noqa: E402
    build_rehearsal_behavior_contract_v2,
)
from gateway.tee.research_lab_runtime_config_v2 import (  # noqa: E402
    ResearchLabRuntimeConfigV2Error,
    research_lab_config_from_document,
    research_lab_execution_config_hash,
    validate_research_lab_execution_config,
)
from leadpoet_canonical.chain_source_v2 import (  # noqa: E402
    chain_source_policy_document,
    chain_source_policy_hash,
)
from leadpoet_canonical.production_parity import (  # noqa: E402
    CONTRACT_SCHEMA_VERSION,
    MIGRATION_RE,
    ProductionParityError,
    SHA_RE,
    migration_sequence,
    sha256_bytes,
    sha256_json,
    validate_contract,
    validate_historical_oracle,
)


ORACLE_PATH = "tests/restart_rehearsal/fixtures/august_9_known_good_v2.json"
ALWAYS_COMMITTED_PATHS = (
    "AGENTS.md",
    "CLAUDE.md",
    ".github/actions/setup-production-parity-controller/action.yml",
    ".github/workflows/attested-v2-release.yml",
    ".github/workflows/deploy-checks.yml",
    ".github/workflows/physical-v2-staging.yml",
    ".github/workflows/production-parity-fast.yml",
    ".github/workflows/production-parity-cleanup.yml",
    "leadpoet_canonical/production_parity.py",
    "gateway/research_lab/api.py",
    "gateway/research_lab/key_vault.py",
    "gateway/research_lab/models.py",
    "neurons/miner.py",
    "research_lab/source_add_miner.py",
    "gw_restart.sh",
    "validator_restart.sh",
    "scripts/build_production_parity_contract.py",
    "scripts/capture_production_parity_runtime_config.py",
    "scripts/cleanup_production_parity_staging.py",
    "scripts/materialize_production_parity_secrets.py",
    "scripts/operate_rebenchmark_iam_policy.py",
    "scripts/production_parity_snapshot.py",
    "scripts/provision_production_parity_staging.py",
    "scripts/resolve_production_parity_deployed_sha.py",
    "scripts/resolve_production_parity_controller_requirements.py",
    "scripts/restart_attested_release_local.sh",
    "scripts/run_production_parity_fast.py",
    "scripts/run_production_parity_full_host.py",
    "scripts/setup_production_parity_staging.py",
    "requirements.txt",
    "setup.py",
    "gateway/tee/protected_workflows.json",
    "leadpoet_canonical/production_parity_boundary_v2.py",
    "gateway/tee/research_lab_runtime_config_v2.py",
    "gateway/tee/execution_job_manager_v2.py",
    "gateway/tee/provider_broker_v2.py",
    "gateway/tee/provider_client_v2.py",
    "gateway/tee/supabase_source_v2.py",
    "gateway/tee/provider_outcome_store_v2.py",
    "gateway/tee/rpc_authority.py",
    "gateway/tee/provider_evidence_cache_store_v2.py",
    "gateway/tee/runtime_identity_v2.py",
    "gateway/tee/tee_service.py",
    "gateway/tee/proxy_transport_preflight_v2.py",
    "gateway/utils/tee_client.py",
    "gateway/utils/tee_egress_forwarder.py",
    "gateway/utils/tee_inter_enclave_relay.py",
    "gateway/utils/tee_v2_bootstrap.py",
    "leadpoet_observability/sentry_operations.py",
    "validator_tee/host/chain_relay_v2.py",
    "validator_tee/host/vsock_client.py",
    ORACLE_PATH,
)
LOW_RISK_PREFIXES = ("docs/", "tests/")
LOW_RISK_EXACT = {"AGENTS.md", "CLAUDE.md", "README.md"}
HIGH_RISK_PREFIXES = (
    ".github/workflows/",
    "gateway/",
    "leadpoet_audit/",
    "leadpoet_canonical/",
    "leadpoet_verifier/",
    "miner_models/",
    "neurons/",
    "qualification/",
    "research_lab/",
    "scripts/",
    "validator_models/",
    "validator_tee/",
)
HIGH_RISK_EXACT = {
    "Dockerfile",
    "Dockerfile.gateway",
    "Dockerfile.validator",
    "gw_restart.sh",
    "validator_restart.sh",
    "requirements.txt",
    "setup.py",
}


def _run_git(root: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=text,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode("utf-8", "replace")
        raise ProductionParityError(f"git {' '.join(args)} failed: {stderr.strip()[-400:]}")
    return result.stdout


def _resolve_sha(root: Path, value: str, *, field_name: str) -> str:
    resolved = str(_run_git(root, "rev-parse", f"{value}^{{commit}}")).strip().lower()
    if not SHA_RE.fullmatch(resolved):
        raise ProductionParityError(f"{field_name} did not resolve to a full Git SHA")
    return resolved


def _tracked_paths(root: Path, sha: str) -> list[str]:
    output = str(_run_git(root, "ls-tree", "-r", "--name-only", sha))
    paths = [line.strip() for line in output.splitlines() if line.strip()]
    if not paths or len(paths) != len(set(paths)):
        raise ProductionParityError("candidate tracked path inventory is invalid")
    return sorted(paths)


def _blob(root: Path, sha: str, path: str) -> bytes:
    return bytes(_run_git(root, "show", f"{sha}:{path}", text=False))


def _source_commitments(
    root: Path,
    sha: str,
    paths: Sequence[str],
) -> list[dict[str, str]]:
    """Bind every selected candidate path to its exact Git blob."""

    return [
        {"path": path, "sha256": sha256_bytes(_blob(root, sha, path))}
        for path in sorted(set(paths))
    ]


def _changed_paths(root: Path, base_sha: str, candidate_sha: str) -> list[str]:
    output = str(
        _run_git(
            root,
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            base_sha,
            candidate_sha,
            "--",
        )
    )
    return sorted({line.strip() for line in output.splitlines() if line.strip()})


def classify_impact(changed_paths: Sequence[str]) -> dict[str, Any]:
    reasons: set[str] = set()
    for path in changed_paths:
        if path in HIGH_RISK_EXACT:
            reasons.add(f"runtime_identity:{path}")
            continue
        if path.startswith(HIGH_RISK_PREFIXES):
            if path.startswith("tests/"):
                continue
            reasons.add(f"runtime_path:{path.split('/', 1)[0]}")
            continue
        if path in LOW_RISK_EXACT or path.startswith(LOW_RISK_PREFIXES):
            continue
        reasons.add(f"unknown_executable_scope:{path}")
    risk_class = "high" if reasons else "low"
    return {
        "class": risk_class,
        "full_physical_required": risk_class == "high",
        "reasons": sorted(reasons or {"documentation_or_test_only"}),
    }


def _runtime_execution_config(
    path: Path | None, *, required: bool
) -> dict[str, Any] | None:
    if path is None:
        if required:
            raise ProductionParityError("sanitized production runtime config is required")
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProductionParityError("sanitized production runtime config is unreadable") from exc
    if not isinstance(raw, Mapping) or not isinstance(
        raw.get("execution_config"), Mapping
    ):
        raise ProductionParityError("canonical production runtime config is invalid")
    try:
        return validate_research_lab_execution_config(raw["execution_config"])
    except ResearchLabRuntimeConfigV2Error as exc:
        raise ProductionParityError(
            "canonical production runtime config differs from candidate classification"
        ) from exc


def _runtime_config_commitment(
    document: Mapping[str, Any] | None,
) -> tuple[str, list[str]]:
    if document is None:
        return sha256_json({"execution_config": None}), ["execution_config:none"]
    normalized = validate_research_lab_execution_config(document)
    keys = [
        "deployment:network",
        "deployment:netuid",
        "epoch_authority:cutover",
        *[f"field:{name}" for name in normalized["fields"]],
        *[
            f"behavior:{name}"
            for name in normalized["behavior_environment"]
        ],
    ]
    return research_lab_execution_config_hash(normalized), sorted(keys)


def _production_policy_commitments(
    execution_config: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = validate_research_lab_execution_config(execution_config)
    config = research_lab_config_from_document(normalized)
    allocation_policy = config.reimbursement_policy_doc(enabled=True)
    chain_policy = chain_source_policy_document()
    return {
        "chain_source": {
            "policy": chain_policy,
            "policy_hash": chain_source_policy_hash(),
        },
        "research_lab_allocation": {
            "policy": allocation_policy,
            "policy_hash": sha256_json(allocation_policy),
        },
    }


def _bind_behavior_to_runtime(
    behavior: Mapping[str, Any],
    execution_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if execution_config is None:
        return dict(behavior)
    body = {
        key: value for key, value in behavior.items() if key != "contract_hash"
    }
    body["policy_commitments"] = _production_policy_commitments(execution_config)
    return {**body, "contract_hash": sha256_json(body)}


def _migration_inventory(root: Path, candidate_sha: str, paths: Sequence[str]) -> list[dict[str, Any]]:
    values = []
    for path in paths:
        if MIGRATION_RE.fullmatch(path) is None:
            continue
        sequence, _ = migration_sequence(path)
        values.append(
            {
                "path": path,
                "sequence": sequence,
                "sha256": sha256_bytes(_blob(root, candidate_sha, path)),
                "transaction_mode": (
                    "autocommit" if path.endswith(".concurrent.sql") else "candidate-file"
                ),
            }
        )
    return sorted(values, key=lambda item: (item["sequence"], item["path"]))


def build_contract(
    *,
    root: Path,
    base_sha: str,
    candidate_sha: str,
    runtime_config: Path | None = None,
    require_runtime_config: bool = False,
) -> dict[str, Any]:
    normalized_root = root.resolve()
    base = _resolve_sha(normalized_root, base_sha, field_name="base_sha")
    candidate = _resolve_sha(
        normalized_root, candidate_sha, field_name="candidate_sha"
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, candidate],
        cwd=normalized_root,
        check=False,
    )
    if ancestor.returncode != 0 or base == candidate:
        raise ProductionParityError("candidate must descend from a distinct base SHA")
    head = _resolve_sha(normalized_root, "HEAD", field_name="HEAD")
    if head != candidate:
        raise ProductionParityError("contract must be built from the exact candidate checkout")

    tracked = _tracked_paths(normalized_root, candidate)
    tracked_set = set(tracked)
    changed = _changed_paths(normalized_root, base, candidate)
    execution_config = _runtime_execution_config(
        runtime_config, required=require_runtime_config
    )
    behavior = _bind_behavior_to_runtime(
        build_rehearsal_behavior_contract_v2(
            source_root=normalized_root,
            candidate_sha=candidate,
            profile="prepush",
            epoch_count=1,
        ),
        execution_config,
    )
    runtime_config_hash, runtime_config_keys = _runtime_config_commitment(
        execution_config
    )

    commitment_paths = set(ALWAYS_COMMITTED_PATHS)
    commitment_paths.update(behavior["production_source_paths"])
    commitment_paths.update(path for path in tracked if MIGRATION_RE.fullmatch(path))
    commitment_paths.update(
        path
        for path in changed
        if path in tracked_set
        and (
            path.endswith((".py", ".sh", ".sql", ".yml", ".yaml", ".json", ".toml"))
            or Path(path).name.startswith("Dockerfile")
        )
    )
    missing = sorted(path for path in commitment_paths if path not in tracked_set)
    if missing:
        raise ProductionParityError(
            "candidate parity source commitments are missing: " + ",".join(missing)
        )
    source_commitments = _source_commitments(
        normalized_root,
        candidate,
        sorted(commitment_paths),
    )
    oracle = validate_historical_oracle(
        json.loads(_blob(normalized_root, candidate, ORACLE_PATH).decode("utf-8"))
    )
    body = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "base_sha": base,
        "candidate_sha": candidate,
        "changed_paths": changed,
        "risk": classify_impact(changed),
        "source_commitments": source_commitments,
        "migrations": _migration_inventory(normalized_root, candidate, tracked),
        "behavior_contract_hash": behavior["contract_hash"],
        "protected_manifest_hash": behavior["protected_manifest_hash"],
        "historical_oracle_hash": sha256_json(oracle),
        "runtime_config_hash": runtime_config_hash,
        "runtime_config_keys": runtime_config_keys,
        "policy_commitments": behavior["policy_commitments"],
    }
    return validate_contract({**body, "contract_hash": sha256_json(body)})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--require-runtime-config", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        contract = build_contract(
            root=args.root,
            base_sha=args.base_sha,
            candidate_sha=args.candidate_sha,
            runtime_config=args.runtime_config,
            require_runtime_config=bool(args.require_runtime_config),
        )
    except (OSError, ValueError, ProductionParityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "candidate_sha": contract["candidate_sha"],
                "contract_hash": contract["contract_hash"],
                "risk_class": contract["risk"]["class"],
                "full_physical_required": contract["risk"]["full_physical_required"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
