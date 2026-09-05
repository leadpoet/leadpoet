#!/usr/bin/env python3.11
"""Strict boundary adapters for isolated restart testing.

The gateway and validator restart shell scripts execute unchanged. Privileged
external services may be adapted. A repository module, script, or long-lived
process that is substituted is recorded explicitly and invalidates a complete
restart rehearsal. Substitutions are permitted only in a clearly labelled
targeted regression run.
"""

from __future__ import annotations

import configparser
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable

# The production gateway import preflight deliberately sets PYTHONSAFEPATH=1
# and replaces PYTHONPATH with the candidate checkout.  Load harness siblings
# explicitly so the boundary adapter remains self-contained in that exact
# invocation shape.
HARNESS_ROOT = Path(__file__).resolve().parent
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from artifact_identity import (
    ALL_ROLES,
    GATEWAY_ROLES,
    VALIDATOR_ROLE,
    docker_save_archive,
    eif_bytes,
    normalized_image_id,
    pcr0 as artifact_pcr0,
)
STATE_ROOT = Path(os.environ.get("REHEARSAL_STATE_ROOT", "/rehearsal-state"))
STATE_PATH = STATE_ROOT / "state.json"
EVENT_PATH = STATE_ROOT / "events.jsonl"
LOCK_PATH = STATE_ROOT / "adapter.lock"
REAL_PYTHON = "/usr/bin/python3.11"
REAL_BASH = "/bin/bash"
REAL_CURL = "/usr/bin/curl"
REAL_GIT = "/usr/bin/git"
_PCR0_CANDIDATE = os.environ.get("REHEARSAL_CANDIDATE_SHA", "").encode("ascii")
PCR0 = artifact_pcr0(_PCR0_CANDIDATE.decode("ascii"))
HASH64 = hashlib.sha256(b"leadpoet-local-restart-rehearsal").hexdigest()
PRODUCTION_SUPABASE_ORIGIN = "https://qplwoislplkcegvdmbim.supabase.co"
LOCAL_POSTGREST_ORIGIN = "http://127.0.0.1:54321"
ACCOUNT = "493765492819"
TARGETED_REGRESSION_SCOPE = "weight_readiness_regression"
PCR0_CACHE_PROVENANCE = "validator_pcr0_cache_v1"
_PCR0_CACHE_RAW_TAG = re.compile(r"validator-enclave-build-[1-9][0-9]*\Z")
_PCR0_CACHE_NORMALIZED_TAG = re.compile(
    r"validator-enclave-build-[1-9][0-9]*-normalized:latest\Z"
)
RUNSC_LOCK_PATH = Path("/opt/leadpoet/runsc-runtime.lock.json")
EXTERNAL_ARTIFACT_ROOT = Path("/opt/leadpoet/external-artifacts")
GITHUB_GIT_FIXTURE_REMOTE = Path("/srv/origin.git")
LOCAL_GIT_FIXTURE_SOURCE = Path("/source")
GIT_FETCH_REPOSITORY_ROOTS = (
    Path("/home/ec2-user/leadpoet_repo"),
    Path("/home/ec2-user/leadpoet/leadpoet"),
    Path("/tmp"),
)
GATEWAY_SECRET_STATE_PATH = Path(
    os.environ.get("REHEARSAL_DURABLE_STATE_ROOT", "/rehearsal-durable-state")
) / "gateway-secret-state.json"
LEGACY_GATEWAY_MINER_MAINTENANCE_FROM_SHA = (
    "0dd3a385a23a3af0fa17210bfe02a39cc4023952"
)
POST_GATEWAY_MINER_MAINTENANCE_FROM_SHA = (
    "7ac1553e32d85d9babda3b3836f4c93cf92e6d60"
)
GATEWAY_MINER_MAINTENANCE_LINEAGE_REPOSITORY = Path("/source")


def _ensure_state() -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.touch(exist_ok=True)
    if not STATE_PATH.exists():
        STATE_PATH.write_text(
            json.dumps(
                {
                    "component": os.environ.get("REHEARSAL_COMPONENT", ""),
                    "candidate_sha": os.environ.get("REHEARSAL_CANDIDATE_SHA", ""),
                    "images": {},
                    "containers": {},
                    "enclaves": [],
                    "processes": {},
                    "docker_ready": True,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _locked_state() -> tuple[io.TextIOWrapper, dict[str, Any]]:
    _ensure_state()
    handle = LOCK_PATH.open("r+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    return handle, value


def _save_state(handle: io.TextIOWrapper, value: dict[str, Any]) -> None:
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, STATE_PATH)
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _event(kind: str, argv: Iterable[str], **details: Any) -> None:
    _ensure_state()
    payload = {
        "at_ns": time.time_ns(),
        "kind": kind,
        "argv": list(argv),
        **details,
    }
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(
        EVENT_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)


def _fail(kind: str, argv: list[str], message: str) -> int:
    _event(kind, argv, status="rejected", reason=message)
    print(f"REHEARSAL CONTRACT ERROR [{kind}]: {message}: {argv!r}", file=sys.stderr)
    return 97


def _arg_value(argv: list[str], name: str, default: str = "") -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def _arg_values(argv: list[str], name: str) -> tuple[str, ...]:
    return tuple(
        argv[index + 1]
        for index, value in enumerate(argv[:-1])
        if value == name
    )


def _candidate_sha() -> str:
    configured = os.environ.get("REHEARSAL_CANDIDATE_SHA", "").strip()
    if re.fullmatch(r"[0-9a-f]{40}", configured):
        return configured
    repo = Path("/home/ec2-user/leadpoet_repo")
    if not repo.exists():
        repo = Path("/home/ec2-user/leadpoet/leadpoet")
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _rehearsal_scope() -> str:
    return os.environ.get("REHEARSAL_SCOPE", "exact").strip()


def _targeted_substitutions_allowed() -> bool:
    return _rehearsal_scope() == TARGETED_REGRESSION_SCOPE


def _route_host_storage_preflight_to_local_postgrest(module: str) -> None:
    if module not in {
        "gateway.main",
        "gateway.research_lab.stateful_epoch_cutover_cli_v1",
        "gateway.tee.bootstrap_active_ancestry_checkpoints_v2",
        "gateway.tee.prepare_active_release_lineage_v2",
        "gateway.tee.verify_weight_submission_ready_v2",
    }:
        return
    configured = os.environ.get("SUPABASE_URL", "").strip()
    if configured not in {PRODUCTION_SUPABASE_ORIGIN, LOCAL_POSTGREST_ORIGIN}:
        raise ValueError("host storage preflight Supabase origin differs")
    os.environ["SUPABASE_URL"] = LOCAL_POSTGREST_ORIGIN


def _candidate_root() -> Path:
    gateway_root = Path("/home/ec2-user/leadpoet_repo")
    if gateway_root.is_dir():
        return gateway_root
    validator_root = Path("/home/ec2-user/leadpoet/leadpoet")
    if validator_root.is_dir():
        return validator_root
    raise RuntimeError("candidate checkout is unavailable")


def _candidate_git_path(resolved: Path, root: Path) -> tuple[Path, str]:
    if resolved == root or root in resolved.parents:
        return resolved.relative_to(root), "candidate_checkout"

    archive_parent = Path("/tmp").resolve()
    for parent in resolved.parents:
        if (
            parent.parent == archive_parent
            and re.fullmatch(
                r"(?:gateway-v2-preflight|leadpoet-local-release-source)\.[A-Za-z0-9]+",
                parent.name,
            )
        ):
            return resolved.relative_to(parent), "candidate_archive"
        if (
            parent.parent == archive_parent
            and re.fullmatch(
                r"gateway-miner-maintenance-bootstrap\.[A-Za-z0-9]+",
                parent.name,
            )
        ):
            candidate_archive = parent / "candidate"
            if candidate_archive in resolved.parents:
                return (
                    resolved.relative_to(candidate_archive),
                    "candidate_archive",
                )
        if (
            parent.parent == archive_parent
            and re.fullmatch(
                r"(?:gateway|validator)-restart-controller-bootstrap\.[A-Za-z0-9]+",
                parent.name,
            )
        ):
            candidate_archive = parent / "authority"
            if candidate_archive in resolved.parents:
                return (
                    resolved.relative_to(candidate_archive),
                    "candidate_archive",
                )

    configured_build_root = Path(
        os.environ.get(
            "GATEWAY_V2_BUILD_WORK_ROOT",
            "/home/ec2-user/.cache/leadpoet/gateway-release-build-v2",
        )
    )
    if configured_build_root.is_absolute():
        candidate_build_root = (
            configured_build_root.resolve() / f"{_candidate_sha()}-local"
        )
        try:
            relative = resolved.relative_to(candidate_build_root)
        except ValueError:
            pass
        else:
            if (
                len(relative.parts) >= 3
                and relative.parts[0] in GATEWAY_ROLES
                and relative.parts[1] == "source"
            ):
                return Path(*relative.parts[2:]), "candidate_archive"

    raise RuntimeError(
        "production source is outside the candidate checkout or a recognized "
        f"candidate archive: {resolved} (candidate root: {root})"
    )


def _source_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    root = _candidate_root().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"production source is unavailable: {resolved}")
    relative, source_kind = _candidate_git_path(resolved, root)
    candidate_sha = _candidate_sha()
    source_commit = candidate_sha
    if source_kind == "candidate_checkout":
        source_commit = subprocess.run(
            ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        from_sha = os.environ.get("REHEARSAL_FROM_SHA", "").strip()
        if source_commit not in {candidate_sha, from_sha}:
            raise RuntimeError(
                "production source checkout is neither the installed nor "
                f"candidate commit: {source_commit}"
            )
        if source_commit == from_sha and from_sha != candidate_sha:
            source_kind = "installed_checkout"
    result = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(root),
            "show",
            f"{source_commit}:{relative.as_posix()}",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "production source is not present at its bound commit: "
            f"{relative.as_posix()}"
        )
    source_bytes = resolved.read_bytes()
    if source_bytes != result.stdout:
        raise RuntimeError(
            "production source bytes differ from its bound commit: "
            f"{relative.as_posix()}"
        )
    return {
        "candidate_sha": candidate_sha,
        "source_commit": source_commit,
        "source_path": str(resolved),
        "source_git_path": relative.as_posix(),
        "source_kind": source_kind,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def _module_source(module: str) -> Path:
    roots = [Path.cwd()]
    roots.extend(
        Path(item)
        for item in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if item
    )
    candidate_root = _candidate_root()
    roots.append(candidate_root)
    seen: set[Path] = set()
    for root in roots:
        resolved_root = root.resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        module_path = resolved_root.joinpath(
            *module.split(".")
        ).with_suffix(".py")
        if module_path.is_file():
            return module_path
        package_main = resolved_root.joinpath(
            *module.split("."),
            "__main__.py",
        )
        if package_main.is_file():
            return package_main
    raise RuntimeError("candidate module source is unavailable: %s" % module)


def _record_production_module(module: str, argv: list[str]) -> None:
    _event(
        "python-module",
        argv,
        status="started",
        module=module,
        implementation="production_module",
        **_source_identity(_module_source(module)),
    )


def _record_production_script(path: Path, argv: list[str]) -> None:
    _event(
        "python-script",
        argv,
        status="started",
        script=path.name,
        implementation="production_script",
        **_source_identity(path),
    )


def _record_internal_substitution(
    *,
    kind: str,
    argv: list[str],
    module: str = "",
    script: str = "",
    process: str = "",
    substitution: str = "",
) -> int:
    details = {
        "status": "substituted",
        "implementation": "internal_substitution",
        "scope": _rehearsal_scope(),
    }
    if module:
        details["module"] = module
    if script:
        details["script"] = script
    if process:
        details["process"] = process
    if substitution:
        details["substitution"] = substitution
    _event(kind, argv, **details)
    if _targeted_substitutions_allowed():
        return 0
    return _fail(
        kind,
        argv,
        "repository implementation substitution invalidates exact rehearsal",
    )


def _record_external_boundary(
    *,
    kind: str,
    argv: list[str],
    boundary: str,
    operation: str,
    status: str = "ok",
    **details: Any,
) -> None:
    contract_path = Path("/harness/boundary_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    definition = (contract.get("boundaries") or {}).get(boundary)
    if not isinstance(definition, dict):
        raise RuntimeError(f"external boundary is not allowlisted: {boundary}")
    allowed = definition.get("allowed_operations") or []
    if operation not in allowed:
        raise RuntimeError(
            f"external boundary operation is not allowlisted: "
            f"{boundary}.{operation}"
        )
    _event(
        kind,
        argv,
        status=status,
        boundary=boundary,
        operation=operation,
        implementation="external_boundary",
        fixture_authenticity="production_shaped_sanitized",
        reject_unknown=bool(definition.get("reject_unknown")),
        **details,
    )


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_commit_is_ancestor(
    ancestor: str,
    descendant: str,
    *,
    repository: Path = GATEWAY_MINER_MAINTENANCE_LINEAGE_REPOSITORY,
) -> bool:
    result = subprocess.run(
        [
            REAL_GIT,
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in {0, 1}:
        raise ValueError("gateway miner-maintenance lineage is unavailable")
    return result.returncode == 0


def _gateway_miner_maintenance_lineage() -> str:
    from_sha = os.environ.get("REHEARSAL_FROM_SHA", "").strip()
    transition = os.environ.get("REHEARSAL_TRANSITION", "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", from_sha):
        raise ValueError("gateway miner-maintenance from-SHA is invalid")
    if transition == "rollback":
        return "rollback"
    if transition != "forward":
        raise ValueError("gateway miner-maintenance transition is invalid")
    if from_sha == LEGACY_GATEWAY_MINER_MAINTENANCE_FROM_SHA:
        return "legacy"
    if from_sha == POST_GATEWAY_MINER_MAINTENANCE_FROM_SHA:
        return "post_rollout"
    if _git_commit_is_ancestor(
        POST_GATEWAY_MINER_MAINTENANCE_FROM_SHA,
        from_sha,
    ):
        return "post_rollout"
    raise ValueError("gateway miner-maintenance from-SHA lineage is unknown")


def _initial_gateway_miner_submissions_state() -> str:
    lineage = _gateway_miner_maintenance_lineage()
    if lineage == "rollback":
        raise ValueError(
            "gateway miner-maintenance rollback requires durable state"
        )
    return "true" if lineage == "legacy" else "false"


def _validate_gateway_miner_submissions_state(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError("durable miner-maintenance setting is invalid")
    lineage = _gateway_miner_maintenance_lineage()
    if lineage == "legacy":
        return "legacy_first_rollout" if normalized == "true" else "legacy_retry"
    if normalized != "false":
        raise ValueError(
            "durable miner-maintenance setting conflicts with release lineage"
        )
    return lineage


def _gateway_secret() -> dict[str, str]:
    values = {
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "BITTENSOR_NETWORK": "finney",
        "BITTENSOR_NETUID": "71",
        "GITHUB_REPO_URL": "/srv/origin.git",
        "GITHUB_BRANCH": "main",
        "SUPABASE_URL": PRODUCTION_SUPABASE_ORIGIN,
        "SUPABASE_ANON_KEY": "rehearsal-public",
        "SUPABASE_SERVICE_ROLE_KEY": "rehearsal-secret",
        "OPENROUTER_API_KEY": "rehearsal-openrouter",
        "EXA_API_KEY": "rehearsal-exa",
        "SCRAPINGDOG_API_KEY": "rehearsal-scrapingdog",
        "DEEPLINE_API_KEY": "rehearsal-deepline",
        "TRUELIST_API_KEY": "rehearsal-truelist",
        "RESEARCH_LAB_TEE_PROTOCOL": "v2",
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "true",
        "RESEARCH_LAB_RAW_TRACE_S3_PREFIX": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            "research-lab/rehearsal/raw-traces"
        ),
        "RESEARCH_LAB_SCORER_TRACE_S3_PREFIX": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            "research-lab/rehearsal/scorer-traces"
        ),
        "RESEARCH_LAB_TRACE_KMS_KEY_ID": (
            "alias/leadpoet-research-lab-trace-encryption"
        ),
        "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            "research-lab/rehearsal/incontainer-traces"
        ),
        "RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID": (
            "alias/leadpoet-research-lab-trace-encryption"
        ),
        "GATEWAY_PYTHON_BIN": "/home/ec2-user/venv311/bin/python3",
        "GATEWAY_PRIVATE_KEY_PATH": (
            "/home/ec2-user/gateway/secrets/gateway_private_key.pem"
        ),
        "ARWEAVE_KEYFILE_PATH": (
            "/home/ec2-user/gateway/secrets/arweave_keyfile.json"
        ),
        "GATEWAY_TEE_TOPOLOGY_MODE": "full",
        "GATEWAY_TEE_ROLE_READY_TIMEOUT_SECONDS": "5",
        "GATEWAY_TEE_ROLE_READY_RETRY_SECONDS": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "RESEARCH_LAB_GATEWAY_API_ENABLED": "true",
        "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED": "true",
        "RESEARCH_LAB_RECEIPTS_ENABLED": "true",
        "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
        "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
        "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": "true",
        "RESEARCH_LAB_INTERNAL_API_KEY": "rehearsal-internal",
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1": (
            "http://legacy-scoring:legacy-password@legacy-proxy.example.com:7421"
        ),
    }
    worker_fleet_mode = os.environ.get(
        "REHEARSAL_GATEWAY_WORKER_FLEET_MODE",
        "active",
    )
    if worker_fleet_mode not in {"active", "deferred"}:
        raise RuntimeError(
            "unknown rehearsal gateway worker fleet mode: "
            + worker_fleet_mode
        )
    if os.environ.get("REHEARSAL_WEIGHT_READINESS_SCENARIO") != (
        "plaintext_proxy_rejected"
    ):
        values.update(
            {
                "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1": (
                    "https://rehearsal-scoring:rehearsal-scoring-password@"
                    "93.184.216.34:443"
                ),
                "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_2": (
                    "https://rehearsal-invalid:invalid-password@"
                    "93.184.216.34:443"
                ),
                "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "1",
            }
        )
    return values


def _current_gateway_secret() -> dict[str, str]:
    if not GATEWAY_SECRET_STATE_PATH.is_file():
        secret = _gateway_secret()
        secret["RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED"] = (
            _initial_gateway_miner_submissions_state()
        )
        return secret
    state = json.loads(GATEWAY_SECRET_STATE_PATH.read_text(encoding="utf-8"))
    if (
        state.get("schema_version")
        != "leadpoet.restart_rehearsal.gateway_secret_state.v1"
        or state.get("secret_id") != "leadpoet/prod/gateway/env"
        or not isinstance(state.get("versions"), dict)
    ):
        raise ValueError("durable rehearsal gateway secret state is invalid")
    current = [
        row
        for row in state["versions"].values()
        if isinstance(row, dict) and "AWSCURRENT" in (row.get("stages") or [])
    ]
    if len(current) != 1 or not isinstance(current[0].get("secret_string"), str):
        raise ValueError("durable rehearsal gateway current secret is invalid")
    secret = json.loads(current[0]["secret_string"])
    if not isinstance(secret, dict) or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in secret.items()
    ):
        raise ValueError("durable rehearsal gateway secret document is invalid")
    _validate_gateway_miner_submissions_state(
        secret.get("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED")
    )
    return secret


def _candidate_worker_ids(
    *,
    section_start: str,
    section_end: str,
    role: str,
) -> tuple[int, ...]:
    """Derive one exercised validator worker fleet from the frozen deployer."""

    source = (
        _candidate_root()
        / "validator_models"
        / "containerizing"
        / "deploy_dynamic.sh"
    )
    try:
        deploy = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"candidate {role} worker deployment source is unavailable"
        ) from exc
    try:
        section = deploy[
            deploy.index(section_start) : deploy.index(section_end)
        ]
    except ValueError as exc:
        raise RuntimeError(
            f"candidate {role} worker deployment section is unavailable"
        ) from exc
    worker_range = re.search(
        r"for\s+i\s+in\s+\{([1-9][0-9]*)\.\.([1-9][0-9]*)\};\s*do",
        section,
    )
    if worker_range is None:
        raise RuntimeError(
            f"candidate {role} worker selection range is unavailable"
        )
    first, last = (int(value) for value in worker_range.groups())
    if first > last:
        raise RuntimeError(
            f"candidate {role} worker selection range is invalid"
        )
    return tuple(range(first, last + 1))


def _candidate_fulfillment_worker_ids() -> tuple[int, ...]:
    return _candidate_worker_ids(
        section_start="# Auto-detect FULFILLMENT proxies",
        section_end="# Wait for containers to start",
        role="fulfillment",
    )


def _validator_secret() -> dict[str, str]:
    values = {
        "ENABLE_FULFILLMENT": "true",
        "FULFILLMENT_LEADERBOARD_EMISSIONS_ENABLED": "false",
        "LEADPOET_WRAPPER_ACTIVE": "1",
        "GATEWAY_URL": "http://gateway.invalid:8000",
        "VALIDATOR_V2_GATEWAY_URL": "http://gateway.invalid:8000",
        "SUPABASE_URL": "http://127.0.0.1:54321",
        "SUPABASE_ANON_KEY": "rehearsal-public",
        "SUPABASE_SERVICE_ROLE_KEY": "rehearsal-secret",
        "OPENROUTER_API_KEY": "rehearsal-openrouter",
        "OPENROUTER_KEY": "rehearsal-openrouter",
        "FULFILLMENT_OPENROUTER_API_KEY": "rehearsal-openrouter",
        "EXA_API_KEY": "rehearsal-exa",
        "SCRAPINGDOG_API_KEY": "rehearsal-scrapingdog",
        "TRUELIST_API_KEY": "rehearsal-truelist",
        "COMPANIES_HOUSE_API_KEY": "rehearsal-companies-house",
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "RESEARCH_LAB_VALIDATOR_FETCH_ENABLED": "true",
        "RESEARCH_LAB_INTERNAL_API_KEY": "rehearsal-internal",
        "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": "true",
        "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED": "true",
        "LEADPOET_SENTRY_RELEASE": "stale-n-minus-one-release",
        "EXPECTED_CHAIN": "wss://entrypoint-finney.opentensor.ai:443",
        "NO_PROXY": "127.0.0.1,localhost",
        "VALIDATOR_WEIGHT_PROTOCOL": "authoritative_v2",
    }
    for worker_id in _candidate_fulfillment_worker_ids():
        values[f"FULFILLMENT_WEBSHARE_PROXY_{worker_id}"] = (
            f"http://rehearsal-ff-{worker_id}.invalid:8080"
        )
    return values


def command_aws(argv: list[str]) -> int:
    if argv[:2] == ["secretsmanager", "get-secret-value"]:
        component = os.environ.get("REHEARSAL_COMPONENT", "")
        expected_secret_id = (
            "leadpoet/prod/gateway/env"
            if component == "gateway"
            else "leadpoet/prod/validator/env"
        )
        if argv != [
            "secretsmanager",
            "get-secret-value",
            "--secret-id",
            expected_secret_id,
            "--query",
            "SecretString",
            "--output",
            "text",
        ]:
            return _fail("aws", argv, "Secrets Manager CLI contract differs")
        secret = (
            _current_gateway_secret()
            if component == "gateway"
            else _validator_secret()
        )
        _record_external_boundary(
            kind="aws",
            argv=argv,
            boundary="aws_cli",
            operation="secretsmanager",
        )
        print(json.dumps(secret, sort_keys=True))
        return 0
    if argv[:2] == ["sts", "get-caller-identity"]:
        _record_external_boundary(
            kind="aws",
            argv=argv,
            boundary="aws_cli",
            operation="sts",
        )
        print(ACCOUNT)
        return 0
    if argv[:2] == ["ecr", "get-login-password"]:
        _record_external_boundary(
            kind="aws",
            argv=argv,
            boundary="aws_cli",
            operation="ecr_login",
        )
        print("rehearsal-ecr-password")
        return 0
    return _fail("aws", argv, "unknown AWS operation")


def _is_allowed_git_fetch_repository(repository: Path) -> bool:
    try:
        resolved = repository.resolve(strict=True)
    except OSError:
        return False
    return any(
        resolved == root.resolve() or root.resolve() in resolved.parents
        for root in GIT_FETCH_REPOSITORY_ROOTS
        if root.is_dir()
    )


def _git_origin_url_without_execution(repository: Path) -> str:
    git_directory = repository / ".git"
    config_path = git_directory / "config"
    if (
        not git_directory.is_dir()
        or git_directory.is_symlink()
        or not config_path.is_file()
        or config_path.is_symlink()
    ):
        raise ValueError("candidate Git repository metadata is unsafe")
    parser = configparser.RawConfigParser(
        interpolation=None,
        strict=True,
    )
    try:
        parser.read_string(config_path.read_text(encoding="utf-8"))
        origin_url = parser.get('remote "origin"', "url")
    except (OSError, UnicodeError, configparser.Error) as exc:
        raise ValueError("candidate Git origin configuration is invalid") from exc
    if not origin_url or origin_url != origin_url.strip():
        raise ValueError("candidate Git origin configuration is invalid")
    return origin_url


def _is_safe_origin_fetch(argv: list[str], fetch_index: int) -> bool:
    values = argv[fetch_index + 1 :]
    if values.count("origin") != 1:
        return False
    origin_index = values.index("origin")
    options = values[:origin_index]
    refs = values[origin_index + 1 :]
    index = 0
    while index < len(options):
        option = options[index]
        if option in {"-q", "--quiet", "--prune", "--no-tags", "-f", "--force"}:
            index += 1
            continue
        if option == "--depth":
            if index + 1 >= len(options) or not options[index + 1].isdigit():
                return False
            index += 2
            continue
        if re.fullmatch(r"--depth=[1-9][0-9]*", option):
            index += 1
            continue
        return False
    for ref in refs:
        if re.fullmatch(r"[0-9a-f]{40}", ref):
            continue
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}", ref) and not any(
            unsafe in ref for unsafe in ("..", "//", "@{")
        ):
            continue
        if re.fullmatch(
            r"\+refs/heads/([A-Za-z0-9][A-Za-z0-9._/-]{0,127}):"
            r"refs/remotes/origin/\1",
            ref,
        ) and not any(unsafe in ref for unsafe in ("..", "//", "@{")):
            continue
        return False
    return True


def _is_local_fixture_seed_fetch(argv: list[str]) -> bool:
    if len(argv) != 5:
        return False
    git_dir, command, quiet, source, refspec = argv
    return (
        git_dir == f"--git-dir={GITHUB_GIT_FIXTURE_REMOTE}"
        and command == "fetch"
        and quiet == "-q"
        and source == str(LOCAL_GIT_FIXTURE_SOURCE)
        and GITHUB_GIT_FIXTURE_REMOTE.is_dir()
        and LOCAL_GIT_FIXTURE_SOURCE.is_dir()
        and re.fullmatch(
            r"[0-9a-f]{40}:refs/heads/(?:main|rehearsal-target|rehearsal-deployed)",
            refspec,
        )
        is not None
    )


def command_git(argv: list[str]) -> int:
    rewritten = list(argv)
    fetch_indexes = [
        index for index, value in enumerate(rewritten) if value == "fetch"
    ]
    if not fetch_indexes:
        os.execv(REAL_GIT, [REAL_GIT, *rewritten])
        return 127
    if len(fetch_indexes) != 1:
        return _fail("git", argv, "candidate Git fetch command is ambiguous")
    if _is_local_fixture_seed_fetch(rewritten):
        os.execv(REAL_GIT, [REAL_GIT, *rewritten])
        return 127

    fetch_index = fetch_indexes[0]
    prefix = rewritten[:fetch_index]
    if len(prefix) == 2 and prefix[0] == "-C":
        repository = Path(prefix[1])
    elif not prefix:
        repository = Path.cwd()
    else:
        return _fail("git", argv, "candidate Git fetch repository differs")
    if not _is_allowed_git_fetch_repository(repository):
        return _fail("git", argv, "candidate Git fetch repository is not allowlisted")
    if not _is_safe_origin_fetch(rewritten, fetch_index):
        return _fail("git", argv, "candidate Git fetch arguments are unsafe")
    try:
        origin_url = _git_origin_url_without_execution(repository)
    except ValueError as exc:
        return _fail("git", argv, str(exc))
    if origin_url not in {
        "https://github.com/leadpoet/leadpoet.git",
        str(GITHUB_GIT_FIXTURE_REMOTE),
    }:
        return _fail("git", argv, "candidate Git fetch origin is not allowlisted")
    if not GITHUB_GIT_FIXTURE_REMOTE.is_dir():
        return _fail("git", argv, "local Git fixture remote is unavailable")
    origin_index = rewritten.index("origin", fetch_index + 1)
    rewritten[origin_index] = str(GITHUB_GIT_FIXTURE_REMOTE)
    if not rewritten[origin_index + 1 :]:
        rewritten.append(
            "+refs/heads/main:refs/remotes/origin/main"
        )
    if origin_url == "https://github.com/leadpoet/leadpoet.git":
        _record_external_boundary(
            kind="git",
            argv=argv,
            boundary="github_git_transport",
            operation="fetch",
            remote_url=origin_url,
            fixture_remote=str(GITHUB_GIT_FIXTURE_REMOTE),
        )
    os.execv(REAL_GIT, [REAL_GIT, *rewritten])
    return 127


def _image_id(name: str) -> str:
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def _image_record_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("id") or "")
    return str(value or "")


def _image_record_by_id(
    images: dict[str, Any],
    image_id: str,
) -> dict[str, Any] | None:
    for value in images.values():
        if (
            isinstance(value, dict)
            and _image_record_id(value) == image_id
        ):
            return dict(value)
    return None


def _external_build_role(argv: list[str], tag: str) -> str:
    if tag == "validator-tee-enclave:raw":
        dockerfile = _arg_value(argv, "-f")
        if not dockerfile.endswith("/validator_tee/Dockerfile.enclave"):
            raise ValueError("validator enclave build used an unexpected Dockerfile")
        return VALIDATOR_ROLE
    match = re.fullmatch(r"tee-enclave:(gateway_[a-z]+)-raw", tag)
    if match is not None:
        role = match.group(1)
        if role not in GATEWAY_ROLES:
            raise ValueError("gateway enclave build used an unknown role")
        build_arg = _arg_value(argv, "--build-arg")
        dockerfile = _arg_value(argv, "-f")
        if (
            build_arg != f"LEADPOET_ENCLAVE_ROLE={role}"
            or not dockerfile.endswith("/gateway/tee/Dockerfile.enclave")
        ):
            raise ValueError("gateway enclave build contract is invalid")
        return role
    if tag.startswith("leadpoet-gateway-verify:"):
        match = re.fullmatch(
            r"leadpoet-gateway-verify:(gateway_[a-z]+)-([0-9a-f]{12})-([1-9][0-9]*)-raw",
            tag,
        )
        if match is None:
            raise ValueError("gateway verification image tag is invalid")
        role, short_commit, _index = match.groups()
        build_args = _arg_values(argv, "--build-arg")
        dockerfile = Path(_arg_value(argv, "-f"))
        context = Path(argv[-1]) if argv else Path()
        if (
            role not in GATEWAY_ROLES
            or short_commit != _candidate_sha()[:12]
            or build_args
            != ("SOURCE_DATE_EPOCH=0", f"LEADPOET_ENCLAVE_ROLE={role}")
            or dockerfile != context / "tee/Dockerfile.enclave"
        ):
            raise ValueError("gateway verification image build contract is invalid")
        return role
    return ""


def _pcr0_cache_build_record(
    argv: list[str],
    tag: str,
) -> dict[str, str] | None:
    """Bind the production PCR0 cache build to its checked-out Git commit."""

    if _PCR0_CACHE_RAW_TAG.fullmatch(tag) is None:
        return None
    expected = [
        "build",
        "--no-cache",
        "-f",
        "validator_tee/Dockerfile.enclave",
        "-t",
        tag,
        ".",
    ]
    if argv != expected:
        raise ValueError("validator PCR0 cache build command differs")

    configured_root = Path(
        os.environ.get("PCR0_BUILD_DIR", "/tmp/pcr0_builder")
    )
    if not configured_root.is_absolute():
        raise ValueError("PCR0_BUILD_DIR must be absolute")
    try:
        build_root = configured_root.resolve(strict=True)
        current_root = Path.cwd().resolve(strict=True)
    except OSError as exc:
        raise ValueError("validator PCR0 cache build directory is unavailable") from exc
    if current_root != build_root:
        raise ValueError("validator PCR0 cache build used the wrong directory")

    commit = _pcr0_cache_git_identity(build_root)
    return {
        "commit": commit,
        "role": VALIDATOR_ROLE,
        "provenance": PCR0_CACHE_PROVENANCE,
        "source_tag": tag,
        "build_root": str(build_root),
    }


def _pcr0_cache_git_identity(build_root: Path) -> str:
    try:
        identity = subprocess.run(
            [
                "/usr/bin/git",
                "-C",
                str(build_root),
                "rev-parse",
                "--show-toplevel",
                "--verify",
                "HEAD^{commit}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("validator PCR0 cache Git identity is unavailable") from exc
    if len(identity) != 2:
        raise ValueError("validator PCR0 cache Git identity is invalid")
    top_level, commit = (value.strip() for value in identity)
    if Path(top_level).resolve() != build_root:
        raise ValueError("validator PCR0 cache build is outside its Git root")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("validator PCR0 cache Git commit is invalid")
    return commit


def _docker_save(
    path: Path,
    *,
    source_tag: str,
    record: dict[str, Any],
    normalizations: dict[str, Any],
) -> None:
    commit = str(record.get("commit") or "")
    role = str(record.get("role") or "")
    if role not in ALL_ROLES:
        raise ValueError("only commit-bound enclave images may be normalized")
    archive_bytes = docker_save_archive(commit, role, source_tag)
    cache_normalization: tuple[str, dict[str, str]] | None = None
    if str(record.get("provenance") or "") == PCR0_CACHE_PROVENANCE:
        build_root = Path(str(record.get("build_root") or ""))
        if (
            _PCR0_CACHE_RAW_TAG.fullmatch(source_tag) is None
            or str(record.get("source_tag") or "") != source_tag
            or not build_root.is_absolute()
            or _pcr0_cache_git_identity(build_root) != commit
        ):
            raise ValueError("validator PCR0 cache source provenance differs")
        normalized_tag = source_tag + "-normalized:latest"
        cache_normalization = (
            normalized_tag,
            {
                "commit": commit,
                "role": role,
                "provenance": PCR0_CACHE_PROVENANCE,
                "source_tag": source_tag,
                "normalized_image_id": normalized_image_id(commit, role),
                "build_root": str(build_root),
            },
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive_bytes)
    if cache_normalization is not None:
        normalized_tag, provenance = cache_normalization
        normalizations[normalized_tag] = provenance


def _docker_load(
    path: Path,
    images: dict[str, Any],
    normalizations: dict[str, Any],
) -> list[str]:
    try:
        with tarfile.open(path, "r:*") as archive:
            manifest_member = archive.getmember("manifest.json")
            manifest_file = archive.extractfile(manifest_member)
            if manifest_file is None:
                raise ValueError("Docker load manifest cannot be read")
            manifest = json.load(manifest_file)
            if not isinstance(manifest, list) or len(manifest) != 1:
                raise ValueError("Docker load requires exactly one image")
            row = manifest[0]
            config_path = str(row.get("Config") or "")
            tags = row.get("RepoTags")
            config_member = archive.getmember(config_path)
            config_file = archive.extractfile(config_member)
            if config_file is None:
                raise ValueError("Docker load config cannot be read")
            config = json.load(config_file)
    except (KeyError, OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise ValueError("Docker load archive is invalid") from exc
    if (
        not re.fullmatch(r"blobs/sha256/[0-9a-f]{64}", config_path)
        or not isinstance(tags, list)
        or not tags
        or not all(isinstance(tag, str) and tag for tag in tags)
        or not isinstance(config, dict)
    ):
        raise ValueError("Docker load archive contract is incomplete")
    labels = (
        config.get("config", {}).get("Labels", {})
        if isinstance(config.get("config"), dict)
        else {}
    )
    rootfs = config.get("rootfs")
    rootfs_layers = (
        rootfs.get("diff_ids") if isinstance(rootfs, dict) else None
    )
    commit = str(labels.get("org.leadpoet.rehearsal.commit") or "")
    role = str(labels.get("org.leadpoet.rehearsal.role") or "")
    image_id = "sha256:" + config_path.rsplit("/", 1)[-1]
    if (
        role not in ALL_ROLES
        or image_id != normalized_image_id(commit, role)
        or not isinstance(rootfs_layers, list)
        or not rootfs_layers
        or any(
            not isinstance(layer, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", layer) is None
            for layer in rootfs_layers
        )
    ):
        raise ValueError("Docker load image identity differs from build contract")
    record = {
        "id": image_id,
        "commit": commit,
        "role": role,
        "rootfs_layers": list(rootfs_layers),
    }
    cache_tags = [
        tag for tag in tags if _PCR0_CACHE_NORMALIZED_TAG.fullmatch(tag)
    ]
    if cache_tags:
        if len(tags) != 1 or len(cache_tags) != 1:
            raise ValueError("validator PCR0 cache load tag set differs")
        cache_tag = cache_tags[0]
        provenance = normalizations.get(cache_tag)
        source_tag = cache_tag.removesuffix("-normalized:latest")
        if not isinstance(provenance, dict) or provenance != {
            "commit": commit,
            "role": role,
            "provenance": PCR0_CACHE_PROVENANCE,
            "source_tag": source_tag,
            "normalized_image_id": image_id,
            "build_root": str(provenance.get("build_root") or ""),
        }:
            raise ValueError("validator PCR0 cache load provenance differs")
        record.update(
            {
                "provenance": PCR0_CACHE_PROVENANCE,
                "source_tag": source_tag,
                "build_root": str(provenance["build_root"]),
            }
        )
    for tag in tags:
        images[tag] = dict(record)
    return list(tags)


def _pcr0_cache_loaded_image_is_bound(
    *,
    image: str,
    record: dict[str, Any],
    normalizations: dict[str, Any],
) -> bool:
    if _PCR0_CACHE_NORMALIZED_TAG.fullmatch(image) is None:
        return False
    source_tag = image.removesuffix("-normalized:latest")
    commit = str(record.get("commit") or "")
    role = str(record.get("role") or "")
    image_id = _image_record_id(record)
    expected = {
        "commit": commit,
        "role": role,
        "provenance": PCR0_CACHE_PROVENANCE,
        "source_tag": source_tag,
        "normalized_image_id": image_id,
        "build_root": str(record.get("build_root") or ""),
    }
    return (
        re.fullmatch(r"[0-9a-f]{40}", commit) is not None
        and role == VALIDATOR_ROLE
        and str(record.get("provenance") or "") == PCR0_CACHE_PROVENANCE
        and str(record.get("source_tag") or "") == source_tag
        and Path(str(record.get("build_root") or "")).is_absolute()
        and image_id == normalized_image_id(commit, role)
        and normalizations.get(image) == expected
    )


def _pcr0_cache_image_is_bound(
    *,
    image: str,
    record: dict[str, Any],
    normalizations: dict[str, Any],
) -> bool:
    if not _pcr0_cache_loaded_image_is_bound(
        image=image,
        record=record,
        normalizations=normalizations,
    ):
        return False
    commit = str(record.get("commit") or "")
    build_root = Path(str(record.get("build_root") or ""))
    try:
        configured_root = Path(
            os.environ.get("PCR0_BUILD_DIR", "/tmp/pcr0_builder")
        ).resolve(strict=True)
        resolved_build_root = build_root.resolve(strict=True)
        observed_commit = _pcr0_cache_git_identity(resolved_build_root)
    except (OSError, ValueError):
        return False
    return (
        resolved_build_root == configured_root
        and observed_commit == commit
    )


def _official_normalized_image_tag(role: str) -> str:
    if role == VALIDATOR_ROLE:
        return "validator-tee-enclave:latest"
    if role in GATEWAY_ROLES:
        return f"tee-enclave:{role}"
    return ""


def _gateway_verification_image_is_bound(
    *,
    image: str,
    record: dict[str, Any],
) -> bool:
    match = re.fullmatch(
        r"leadpoet-gateway-verify:(gateway_[a-z]+)-([0-9a-f]{12})-([1-9][0-9]*)",
        image,
    )
    if match is None:
        return False
    tagged_role, tagged_commit, _index = match.groups()
    commit = str(record.get("commit") or "")
    role = str(record.get("role") or "")
    return (
        commit == _candidate_sha()
        and role == tagged_role
        and role in GATEWAY_ROLES
        and tagged_commit == commit[:12]
        and not str(record.get("provenance") or "")
        and _image_record_id(record) == normalized_image_id(commit, role)
    )


def _docker_run_contract(
    argv: list[str],
) -> tuple[str, dict[str, str], list[str], list[str]]:
    """Parse the exact Docker run options used by production launchers."""

    environment: dict[str, str] = {}
    mounts: list[str] = []
    name = ""
    image_index = -1
    options_with_value = {
        "--entrypoint",
        "--name",
        "--network",
        "--restart",
        "--log-driver",
        "--log-opt",
        "--device",
        "-e",
        "--env",
        "-v",
        "--volume",
        "--platform",
    }
    flags = {"--rm", "-d", "--privileged", "-i"}
    index = 1
    while index < len(argv):
        item = argv[index]
        if item in flags:
            index += 1
            continue
        if item.startswith("--log-driver="):
            if not item.removeprefix("--log-driver="):
                raise ValueError(
                    "Docker run --log-driver omitted its value"
                )
            index += 1
            continue
        if item in options_with_value:
            if index + 1 >= len(argv):
                raise ValueError(f"Docker run option omitted value: {item}")
            value = argv[index + 1]
            if item == "--name":
                name = value
            elif item in {"-e", "--env"}:
                key, separator, env_value = value.partition("=")
                if not key or not re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*", key
                ):
                    raise ValueError("Docker run environment entry is invalid")
                if separator:
                    environment[key] = env_value
                elif key in os.environ:
                    # Docker's name-only form inherits the value without
                    # exposing it in process arguments. An absent host value
                    # remains absent in the container.
                    environment[key] = os.environ[key]
            elif item in {"-v", "--volume"}:
                mounts.append(value)
            index += 2
            continue
        if item.startswith("-"):
            raise ValueError(f"Docker run used an unknown option: {item}")
        image_index = index
        break
    if image_index < 0:
        raise ValueError("Docker run image is missing")
    return name, environment, mounts, argv[image_index:]


def _docker_image_inspect_contract(argv: list[str]) -> tuple[str, str]:
    """Parse Docker's image-inspect target and optional format argument."""

    if argv[:2] != ["image", "inspect"]:
        raise ValueError("Docker image inspect operation is invalid")
    target = ""
    template = ""
    index = 2
    while index < len(argv):
        item = argv[index]
        if item in {"-f", "--format"}:
            if template or index + 1 >= len(argv) or not argv[index + 1]:
                raise ValueError("Docker image inspect format is invalid")
            template = argv[index + 1]
            index += 2
            continue
        if item.startswith("-"):
            raise ValueError(
                f"Docker image inspect used an unknown option: {item}"
            )
        if target:
            raise ValueError(
                "Docker image inspect requires exactly one target"
            )
        target = item
        index += 1
    if not target:
        raise ValueError("Docker image inspect target is missing")
    return target, template


def _process_is_alive(pid: Any) -> bool:
    try:
        normalized = int(pid)
        os.kill(normalized, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _stop_container_process(row: dict[str, Any]) -> None:
    pid = row.get("pid")
    if not _process_is_alive(pid):
        row["running"] = False
        return
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except (OSError, ValueError):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (OSError, ValueError):
            pass
    deadline = time.monotonic() + 5
    while _process_is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_is_alive(pid):
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except (OSError, ValueError):
            pass
    row["running"] = False


def _run_drand_builder_boundary(
    argv: list[str],
    *,
    mounts: list[str],
) -> None:
    if (
        "--rm" not in argv
        or _arg_value(argv, "--platform") != "linux/amd64"
        or _arg_value(argv, "--network") != "bridge"
    ):
        raise ValueError("drand builder Docker run contract differs")
    work_mounts = [
        value for value in mounts if value.endswith(":/work")
    ]
    cache_mounts = [
        value for value in mounts if value.endswith(":/cargo-cache")
    ]
    if len(work_mounts) != 1 or len(cache_mounts) != 1:
        raise ValueError("drand builder Docker mounts differ")
    work_root = Path(work_mounts[0].removesuffix(":/work"))
    source = Path(
        "/opt/leadpoet/drand-cabi-v2/libbittensor_drand_v2.so"
    )
    if not source.is_file():
        raise ValueError("independently rebuilt drand artifact is unavailable")
    output = work_root / "output"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / "libbittensor_drand_v2.so")
    (output / "max-glibc.txt").write_text("2.26\n", encoding="ascii")


def command_docker(argv: list[str]) -> int:
    handle, state = _locked_state()
    images = state.setdefault("images", {})
    containers = state.setdefault("containers", {})
    normalizations = state.setdefault("pcr0_cache_normalizations", {})
    operation = "state"
    status = 0
    output = ""
    try:
        if not argv:
            return _fail("docker", argv, "missing Docker operation")
        if argv[0] == "info":
            operation = "state"
            if not state.get("docker_ready", True):
                status = 1
            elif "--format" in argv:
                output = "/var/lib/docker"
            else:
                output = "Server Version: rehearsal"
        elif argv[:2] in (["images", "-q"], ["image", "ls"]):
            operation = "state"
            target = argv[-1] if argv[:2] == ["images", "-q"] and len(argv) > 2 else ""
            if target:
                output = _image_record_id(images.get(target))
            else:
                output = "\n".join(
                    sorted(
                        {
                            _image_record_id(value)
                            for value in images.values()
                            if _image_record_id(value)
                        }
                    )
                )
        elif argv[:2] == ["image", "inspect"]:
            operation = "inspect"
            try:
                target, template = _docker_image_inspect_contract(argv)
            except ValueError as exc:
                return _fail("docker", argv, str(exc))
            record = images.get(target)
            if not isinstance(record, dict):
                status = 1
            else:
                if not template:
                    output = json.dumps([record], sort_keys=True)
                elif "org.opencontainers.image.revision" in template:
                    output = str(record.get("commit") or "")
                elif template == "{{json .RootFS.Layers}}":
                    layers = record.get("rootfs_layers")
                    if not isinstance(layers, list) or not layers:
                        return _fail(
                            "docker",
                            argv,
                            "Docker image rootfs layers are unavailable",
                        )
                    output = json.dumps(layers, separators=(",", ":"))
                elif ".Id" in template:
                    output = _image_record_id(record)
                else:
                    return _fail(
                        "docker",
                        argv,
                        "unknown Docker image inspect format",
                    )
        elif argv[0] == "inspect":
            operation = "inspect"
            target = argv[-1]
            row = containers.get(target, {})
            if row.get("running") and not _process_is_alive(row.get("pid")):
                row["running"] = False
            template = _arg_value(argv, "-f")
            if ".State.Running" in template:
                output = "true" if row.get("running") else "false"
            elif ".RestartCount" in template:
                output = str(row.get("restart_count", 0))
            elif template == "{{.Image}}":
                output = str(row.get("image_id") or "")
            elif "org.opencontainers.image.revision" in template:
                output = str(row.get("image_revision") or "")
            elif ".Config.Env" in template:
                output = "\n".join(row.get("environment") or [])
            else:
                output = json.dumps([row])
        elif argv[0] == "build":
            operation = "build"
            tag = _arg_value(argv, "-t")
            if not tag:
                return _fail("docker", argv, "Docker build omitted -t")
            try:
                cache_record = _pcr0_cache_build_record(argv, tag)
                role = (
                    str(cache_record["role"])
                    if cache_record is not None
                    else _external_build_role(argv, tag)
                )
            except ValueError as exc:
                return _fail("docker", argv, str(exc))
            commit = (
                str(cache_record["commit"])
                if cache_record is not None
                else _candidate_sha()
            )
            record = {
                "id": (
                    _image_id(
                        "raw:"
                        + commit
                        + ":"
                        + role
                        + (
                            ":" + str(cache_record["provenance"])
                            if cache_record is not None
                            else ""
                        )
                    )
                    if role
                    else _image_id(
                        "build:"
                        + commit
                        + ":"
                        + json.dumps(argv, separators=(",", ":"))
                    )
                ),
                "commit": commit,
                "role": role,
            }
            if cache_record is not None:
                record.update(cache_record)
            images[tag] = record
        elif argv[0] in {"rmi", "rm", "stop"}:
            operation = "stop" if argv[0] == "stop" else "remove"
            for item in argv[1:]:
                if item.startswith("-"):
                    continue
                if argv[0] == "rmi":
                    images.pop(item, None)
                elif argv[0] == "stop" and item in containers:
                    _stop_container_process(containers[item])
                elif argv[0] == "rm":
                    if item in containers:
                        _stop_container_process(containers[item])
                    containers.pop(item, None)
        elif argv[:2] in (
            ["container", "prune"],
            ["builder", "prune"],
            ["system", "prune"],
            ["image", "prune"],
        ):
            operation = "prune"
            output = "rehearsal prune complete"
        elif argv[:2] == ["system", "df"]:
            operation = "state"
            output = "TYPE TOTAL ACTIVE SIZE RECLAIMABLE"
        elif argv[:2] == ["volume", "ls"]:
            operation = "state"
            output = ""
        elif argv[0] == "ps":
            operation = "state"
            names = []
            name_filters = tuple(
                value.removeprefix("name=")
                for value in _arg_values(argv, "--filter")
                if value.startswith("name=")
            )
            for name, row in sorted(containers.items()):
                if row.get("running") and not _process_is_alive(row.get("pid")):
                    row["running"] = False
                if name_filters and not any(
                    value in name for value in name_filters
                ):
                    continue
                if argv[1:2] == ["-aq"] or "-a" in argv:
                    names.append(name)
                elif row.get("running"):
                    names.append(name)
            output = "\n".join(names)
        elif argv[0] == "login":
            operation = "login"
            sys.stdin.read()
            output = "Login Succeeded"
        elif argv[0] == "save":
            operation = "save"
            destination = _arg_value(argv, "-o")
            source_tag = (
                argv[1]
                if len(argv) > 1 and argv[1] != "-o"
                else (argv[-1] if argv else "")
            )
            record = images.get(source_tag)
            if not destination or not isinstance(record, dict):
                return _fail(
                    "docker",
                    argv,
                    "Docker save source or destination is unavailable",
                )
            try:
                _docker_save(
                    Path(destination),
                    source_tag=source_tag,
                    record=record,
                    normalizations=normalizations,
                )
            except ValueError as exc:
                return _fail("docker", argv, str(exc))
        elif argv[0] == "load":
            operation = "load"
            source = _arg_value(argv, "-i")
            if not source:
                return _fail("docker", argv, "Docker load omitted -i")
            try:
                tags = _docker_load(Path(source), images, normalizations)
            except ValueError as exc:
                return _fail("docker", argv, str(exc))
            output = "\n".join(f"Loaded image: {tag}" for tag in tags)
        elif argv[0] == "tag":
            operation = "tag"
            if len(argv) < 3:
                return _fail("docker", argv, "Docker tag is incomplete")
            target = argv[2]
            if _PCR0_CACHE_NORMALIZED_TAG.fullmatch(target) is not None:
                source = images.get(target)
                provenance = normalizations.get(target)
                if (
                    not isinstance(source, dict)
                    or not isinstance(provenance, dict)
                    or argv[1] != str(provenance.get("normalized_image_id") or "")
                    or not _pcr0_cache_loaded_image_is_bound(
                        image=target,
                        record=source,
                        normalizations=normalizations,
                    )
                ):
                    return _fail(
                        "docker",
                        argv,
                        "validator PCR0 cache image tag provenance differs",
                    )
            else:
                source = images.get(argv[1])
                if not isinstance(source, dict) and argv[1].startswith("sha256:"):
                    source = _image_record_by_id(images, argv[1])
            if not isinstance(source, dict):
                return _fail("docker", argv, "Docker tag source is unavailable")
            images[target] = dict(source)
        elif argv[0] == "run":
            operation = "run"
            try:
                name, environment, mounts, invocation = (
                    _docker_run_contract(argv)
                )
            except ValueError as exc:
                return _fail("docker", argv, str(exc))
            image = invocation[0]
            tail = invocation[1:]
            if image == "validator-drand-builder:v2":
                try:
                    _run_drand_builder_boundary(argv, mounts=mounts)
                except ValueError as exc:
                    return _fail("docker", argv, str(exc))
            elif "-c" in tail and (
                image == "validator-tee-enclave:latest"
                or image.startswith(
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/"
                )
            ):
                if _arg_value(argv, "--entrypoint") != "python3":
                    return _fail(
                        "docker",
                        argv,
                        "validator enclave metadata command used the wrong entrypoint",
                    )
                if image == "validator-tee-enclave:latest":
                    record = images.get(image)
                    if (
                        not isinstance(record, dict)
                        or str(record.get("commit") or "")
                        != _candidate_sha()
                        or str(record.get("role") or "") != VALIDATOR_ROLE
                    ):
                        return _fail(
                            "docker",
                            argv,
                            "validator enclave metadata image is not candidate-bound",
                        )
                snippet = _arg_value(tail, "-c")
                release_input = json.loads(
                    (STATE_ROOT / "release-build-input.json").read_text(
                        encoding="utf-8"
                    )
                )
                if "compute_app_manifest_hash" in snippet:
                    output = str(
                        release_input["validator_app_manifest_hash"]
                    )
                elif "dependency_lock_hash" in snippet:
                    output = str(
                        release_input["validator_dependency_lock_hash"]
                    )
                else:
                    return _fail(
                        "docker",
                        argv,
                        "unknown validator enclave metadata command",
                    )
            elif image == "leadpoet-validator:latest":
                image_record = images.get(image)
                candidate_sha = _candidate_sha()
                image_id = _image_record_id(image_record)
                if (
                    not isinstance(image_record, dict)
                    or str(image_record.get("commit") or "") != candidate_sha
                    or not image_id
                    or not name
                    or "-d" not in argv
                    or _arg_value(argv, "--network") != "host"
                    or _arg_value(argv, "--restart") != "unless-stopped"
                    or environment.get("LEADPOET_CONTAINER_MODE") != "1"
                    or environment.get("LEADPOET_WRAPPER_ACTIVE") != "1"
                    or environment.get("VALIDATOR_RUNTIME_GENERATION", "")
                    == ""
                    or environment.get("LEADPOET_SUBNET_EPOCH_CUTOVER_JSON", "")
                    == ""
                    or any(
                        environment.get(variable) != candidate_sha
                        for variable in (
                            "LEADPOET_SENTRY_RELEASE",
                            "VALIDATOR_V2_DEPLOY_COMMIT",
                            "GITHUB_SHA",
                            "GIT_COMMIT",
                        )
                    )
                ):
                    return _fail(
                        "docker",
                        argv,
                        "validator role Docker release contract differs",
                    )
                worker_id = ""
                if name == "leadpoet-validator-main":
                    role = "validator.coordinator"
                    if (
                        _arg_value(tail, "--mode") != "coordinator"
                        or _arg_value(tail, "--container-id") != "0"
                        or environment.get("VALIDATOR_WEIGHT_PROTOCOL")
                        != "authoritative_v2"
                        or environment.get(
                            "FULFILLMENT_LEADERBOARD_EMISSIONS_ENABLED"
                        )
                        != "false"
                    ):
                        return _fail(
                            "docker",
                            argv,
                            "validator coordinator Docker run contract differs",
                        )
                else:
                    worker_match = re.fullmatch(
                        r"leadpoet-(validator|ff)-worker-([1-9][0-9]*)",
                        name,
                    )
                    if not worker_match:
                        return _fail(
                            "docker", argv, "validator role name is invalid"
                        )
                    worker_kind, worker_id = worker_match.groups()
                    expected_mode = {
                        "validator": "worker",
                        "ff": "fulfillment_worker",
                    }[worker_kind]
                    role = {
                        "validator": "validator.sourcing_worker",
                        "ff": "validator.fulfillment_worker",
                    }[worker_kind]
                    if (
                        _arg_value(tail, "--mode") != expected_mode
                        or _arg_value(tail, "--container-id") != worker_id
                        or (
                            worker_kind == "ff"
                            and environment.get("ENABLE_FULFILLMENT") != "true"
                        )
                    ):
                        return _fail(
                            "docker",
                            argv,
                            f"{role} Docker run contract differs",
                        )
                source = Path(
                    "/home/ec2-user/leadpoet/leadpoet/neurons/validator.py"
                )
                if not source.is_file():
                    return _fail(
                        "docker", argv, "validator role source is absent"
                    )
                run_ordinal = os.environ.get("REHEARSAL_RUN_ORDINAL", "1")
                transition = os.environ.get(
                    "REHEARSAL_TRANSITION", "forward"
                )
                if role == "validator.coordinator":
                    log_name = (
                        f"validator-main-{run_ordinal}-{transition}.log"
                    )
                else:
                    log_name = (
                        f"{name}-{run_ordinal}-{transition}.log"
                    )
                log_path = Path("/evidence") / log_name
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.unlink(missing_ok=True)
                child_environment = os.environ.copy()
                child_environment.update(environment)
                child_environment.update(
                    {
                        "HOME": "/home/ec2-user",
                        "PYTHONPATH": (
                            "/harness:/home/ec2-user/leadpoet/leadpoet"
                        ),
                        "REHEARSAL_SCOPE": _rehearsal_scope(),
                        "REHEARSAL_COMPONENT": "validator",
                    }
                )
                log_handle = log_path.open("ab")
                child = subprocess.Popen(
                    [REAL_PYTHON, str(source), *tail],
                    cwd="/home/ec2-user/leadpoet/leadpoet",
                    env=child_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log_handle.close()
                containers[name] = {
                    "running": True,
                    "restart_count": 0,
                    "pid": child.pid,
                    "environment": [
                        f"{key}={value}"
                        for key, value in sorted(environment.items())
                    ],
                    "image_id": image_id,
                    "image_revision": candidate_sha,
                    "mounts": mounts,
                    "role": role,
                    "worker_id": worker_id,
                    "argv": [str(source), *tail],
                    "log_path": str(log_path),
                }
                _event(
                    "validator-process",
                    [str(source), *tail],
                    status="started",
                    process=role,
                    pid=child.pid,
                    container_name=name,
                    image_id=image_id,
                    image_revision=candidate_sha,
                    release_environment={
                        variable: environment[variable]
                        for variable in (
                            "LEADPOET_SENTRY_RELEASE",
                            "VALIDATOR_V2_DEPLOY_COMMIT",
                            "GITHUB_SHA",
                            "GIT_COMMIT",
                        )
                    },
                    worker_id=worker_id,
                    implementation="production_script",
                    scope=_rehearsal_scope(),
                    **_source_identity(source),
                )
                output = name
            else:
                return _fail("docker", argv, "unknown Docker run image")
        elif argv[0] == "exec":
            operation = "run"
            interactive = "-i" in argv[1:]
            filtered = [item for item in argv[1:] if item != "-i"]
            if len(filtered) < 2:
                return _fail("docker", argv, "Docker exec is incomplete")
            target, command = filtered[0], filtered[1:]
            row = containers.get(target)
            if not isinstance(row, dict) or not row.get("running"):
                return _fail(
                    "docker", argv, "Docker exec target is not running"
                )
            # A real Docker daemon permits independent read-only exec probes to
            # overlap. Release the adapter state lock before running the child
            # so the exact launcher can exercise that production concurrency.
            # Docker exec does not mutate the emulated container inventory.
            row = dict(row)
            _save_state(handle, state)
            handle = None
            if (
                len(command) >= 3
                and command[:2] == ["sh", "-c"]
                and "proc/1/cmdline" in command[2]
            ):
                if "validator.py" not in " ".join(row.get("argv") or []):
                    status = 1
            elif (
                command[:2] == ["sh", "-c"]
                and len(command) == 3
                and command[2]
                == 'test -n "${LEADPOET_SUBNET_EPOCH_CUTOVER_JSON:-}"'
            ):
                environment = dict(
                    line.split("=", 1)
                    for line in row.get("environment") or []
                    if "=" in line
                )
                if not environment.get("LEADPOET_SUBNET_EPOCH_CUTOVER_JSON"):
                    status = 1
            elif (
                command
                == [
                    "sh",
                    "-c",
                    "test -s /app/validator_weights/current_block.json",
                ]
            ):
                weight_mounts = [
                    value
                    for value in row.get("mounts") or []
                    if value.split(":", 1)[-1]
                    == "/app/validator_weights"
                ]
                if len(weight_mounts) != 1 or not (
                    Path(weight_mounts[0].split(":", 1)[0])
                    / "current_block.json"
                ).is_file():
                    status = 1
            elif command == ["curl", "-s", "--max-time", "10", "https://api.ipify.org"]:
                output = "203.0.113.10"
            elif command in (
                ["python3", "-"],
                ["sh", "-c", "cd /app && python3 -"],
            ):
                if not interactive:
                    return _fail(
                        "docker",
                        argv,
                        "Docker exec stdin script requires -i",
                    )
                child_environment = os.environ.copy()
                child_environment.update(
                    line.split("=", 1)
                    for line in row.get("environment") or []
                    if "=" in line
                )
                child_environment["PYTHONPATH"] = (
                    "/harness:/home/ec2-user/leadpoet/leadpoet"
                )
                result = subprocess.run(
                    [REAL_PYTHON, "-"],
                    input=sys.stdin.buffer.read(),
                    cwd="/home/ec2-user/leadpoet/leadpoet",
                    env=child_environment,
                    check=False,
                )
                status = result.returncode
            else:
                return _fail("docker", argv, "unknown Docker exec command")
        elif argv[0] == "logs":
            operation = "state"
            target = argv[-1]
            row = containers.get(target, {})
            log_path = Path(str(row.get("log_path") or ""))
            output = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.is_file()
                else ""
            )
        else:
            return _fail("docker", argv, "unknown Docker operation")
        _record_external_boundary(
            kind="docker",
            argv=argv,
            boundary="docker_daemon",
            operation=operation,
            status="ok" if status == 0 else "failed",
        )
        if output:
            print(output)
        return status
    finally:
        if handle is not None:
            _save_state(handle, state)


def command_nitro(argv: list[str]) -> int:
    handle, state = _locked_state()
    enclaves = state.setdefault("enclaves", [])
    images = state.setdefault("images", {})
    normalizations = state.setdefault("pcr0_cache_normalizations", {})
    try:
        if argv[:2] == ["terminate-enclave", "--all"]:
            enclaves.clear()
            _record_external_boundary(
                kind="nitro",
                argv=argv,
                boundary="nitro_enclaves",
                operation="terminate_enclave",
            )
            return 0
        if argv and argv[0] == "build-enclave":
            output = _arg_value(argv, "--output-file")
            image = _arg_value(argv, "--docker-uri")
            if not output or not image:
                return _fail("nitro", argv, "build-enclave arguments are incomplete")
            record = images.get(image)
            candidate_bound = False
            cache_bound = False
            if isinstance(record, dict):
                commit = str(record.get("commit") or "")
                role = str(record.get("role") or "")
                candidate_bound = (
                    commit == _candidate_sha()
                    and role in ALL_ROLES
                    and (
                        image == _official_normalized_image_tag(role)
                        or _gateway_verification_image_is_bound(
                            image=image,
                            record=record,
                        )
                    )
                    and not str(record.get("provenance") or "")
                    and _image_record_id(record)
                    == normalized_image_id(commit, role)
                )
                cache_bound = _pcr0_cache_image_is_bound(
                    image=image,
                    record=record,
                    normalizations=normalizations,
                )
            if not candidate_bound and not cache_bound:
                return _fail(
                    "nitro",
                    argv,
                    "build-enclave image identity is not commit-bound: "
                    + json.dumps(
                        {
                            "image": image,
                            "record": record,
                            "expected_commit": _candidate_sha(),
                            "expected_role": VALIDATOR_ROLE,
                        },
                        sort_keys=True,
                    ),
                )
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if candidate_bound and str(record["role"]) == VALIDATOR_ROLE:
                validator_app = STATE_ROOT / "validator-app"
                if not validator_app.is_dir():
                    return _fail(
                        "nitro",
                        argv,
                        "candidate validator application filesystem is absent",
                    )
                runtime_app = Path("/app")
                shutil.rmtree(runtime_app, ignore_errors=True)
                shutil.copytree(validator_app, runtime_app)
                for path in runtime_app.rglob("*"):
                    path.chmod(0o755 if path.is_dir() else 0o644)
            destination.write_bytes(
                eif_bytes(str(record["commit"]), str(record["role"]))
            )
            _record_external_boundary(
                kind="nitro",
                argv=argv,
                boundary="nitro_enclaves",
                operation="build_enclave",
            )
            print(
                json.dumps(
                    {
                        "Measurements": {
                            "PCR0": artifact_pcr0(str(record["commit"])),
                            "PCR1": hashlib.sha384(b"pcr1").hexdigest(),
                            "PCR2": hashlib.sha384(b"pcr2").hexdigest(),
                        }
                    },
                    sort_keys=True,
                )
            )
            return 0
        if argv and argv[0] == "run-enclave":
            cid = _arg_value(argv, "--enclave-cid", "16")
            eif = _arg_value(argv, "--eif-path")
            if not eif or not Path(eif).is_file():
                return _fail("nitro", argv, "run-enclave EIF is unavailable")
            row = {
                "EnclaveCID": int(cid),
                "EnclaveID": f"rehearsal-{cid}",
                "State": "RUNNING",
                "Measurements": {"PCR0": PCR0},
            }
            enclaves.append(row)
            _record_external_boundary(
                kind="nitro",
                argv=argv,
                boundary="nitro_enclaves",
                operation="run_enclave",
            )
            print(json.dumps(row, sort_keys=True))
            return 0
        if argv and argv[0] == "describe-eif":
            eif = _arg_value(argv, "--eif-path")
            if not eif or not Path(eif).is_file():
                return _fail("nitro", argv, "describe-eif EIF is unavailable")
            _record_external_boundary(
                kind="nitro",
                argv=argv,
                boundary="nitro_enclaves",
                operation="describe_enclaves",
            )
            print(
                json.dumps(
                    {
                        "Measurements": {
                            "PCR0": PCR0,
                            "PCR1": hashlib.sha384(b"pcr1").hexdigest(),
                            "PCR2": hashlib.sha384(b"pcr2").hexdigest(),
                        }
                    },
                    sort_keys=True,
                )
            )
            return 0
        if argv and argv[0] == "describe-enclaves":
            _record_external_boundary(
                kind="nitro",
                argv=argv,
                boundary="nitro_enclaves",
                operation="describe_enclaves",
            )
            print(json.dumps(enclaves, sort_keys=True))
            return 0
        return _fail("nitro", argv, "unknown Nitro operation")
    finally:
        _save_state(handle, state)


def command_systemctl(argv: list[str]) -> int:
    accepted = {"start", "stop", "restart", "reset-failed", "is-active"}
    if not argv or argv[0] not in accepted:
        return _fail("systemctl", argv, "unknown systemctl operation")
    _record_external_boundary(
        kind="systemctl",
        argv=argv,
        boundary="host_kernel",
        operation="systemd",
    )
    return 0


def command_curl(argv: list[str]) -> int:
    output_path = _arg_value(argv, "--output") or _arg_value(argv, "-o")
    if argv in (
        ["-s", "ifconfig.me"],
        ["-s", "https://ipinfo.io"],
        ["-s", "myip.dnsomatic.com"],
    ):
        print("203.0.113.10")
        _record_external_boundary(
            kind="curl",
            argv=argv,
            boundary="http_service",
            operation="external_ip",
        )
        return 0
    urls = [arg for arg in argv if re.match(r"^https?://", arg)]
    if len(urls) != 1:
        return _fail("curl", argv, "curl must contain exactly one URL")
    url = urls[0]
    if url.startswith(("http://localhost", "http://127.0.0.1")):
        _event(
            "gateway-http",
            argv,
            status="started",
            implementation="production_http_route",
            url=url,
        )
        os.execv(REAL_CURL, [REAL_CURL, *argv])
        return 127
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if RUNSC_LOCK_PATH.is_file():
            runsc_lock = json.loads(
                RUNSC_LOCK_PATH.read_text(encoding="utf-8")
            )
        else:
            runsc_lock = {}
        if url == runsc_lock.get("source_url"):
            artifact = (
                EXTERNAL_ARTIFACT_ROOT
                / str(runsc_lock.get("artifact_filename") or "")
            )
            if not artifact.is_file():
                return _fail(
                    "curl",
                    argv,
                    "verified local runsc mirror is unavailable",
                )
            shutil.copy2(artifact, output_path)
        else:
            Path(output_path).write_bytes(b"rehearsal-downloaded-artifact\n")
        _record_external_boundary(
            kind="curl",
            argv=argv,
            boundary="http_service",
            operation="download",
        )
        return 0
    response_details: dict[str, Any] = {"url": url}
    if url.endswith("/build-info"):
        served_commit = _candidate_sha()
        response_details["served_commit"] = served_commit
        print(json.dumps({"git_commit": served_commit}, sort_keys=True))
    elif url.endswith("/health/v2-authority"):
        handle, state = _locked_state()
        health_probe_attempt = int(
            state.get("gateway_v2_authority_health_probe_attempts", 0)
        ) + 1
        state["gateway_v2_authority_health_probe_attempts"] = (
            health_probe_attempt
        )
        _save_state(handle, state)
        candidate_sha = _candidate_sha()
        from_sha = os.environ.get("REHEARSAL_FROM_SHA", "").strip()
        served_commit = (
            from_sha
            if health_probe_attempt == 1
            and re.fullmatch(r"[0-9a-f]{40}", from_sha)
            and from_sha != candidate_sha
            else candidate_sha
        )
        response_details.update(
            {
                "gateway_probe_attempt": health_probe_attempt,
                "served_commit": served_commit,
            }
        )
        print(
            json.dumps(
                {
                    "schema_version": "leadpoet.gateway_v2_authority_health.v2",
                    "status": "ready",
                    "commit_sha": served_commit,
                },
                sort_keys=True,
            )
        )
    elif re.search(r"/weights/v2/release-evidence/[0-9a-f]{40}$", url):
        served_commit = _candidate_sha()
        response_details["served_commit"] = served_commit
        print(
            json.dumps(
                {
                    "schema_version": "leadpoet.auditor_release_evidence.v2",
                    "commit_sha": served_commit,
                    "release_channel_version_id": "rehearsal-version",
                    "release_channel_get_url": "https://release.invalid/get",
                    "release_channel_head_url": "https://release.invalid/head",
                },
                sort_keys=True,
            )
        )
    elif url.endswith(("/health", "/research-lab/status", "/attest")):
        print(json.dumps({"status": "ok"}, sort_keys=True))
    else:
        return _fail("curl", argv, "unknown HTTP endpoint")
    _record_external_boundary(
        kind="curl",
        argv=argv,
        boundary="http_service",
        operation="gateway_request",
        **response_details,
    )
    return 0


def command_sudo(argv: list[str]) -> int:
    while argv and argv[0].startswith("-"):
        argv = argv[1:]
    if not argv:
        return _fail("sudo", argv, "sudo command is missing")
    _event("sudo", argv, status="delegated")
    os.execvpe(argv[0], argv, os.environ.copy())
    return 127


def command_df(argv: list[str]) -> int:
    _record_external_boundary(
        kind="host-command",
        argv=argv,
        boundary="host_kernel",
        operation="filesystem_capacity",
    )
    if any("output=avail" in arg for arg in argv):
        print("Avail")
        print("107374182400" if "-B1" in argv else "104857600")
        return 0
    print("Filesystem Size Used Avail Use% Mounted on")
    print("rehearsal 120G 1G 119G 1% /")
    return 0


def command_getconf(argv: list[str]) -> int:
    _record_external_boundary(
        kind="host-command",
        argv=argv,
        boundary="host_kernel",
        operation="cpu_capacity",
    )
    if argv == ["_NPROCESSORS_CONF"]:
        print("16")
        return 0
    return _fail("getconf", argv, "unknown getconf query")


def command_awk(argv: list[str]) -> int:
    if argv and argv[-1] == "/proc/meminfo" and "MemTotal" in " ".join(argv):
        _record_external_boundary(
            kind="host-command",
            argv=argv,
            boundary="host_kernel",
            operation="memory_capacity",
        )
        print("131072")
        return 0
    os.execv("/usr/bin/awk", ["awk", *argv])
    return 127


def command_sleep(argv: list[str]) -> int:
    _record_external_boundary(
        kind="host-command",
        argv=argv,
        boundary="host_kernel",
        operation="timing",
        status="shortened",
    )
    time.sleep(0.01)
    return 0


def command_ss(argv: list[str]) -> int:
    _record_external_boundary(
        kind="host-command",
        argv=argv,
        boundary="host_kernel",
        operation="socket_state",
    )
    return 0


def command_ctr(argv: list[str]) -> int:
    allowed_tokens = {"containers", "tasks", "namespaces", "list", "-q", "-n", "moby"}
    if any(item not in allowed_tokens for item in argv):
        return _fail("ctr", argv, "unknown containerd operation")
    _record_external_boundary(
        kind="host-command",
        argv=argv,
        boundary="host_kernel",
        operation="containerd_state",
    )
    return 0


def command_nsenter(argv: list[str]) -> int:
    if "--" not in argv:
        return _fail("nsenter", argv, "nsenter omitted --")
    command = argv[argv.index("--") + 1 :]
    if not command:
        return _fail("nsenter", argv, "nsenter command is empty")
    _record_external_boundary(
        kind="host-command",
        argv=argv,
        boundary="host_kernel",
        operation="mount_namespace",
        status="delegated",
    )
    os.execvpe(command[0], command, os.environ.copy())
    return 127


def command_pgrep(argv: list[str]) -> int:
    _record_external_boundary(
        kind="host-command",
        argv=argv,
        boundary="host_kernel",
        operation="process_lookup",
    )
    pattern = argv[-1] if argv else ""
    handle, state = _locked_state()
    try:
        if "containerd-shim-runc-v2" in pattern:
            if "-c" in "".join(argv) or "-fc" in argv:
                print("0")
                return 0
            return 1
        process_key = ""
        if "gateway[.]main" in pattern or "gateway.main" in pattern:
            process_key = "gateway.main"
        elif "chain_relay_v2" in pattern:
            process_key = "validator.chain_relay"
        pid = state.get("processes", {}).get(process_key)
        if pid and Path(f"/proc/{pid}").exists():
            print(pid)
            return 0
        return 1
    finally:
        _save_state(handle, state)


def command_pkill(argv: list[str]) -> int:
    _record_external_boundary(
        kind="host-command",
        argv=argv,
        boundary="host_kernel",
        operation="process_termination",
    )
    pattern = argv[-1] if argv else ""
    handle, state = _locked_state()
    try:
        processes = state.setdefault("processes", {})
        for key, pid in list(processes.items()):
            if key in pattern or ("gateway" in key and "gateway" in pattern):
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
                processes.pop(key, None)
        return 0
    finally:
        _save_state(handle, state)


def _long_lived_process(key: str, argv: list[str]) -> int:
    environment_contract: dict[str, str] = {}
    if _record_internal_substitution(
        kind="process",
        argv=argv,
        process=key,
    ) != 0:
        return 97
    handle, state = _locked_state()
    state.setdefault("processes", {})[key] = os.getpid()
    _save_state(handle, state)
    _event(
        "process",
        argv,
        status="started",
        process=key,
        pid=os.getpid(),
        implementation="internal_substitution",
        scope=_rehearsal_scope(),
    )

    def stop(_signum: int, _frame: Any) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        time.sleep(60)


def _exec_long_lived_production_module(
    key: str,
    module: str,
    argv: list[str],
) -> int:
    environment_contract: dict[str, str] = {}
    handle, state = _locked_state()
    state.setdefault("processes", {})[key] = os.getpid()
    _save_state(handle, state)
    _event(
        "process",
        argv,
        status="started",
        process=key,
        pid=os.getpid(),
        implementation="production_module",
        scope=_rehearsal_scope(),
        **_source_identity(_module_source(module)),
    )
    current_python_path = os.environ.get("PYTHONPATH", "")
    python_paths = [item for item in current_python_path.split(":") if item]
    if "/harness" not in python_paths:
        python_paths.insert(0, "/harness")
    os.environ["PYTHONPATH"] = ":".join(python_paths)
    _route_host_storage_preflight_to_local_postgrest(module)
    os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
    return 127


def _release_manifest(role: str) -> dict[str, Any]:
    return {
        "schema_version": f"leadpoet.{role}_release_manifest.v2",
        "commit_sha": _candidate_sha(),
        "pcr0": PCR0,
        "release_hash": "sha256:" + HASH64,
        "release_manifest_hash": "sha256:" + HASH64,
        "verified_build_count": 6,
    }


def _module_release_channel(argv: list[str]) -> int:
    expected = _arg_value(argv, "--expected-commit")
    if expected != _candidate_sha():
        return _fail("python-module", argv, "release expected commit differs")
    gateway_output = _arg_value(argv, "--gateway-output")
    validator_output = _arg_value(argv, "--validator-output")
    lineage_output = _arg_value(argv, "--lineage-output")
    if gateway_output:
        _write_json(gateway_output, _release_manifest("gateway"))
    if validator_output:
        _write_json(validator_output, _release_manifest("validator"))
    if lineage_output:
        _write_json(
            lineage_output,
            {
                "schema_version": "leadpoet.gateway_release_lineage.v2",
                "commit_sha": expected,
                "lineage_hash": "sha256:" + HASH64,
                "releases": [{"commit_sha": expected}],
            },
        )
    print(json.dumps({"status": "local_verified", "commit_sha": expected}))
    return 0


def _module_restart_gate(argv: list[str]) -> int:
    capture = _arg_value(argv, "--capture-output")
    report = {
        "schema_version": "leadpoet.restart_epoch_start.v1",
        "maximum_restart_epoch_block": 300,
        "restart_allowed": True,
        "snapshot": {
            "netuid": 71,
            "epoch_id": 99999,
            "epoch_block": 42,
            "tempo": 360,
            "block_hash": "0x" + "1" * 64,
        },
    }
    if capture:
        _write_json(capture, report)
    captured = _arg_value(argv, "--captured-report")
    if captured and not Path(captured).is_file():
        return _fail("python-module", argv, "captured restart report is missing")
    print(json.dumps(report, sort_keys=True))
    return 0


def _module_envelopes(argv: list[str]) -> int:
    output_dir = Path(_arg_value(argv, "--output-dir"))
    deploy_commit = _arg_value(argv, "--deploy-commit")
    if deploy_commit != _candidate_sha():
        return _fail("python-module", argv, "envelope commit differs")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "artifact_master_key",
        "openrouter",
        "exa",
        "scrapingdog",
        "deepline",
        "supabase_service_role",
        "truelist",
    )
    for name in names:
        _write_json(
            output_dir / f"{name}.json",
            {
                "schema_version": "leadpoet.kms_credential_envelope.v2",
                "deploy_commit": deploy_commit,
                "ciphertext_b64": "cmVoZWFyc2Fs",
                "credential_reference_hash": "sha256:" + HASH64,
            },
        )
    _write_json(
        output_dir / "gateway-v2-env-transition.json",
        {"schema_version": "leadpoet.gateway_env_transition.v2", "status": "ready"},
    )
    print(json.dumps({"status": "installed", "deploy_commit": deploy_commit}))
    return 0


def _module_stage_artifacts(argv: list[str]) -> int:
    output = _arg_value(argv, "--output-dir")
    if not output:
        return _fail("python-module", argv, "artifact output directory is missing")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = _arg_value(argv, "--lock")
    if lock_path and Path(lock_path).is_file():
        try:
            lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            lock = {}
        for row in (lock.get("artifacts") or {}).values():
            filename = str(row.get("filename") or "")
            if filename:
                (destination / filename).write_bytes(b"rehearsal-runtime-artifact\n")
    print(json.dumps({"status": "staged", "output_dir": str(destination)}))
    return 0


def _scrub_parent_env(argv: list[str]) -> int:
    if len(argv) < 2:
        return _fail("python-inline", argv, "scrub-parent-env arguments are missing")
    env_path = Path(argv[0])
    report_path = Path(argv[1])
    secret_names = {
        "OPENROUTER_API_KEY",
        "EXA_API_KEY",
        "SCRAPINGDOG_API_KEY",
        "DEEPLINE_API_KEY",
        "TRUELIST_API_KEY",
    }
    kept = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        candidate = line.removeprefix("export ").strip()
        key = candidate.split("=", 1)[0] if "=" in candidate else ""
        if key not in secret_names:
            kept.append(line)
    env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    _write_json(
        report_path,
        {"schema_version": "leadpoet.gateway_env_transition.v2", "status": "scrubbed"},
    )
    print("Scrubbed commit-bound provider plaintext from prepared parent environment")
    return 0


def _python_inline(argv: list[str]) -> int:
    source = sys.stdin.read()
    _event(
        "python-inline",
        argv,
        status="started",
        implementation="production_inline",
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    result = subprocess.run([REAL_PYTHON, *argv], input=source, text=True, check=False)
    return result.returncode


def _python_script(argv: list[str]) -> int | None:
    path = Path(argv[0])
    name = path.name
    if path.resolve() == Path("/tmp/get-pip.py"):
        expected = b"rehearsal-downloaded-artifact\n"
        if not path.is_file() or path.read_bytes() != expected:
            return _fail(
                "python-script",
                argv,
                "downloaded get-pip.py does not match the strict rehearsal fixture",
            )
        _record_external_boundary(
            kind="python-script",
            argv=argv,
            boundary="python_package_index",
            operation="bootstrap",
        )
        return 0
    if name in {"gateway_git_deploy.py", "write_gateway_build_info.py"}:
        return None
    return None


def command_python(argv: list[str]) -> int:
    current_python_path = os.environ.get("PYTHONPATH", "")
    python_paths = [
        item for item in current_python_path.split(":") if item
    ]
    if str(HARNESS_ROOT) not in python_paths:
        python_paths.insert(0, str(HARNESS_ROOT))
    os.environ["PYTHONPATH"] = ":".join(python_paths)
    if not argv:
        return _python_inline(argv)
    if argv[0] == "-u":
        if len(argv) == 1:
            return _fail("python", argv, "-u omitted a Python operation")
        return command_python(argv[1:])
    if argv[0] == "-":
        return _python_inline(argv)
    if argv[0] == "-m":
        if len(argv) < 2:
            return _fail("python", argv, "-m omitted a module")
        module = argv[1]
        module_argv = argv[2:]
        if module == "pip":
            if module_argv and module_argv[0] == "download":
                _record_external_boundary(
                    kind="python-dependencies",
                    argv=module_argv,
                    boundary="python_package_index",
                    operation="download",
                )
                destination_value = _arg_value(module_argv, "--dest")
                if not destination_value:
                    return _fail("pip", module_argv, "pip download omitted --dest")
                destination = Path(destination_value)
                destination.mkdir(parents=True, exist_ok=True)
                mirror = Path("/opt/leadpoet/scoring-wheelhouse")
                wheels = sorted(mirror.glob("*.whl"))
                if not wheels:
                    return _fail(
                        "pip",
                        module_argv,
                        "local scoring package mirror is empty",
                    )
                if any(destination.iterdir()):
                    return _fail(
                        "pip",
                        module_argv,
                        "pip download destination is not empty",
                    )
                for wheel in wheels:
                    shutil.copy2(wheel, destination / wheel.name)
                _event("pip", module_argv, status="ok", operation="download")
                return 0
            if module_argv and module_argv[0] == "install":
                _record_external_boundary(
                    kind="python-dependencies",
                    argv=module_argv,
                    boundary="python_package_index",
                    operation="install",
                )
                requirement_paths: list[Path] = []
                for option in ("--requirement", "-r"):
                    start = 0
                    while True:
                        try:
                            index = module_argv.index(option, start)
                        except ValueError:
                            break
                        if index + 1 >= len(module_argv):
                            return _fail(
                                "pip",
                                module_argv,
                                f"{option} omitted its requirement path",
                            )
                        requirement_paths.append(Path(module_argv[index + 1]))
                        start = index + 2
                if not requirement_paths:
                    return _fail(
                        "pip",
                        module_argv,
                        "offline install contract omitted a requirement file",
                    )
                missing = [
                    str(path)
                    for path in requirement_paths
                    if not path.is_file() or path.stat().st_size == 0
                ]
                if missing:
                    return _fail(
                        "pip",
                        module_argv,
                        f"requirement file is unavailable: {missing}",
                    )
                result = subprocess.run(
                    [REAL_PYTHON, "-m", "pip", "check"],
                    check=False,
                )
                _event(
                    "pip",
                    module_argv,
                    status="ok" if result.returncode == 0 else "failed",
                    operation="offline-install-contract",
                    requirement_paths=[str(path) for path in requirement_paths],
                )
                return result.returncode
            if module_argv and module_argv[0] == "uninstall":
                _record_external_boundary(
                    kind="python-dependencies",
                    argv=module_argv,
                    boundary="python_package_index",
                    operation="uninstall",
                )
                _event(
                    "pip",
                    module_argv,
                    status="ok",
                    operation="offline-uninstall-contract",
                )
                return 0
            _event("pip", module_argv, status="real")
            os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
        if module == "Leadpoet.utils.restart_epoch_gate":
            _record_production_module(module, argv)
            current_python_path = os.environ.get("PYTHONPATH", "")
            python_paths = [
                item for item in current_python_path.split(":") if item
            ]
            if "/harness" not in python_paths:
                python_paths.insert(0, "/harness")
            os.environ["PYTHONPATH"] = ":".join(python_paths)
            if os.environ.get("REHEARSAL_TRANSITION", "forward") == "forward":
                os.environ[
                    "LEADPOET_REHEARSAL_RESTART_EPOCH_TRANSIENT_FAILURES"
                ] = "1"
            os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
        elif module == "gateway.tee.release_channel_v2":
            _record_production_module(module, argv)
            current_python_path = os.environ.get("PYTHONPATH", "")
            python_paths = [
                item for item in current_python_path.split(":") if item
            ]
            if "/harness" not in python_paths:
                python_paths.insert(0, "/harness")
            os.environ["PYTHONPATH"] = ":".join(python_paths)
            os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
        elif module == "gateway.tee.prepare_gateway_envelopes_v2":
            _record_production_module(module, argv)
            current_python_path = os.environ.get("PYTHONPATH", "")
            python_paths = [
                item for item in current_python_path.split(":") if item
            ]
            if "/harness" not in python_paths:
                python_paths.insert(0, "/harness")
            os.environ["PYTHONPATH"] = ":".join(python_paths)
            os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
        elif module in {
            "gateway.research_lab.stateful_epoch_cutover_cli_v1",
            "gateway.tee.bootstrap_active_ancestry_checkpoints_v2",
            "gateway.tee.prepare_active_release_lineage_v2",
            "gateway.tee.restart_preflight_v2",
            "validator_tee.host.docker_operation_guard_v2",
            "gateway.research_lab.provider_profiles_v2",
            "gateway.utils.tee_v2_bootstrap",
            "gateway.utils.tee_kms_provision_v2",
            "gateway.tee.verify_v2_runtime_ready",
            "validator_tee.host.refresh_hotkey_config_v2",
            "validator_tee.host.restart_preflight_v2",
            "validator_tee.host.verify_chain_signing_profile_v2",
            "validator_tee.host.verify_release_gate_v2",
            "validator_tee.host.release_archive_v2",
            "validator_tee.host.runtime_v2_bootstrap",
            "validator_tee.host.hotkey_bootstrap_v2",
            "gateway.tee.release_archive_v2",
        }:
            _record_production_module(module, argv)
            _route_host_storage_preflight_to_local_postgrest(module)
            current_python_path = os.environ.get("PYTHONPATH", "")
            python_paths = [
                item for item in current_python_path.split(":") if item
            ]
            if "/harness" not in python_paths:
                python_paths.insert(0, "/harness")
            os.environ["PYTHONPATH"] = ":".join(python_paths)
            os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
        elif module == "validator_tee.scripts.stage_runtime_artifacts_v2":
            _record_production_module(module, argv)
            current_python_path = os.environ.get("PYTHONPATH", "")
            python_paths = [
                item for item in current_python_path.split(":") if item
            ]
            if "/harness" not in python_paths:
                python_paths.insert(0, "/harness")
            os.environ["PYTHONPATH"] = ":".join(python_paths)
            os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
        elif module == "gateway.tee.verify_weight_submission_ready_v2":
            _record_production_module(module, argv)
            _route_host_storage_preflight_to_local_postgrest(module)
            if (
                "--repair" in argv
                and os.environ.get(
                    "REHEARSAL_WEIGHT_READINESS_FAIL_ONCE", ""
                )
                == "1"
            ):
                handle, state = _locked_state()
                attempts = int(
                    state.get("weight_readiness_repair_attempts", 0)
                ) + 1
                state["weight_readiness_repair_attempts"] = attempts
                _save_state(handle, state)
                if attempts == 1:
                    _event(
                        "fault-injection",
                        argv,
                        status="injected-transient-failure",
                        module=module,
                        implementation="real-module-process-boundary",
                        scope=_rehearsal_scope(),
                    )
                    print(
                        "REHEARSAL_INJECTED_WEIGHT_READINESS_FAILURE",
                        file=sys.stderr,
                    )
                    return 75
            os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
        elif module == "gateway.main":
            return _exec_long_lived_production_module(
                "gateway.main", module, argv
            )
        elif module == "gateway.utils.tee_egress_forwarder":
            return _exec_long_lived_production_module(
                "gateway.tee_egress", module, argv
            )
        elif module == "gateway.utils.tee_inter_enclave_relay":
            return _exec_long_lived_production_module(
                "gateway.tee_relay", module, argv
            )
        elif module == "validator_tee.host.chain_relay_v2":
            return _exec_long_lived_production_module(
                "validator.chain_relay", module, argv
            )
        else:
            _record_production_module(module, argv)
            os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
        _event(
            "python-module",
            argv,
            status="ok" if result == 0 else "failed",
            module=module,
            implementation="internal_substitution",
            scope=_rehearsal_scope(),
        )
        return result
    if argv[0].endswith(".py"):
        intercepted = _python_script(argv)
        if intercepted is not None:
            return intercepted
        _record_production_script(Path(argv[0]), argv)
    _event("python", argv, status="real")
    os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
    return 127


def command_bash(argv: list[str]) -> int:
    if not argv:
        os.execv(REAL_BASH, [REAL_BASH])
    script = Path(argv[0]).name
    _event("bash", argv, status="real", script=script)
    os.execv(REAL_BASH, [REAL_BASH, *argv])
    return 127


COMMANDS = {
    "aws": command_aws,
    "docker": command_docker,
    "nitro-cli": command_nitro,
    "systemctl": command_systemctl,
    "curl": command_curl,
    "git": command_git,
    "sudo": command_sudo,
    "df": command_df,
    "getconf": command_getconf,
    "awk": command_awk,
    "sleep": command_sleep,
    "ss": command_ss,
    "ctr": command_ctr,
    "nsenter": command_nsenter,
    "pgrep": command_pgrep,
    "pkill": command_pkill,
    "python3": command_python,
    "python3.11": command_python,
    "bash": command_bash,
}


def main() -> int:
    if len(sys.argv) < 2:
        print("adapter command is missing", file=sys.stderr)
        return 2
    command = sys.argv[1]
    argv = sys.argv[2:]
    handler = COMMANDS.get(command)
    if handler is None:
        return _fail("adapter", sys.argv[1:], "unknown adapter command")
    return int(handler(argv))


if __name__ == "__main__":
    raise SystemExit(main())
