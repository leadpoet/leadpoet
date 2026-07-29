#!/usr/bin/env python3.11
"""Validate that the exact restart launcher consumed every required contract."""

from __future__ import annotations

import ast
import json
import hashlib
from pathlib import Path
import subprocess
import sys


TARGETED_REGRESSION_SCOPE = "weight_readiness_regression"
EXPECTED_GATEWAY_PRIVATE_MODEL_ENV = {
    "RESEARCH_LAB_PRIVATE_REPO_BRANCH": "leadpoet-lab",
    "RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI": (
        "s3://leadpoet-private-model-artifacts-493765492819/"
        "research-lab/sourcing-model/branches/leadpoet-lab/current.json"
    ),
    "RESEARCH_LAB_PRIVATE_MODEL_KMS_KEY_ID": (
        "alias/leadpoet-research-lab-artifact-signing"
    ),
}
KNOWN_INTERNAL_SUBSTITUTION_MODULES = {
    "Leadpoet.utils.restart_epoch_gate",
    "gateway.tee.prepare_gateway_envelopes_v2",
    "gateway.tee.release_archive_v2",
    "gateway.tee.release_channel_v2",
    "gateway.tee.restart_preflight_v2",
    "gateway.tee.verify_v2_runtime_ready",
    "gateway.research_lab.provider_profiles_v2",
    "gateway.utils.tee_kms_provision_v2",
    "gateway.utils.tee_v2_bootstrap",
    "validator_tee.host.docker_operation_guard_v2",
    "validator_tee.host.hotkey_bootstrap_v2",
    "validator_tee.host.refresh_hotkey_config_v2",
    "validator_tee.host.release_archive_v2",
    "validator_tee.host.restart_preflight_v2",
    "validator_tee.host.runtime_v2_bootstrap",
    "validator_tee.host.verify_chain_signing_profile_v2",
    "validator_tee.host.verify_release_gate_v2",
    "validator_tee.scripts.stage_runtime_artifacts_v2",
}
KNOWN_INTERNAL_SUBSTITUTION_SCRIPTS = {
    "docker_image_normalizer_v2.py",
    "host_memory_guard_v2.py",
    "release_manifest_v2.py",
    "sandbox_runtime_artifact.py",
    "scoring_wheelhouse.py",
    "stage_runtime_artifacts_v2.py",
    "verify_release_artifacts_v2.py",
    "verify_topology.py",
}
KNOWN_INTERNAL_SUBSTITUTION_PROCESSES = {
    "gateway.main",
    "gateway.tee_egress",
    "gateway.tee_relay",
    "validator.chain_relay",
}
KNOWN_INTERNAL_SUBSTITUTION_BOUNDARIES = {
    "bash.build_drand_cabi_v2",
    "http.local_gateway",
    "host.containerd_state",
    "host.cpu_capacity",
    "host.filesystem_capacity",
    "host.memory_capacity",
    "host.mount_namespace",
    "host.process_lookup",
    "host.process_termination",
    "host.socket_state",
    "host.systemd",
    "host.timing",
    "python.scrub_parent_environment",
    "python.validator_coordinator",
    "python_dependencies.download",
    "python_dependencies.bootstrap",
    "python_dependencies.install",
    "python_dependencies.uninstall",
}


def events(
    state_root: Path = Path("/rehearsal-state"),
) -> list[dict]:
    rows: list[dict] = []
    for name in ("events.jsonl", "local-postgrest-events.jsonl"):
        path = state_root / name
        if not path.is_file():
            continue
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return sorted(rows, key=lambda row: int(row.get("at_ns") or 0))


def require_order(values: list[str], required: list[str]) -> None:
    cursor = -1
    for expected in required:
        try:
            cursor = values.index(expected, cursor + 1)
        except ValueError as exc:
            raise SystemExit(
                f"required rehearsal event is missing or out of order: {expected}"
            ) from exc


def _substitution_identity(row: dict) -> str:
    return str(
        row.get("substitution")
        or row.get("module")
        or row.get("script")
        or row.get("process")
        or ""
    )


def _git_blob(
    roots: tuple[Path, ...],
    commit: str,
    git_path: str,
) -> bytes | None:
    for root in roots:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"{commit}:{git_path}",
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout
    return None


def _verify_installed_checkout_handoff(
    *,
    source_path: Path,
    source_git_path: str,
    candidate_sha: str,
    candidate_roots: tuple[Path, ...],
) -> bool:
    resolved_source = source_path.resolve()
    for root in candidate_roots:
        resolved_root = root.resolve()
        if resolved_source != resolved_root and resolved_root not in resolved_source.parents:
            continue
        head = subprocess.run(
            ["git", "-C", str(resolved_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if head.returncode != 0 or head.stdout.strip() != candidate_sha:
            return False
        candidate_blob = _git_blob(
            (resolved_root,),
            candidate_sha,
            source_git_path,
        )
        if candidate_blob is None:
            return not source_path.exists()
        return (
            source_path.is_file()
            and source_path.read_bytes() == candidate_blob
        )
    return False


def _verify_production_identity(
    row: dict,
    from_sha: str | None,
    candidate_sha: str,
    candidate_roots: tuple[Path, ...],
) -> None:
    source_path = Path(str(row.get("source_path") or ""))
    source_git_path = str(row.get("source_git_path") or "")
    source_kind = str(row.get("source_kind") or "")
    source_hash = str(row.get("source_sha256") or "")
    source_commit = str(row.get("source_commit") or candidate_sha)
    allowed_source_commits = {candidate_sha}
    if from_sha:
        allowed_source_commits.add(from_sha)
    if (
        row.get("candidate_sha") != candidate_sha
        or source_commit not in allowed_source_commits
        or not source_git_path
        or source_kind
        not in {"installed_checkout", "candidate_checkout", "candidate_archive"}
        or (
            source_kind == "installed_checkout"
            and (not from_sha or source_commit != from_sha)
        )
        or (
            source_kind in {"candidate_checkout", "candidate_archive"}
            and source_commit != candidate_sha
        )
    ):
        raise SystemExit(
            "candidate production source identity is invalid: %r" % row
        )

    git_blob = _git_blob(candidate_roots, source_commit, source_git_path)
    if git_blob is None or source_hash != hashlib.sha256(git_blob).hexdigest():
        raise SystemExit(
            "candidate production source Git identity is invalid: %r" % row
        )

    if source_path.is_file():
        if source_hash != hashlib.sha256(source_path.read_bytes()).hexdigest():
            if source_kind != "installed_checkout" or not (
                _verify_installed_checkout_handoff(
                    source_path=source_path,
                    source_git_path=source_git_path,
                    candidate_sha=candidate_sha,
                    candidate_roots=candidate_roots,
                )
            ):
                raise SystemExit(
                    "candidate production source bytes changed after execution: %r"
                    % row
                )
    elif source_kind == "installed_checkout":
        if not _verify_installed_checkout_handoff(
            source_path=source_path,
            source_git_path=source_git_path,
            candidate_sha=candidate_sha,
            candidate_roots=candidate_roots,
        ):
            raise SystemExit(
                "installed production source disappeared without an exact "
                "candidate checkout handoff: %r" % row
            )
    elif source_kind != "candidate_archive":
        raise SystemExit(
            "candidate checkout source disappeared after execution: %r" % row
        )


def verify_rehearsal_integrity(
    rows: list[dict],
    *,
    from_sha: str | None = None,
    candidate_sha: str,
    scope: str,
    candidate_roots: tuple[Path, ...] = (
        Path("/home/ec2-user/leadpoet_repo"),
        Path("/home/ec2-user/leadpoet/leadpoet"),
    ),
) -> None:
    substitutions = [
        row
        for row in rows
        if row.get("implementation") == "internal_substitution"
    ]
    synthetic_external_fixtures = [
        row
        for row in rows
        if row.get("fixture_authenticity") == "synthetic"
    ]
    if scope == "exact" and substitutions:
        identities = sorted(
            {
                _substitution_identity(row) or "<unknown>"
                for row in substitutions
            }
        )
        raise SystemExit(
            "exact restart rehearsal used repository-code substitutions: "
            + ", ".join(identities)
        )
    if scope == "exact" and synthetic_external_fixtures:
        identities = sorted(
            {
                str(
                    row.get("operation")
                    or row.get("boundary")
                    or row.get("kind")
                    or "<unknown>"
                )
                for row in synthetic_external_fixtures
            }
        )
        raise SystemExit(
            "exact restart rehearsal used synthetic external fixtures: "
            + ", ".join(identities)
        )
    if scope == TARGETED_REGRESSION_SCOPE:
        for row in substitutions:
            identity = _substitution_identity(row)
            allowed = (
                identity in KNOWN_INTERNAL_SUBSTITUTION_MODULES
                or identity in KNOWN_INTERNAL_SUBSTITUTION_SCRIPTS
                or identity in KNOWN_INTERNAL_SUBSTITUTION_PROCESSES
                or identity in KNOWN_INTERNAL_SUBSTITUTION_BOUNDARIES
            )
            if not allowed:
                raise SystemExit(
                    "targeted regression used an unclassified internal "
                    f"substitution: {identity or '<unknown>'}"
                )
    elif scope != "exact":
        raise SystemExit("unknown rehearsal scope: %s" % scope)

    for row in rows:
        implementation = row.get("implementation")
        if implementation in {"production_module", "production_script"}:
            _verify_production_identity(
                row,
                from_sha,
                candidate_sha,
                candidate_roots,
            )


def verify_migration_backed_database_contract(candidate_sha: str) -> str:
    path = Path("/rehearsal-state/postgres-v2-schema-contract.json")
    if not path.is_file():
        raise SystemExit(
            "migration-backed PostgreSQL contract evidence is missing"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    required_checks = {
        "pre_128_transport_rejected",
        "post_128_transport_persisted",
        "transport_contract_valid",
        "pre_129_attested_local_transport_rejected",
        "post_129_attested_local_transport_persisted",
        "transport_terminal_contract_valid",
        "pre_133_provider_outcome_contract_rejected",
        "post_133_provider_outcome_contract_valid",
        "pre_134_provider_outcome_head_contract_rejected",
        "post_134_provider_outcome_head_contract_valid",
        "provider_outcome_append_atomic",
        "provider_outcome_contention_zero_rollback",
        "provider_outcome_conflict_head_exact",
        "pre_132_lifetime_credit_rejected",
        "post_132_lifetime_credit_persisted",
        "lifetime_credit_rpc_idempotent",
        "grandfathered_credit_unchanged",
        "lifetime_credit_contract_valid",
        "finalized_view_projection_exact",
        "finalized_view_seed_available",
        "settlement_authority_parsed",
        "measured_settlement_receipt_projection_exact",
        "tampered_weight_receipt_rejected",
        "required_schema_migrations_declared",
    }
    checks = document.get("checks")
    expected_provider_outcome_migrations = [
        "130-research-lab-provider-outcome-append.sql",
        "131-research-lab-provider-outcome-backpressure.sql",
        "132-research-lab-champion-lifetime-credit.sql",
        "133-research-lab-provider-outcome-contention-status.sql",
        "134-research-lab-provider-outcome-head-contention.sql",
    ]
    applied_migrations = document.get("applied_migrations")
    if (
        document.get("schema_version")
        != "leadpoet.restart_rehearsal.postgres_contract.v1"
        or document.get("candidate_sha") != candidate_sha
        or not isinstance(checks, dict)
        or set(checks) != required_checks
        or any(checks[name] is not True for name in required_checks)
        or not isinstance(applied_migrations, list)
        or applied_migrations[-5:] != expected_provider_outcome_migrations
    ):
        raise SystemExit(
            "migration-backed PostgreSQL contract evidence is incomplete"
        )
    relations = document.get("relations")
    if (
        not isinstance(relations, dict)
        or "research_lab_finalized_allocation_epochs_v2" not in relations
    ):
        raise SystemExit(
            "migration-backed finalized allocation view evidence is missing"
        )
    seed_rows = document.get("seed_rows")
    finalized_rows = (
        seed_rows.get("research_lab_finalized_allocation_epochs_v2")
        if isinstance(seed_rows, dict)
        else None
    )
    if (
        not isinstance(finalized_rows, list)
        or len(finalized_rows) != 1
        or not isinstance(finalized_rows[0], dict)
        or set(finalized_rows[0])
        != set(
            relations["research_lab_finalized_allocation_epochs_v2"][
                "columns"
            ]
        )
    ):
        raise SystemExit(
            "migration-backed finalized authority seed is missing"
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_gateway_private_model_environment(rows: list[dict]) -> None:
    gateway_processes = [
        row
        for row in rows
        if row.get("kind") == "process"
        and row.get("process") == "gateway.main"
        and row.get("status") == "started"
    ]
    if len(gateway_processes) != 1:
        raise SystemExit(
            "gateway rehearsal did not launch exactly one gateway.main process"
        )
    if (
        gateway_processes[0].get("environment_contract")
        != EXPECTED_GATEWAY_PRIVATE_MODEL_ENV
    ):
        raise SystemExit(
            "gateway.main private-model source environment differs from "
            "the canonical restart contract"
        )


def verify_gateway_provider_preflight(
    rows: list[dict],
    *,
    transition: str,
) -> None:
    if transition != "forward":
        return
    expected = {
        ("api.exa.ai", "/search"),
        ("api.scrapingdog.com", "/account"),
    }
    observed = {
        (str(row.get("host") or ""), str(row.get("path") or ""))
        for row in rows
        if row.get("operation") == "provider_transport"
        and row.get("status") == 200
    }
    if not expected <= observed:
        raise SystemExit(
            "gateway provider preflight did not complete through both "
            "authenticated provider boundaries"
        )


def verify_chain_settlement_durable_readback(rows: list[dict]) -> None:
    persistence_ordinals = [
        ordinal
        for ordinal, row in enumerate(rows)
        if row.get("kind") == "local-postgrest"
        and row.get("operation") == "chain_settlement_persisted"
        and row.get("status") == "ok"
        and row.get("target")
        in {
            "persist_research_lab_chain_realized_settlement_v1",
            "persist_research_lab_chain_realized_unattributed_v2",
            "persist_research_lab_chain_realized_lifetime_settlement_v2",
        }
    ]
    if not persistence_ordinals:
        raise SystemExit(
            "gateway rehearsal did not persist a chain-realized settlement"
        )
    settlement_reads = [
        ordinal
        for ordinal, row in enumerate(rows)
        if row.get("kind") == "local-postgrest"
        and row.get("operation") == "select"
        and row.get("status") == "ok"
        and row.get("target")
        == "research_lab_chain_realized_epoch_settlements_v1"
    ]
    credit_reads = [
        ordinal
        for ordinal, row in enumerate(rows)
        if row.get("kind") == "local-postgrest"
        and row.get("operation") == "select"
        and row.get("status") == "ok"
        and row.get("target")
        == "research_lab_chain_realized_obligation_credits_v1"
    ]
    first_persistence = min(persistence_ordinals)
    if not any(ordinal > first_persistence for ordinal in settlement_reads):
        raise SystemExit(
            "gateway rehearsal did not read back its durable settlement"
        )
    if not any(ordinal > first_persistence for ordinal in credit_reads):
        raise SystemExit(
            "gateway rehearsal did not read back durable settlement credits"
        )


def verify_restart_epoch_transient_recovery(rows: list[dict]) -> None:
    head_reads = [
        row
        for row in rows
        if row.get("boundary") == "stateful_subnet_chain"
        and row.get("operation") == "epoch_snapshot"
        and row.get("method") == "get_chain_head"
    ]
    failed_ordinals = [
        ordinal
        for ordinal, row in enumerate(head_reads)
        if row.get("injected_failure") is True
    ]
    successful_ordinals = [
        ordinal
        for ordinal, row in enumerate(head_reads)
        if row.get("injected_failure") is False
    ]
    if not failed_ordinals or not successful_ordinals:
        raise SystemExit(
            "restart launcher did not exercise transient epoch-read recovery"
        )
    if not any(
        successful > failed
        for failed in failed_ordinals
        for successful in successful_ordinals
    ):
        raise SystemExit(
            "restart launcher did not recover after the injected epoch-read failure"
        )


def selected_weight_storage_preflight_capability(
    candidate_roots: tuple[Path, ...],
) -> bool:
    relative = Path("gateway/tee/verify_weight_submission_ready_v2.py")
    for root in candidate_roots:
        source_path = root / relative
        if not source_path.is_file():
            continue
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--storage-read-preflight"
            for node in ast.walk(tree)
        )
    raise SystemExit(
        "candidate weight-readiness source is unavailable for capability proof"
    )


def verify_gateway_weight_readiness_invocations(
    rows: list[dict],
    *,
    candidate_sha: str,
    transition: str = "forward",
    storage_preflight_supported: bool = True,
) -> None:
    module = "gateway.tee.verify_weight_submission_ready_v2"
    if not storage_preflight_supported and transition != "rollback":
        raise SystemExit(
            "current release does not declare the required weight storage preflight"
        )
    expected_prefix = []
    if storage_preflight_supported:
        expected_prefix.append(
            {
                "argv": ["-m", module, "--storage-read-preflight"],
                "source_kind": "candidate_archive",
            }
        )
    repair_contract = {
        "argv": ["-m", module, "--repair"],
        "source_kind": "candidate_checkout",
    }
    http_contract = {
        "argv": [
            "-m",
            module,
            "--gateway-url",
            "http://localhost:8000",
            "--http-timeout-seconds",
            "360",
        ],
        "source_kind": "candidate_checkout",
    }
    observed = [
        row
        for row in rows
        if row.get("kind") == "python-module"
        and row.get("module") == module
    ]
    prefix_count = len(expected_prefix)
    if len(observed) < prefix_count + 2:
        raise SystemExit(
            "gateway launcher did not execute the exact production weight "
            f"readiness invocation contract: {observed!r}"
        )
    repair_rows = observed[prefix_count:-1]
    if not repair_rows:
        raise SystemExit(
            "gateway launcher did not execute weight readiness repair"
        )
    expected = [
        *expected_prefix,
        *([repair_contract] * len(repair_rows)),
        http_contract,
    ]
    injected = any(
        row.get("kind") == "fault-injection"
        and row.get("module") == module
        and row.get("status") == "injected-transient-failure"
        for row in rows
    )
    if injected and len(repair_rows) < 2:
        raise SystemExit(
            "gateway launcher did not retry weight readiness after the "
            "injected transient failure"
        )
    for ordinal, (row, contract) in enumerate(
        zip(observed, expected),
        start=1,
    ):
        if (
            row.get("status") != "started"
            or row.get("implementation") != "production_module"
            or row.get("candidate_sha") != candidate_sha
            or row.get("source_commit") != candidate_sha
            or row.get("source_git_path")
            != "gateway/tee/verify_weight_submission_ready_v2.py"
            or row.get("source_kind") != contract["source_kind"]
            or row.get("argv") != contract["argv"]
        ):
            raise SystemExit(
                "gateway production weight readiness invocation differs from "
                f"the launcher contract at ordinal {ordinal}: {row!r}"
            )
def main() -> int:
    component, from_sha, candidate_sha = sys.argv[1:4]
    scenario = (
        sys.argv[4]
        if len(sys.argv) > 4
        else "production_success"
    )
    scope = sys.argv[5] if len(sys.argv) > 5 else "exact"
    transition = sys.argv[6] if len(sys.argv) > 6 else "forward"
    if transition not in {"forward", "rollback"}:
        raise SystemExit("unknown rehearsal transition: %s" % transition)
    rows = events()
    rejected = [row for row in rows if row.get("status") == "rejected"]
    if rejected:
        raise SystemExit(f"contract adapter rejected operations: {rejected!r}")
    verify_rehearsal_integrity(
        rows,
        from_sha=from_sha,
        candidate_sha=candidate_sha,
        scope=scope,
    )
    postgres_contract_sha256 = verify_migration_backed_database_contract(
        candidate_sha
    )
    if transition == "forward":
        verify_restart_epoch_transient_recovery(rows)

    labels: list[str] = []
    for row in rows:
        kind = row.get("kind")
        module = row.get("module")
        operation = row.get("operation")
        stage = row.get("stage")
        process = row.get("process")
        if module:
            labels.append(f"module:{module}")
        elif kind == "nitro":
            labels.append(f"nitro:{operation}")
        elif kind == "docker":
            labels.append(f"docker:{operation}")
        elif kind == "process":
            labels.append(f"process:{process}")

    if component == "gateway":
        if scenario != "production_success":
            raise SystemExit(
                "targeted fault scenario cannot satisfy exact restart evidence"
            )
        storage_preflight_supported = (
            selected_weight_storage_preflight_capability(
                (
                    Path("/home/ec2-user/leadpoet_repo"),
                    Path("/home/ec2-user/leadpoet/leadpoet"),
                )
            )
        )
        verify_gateway_weight_readiness_invocations(
            rows,
            candidate_sha=candidate_sha,
            transition=transition,
            storage_preflight_supported=storage_preflight_supported,
        )
        verify_chain_settlement_durable_readback(rows)
        required_gateway_order = [
            "module:gateway.tee.release_channel_v2",
            "module:gateway.tee.prepare_gateway_envelopes_v2",
        ]
        if storage_preflight_supported:
            required_gateway_order.append(
                "module:gateway.tee.verify_weight_submission_ready_v2"
            )
        required_gateway_order.extend(
            [
                "module:gateway.tee.restart_preflight_v2",
                "nitro:build_enclave",
                "nitro:run_enclave",
                "module:gateway.utils.tee_v2_bootstrap",
                "module:gateway.utils.tee_kms_provision_v2",
                "module:gateway.tee.verify_v2_runtime_ready",
                "module:gateway.tee.verify_weight_submission_ready_v2",
                "process:gateway.main",
                "module:gateway.tee.verify_weight_submission_ready_v2",
            ]
        )
        require_order(labels, required_gateway_order)
        verify_gateway_private_model_environment(rows)
        verify_gateway_provider_preflight(rows, transition=transition)
        state = json.loads(
            Path("/rehearsal-state/state.json").read_text(encoding="utf-8")
        )
        if len(state.get("enclaves", [])) != 3:
            raise SystemExit("gateway did not start the exact three-enclave topology")
    else:
        require_order(
            labels,
            [
                "module:gateway.tee.release_channel_v2",
                "module:validator_tee.host.refresh_hotkey_config_v2",
                "module:validator_tee.host.restart_preflight_v2",
                "nitro:build_enclave",
                "nitro:run_enclave",
                "process:validator.chain_relay",
                "module:validator_tee.host.runtime_v2_bootstrap",
                "module:validator_tee.host.hotkey_bootstrap_v2",
            ],
        )
        state = json.loads(
            Path("/rehearsal-state/state.json").read_text(encoding="utf-8")
        )
        validator = state.get("containers", {}).get("leadpoet-validator-main", {})
        validator_log = Path(str(validator.get("log_path") or ""))
        if (
            validator.get("running") is not True
            or validator.get("restart_count") != 0
            or not isinstance(validator.get("pid"), int)
            or int(validator["pid"]) <= 0
            or not validator_log.is_file()
        ):
            raise SystemExit("validator final container state is invalid")

    pcr0_values = {
        str((enclave.get("Measurements") or {}).get("PCR0") or "")
        for enclave in state.get("enclaves", [])
    }
    if len(pcr0_values) != 1 or "" in pcr0_values:
        raise SystemExit("launcher enclave PCR0 evidence is missing or ambiguous")

    print(
        json.dumps(
            {
                "schema_version": "leadpoet.local_restart_rehearsal.v2",
                "status": (
                    "passed"
                    if scope == "exact"
                    else "targeted_regression_passed"
                ),
                "component": component,
                "scope": scope,
                "from_sha": from_sha,
                "candidate_sha": candidate_sha,
                "scenario": scenario,
                "transition": transition,
                "event_count": len(rows),
                "pcr0": next(iter(pcr0_values)),
                "postgres_contract_sha256": postgres_contract_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
