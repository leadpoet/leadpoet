#!/usr/bin/env python3.11
"""Validate that the exact restart launcher consumed every required contract."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys


TARGETED_REGRESSION_SCOPE = "weight_readiness_regression"
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


def events() -> list[dict]:
    path = Path("/rehearsal-state/events.jsonl")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def _verify_production_identity(
    row: dict,
    candidate_sha: str,
    candidate_roots: tuple[Path, ...],
) -> None:
    source_path = Path(str(row.get("source_path") or ""))
    source_git_path = str(row.get("source_git_path") or "")
    source_kind = str(row.get("source_kind") or "")
    source_hash = str(row.get("source_sha256") or "")
    if (
        row.get("candidate_sha") != candidate_sha
        or not source_git_path
        or source_kind not in {"candidate_checkout", "candidate_archive"}
    ):
        raise SystemExit(
            "candidate production source identity is invalid: %r" % row
        )

    git_blob: bytes | None = None
    for root in candidate_roots:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"{candidate_sha}:{source_git_path}",
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            git_blob = result.stdout
            break
    if git_blob is None or source_hash != hashlib.sha256(git_blob).hexdigest():
        raise SystemExit(
            "candidate production source Git identity is invalid: %r" % row
        )

    if source_path.is_file():
        if source_hash != hashlib.sha256(source_path.read_bytes()).hexdigest():
            raise SystemExit(
                "candidate production source bytes changed after execution: %r"
                % row
            )
    elif source_kind != "candidate_archive":
        raise SystemExit(
            "candidate checkout source disappeared after execution: %r" % row
        )


def verify_rehearsal_integrity(
    rows: list[dict],
    *,
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
                candidate_sha,
                candidate_roots,
            )


def main() -> int:
    component, from_sha, candidate_sha = sys.argv[1:4]
    scenario = (
        sys.argv[4]
        if len(sys.argv) > 4
        else "transient_503_recovery"
    )
    scope = sys.argv[5] if len(sys.argv) > 5 else "exact"
    rows = events()
    rejected = [row for row in rows if row.get("status") == "rejected"]
    if rejected:
        raise SystemExit(f"contract adapter rejected operations: {rejected!r}")
    verify_rehearsal_integrity(
        rows,
        candidate_sha=candidate_sha,
        scope=scope,
    )

    labels: list[str] = []
    for row in rows:
        kind = row.get("kind")
        module = row.get("module")
        operation = row.get("operation")
        stage = row.get("stage")
        process = row.get("process")
        if module:
            labels.append(f"module:{module}")
        elif kind == "weight-readiness":
            labels.append(f"weight:{stage}")
        elif kind == "nitro":
            labels.append(f"nitro:{operation}")
        elif kind == "docker":
            labels.append(f"docker:{operation}")
        elif kind == "process":
            labels.append(f"process:{process}")

    if component == "gateway":
        supabase_rows = [
            row
            for row in rows
            if row.get("kind") == "weight-readiness-supabase"
        ]
        expected_supabase_runs = (
            2 if scenario == "transient_503_recovery" else 1
        )
        if (
            len(supabase_rows) != expected_supabase_runs
            or any(row.get("status") != "ok" for row in supabase_rows)
            or any(row.get("row_count") != 16 for row in supabase_rows)
            or any(row.get("page_count") != 9 for row in supabase_rows)
        ):
            raise SystemExit(
                "finalized allocation history pagination was not rehearsed"
            )
        weight_rows = [
            row for row in rows if row.get("kind") == "weight-readiness"
        ]
        if not weight_rows or any(
            row.get("implementation") != "production_module"
            or row.get("candidate_sha") != candidate_sha
            for row in weight_rows
        ):
            raise SystemExit(
                "weight readiness did not execute the candidate production module"
            )
        module_path = Path(
            "/home/ec2-user/leadpoet_repo/"
            "gateway/tee/verify_weight_submission_ready_v2.py"
        )
        module_hash = __import__("hashlib").sha256(
            module_path.read_bytes()
        ).hexdigest()
        if any(
            row.get("module_path") != str(module_path)
            or row.get("module_sha256") != module_hash
            for row in weight_rows
        ):
            raise SystemExit(
                "weight readiness source identity differs from the candidate checkout"
            )

        repair_rows = [
            row for row in weight_rows if row.get("stage") == "repair"
        ]
        boundaries = [
            row
            for row in rows
            if row.get("kind") == "weight-readiness-boundary"
        ]
        persistence_rows = [
            row
            for row in rows
            if row.get("kind") == "weight-readiness-persistence"
        ]
        if not repair_rows or repair_rows[0].get("status") != "started":
            raise SystemExit("real weight repair did not start")

        if scenario != "transient_503_recovery":
            if any(
                row.get("process") == "gateway.main"
                for row in rows
            ):
                raise SystemExit(
                    "gateway launched after a failed weight-readiness gate"
                )
            if any(
                row.get("stage") == "http_handoff"
                for row in weight_rows
            ):
                raise SystemExit(
                    "post-launch handoff ran after pre-launch readiness failed"
                )
            if repair_rows[-1].get("status") != "failed":
                raise SystemExit(
                    "failure scenario did not fail the real readiness module"
                )
            direct_failures = [
                row
                for row in boundaries
                if row.get("boundary") == "direct_allocation"
                and row.get("status") == "failed"
            ]
            expected_attempts = 3 if scenario == "exhausted_503" else 1
            expected_code = (
                "authenticated_http_503"
                if scenario == "exhausted_503"
                else "authenticated_http_403"
            )
            if (
                len(direct_failures) != expected_attempts
                or any(
                    row.get("failure_code") != expected_code
                    for row in direct_failures
                )
                or len(persistence_rows) != expected_attempts
            ):
                raise SystemExit(
                    "readiness failure retry cardinality is invalid"
                )
            if any(
                row.get("failure_code") != expected_code
                for row in persistence_rows
            ):
                raise SystemExit(
                    "persistence failure evidence differs from the scenario"
                )
            deployment = json.loads(
                Path(
                    "/home/ec2-user/.config/leadpoet/deployments/"
                    "gateway-current.json"
                ).read_text(encoding="utf-8")
            )
            if (
                deployment.get("target_sha") != candidate_sha
                or deployment.get("status") != "failed"
                or deployment.get("stage") != "validator_weight_input_repair"
            ):
                raise SystemExit(
                    "failed restart deployment evidence is invalid"
                )
            print(
                json.dumps(
                    {
                        "schema_version": (
                            "leadpoet.local_restart_rehearsal.v2"
                        ),
                        "status": "targeted_expected_failure_passed",
                        "component": component,
                        "scope": scope,
                        "scenario": scenario,
                        "from_sha": from_sha,
                        "candidate_sha": candidate_sha,
                        "event_count": len(rows),
                    },
                    sort_keys=True,
                )
            )
            return 0

        require_order(
            labels,
            [
                "module:gateway.tee.release_channel_v2",
                "module:gateway.tee.prepare_gateway_envelopes_v2",
                "module:gateway.tee.restart_preflight_v2",
                "nitro:build",
                "nitro:run",
                "module:gateway.utils.tee_v2_bootstrap",
                "module:gateway.utils.tee_kms_provision_v2",
                "module:gateway.tee.verify_v2_runtime_ready",
                "weight:repair",
                "process:gateway.main",
                "weight:http_handoff",
            ],
        )
        if repair_rows[-1].get("status") != "ok":
            raise SystemExit("real weight repair did not recover")
        direct_rows = [
            row
            for row in boundaries
            if row.get("boundary") == "direct_allocation"
        ]
        if [
            (row.get("ordinal"), row.get("status"), row.get("failure_code"))
            for row in direct_rows
        ] != [
            (1, "failed", "authenticated_http_503"),
            (2, "ok", None),
        ]:
            raise SystemExit(
                "transient allocation recovery sequence is invalid"
            )
        if (
            len(persistence_rows) != 2
            or persistence_rows[0].get("status") != "failed"
            or persistence_rows[0].get("failure_code")
            != "authenticated_http_503"
            or persistence_rows[1].get("status") != "persisted"
        ):
            raise SystemExit(
                "measured artifact persistence recovery was not exercised"
            )
        recovered_attempts = persistence_rows[1].get("attempts") or []
        if [
            (row.get("method"), row.get("http_status"))
            for row in recovered_attempts
        ] != [
            ("GET", 503),
            ("GET", 200),
            ("HEAD", 503),
            ("HEAD", 200),
        ]:
            raise SystemExit(
                "measured GET/HEAD transient recovery sequence is invalid"
            )
        http_rows = [
            row
            for row in weight_rows
            if row.get("stage") == "http_handoff"
        ]
        if (
            not http_rows
            or http_rows[0].get("status") != "started"
            or http_rows[-1].get("status") != "ok"
            or not any(
                row.get("boundary") == "localhost_allocation_http"
                and row.get("status") == "ok"
                for row in boundaries
            )
        ):
            raise SystemExit(
                "real post-launch allocation handoff was not validated"
            )
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
                "nitro:build",
                "nitro:run",
                "process:validator.chain_relay",
                "module:validator_tee.host.runtime_v2_bootstrap",
                "module:validator_tee.host.hotkey_bootstrap_v2",
            ],
        )
        state = json.loads(
            Path("/rehearsal-state/state.json").read_text(encoding="utf-8")
        )
        validator = state.get("containers", {}).get("leadpoet-validator-main", {})
        if validator != {"running": True, "restart_count": 0}:
            raise SystemExit("validator final container state is invalid")

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
                "event_count": len(rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
