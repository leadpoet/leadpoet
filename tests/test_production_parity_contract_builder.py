from pathlib import Path
import subprocess

import pytest

from leadpoet_canonical.production_parity import (
    ProductionParityError,
    migration_delta,
    sha256_bytes,
    verify_contract_checkout,
)
from scripts.build_production_parity_contract import (
    ALWAYS_COMMITTED_PATHS,
    _migration_inventory,
    _source_commitments,
    _tracked_paths,
    build_contract,
)


ROOT = Path(__file__).resolve().parents[1]
RETAINED_ARENA_BASE_SHA = "1e15195e44e5be2880ccd82b70a25c2c7bcf35de"
RETAINED_ARENA_UPLOAD_PATH = "scripts/191-lab-arena-upload-recovery.sql"
RETAINED_ARENA_UPLOAD_SHA256 = (
    "sha256:42913cf44d0d1f69a465731e75045af634c1b2600ab0e8fba24530ada979f8d7"
)
PHYSICAL_STAGING_PATH = "scripts/run_production_parity_full_host.py"
HOST_RPC_TRANSPORT_PATHS = {
    "gateway/tee/proxy_transport_preflight_v2.py",
    "gateway/utils/tee_client.py",
    "gateway/utils/tee_egress_forwarder.py",
    "gateway/utils/tee_inter_enclave_relay.py",
    "scripts/run_production_parity_full_host.py",
    "validator_tee/host/chain_relay_v2.py",
    "validator_tee/host/vsock_client.py",
}
REBENCHMARK_TRANSPORT_EVIDENCE_PATHS = {
    "gateway/tee/execution_job_manager_v2.py",
    "gateway/tee/provider_broker_v2.py",
    "gateway/tee/provider_client_v2.py",
    "gateway/tee/rpc_authority.py",
    "leadpoet_observability/sentry_operations.py",
}


def test_candidate_retains_real_applied_arena_migration_history() -> None:
    candidate_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    snapshot = _migration_inventory(
        ROOT,
        RETAINED_ARENA_BASE_SHA,
        _tracked_paths(ROOT, RETAINED_ARENA_BASE_SHA),
    )
    candidate = _migration_inventory(
        ROOT,
        candidate_sha,
        _tracked_paths(ROOT, candidate_sha),
    )
    retained = next(
        item for item in snapshot if item["path"] == RETAINED_ARENA_UPLOAD_PATH
    )
    assert retained["sha256"] == RETAINED_ARENA_UPLOAD_SHA256
    delta = migration_delta(
        snapshot_migrations=snapshot,
        candidate_migrations=candidate,
    )
    delta_paths = {item["path"] for item in delta}
    assert {
        "scripts/193-lab-arena-upload-recovery.sql",
        "scripts/194-lab-arena-open-scorer-refresh.sql",
    } <= delta_paths

    without_history = [
        item for item in candidate if item["path"] != RETAINED_ARENA_UPLOAD_PATH
    ]
    with pytest.raises(ProductionParityError, match="candidate removed applied migration"):
        migration_delta(
            snapshot_migrations=snapshot,
            candidate_migrations=without_history,
        )

    rewritten = [dict(item) for item in candidate]
    next(
        item for item in rewritten if item["path"] == RETAINED_ARENA_UPLOAD_PATH
    )["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ProductionParityError, match="candidate rewrote applied migration"):
        migration_delta(
            snapshot_migrations=snapshot,
            candidate_migrations=rewritten,
        )


def test_host_rpc_transports_are_exact_candidate_git_blobs() -> None:
    assert HOST_RPC_TRANSPORT_PATHS <= set(ALWAYS_COMMITTED_PATHS)
    candidate_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = []
    for path in sorted(HOST_RPC_TRANSPORT_PATHS):
        candidate_blob = subprocess.run(
            ["git", "show", "%s:%s" % (candidate_sha, path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        expected.append({"path": path, "sha256": sha256_bytes(candidate_blob)})
    assert _source_commitments(
        ROOT,
        candidate_sha,
        sorted(HOST_RPC_TRANSPORT_PATHS),
    ) == expected


def test_rebenchmark_transport_evidence_sources_are_exact_candidate_git_blobs() -> None:
    assert REBENCHMARK_TRANSPORT_EVIDENCE_PATHS <= set(ALWAYS_COMMITTED_PATHS)
    candidate_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = []
    for path in sorted(REBENCHMARK_TRANSPORT_EVIDENCE_PATHS):
        candidate_blob = subprocess.run(
            ["git", "show", "%s:%s" % (candidate_sha, path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        expected.append({"path": path, "sha256": sha256_bytes(candidate_blob)})
    assert _source_commitments(
        ROOT,
        candidate_sha,
        sorted(REBENCHMARK_TRANSPORT_EVIDENCE_PATHS),
    ) == expected


def test_physical_staging_is_bound_as_an_exact_candidate_git_blob() -> None:
    candidate_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    candidate_blob = subprocess.run(
        ["git", "show", f"{candidate_sha}:{PHYSICAL_STAGING_PATH}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout

    assert PHYSICAL_STAGING_PATH in ALWAYS_COMMITTED_PATHS
    assert _source_commitments(
        ROOT,
        candidate_sha,
        [PHYSICAL_STAGING_PATH],
    ) == [
        {
            "path": PHYSICAL_STAGING_PATH,
            "sha256": sha256_bytes(candidate_blob),
        }
    ]


def test_host_transport_contract_is_enforced_against_each_checkout_blob(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "candidate"
    subprocess.run(
        ["git", "clone", "--shared", "--quiet", str(ROOT), str(checkout)],
        check=True,
    )
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Leadpoet Test",
            "-c",
            "user.email=leadpoet-test@example.invalid",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "candidate",
        ],
        cwd=checkout,
        check=True,
    )
    candidate_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    contract = build_contract(
        root=checkout,
        base_sha=base_sha,
        candidate_sha=candidate_sha,
    )

    assert HOST_RPC_TRANSPORT_PATHS <= {
        item["path"] for item in contract["source_commitments"]
    }
    assert REBENCHMARK_TRANSPORT_EVIDENCE_PATHS <= {
        item["path"] for item in contract["source_commitments"]
    }
    assert verify_contract_checkout(checkout, contract) == contract
    for relative_path in sorted(
        HOST_RPC_TRANSPORT_PATHS | REBENCHMARK_TRANSPORT_EVIDENCE_PATHS
    ):
        target = checkout / relative_path
        original = target.read_bytes()
        target.write_bytes(original + b"\n# transport-tamper\n")
        with pytest.raises(
            ProductionParityError,
            match="candidate worktree source differs: " + relative_path,
        ):
            verify_contract_checkout(checkout, contract)
        target.write_bytes(original)
        assert verify_contract_checkout(checkout, contract) == contract
