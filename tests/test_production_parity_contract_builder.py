from pathlib import Path
import subprocess

import pytest

from leadpoet_canonical.production_parity import (
    ProductionParityError,
    sha256_bytes,
    verify_contract_checkout,
)
from scripts.build_production_parity_contract import (
    ALWAYS_COMMITTED_PATHS,
    _source_commitments,
    build_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_STAGING_PATH = "scripts/run_physical_v2_staging.py"
HOST_RPC_TRANSPORT_PATHS = {
    "gateway/tee/proxy_transport_preflight_v2.py",
    "gateway/utils/tee_client.py",
    "gateway/utils/tee_egress_forwarder.py",
    "gateway/utils/tee_inter_enclave_relay.py",
    "scripts/run_physical_v2_staging.py",
    "validator_tee/host/chain_relay_v2.py",
    "validator_tee/host/vsock_client.py",
}


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
    assert verify_contract_checkout(checkout, contract) == contract
    for relative_path in sorted(HOST_RPC_TRANSPORT_PATHS):
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
