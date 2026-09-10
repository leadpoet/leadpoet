from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]

def test_arena_signer_build_uses_frozen_git_archive():
    script=(ROOT / "validator_tee/scripts/build_enclave.sh").read_text()
    assert 'git -C "$REPO_ROOT" archive --format=tar "$COMMIT_SHA"' in script
    assert "git clean" not in script
    assert "protected_workflows" not in script
    assert "Dockerfile.arena-signer" in script
    assert "Dockerfile.release" not in script

def test_arena_signer_build_emits_restart_contract_and_policy():
    script=(ROOT / "validator_tee/scripts/build_enclave.sh").read_text()
    assert "leadpoet.arena.signer_manifest.v1" in script
    assert '"commit_sha"' not in script
    assert '"eif_sha256":"sha256:"+digest' in script
    assert '"pcr0":pcr0' in script
    assert '"policy_sha256":"sha256:"+hashlib.sha256(policy_bytes).hexdigest()' in script
    dockerfile=(ROOT / "validator_tee/Dockerfile.arena-signer").read_text()
    assert "/app/validator_tee/enclave/arena_signer_policy.json" in dockerfile

def test_arena_signer_build_is_valid_bash():
    subprocess.run(["bash","-n",str(ROOT / "validator_tee/scripts/build_enclave.sh")],check=True)

def test_arena_signer_build_serializes_docker_and_freezes_policy():
    script=(ROOT / "validator_tee/scripts/build_enclave.sh").read_text()
    assert "leadpoet_acquire_docker_operation_lock_v2" in script
    assert 'exec 9>"$OUTPUT_ROOT/.publication.lock"' in script
    assert 'install -m 0644 "$FROZEN_POLICY" "$CONTEXT/.validator-tee-artifacts/arena_signer_policy.json"' in script
    assert 'install -m 0644 "$FROZEN_POLICY" "$OUTPUT_ROOT/arena_signer_policy.json"' in script
