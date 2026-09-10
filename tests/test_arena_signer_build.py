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
    assert 'install -m 0600 "$FROZEN_POLICY" "$OUTPUT_ROOT/arena_signer_policy.json"' in script

def test_published_policy_is_accepted_by_private_bootstrap(tmp_path):
    from validator_tee.host.arena_hotkey_bootstrap import _private_read

    script = (ROOT / "validator_tee/scripts/build_enclave.sh").read_text()
    source = tmp_path / "frozen.json"
    source.write_text('{"public_policy":true}\n')
    output = tmp_path / "output"
    output.mkdir()
    line = next(line for line in script.splitlines()
                if line.startswith('install ') and '"$OUTPUT_ROOT/arena_signer_policy.json"' in line)
    subprocess.run(["bash", "-c", line], check=True,
                   env={"PATH":"/usr/bin:/bin", "FROZEN_POLICY":str(source), "OUTPUT_ROOT":str(output)})
    published = output / "arena_signer_policy.json"
    assert published.stat().st_mode & 0o777 == 0o600
    assert _private_read(published) == source.read_bytes()
