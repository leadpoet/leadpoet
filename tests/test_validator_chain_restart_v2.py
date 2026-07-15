from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_validator_restart_preserves_build_order_and_starts_chain_relay():
    script = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    preflight = script.index("python3 -m validator_tee.host.restart_preflight_v2")
    artifact_prepare = script.index(
        "python3 -m validator_tee.scripts.stage_runtime_artifacts_v2"
    )
    shutdown = script.index('echo "Stopping validator processes and containers"')
    build = script.index("bash validator_tee/scripts/build_enclave.sh")
    release_gate = script.index("python3 -m validator_tee.host.verify_release_gate_v2")
    release_archive = script.index("python3 -m validator_tee.host.release_archive_v2")
    enclave = script.index("sudo nitro-cli run-enclave")
    relay = script.index("python3 -m validator_tee.host.chain_relay_v2")
    runtime = script.index("python3 -m validator_tee.host.runtime_v2_bootstrap")
    hotkey = script.index("python3 -m validator_tee.host.hotkey_bootstrap_v2")
    validator = script.index('echo "Starting validator"')
    assert artifact_prepare < preflight < shutdown < build < release_gate < release_archive < enclave < relay < runtime < hotkey < validator
    assert script.index("--allow-download") < shutdown
    build_script = (
        ROOT / "validator_tee" / "scripts" / "build_enclave.sh"
    ).read_text(encoding="utf-8")
    assert "--offline-artifact-root" in build_script
    assert "--allow-download" not in build_script
    assert 'pkill -TERM -f "validator_tee.host.chain_relay_v2"' in script
    assert 'pkill -KILL -f "validator_tee.host.chain_relay_v2"' in script
    assert 'VALIDATOR_PYTHON_BIN="${VALIDATOR_PYTHON_BIN:-python3}"' in script
    assert '"$VALIDATOR_PYTHON_BIN" neurons/validator.py' in script
    assert "usable validator hotkey material remains on the parent" in script
    assert 'HOST_HOTKEY_DIR="$VALIDATOR_WALLET_ROOT/$VALIDATOR_WALLET_NAME/hotkeys"' in script
    assert 'find "$HOST_HOTKEY_DIR" -mindepth 1 -maxdepth 1 -print -quit' in script
    assert "legacy_v1_compat" not in script
    assert "VALIDATOR_V2_GATEWAY_URL" in script
    assert "VALIDATOR_V2_RELEASE_MANIFEST" in script
    assert '--validator-release "$VALIDATOR_V2_RELEASE_MANIFEST"' in script
    assert '--host-hotkey-directory "$HOST_HOTKEY_DIR"' in script
    assert '--retain 3' in script
    assert "unset ENABLE_TEE_SUBMISSION VALIDATOR_ATTESTED_WEIGHT_MODE" in script
    assert 'export VALIDATOR_V2_DEPLOY_COMMIT="$VALIDATOR_DEPLOY_SHA"' in script
    assert 'export LEADPOET_WRAPPER_ACTIVE=1' in script
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")
    assert 'git -C "$REPO_ROOT" archive "$VALIDATOR_V2_DEPLOY_COMMIT"' in deploy
    assert '--build-arg "LEADPOET_BUILD_COMMIT=$VALIDATOR_V2_DEPLOY_COMMIT"' in deploy
    assert "org.opencontainers.image.revision" in deploy
    assert "git fetch" not in deploy
    assert "git reset --hard" not in deploy
    assert '-e VALIDATOR_V2_DEPLOY_COMMIT="$VALIDATOR_V2_DEPLOY_COMMIT"' in deploy
    assert "ENABLE_TEE_SUBMISSION" not in deploy
    assert "VALIDATOR_REQUIRE_GATEWAY_WEIGHT_SUBMISSION" not in deploy
    subprocess.run(["bash", "-n", str(ROOT / "validator_restart.sh")], check=True)
    subprocess.run(
        [
            "bash",
            "-n",
            str(ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"),
        ],
        check=True,
    )


def test_validator_restart_and_container_are_v2_only():
    script = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")

    assert 'VALIDATOR_WEIGHT_PROTOCOL="${VALIDATOR_WEIGHT_PROTOCOL:-authoritative_v2}"' in script
    assert 'authoritative_v2)' in script
    assert 'EXPECTED_CHAIN="${EXPECTED_CHAIN:-wss://entrypoint-finney.opentensor.ai:443}"' in script
    assert "legacy_v1_compat" not in script
    assert "verify_legacy_v1_enclave" not in script
    assert "APPROVED_LEGACY_V1_PCR0=" not in script
    assert '-e VALIDATOR_WEIGHT_PROTOCOL=authoritative_v2' in deploy
    assert 'VALIDATOR_DEPLOY_SHA="$(git -C "$REPO_DIR" rev-parse HEAD)"' in deploy
    assert '-e GIT_COMMIT_HASH="$VALIDATOR_DEPLOY_SHA"' in deploy
    assert '-e EXPECTED_CHAIN="$EXPECTED_CHAIN"' in deploy
    assert '-e BITTENSOR_NETUID="$VALIDATOR_NETUID"' in deploy
    assert '--netuid "$VALIDATOR_NETUID"' in deploy
    subprocess.run(["bash", "-n", str(ROOT / "validator_restart.sh")], check=True)
    subprocess.run(
        [
            "bash",
            "-n",
            str(
                ROOT
                / "validator_models"
                / "containerizing"
                / "deploy_dynamic.sh"
            ),
        ],
        check=True,
    )


def test_validator_deploy_preserves_secret_manager_gateway_routing():
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")

    capture = deploy.index('INHERITED_GATEWAY_URL_SET="${GATEWAY_URL+x}"')
    main_env = deploy.index('source "$MAIN_ENV_PATH"')
    docker_env = deploy.index("source .env.docker")
    restore = deploy.index('GATEWAY_URL="$INHERITED_GATEWAY_URL"')
    first_container = deploy.index("docker run -d")

    assert capture < main_env < docker_env < restore < first_container
    assert deploy.count(
        '-e GATEWAY_URL="${GATEWAY_URL:-https://gateway.subnet71.com}"'
    ) == 3
    assert (
        '-e VALIDATOR_V2_GATEWAY_URL="${VALIDATOR_V2_GATEWAY_URL:-}"'
        in deploy
    )
    assert "http://52.91.135.79:8000" not in deploy


def test_validator_eif_measures_chain_source_without_base_image_change():
    dockerfile = (ROOT / "validator_tee" / "Dockerfile.enclave").read_text(
        encoding="utf-8"
    )
    build_script = (
        ROOT / "validator_tee" / "scripts" / "build_enclave.sh"
    ).read_text(encoding="utf-8")
    assert "COPY validator_tee/enclave/ /app/validator_tee/enclave/" in dockerfile
    assert "COPY leadpoet_canonical/ /app/leadpoet_canonical/" in dockerfile
    assert '"validator_tee/enclave"' in build_script
    assert '"leadpoet_canonical"' in build_script
    assert (ROOT / "validator_tee" / "enclave" / "chain_source_v2.py").is_file()
    assert (ROOT / "leadpoet_canonical" / "chain_source_v2.py").is_file()
