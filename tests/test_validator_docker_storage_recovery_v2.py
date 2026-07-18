from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_validator_docker_recovery_is_guarded_and_runs_after_shutdown():
    recovery = (
        ROOT / "validator_tee" / "scripts" / "reclaim_docker_storage_v2.sh"
    ).read_text(encoding="utf-8")
    restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    workflow = (
        ROOT / ".github" / "workflows" / "attested-v2-release.yml"
    ).read_text(encoding="utf-8")

    assert 'CONTAINER_COUNT="$(docker ps -aq' in recovery
    assert 'if [ "$CONTAINER_COUNT" -ne 0 ]' in recovery
    assert 'if [ "$DOCKER_ROOT" != "/var/lib/docker" ]' in recovery
    assert "systemctl stop docker.service docker.socket" in recovery
    assert 'rm -rf --one-file-system "$DOCKER_ROOT"' in recovery

    remove_containers = restart.index("| xargs -r docker rm")
    reclaim = restart.index("reclaim_docker_storage_v2.sh")
    build = restart.index("bash validator_tee/scripts/build_enclave.sh")
    assert remove_containers < reclaim < build
    assert "VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1" in restart
    assert "VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1" in workflow

    subprocess.run(
        [
            "bash",
            "-n",
            str(
                ROOT
                / "validator_tee"
                / "scripts"
                / "reclaim_docker_storage_v2.sh"
            ),
        ],
        check=True,
    )
