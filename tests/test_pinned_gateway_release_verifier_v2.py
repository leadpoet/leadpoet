from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from gateway.tee.release_channel_v2 import build_release_channel_v2
from tests.test_release_channel_v2 import _gateway_manifest, _validator_manifest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = (
    ROOT / "validator_tee" / "scripts" / "verify_pinned_gateway_release_v2.sh"
)
COMMIT = "a" * 40


def _local_release_evidence() -> str:
    import json

    return json.dumps(
        {
            "schema_version": "leadpoet.auditor_local_release_evidence.v1",
            "commit_sha": COMMIT,
            "release_channel": build_release_channel_v2(
                gateway_release_manifest=_gateway_manifest(COMMIT),
                validator_release_manifest=_validator_manifest(COMMIT),
            ),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _fake_commands(tmp_path: Path, *, transient_first: bool) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "curl-count"
    curl = bin_dir / "curl"
    curl.write_text(
        """#!/bin/bash
set -euo pipefail
count=0
if [ -f "$FAKE_CURL_STATE" ]; then
  count="$(cat "$FAKE_CURL_STATE")"
fi
count=$((count + 1))
printf '%s\\n' "$count" > "$FAKE_CURL_STATE"
if [ "${FAKE_CURL_DELAY_SECONDS:-0}" != "0" ]; then
  /bin/sleep "$FAKE_CURL_DELAY_SECONDS"
fi
if [ "${TRANSIENT_FIRST:-0}" = "1" ] && [ "$count" -eq 1 ]; then
  exit 7
fi
if [ "${FAKE_REVOKE_AFTER_REQUESTS:-0}" = "$count" ]; then
  printf 'failed:%s\\n' "$FAKE_COMMIT" > "$FAKE_COORDINATION_FILE"
fi
url="${!#}"
if [[ "$url" == */health/v2-authority ]]; then
  printf '{"status":"ready","commit_sha":"%s"}\\n' "$FAKE_COMMIT"
elif [[ "$url" == */build-info ]]; then
  printf '{"git_commit":"%s"}\\n' "$FAKE_COMMIT"
elif [[ "$url" == */weights/v2/release-evidence/* ]]; then
  if [ -n "${FAKE_RELEASE_EVIDENCE:-}" ]; then
    printf '%s\\n' "$FAKE_RELEASE_EVIDENCE"
  else
    printf '{"schema_version":"leadpoet.auditor_release_evidence.v2","commit_sha":"%s"}\\n' "$FAKE_COMMIT"
  fi
else
  exit 22
fi
""",
        encoding="utf-8",
    )
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    curl.chmod(0o755)
    sleep.chmod(0o755)
    return bin_dir, state


def _run(
    tmp_path: Path,
    *,
    transient_first: bool,
    returned_commit: str = COMMIT,
    coordination_file: Path | None = None,
    coordination_max_attempts: int | None = None,
    max_attempts: int | None = None,
    timeout_seconds: int | None = None,
    curl_delay_seconds: int = 0,
    revoke_after_requests: int = 0,
    release_evidence: str = "",
) -> tuple[subprocess.CompletedProcess[str], int]:
    bin_dir, state = _fake_commands(
        tmp_path,
        transient_first=transient_first,
    )
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "FAKE_CURL_STATE": str(state),
        "FAKE_COMMIT": returned_commit,
        "FAKE_CURL_DELAY_SECONDS": str(curl_delay_seconds),
        "FAKE_COORDINATION_FILE": str(coordination_file or ""),
        "FAKE_REVOKE_AFTER_REQUESTS": str(revoke_after_requests),
        "FAKE_RELEASE_EVIDENCE": release_evidence,
        "TRANSIENT_FIRST": "1" if transient_first else "0",
    }
    if coordination_file is not None:
        env["VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE"] = str(
            coordination_file
        )
    if coordination_max_attempts is not None:
        env["VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS"] = str(
            coordination_max_attempts
        )
    if max_attempts is not None:
        env["VALIDATOR_PINNED_GATEWAY_MAX_ATTEMPTS"] = str(max_attempts)
    if timeout_seconds is not None:
        env["VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS"] = str(timeout_seconds)
    result = subprocess.run(
        ["bash", str(VERIFIER), "http://gateway.invalid:8000", COMMIT],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    requests = (
        int(state.read_text(encoding="utf-8").strip())
        if state.exists()
        else 0
    )
    return result, requests


def test_pinned_gateway_verifier_accepts_valid_local_release_evidence(
    tmp_path: Path,
) -> None:
    result, requests = _run(
        tmp_path,
        transient_first=False,
        release_evidence=_local_release_evidence(),
    )

    assert result.returncode == 0
    assert "pinned_gateway_release_aligned" in result.stdout
    assert requests == 3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra="not-allowed"),
        lambda value: value["release_channel"].update(
            channel_hash="sha256:" + "0" * 64
        ),
        lambda value: value["release_channel"].update(commit_sha="b" * 40),
    ],
)
def test_pinned_gateway_verifier_rejects_invalid_local_release_evidence(
    tmp_path: Path,
    mutation,
) -> None:
    import json

    evidence = json.loads(_local_release_evidence())
    mutation(evidence)
    result, requests = _run(
        tmp_path,
        transient_first=False,
        max_attempts=1,
        release_evidence=json.dumps(evidence, separators=(",", ":")),
    )

    assert result.returncode == 1
    assert "did not align after 1 attempts" in result.stderr
    assert requests == 3


def test_pinned_gateway_verifier_recovers_from_transient_transport_failure(
    tmp_path: Path,
) -> None:
    result, requests = _run(tmp_path, transient_first=True)

    assert result.returncode == 0
    assert "pinned_gateway_release_aligned" in result.stdout
    assert "retrying (1/12)" in result.stderr
    assert "endpoint=v2_authority curl_status=7" in result.stderr
    assert requests == 4


def test_pinned_gateway_verifier_fails_closed_after_bounded_mismatch(
    tmp_path: Path,
) -> None:
    result, requests = _run(
        tmp_path,
        transient_first=False,
        returned_commit="b" * 40,
    )

    assert result.returncode == 1
    assert "did not align after 12 attempts" in result.stderr
    assert requests == 36


def test_pinned_gateway_verifier_honors_restart_wait_budget(
    tmp_path: Path,
) -> None:
    result, requests = _run(
        tmp_path,
        transient_first=False,
        returned_commit="b" * 40,
        max_attempts=2,
    )

    assert result.returncode == 1
    assert "did not align after 2 attempts" in result.stderr
    assert requests == 6


def test_pinned_gateway_verifier_rejects_unbounded_wait(
    tmp_path: Path,
) -> None:
    bin_dir, _ = _fake_commands(tmp_path, transient_first=False)
    result = subprocess.run(
        ["bash", str(VERIFIER), "http://gateway.invalid:8000", COMMIT],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS": "3001",
        },
        timeout=10,
    )

    assert result.returncode == 2
    assert "must be between 1 and 3000" in result.stderr


def test_pinned_gateway_verifier_rejects_unbounded_release_attempts(
    tmp_path: Path,
) -> None:
    bin_dir, _ = _fake_commands(tmp_path, transient_first=False)
    result = subprocess.run(
        ["bash", str(VERIFIER), "http://gateway.invalid:8000", COMMIT],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "VALIDATOR_PINNED_GATEWAY_MAX_ATTEMPTS": "3001",
        },
        timeout=10,
    )

    assert result.returncode == 2
    assert "VALIDATOR_PINNED_GATEWAY_MAX_ATTEMPTS" in result.stderr


def test_pinned_gateway_verifier_rejects_foreign_success_marker(
    tmp_path: Path,
) -> None:
    barrier = tmp_path / "gateway-restart-complete"
    barrier.write_text("b" * 40 + "\n", encoding="utf-8")
    result, requests = _run(
        tmp_path,
        transient_first=False,
        coordination_file=barrier,
        coordination_max_attempts=2,
    )

    assert result.returncode == 1
    assert "success marker differs from the selected commit" in result.stderr
    assert requests == 0


def test_pinned_gateway_verifier_accepts_matching_coordinator_completion(
    tmp_path: Path,
) -> None:
    barrier = tmp_path / "gateway-restart-complete"
    barrier.write_text(COMMIT + "\n", encoding="utf-8")
    result, requests = _run(
        tmp_path,
        transient_first=False,
        coordination_file=barrier,
    )

    assert result.returncode == 0
    assert "pinned_gateway_release_aligned" in result.stdout
    assert requests == 3


def test_pinned_gateway_verifier_accepts_extended_restart_budget(
    tmp_path: Path,
) -> None:
    barrier = tmp_path / "gateway-restart-complete"
    barrier.write_text(COMMIT + "\n", encoding="utf-8")
    result, requests = _run(
        tmp_path,
        transient_first=False,
        coordination_file=barrier,
        coordination_max_attempts=3000,
        max_attempts=3000,
        timeout_seconds=9300,
    )

    assert result.returncode == 0
    assert "pinned_gateway_release_aligned" in result.stdout
    assert requests == 3


def test_pinned_gateway_verifier_rechecks_marker_after_live_contract(
    tmp_path: Path,
) -> None:
    barrier = tmp_path / "gateway-restart-complete"
    barrier.write_text(COMMIT + "\n", encoding="utf-8")
    result, requests = _run(
        tmp_path,
        transient_first=False,
        coordination_file=barrier,
        revoke_after_requests=3,
    )

    assert result.returncode == 1
    assert "restart failed for the selected commit" in result.stderr
    assert "pinned_gateway_release_aligned" not in result.stdout
    assert requests == 3


def test_pinned_gateway_verifier_rejects_matching_failure_marker_without_http(
    tmp_path: Path,
) -> None:
    barrier = tmp_path / "gateway-restart-complete"
    barrier.write_text(f"failed:{COMMIT}\n", encoding="utf-8")
    result, requests = _run(
        tmp_path,
        transient_first=False,
        coordination_file=barrier,
    )

    assert result.returncode == 1
    assert "restart failed for the selected commit" in result.stderr
    assert requests == 0


def test_pinned_gateway_verifier_rejects_foreign_failure_marker_without_http(
    tmp_path: Path,
) -> None:
    barrier = tmp_path / "gateway-restart-complete"
    barrier.write_text(f"failed:{'b' * 40}\n", encoding="utf-8")
    result, requests = _run(
        tmp_path,
        transient_first=False,
        coordination_file=barrier,
    )

    assert result.returncode == 1
    assert "failure marker differs from the selected commit" in result.stderr
    assert requests == 0


def test_pinned_gateway_verifier_rejects_unbounded_total_timeout(
    tmp_path: Path,
) -> None:
    bin_dir, _ = _fake_commands(tmp_path, transient_first=False)
    result = subprocess.run(
        ["bash", str(VERIFIER), "http://gateway.invalid:8000", COMMIT],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS": "10801",
        },
        timeout=10,
    )

    assert result.returncode == 2
    assert "VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS" in result.stderr


def test_pinned_gateway_verifier_outer_timeout_kills_slow_request(
    tmp_path: Path,
) -> None:
    result, requests = _run(
        tmp_path,
        transient_first=False,
        timeout_seconds=1,
        curl_delay_seconds=5,
    )

    assert result.returncode == 124
    assert "verification exceeded 1s" in result.stderr
    assert requests == 1
