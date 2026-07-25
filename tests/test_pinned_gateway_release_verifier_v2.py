from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = (
    ROOT / "validator_tee" / "scripts" / "verify_pinned_gateway_release_v2.sh"
)
COMMIT = "a" * 40


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
if [ "${TRANSIENT_FIRST:-0}" = "1" ] && [ "$count" -eq 1 ]; then
  exit 7
fi
url="${!#}"
if [[ "$url" == */health/v2-authority ]]; then
  printf '{"status":"ready","commit_sha":"%s"}\\n' "$FAKE_COMMIT"
elif [[ "$url" == */build-info ]]; then
  printf '{"git_commit":"%s"}\\n' "$FAKE_COMMIT"
elif [[ "$url" == */weights/v2/release-evidence/* ]]; then
  printf '{"schema_version":"leadpoet.auditor_release_evidence.v2","commit_sha":"%s"}\\n' "$FAKE_COMMIT"
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
) -> tuple[subprocess.CompletedProcess[str], int]:
    bin_dir, state = _fake_commands(
        tmp_path,
        transient_first=transient_first,
    )
    result = subprocess.run(
        ["bash", str(VERIFIER), "http://gateway.invalid:8000", COMMIT],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "FAKE_CURL_STATE": str(state),
            "FAKE_COMMIT": returned_commit,
            "TRANSIENT_FIRST": "1" if transient_first else "0",
        },
        timeout=10,
    )
    return result, int(state.read_text(encoding="utf-8").strip())


def test_pinned_gateway_verifier_recovers_from_transient_transport_failure(
    tmp_path: Path,
) -> None:
    result, requests = _run(tmp_path, transient_first=True)

    assert result.returncode == 0
    assert "pinned_gateway_release_aligned" in result.stdout
    assert "retrying (1/12)" in result.stderr
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
