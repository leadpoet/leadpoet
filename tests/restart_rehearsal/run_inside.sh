#!/bin/bash
set -euo pipefail

FROM_SHA="${REHEARSAL_FROM_SHA:?REHEARSAL_FROM_SHA is required}"
CANDIDATE_SHA="${REHEARSAL_CANDIDATE_SHA:?REHEARSAL_CANDIDATE_SHA is required}"
TRANSITION="${REHEARSAL_TRANSITION:?REHEARSAL_TRANSITION is required}"
COMPONENT="${REHEARSAL_COMPONENT:?REHEARSAL_COMPONENT is required}"
WEIGHT_READINESS_SCENARIO="${REHEARSAL_WEIGHT_READINESS_SCENARIO:-production_success}"
REHEARSAL_SCOPE="${REHEARSAL_SCOPE:-exact}"
REHEARSAL_PROFILE="${REHEARSAL_PROFILE:-prepush}"
RUN_ORDINAL="${REHEARSAL_RUN_ORDINAL:-1}"
GATEWAY_WORKER_FLEET_MODE="${REHEARSAL_GATEWAY_WORKER_FLEET_MODE:-active}"
case "$GATEWAY_WORKER_FLEET_MODE" in
  active)
    GATEWAY_DEFER_WORKER_FLEETS=""
    ;;
  deferred)
    GATEWAY_DEFER_WORKER_FLEETS="all"
    ;;
  *)
    echo "ERROR: invalid rehearsal gateway worker fleet mode" >&2
    exit 2
    ;;
esac

if ! [[ "$RUN_ORDINAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: REHEARSAL_RUN_ORDINAL must be a positive integer" >&2
  exit 2
fi

case "$COMPONENT" in
  gateway|validator|workflow) ;;
  *)
    echo "ERROR: REHEARSAL_COMPONENT must be gateway, validator, or workflow" >&2
    exit 2
    ;;
esac

export REHEARSAL_STATE_ROOT=/rehearsal-state
mkdir -p "$REHEARSAL_STATE_ROOT" /harness/bin /home/ec2-user /evidence

BOUNDARY_SERVICE_PID=""
GATEWAY_ENCLAVE_SERVICE_PIDS=""
VALIDATOR_ENCLAVE_SERVICE_PID=""
TLS_PROXY_SERVICE_PID=""
preserve_rehearsal_evidence() {
  if [ -f "$REHEARSAL_STATE_ROOT/events.jsonl" ]; then
    cp "$REHEARSAL_STATE_ROOT/events.jsonl" \
      "/evidence/${RUN_ORDINAL}-${COMPONENT}-${TRANSITION}-${CANDIDATE_SHA}-events.jsonl" \
      2>/dev/null || true
  fi
  if [ "$COMPONENT" = "gateway" ] \
    && [ -f /home/ec2-user/gateway/gateway.log ]; then
    cp /home/ec2-user/gateway/gateway.log \
      "/evidence/${RUN_ORDINAL}-gateway-${TRANSITION}-${CANDIDATE_SHA}.log" \
      2>/dev/null || true
  fi
}
cleanup_boundary_service() {
  if [ -n "$GATEWAY_ENCLAVE_SERVICE_PIDS" ]; then
    for pid in $GATEWAY_ENCLAVE_SERVICE_PIDS; do
      kill "$pid" 2>/dev/null || true
    done
    for pid in $GATEWAY_ENCLAVE_SERVICE_PIDS; do
      wait "$pid" 2>/dev/null || true
    done
  fi
  if [ -n "$VALIDATOR_ENCLAVE_SERVICE_PID" ]; then
    kill "$VALIDATOR_ENCLAVE_SERVICE_PID" 2>/dev/null || true
    wait "$VALIDATOR_ENCLAVE_SERVICE_PID" 2>/dev/null || true
  fi
  if [ -n "$BOUNDARY_SERVICE_PID" ]; then
    kill "$BOUNDARY_SERVICE_PID" 2>/dev/null || true
    wait "$BOUNDARY_SERVICE_PID" 2>/dev/null || true
  fi
  if [ -n "$TLS_PROXY_SERVICE_PID" ]; then
    kill "$TLS_PROXY_SERVICE_PID" 2>/dev/null || true
    wait "$TLS_PROXY_SERVICE_PID" 2>/dev/null || true
  fi
}
finalize_rehearsal() {
  local status=$?
  trap - EXIT INT TERM
  set +e
  preserve_rehearsal_evidence
  cleanup_boundary_service
  exit "$status"
}
trap finalize_rehearsal EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$COMPONENT" = "workflow" ]; then
  case "$REHEARSAL_PROFILE" in
    prepush|release) ;;
    *)
      echo "ERROR: REHEARSAL_PROFILE must be prepush or release" >&2
      exit 2
      ;;
  esac
  exec /usr/bin/python3.11 /harness/production_workflow_runner.py \
    --profile "$REHEARSAL_PROFILE" \
    --candidate-sha "$CANDIDATE_SHA" \
    --epochs "${REHEARSAL_EPOCHS:?REHEARSAL_EPOCHS is required}" \
    --fixture /harness/fixtures/production_shaped_v2.json \
    --boundary-contract /harness/boundary_contract.json \
    --output /evidence/workflow.json
fi

make_adapter() {
  local command="$1"
  cat >"/harness/bin/$command" <<EOF
#!/bin/bash
exec /usr/bin/python3.11 /harness/contract_adapter.py "$command" "\$@"
EOF
  chmod 755 "/harness/bin/$command"
}

for command in \
  aws docker nitro-cli systemctl curl sudo df getconf awk sleep ss ctr nsenter \
  pgrep pkill python3 python3.11 bash; do
  make_adapter "$command"
done

export PATH="/harness/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="/harness${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /home/ec2-user/venv311/bin
ln -sf /harness/bin/python3 /home/ec2-user/venv311/bin/python3

git config --global user.email "restart-rehearsal@leadpoet.invalid"
git config --global user.name "Leadpoet Restart Rehearsal"
git config --global --add safe.directory '*'

git -C /source cat-file -e "$FROM_SHA^{commit}"
git -C /source cat-file -e "$CANDIDATE_SHA^{commit}"
case "$TRANSITION" in
  forward)
    git -C /source merge-base --is-ancestor "$FROM_SHA" "$CANDIDATE_SHA"
    ;;
  rollback)
    test "$FROM_SHA" != "$CANDIDATE_SHA"
    git -C /source merge-base --is-ancestor "$CANDIDATE_SHA" "$FROM_SHA"
    ;;
  *)
    echo "ERROR: unsupported restart transition: $TRANSITION" >&2
    exit 1
    ;;
esac

git init --bare -q /srv/origin.git
if [ "$TRANSITION" = "rollback" ]; then
  git --git-dir=/srv/origin.git fetch -q /source \
    "$FROM_SHA:refs/heads/main"
  git --git-dir=/srv/origin.git fetch -q /source \
    "$CANDIDATE_SHA:refs/heads/rehearsal-target"
else
  git --git-dir=/srv/origin.git fetch -q /source \
    "$CANDIDATE_SHA:refs/heads/main"
  git --git-dir=/srv/origin.git fetch -q /source \
    "$FROM_SHA:refs/heads/rehearsal-deployed"
fi

mkdir -p \
  /home/ec2-user/.config/leadpoet \
  /home/ec2-user/.config/leadpoet/env-backups \
  /home/ec2-user/.config/leadpoet/v2 \
  /home/ec2-user/.cache/leadpoet-v2-artifacts \
  /home/ec2-user/gateway/secrets \
  /home/ec2-user/tee \
  /var/lib/docker/overlay2 \
  /var/lib/containerd \
  /etc/nitro_enclaves

printf '%s\n' "rehearsal-private-key" \
  >/home/ec2-user/gateway/secrets/gateway_private_key.pem
printf '%s\n' '{}' \
  >/home/ec2-user/gateway/secrets/arweave_keyfile.json
chmod 600 /home/ec2-user/gateway/secrets/*

git -C /source show \
  "$FROM_SHA:config/stateful-epoch-cutover-sn71.json" \
  >/home/ec2-user/.config/leadpoet/stateful-epoch-cutover.json
PYTHONPATH="/source:/harness" /usr/bin/python3.11 - <<'PY'
import hashlib
import json
from pathlib import Path

from bittensor_wallet import Keypair
from validator_tee.enclave.hotkey_authority_v2 import (
    HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
    MEASURED_DRAND_LIBRARY_PATH,
)
from validator_tee.host.hotkey_bootstrap_v2 import build_hotkey_envelope_v2

root = Path("/home/ec2-user/.config/leadpoet")
seed = hashlib.sha256(b"leadpoet-local-validator-hotkey").digest()
keypair = Keypair.create_from_seed(
    "0x" + seed.hex(),
)
hotkey = keypair.ss58_address
public_key = keypair.public_key.hex()
drand_hash = Path(
    "/source/validator_tee/enclave/libbittensor_drand_v2.sha256"
).read_text(encoding="ascii").strip()
configuration = {
    "schema_version": HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
    "validator_hotkey": hotkey,
    "hotkey_public_key": public_key,
    "chain_signing_profile_hash": "sha256:" + "0" * 64,
    "drand_library_path": MEASURED_DRAND_LIBRARY_PATH,
    "drand_library_sha256": drand_hash,
}
envelope = build_hotkey_envelope_v2(
    validator_hotkey=hotkey,
    hotkey_public_key=public_key,
    seed=seed,
    kms_key_id="rehearsal-validator-kms",
    encryption_context={
        "leadpoet:purpose": "validator-hotkey-v2",
        "leadpoet:validator_hotkey": hotkey,
    },
)
for name, value in (
    ("validator-hotkey-config-v2.json", configuration),
    ("validator-hotkey-envelope-v2.json", envelope),
):
    path = root / name
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
PY

if [ "$COMPONENT" = "gateway" ] || [ "$COMPONENT" = "validator" ]; then
  /usr/bin/python3.11 /harness/gateway_boundary_service.py \
    --host 127.0.0.1 \
    --port 54321 \
    --fixture /harness/fixtures/production_shaped_v2.json \
    --source-root /source \
    --state-root "$REHEARSAL_STATE_ROOT" &
  BOUNDARY_SERVICE_PID=$!
  for _attempt in $(seq 1 100); do
    [ -f "$REHEARSAL_STATE_ROOT/local-postgrest.ready" ] && break
    kill -0 "$BOUNDARY_SERVICE_PID" 2>/dev/null || {
      echo "ERROR: strict local PostgREST service exited during startup" >&2
      exit 1
    }
    /bin/sleep 0.05
  done
  if [ ! -f "$REHEARSAL_STATE_ROOT/local-postgrest.ready" ]; then
    echo "ERROR: strict local PostgREST service did not become ready" >&2
    exit 1
  fi
fi

if [ "$COMPONENT" = "validator" ]; then
  test -s /opt/leadpoet/drand-cabi-v2/libbittensor_drand_v2.so
  mkdir -p /app/validator_tee/enclave
  install -m 0444 \
    /opt/leadpoet/drand-cabi-v2/libbittensor_drand_v2.so \
    /app/validator_tee/enclave/libbittensor_drand_v2.so
  DRAND_LIBRARY_SHA256="$(
    tr -d '[:space:]' \
      </source/validator_tee/enclave/libbittensor_drand_v2.sha256
  )"
  echo \
    "$DRAND_LIBRARY_SHA256  /app/validator_tee/enclave/libbittensor_drand_v2.so" \
    | sha256sum -c -

  PYTHONPATH="/source:/harness" \
    /usr/bin/python3.11 /harness/validator_enclave_service.py &
  VALIDATOR_ENCLAVE_SERVICE_PID=$!
  for _attempt in $(seq 1 100); do
    [ -S "$REHEARSAL_STATE_ROOT/validator-enclave.sock" ] \
      && [ -f "$REHEARSAL_STATE_ROOT/validator-enclave.ready" ] \
      && break
    kill -0 "$VALIDATOR_ENCLAVE_SERVICE_PID" 2>/dev/null || {
      echo "ERROR: persistent validator enclave service exited during startup" >&2
      exit 1
    }
    /bin/sleep 0.05
  done
  if [ ! -S "$REHEARSAL_STATE_ROOT/validator-enclave.sock" ]; then
    echo "ERROR: persistent validator enclave service did not become ready" >&2
    exit 1
  fi
fi

FIXTURE_SEED_ROOT=/rehearsal-fixture-seed
if [ -d "$FIXTURE_SEED_ROOT" ]; then
  /usr/bin/python3.11 - "$FIXTURE_SEED_ROOT" "$CANDIDATE_SHA" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidate = sys.argv[2]
manifest = json.loads(
    (root / "fixture-seed.json").read_text(encoding="utf-8")
)
release = json.loads(
    (root / "release-build-input.json").read_text(encoding="utf-8")
)
if (
    manifest
    != {
        "schema_version": "leadpoet.local_fixture_seed.v1",
        "candidate_sha": candidate,
    }
    or release.get("commit_sha") != candidate
    or not (root / "config-v2/acceptance-corpus-v2.json").is_file()
    or not (root / "validator-app").is_dir()
    or not (
        root / "gateway-attested-runtime/scoring_import_closure.json"
    ).is_file()
    or {
        path.name
        for path in (root / "gateway-enclave-build-identities").glob("*.json")
    }
    != {
        "gateway_autoresearch.json",
        "gateway_coordinator.json",
        "gateway_scoring.json",
    }
):
    raise SystemExit("sanitized fixture seed differs from the target release")
PY
  cp -a "$FIXTURE_SEED_ROOT/config-v2/." \
    /home/ec2-user/.config/leadpoet/v2/
  cp -a "$FIXTURE_SEED_ROOT/release-build-input.json" \
    "$REHEARSAL_STATE_ROOT/release-build-input.json"
  cp -a "$FIXTURE_SEED_ROOT/validator-app" \
    "$REHEARSAL_STATE_ROOT/validator-app"
  cp -a "$FIXTURE_SEED_ROOT/gateway-enclave-build-identities" \
    "$REHEARSAL_STATE_ROOT/gateway-enclave-build-identities"
  cp -a "$FIXTURE_SEED_ROOT/gateway-attested-runtime" \
    "$REHEARSAL_STATE_ROOT/gateway-attested-runtime"
else
  PYTHONPATH="/source:/harness" \
    /usr/bin/python3.11 /harness/prepare_host_fixtures.py \
    --output-dir /home/ec2-user/.config/leadpoet/v2 \
    --candidate-sha "$CANDIDATE_SHA"
fi

if [ "$COMPONENT" = "gateway" ]; then
  PYTHONPATH="" /usr/bin/python3.11 \
    /harness/tls_connect_proxy_service.py &
  TLS_PROXY_SERVICE_PID=$!
  for _attempt in $(seq 1 200); do
    [ -f "$REHEARSAL_STATE_ROOT/tls-connect-proxy.ready" ] && break
    kill -0 "$TLS_PROXY_SERVICE_PID" 2>/dev/null || {
      echo "ERROR: rehearsal TLS CONNECT proxy exited during startup" >&2
      exit 1
    }
    /bin/sleep 0.05
  done
  if [ ! -f "$REHEARSAL_STATE_ROOT/tls-connect-proxy.ready" ]; then
    echo "ERROR: rehearsal TLS CONNECT proxy did not become ready" >&2
    exit 1
  fi
  CERTIFI_BUNDLE="$(
    PYTHONPATH="" /usr/bin/python3.11 -c 'import certifi; print(certifi.where())'
  )"
  cat "$REHEARSAL_STATE_ROOT/tls-connect-proxy-ca.pem" >>"$CERTIFI_BUNDLE"

  echo "Materializing the exact measured gateway-enclave filesystem"
  rm -rf /app/gateway
  mkdir -p /app
  cp -a /source/gateway /app/gateway
  rm -rf /app/gateway/_attested_runtime
  cp -a "$REHEARSAL_STATE_ROOT/gateway-attested-runtime" \
    /app/gateway/_attested_runtime
  RUNSC_ARTIFACT_NAME="$(
    /usr/bin/python3.11 - <<'PY'
import json
print(
    json.load(
        open(
            "/app/gateway/tee/runsc-runtime.lock.json",
            encoding="utf-8",
        )
    )["artifact_filename"]
)
PY
  )"
  install -m 0555 \
    "/opt/leadpoet/external-artifacts/$RUNSC_ARTIFACT_NAME" \
    /usr/local/bin/runsc
  PYTHONPATH=/source /usr/bin/python3.11 \
    /app/gateway/tee/sandbox_runtime_artifact.py verify \
      --lock /app/gateway/tee/runsc-runtime.lock.json \
      --artifact /usr/local/bin/runsc
  PYTHONPATH=/source /usr/bin/python3.11 \
    /app/gateway/tee/sandbox_runtime_artifact.py write-rootfs-manifest \
      --lock /app/gateway/tee/runsc-runtime.lock.json \
      --requirements-lock \
        /app/gateway/tee/requirements-scoring-py39.lock \
      --python-version 3.9.24 \
      --output /leadpoet-model-rootfs.manifest.json

  for role in gateway_coordinator gateway_scoring gateway_autoresearch; do
    REHEARSAL_GATEWAY_ENCLAVE_ROLE="$role" \
      REHEARSAL_GATEWAY_CANDIDATE_ROOT="/source/gateway" \
      REHEARSAL_GATEWAY_CANONICAL_APP_ROOT="/app/gateway" \
      PYTHONPATH="/source:/harness" \
      /usr/bin/python3.11 /harness/gateway_enclave_service.py &
    pid=$!
    GATEWAY_ENCLAVE_SERVICE_PIDS="$GATEWAY_ENCLAVE_SERVICE_PIDS $pid"
    ready="$REHEARSAL_STATE_ROOT/gateway-enclave-${role}.ready"
    socket_path="$REHEARSAL_STATE_ROOT/gateway-enclave-${role}.sock"
    for _attempt in $(seq 1 600); do
      [ -S "$socket_path" ] && [ -f "$ready" ] && break
      kill -0 "$pid" 2>/dev/null || {
        echo "ERROR: persistent $role enclave service exited during startup" >&2
        exit 1
      }
      /bin/sleep 0.05
    done
    if [ ! -S "$socket_path" ]; then
      echo "ERROR: persistent $role enclave service did not become ready" >&2
      exit 1
    fi
  done
fi

if [ "$COMPONENT" = "gateway" ]; then
  git clone -q /srv/origin.git /home/ec2-user/leadpoet_repo
  git -C /home/ec2-user/leadpoet_repo checkout -q --detach "$FROM_SHA"
  git -C /home/ec2-user/leadpoet_repo branch -f main "$FROM_SHA"
  git -C /home/ec2-user/leadpoet_repo checkout -q main
  git -C /home/ec2-user/leadpoet_repo remote set-url origin /srv/origin.git
  git -C /home/ec2-user/leadpoet_repo show "$FROM_SHA:gw_restart.sh" \
    >/home/ec2-user/gw_restart.sh
  chmod 700 /home/ec2-user/gw_restart.sh

  echo "REHEARSAL_START component=gateway from=$FROM_SHA candidate=$CANDIDATE_SHA transition=$TRANSITION scenario=$WEIGHT_READINESS_SCENARIO scope=$REHEARSAL_SCOPE"
  set +e
  if [ "$TRANSITION" = "rollback" ]; then
    env \
      HOME=/home/ec2-user \
      LEADPOET_REPO_ROOT=/home/ec2-user/leadpoet_repo \
      GATEWAY_ROOT=/home/ec2-user/leadpoet_repo/gateway \
      GATEWAY_LOG_ROOT=/home/ec2-user/gateway \
      GATEWAY_LOG_FILE=/home/ec2-user/gateway/gateway.log \
      GATEWAY_HOST_RESTART_SCRIPT=/home/ec2-user/gw_restart.sh \
      GATEWAY_TEE_EIF_ROOT=/home/ec2-user/tee \
      GATEWAY_PYTHON_BIN=/home/ec2-user/venv311/bin/python3 \
      GATEWAY_TEE_TOPOLOGY_MODE=full \
      RESEARCH_LAB_TEE_PROTOCOL=v2 \
      GATEWAY_V2_DEFER_WORKER_FLEETS="$GATEWAY_DEFER_WORKER_FLEETS" \
      bash /home/ec2-user/gw_restart.sh --commit "$CANDIDATE_SHA"
  else
    env \
      HOME=/home/ec2-user \
      LEADPOET_REPO_ROOT=/home/ec2-user/leadpoet_repo \
      GATEWAY_ROOT=/home/ec2-user/leadpoet_repo/gateway \
      GATEWAY_LOG_ROOT=/home/ec2-user/gateway \
      GATEWAY_LOG_FILE=/home/ec2-user/gateway/gateway.log \
      GATEWAY_HOST_RESTART_SCRIPT=/home/ec2-user/gw_restart.sh \
      GATEWAY_TEE_EIF_ROOT=/home/ec2-user/tee \
      GATEWAY_PYTHON_BIN=/home/ec2-user/venv311/bin/python3 \
      GATEWAY_TEE_TOPOLOGY_MODE=full \
      RESEARCH_LAB_TEE_PROTOCOL=v2 \
      GATEWAY_V2_DEFER_WORKER_FLEETS="$GATEWAY_DEFER_WORKER_FLEETS" \
      bash /home/ec2-user/gw_restart.sh
  fi
  RESTART_STATUS=$?
  set -e

  if [ "$WEIGHT_READINESS_SCENARIO" = "plaintext_proxy_rejected" ]; then
    if [ "$RESTART_STATUS" -ne 75 ]; then
      echo "ERROR: plaintext proxy regression did not fail closed with status 75" >&2
      exit 1
    fi
    /usr/bin/python3.11 - "$CANDIDATE_SHA" <<'PY'
import json
import sys
from pathlib import Path

candidate = sys.argv[1]
deployment = json.loads(
    Path(
        "/home/ec2-user/.config/leadpoet/deployments/gateway-current.json"
    ).read_text()
)
installed_controller = Path("/home/ec2-user/gw_restart.sh").read_text()
envelope_stage = 'GATEWAY_DEPLOY_STAGE="v2_credential_envelope_preparation"'
envelope_call = (
    "run_prepared_gateway_module "
    "gateway.tee.prepare_gateway_envelopes_v2"
)
expected_stage = (
    "v2_credential_envelope_preparation"
    if envelope_stage in installed_controller
    and installed_controller.index(envelope_stage)
    < installed_controller.index(envelope_call)
    else "git_prepared_tree_verification"
)
if deployment.get("target_sha") != candidate:
    raise SystemExit("failed proxy preflight was not bound to the candidate")
if deployment.get("status") != "failed":
    raise SystemExit("failed proxy preflight did not record a failed deployment")
if deployment.get("stage") != expected_stage:
    raise SystemExit("failed proxy preflight recorded the wrong restart stage")
transition = Path(
    "/home/ec2-user/.config/leadpoet/v2/gateway-v2-env-transition.json"
)
if transition.exists():
    raise SystemExit("plaintext proxy failure wrote a V2 transition report")
PY
    preserve_rehearsal_evidence
    echo "TARGETED_RESTART_REGRESSION_SUCCESS component=gateway candidate=$CANDIDATE_SHA scenario=$WEIGHT_READINESS_SCENARIO"
    exit 0
  fi
  if [ "$WEIGHT_READINESS_SCENARIO" != "production_success" ]; then
    echo "ERROR: full restart rehearsal received a targeted-only scenario: $WEIGHT_READINESS_SCENARIO" >&2
    exit 1
  fi
  if [ "$RESTART_STATUS" -ne 0 ]; then
    echo "REHEARSAL_FAILURE_DIAGNOSTICS component=gateway status=$RESTART_STATUS" >&2
    for endpoint in /research-lab/status /attest; do
      body_file="$(mktemp)"
      http_status="$(
        /usr/bin/curl \
          --silent \
          --show-error \
          --max-time 10 \
          --output "$body_file" \
          --write-out '%{http_code}' \
          "http://127.0.0.1:8000${endpoint}" || true
      )"
      echo "REHEARSAL_HTTP_DIAGNOSTIC endpoint=$endpoint status=${http_status:-curl_failed}" >&2
      head -c 4096 "$body_file" >&2 || true
      echo >&2
      rm -f "$body_file"
    done
    if [ -f /home/ec2-user/gateway/gateway.log ]; then
      echo "REHEARSAL_GATEWAY_LOG_TAIL_BEGIN" >&2
      tail -n 200 /home/ec2-user/gateway/gateway.log >&2
      echo "REHEARSAL_GATEWAY_LOG_TAIL_END" >&2
    fi
    if [ -f "$REHEARSAL_STATE_ROOT/local-postgrest-events.jsonl" ]; then
      echo "REHEARSAL_POSTGREST_EVENT_TAIL_BEGIN" >&2
      tail -n 100 "$REHEARSAL_STATE_ROOT/local-postgrest-events.jsonl" >&2
      echo "REHEARSAL_POSTGREST_EVENT_TAIL_END" >&2
    fi
    echo "ERROR: exact gateway launcher failed" >&2
    exit "$RESTART_STATUS"
  fi

  test "$(git -C /home/ec2-user/leadpoet_repo rev-parse HEAD)" = "$CANDIDATE_SHA"
  test "$(git -C /home/ec2-user/leadpoet_repo status --porcelain)" = ""
  EXPECTED_CONTROLLER_SHA="$CANDIDATE_SHA"
  if [ "$TRANSITION" = "rollback" ]; then
    EXPECTED_CONTROLLER_SHA="$FROM_SHA"
  fi
  test "$(git -C /home/ec2-user/leadpoet_repo show "$EXPECTED_CONTROLLER_SHA:gw_restart.sh" | sha256sum | cut -d' ' -f1)" = \
    "$(sha256sum /home/ec2-user/gw_restart.sh | cut -d' ' -f1)"
  python3 - "$CANDIDATE_SHA" <<'PY'
import json
import sys
from pathlib import Path

candidate = sys.argv[1]
build_info = json.loads(
    Path("/home/ec2-user/leadpoet_repo/gateway/BUILD_INFO.json").read_text()
)
if build_info.get("git_commit") != candidate:
    raise SystemExit("gateway build info does not match candidate")
deployment = json.loads(
    Path(
        "/home/ec2-user/.config/leadpoet/deployments/gateway-current.json"
    ).read_text()
)
if deployment.get("target_sha") != candidate or deployment.get("status") != "succeeded":
    raise SystemExit("gateway deployment record is not successful")
PY
else
  echo \
    "$DRAND_LIBRARY_SHA256  /app/validator_tee/enclave/libbittensor_drand_v2.so" \
    | sha256sum -c -
  mkdir -p /home/ec2-user/leadpoet
  git clone -q /srv/origin.git /home/ec2-user/leadpoet/leadpoet
  git -C /home/ec2-user/leadpoet/leadpoet checkout -q --detach "$FROM_SHA"
  git -C /home/ec2-user/leadpoet/leadpoet branch -f main "$FROM_SHA"
  git -C /home/ec2-user/leadpoet/leadpoet checkout -q main
  git -C /home/ec2-user/leadpoet/leadpoet remote set-url origin /srv/origin.git
  printf '%s\n' \
    '# Sanitized production-shaped fallback; inherited V2 env remains authoritative.' \
    >/home/ec2-user/leadpoet/leadpoet/.env
  chmod 600 /home/ec2-user/leadpoet/leadpoet/.env
  git -C /home/ec2-user/leadpoet/leadpoet show "$FROM_SHA:validator_restart.sh" \
    >/home/ec2-user/validator_restart.sh
  chmod 700 /home/ec2-user/validator_restart.sh

  mkdir -p \
    /home/ec2-user/.bittensor/wallets/validator_72/hotkeys \
    /home/ec2-user/.bittensor/wallets/validator_72
  printf '%s\n' \
    '{"ss58Address":"5CUxhqZ2ewLA61PtdKYzdnLXq1jyFxsvjMg8mRsim4Ni8T3p"}' \
    >/home/ec2-user/.bittensor/wallets/validator_72/coldkeypub.txt

  echo "REHEARSAL_START component=validator from=$FROM_SHA candidate=$CANDIDATE_SHA transition=$TRANSITION"
  if [ "$TRANSITION" = "rollback" ]; then
    env \
      HOME=/home/ec2-user \
      VALIDATOR_ROOT=/home/ec2-user/leadpoet/leadpoet \
      VALIDATOR_PYTHON_BIN=/home/ec2-user/venv311/bin/python3 \
      VALIDATOR_DOCKER_MIN_FREE_BYTES=1000000000 \
      bash /home/ec2-user/validator_restart.sh \
        --commit "$CANDIDATE_SHA"
  else
    env \
      HOME=/home/ec2-user \
      VALIDATOR_ROOT=/home/ec2-user/leadpoet/leadpoet \
      VALIDATOR_PYTHON_BIN=/home/ec2-user/venv311/bin/python3 \
      VALIDATOR_DOCKER_MIN_FREE_BYTES=1000000000 \
      bash /home/ec2-user/validator_restart.sh
  fi

  test "$(git -C /home/ec2-user/leadpoet/leadpoet rev-parse HEAD)" = "$CANDIDATE_SHA"
  test "$(
    git -C /home/ec2-user/leadpoet/leadpoet \
      status --porcelain --untracked-files=no
  )" = ""
  EXPECTED_CONTROLLER_SHA="$CANDIDATE_SHA"
  if [ "$TRANSITION" = "rollback" ]; then
    EXPECTED_CONTROLLER_SHA="$FROM_SHA"
  fi
  test "$(
    git -C /home/ec2-user/leadpoet/leadpoet \
      show "$EXPECTED_CONTROLLER_SHA:validator_restart.sh" \
      | sha256sum | cut -d' ' -f1
  )" = "$(
    sha256sum /home/ec2-user/validator_restart.sh | cut -d' ' -f1
  )"
fi

preserve_rehearsal_evidence

/usr/bin/python3.11 /harness/verify_evidence.py \
  "$COMPONENT" "$FROM_SHA" "$CANDIDATE_SHA" "$WEIGHT_READINESS_SCENARIO" "$REHEARSAL_SCOPE" "$TRANSITION" \
  | tee "/evidence/${RUN_ORDINAL}-${COMPONENT}-${TRANSITION}-${CANDIDATE_SHA}.json"
if [ "$REHEARSAL_SCOPE" = "exact" ]; then
  echo "REHEARSAL_SUCCESS component=$COMPONENT candidate=$CANDIDATE_SHA"
else
  echo "TARGETED_RESTART_REGRESSION_SUCCESS component=$COMPONENT candidate=$CANDIDATE_SHA scenario=$WEIGHT_READINESS_SCENARIO"
fi
