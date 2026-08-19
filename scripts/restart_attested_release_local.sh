#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_KEY="${LEADPOET_GATEWAY_SSH_KEY:-$HOME/Downloads/leadpoet-2026-07-28.pem}"
VALIDATOR_KEY="${LEADPOET_VALIDATOR_SSH_KEY:-$HOME/Downloads/leadpoet-2026-07-28.pem}"
GATEWAY_HOST="${LEADPOET_GATEWAY_SSH_HOST:-ec2-user@52.91.135.79}"
VALIDATOR_HOST="${LEADPOET_VALIDATOR_SSH_HOST:-ec2-user@100.59.201.156}"
GATEWAY_RESTART="${LEADPOET_GATEWAY_RESTART_PATH:-/home/ec2-user/gw_restart.sh}"
VALIDATOR_RESTART="${LEADPOET_VALIDATOR_RESTART_PATH:-/home/ec2-user/validator_restart.sh}"
GATEWAY_REPO_ROOT="${LEADPOET_GATEWAY_REPO_ROOT:-/home/ec2-user/leadpoet_repo}"
GATEWAY_PYTHON_BIN="${LEADPOET_GATEWAY_PYTHON_BIN:-/home/ec2-user/venv311/bin/python3}"
VALIDATOR_REPO_ROOT="${LEADPOET_VALIDATOR_REPO_ROOT:-/home/ec2-user/leadpoet/leadpoet}"
GATEWAY_DEPLOY_READINESS_PATH="${LEADPOET_GATEWAY_DEPLOY_READINESS_PATH:-/home/ec2-user/gateway/deploy_readiness.json}"
GATEWAY_ENV_SECRET_ID="${LEADPOET_GATEWAY_ENV_SECRET_ID:-}"
VALIDATOR_ENV_SECRET_ID="${LEADPOET_VALIDATOR_ENV_SECRET_ID:-}"
RELEASE_PREFIX="${LEADPOET_RELEASE_PREFIX:-attested-v2/releases}"
# Three-second retries allow up to 2.5 hours for the gateway rebuild, plus a
# five-minute margin for the final bounded release probes.
VALIDATOR_COORDINATION_ATTEMPTS=3000
VALIDATOR_COORDINATION_TIMEOUT_SECONDS=9300
VALIDATOR_FAILURE_CLEANUP_ATTEMPTS=60
VALIDATOR_FAILURE_MARKER_ATTEMPTS=20

commit=""
component="all"
validator_job=""
failure_marker_job=""
temporary_root=""
coordination_file=""
gateway_observation=""
gateway_evidence=""
validator_observation=""
validator_evidence=""
transition_manifest=""
final_manifest=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/restart_attested_release_local.sh \
    --commit <full-40-character-sha> \
    [--component all|gateway|validator]

The default "all" mode starts both exact-commit restarts in one invocation.
A single-component restart is accepted only when the other component is
already running the selected commit.
EOF
}

cleanup() {
  local status="$?"
  local failure_marker_command=""
  set +e
  if [ -n "$validator_job" ] && kill -0 "$validator_job" 2>/dev/null; then
    # Cancellation is the authority revocation. Signal the remote validator
    # before attempting the durable failure marker so a slow SSH connection
    # cannot delay cleanup past the late activation boundary.
    kill -TERM "$validator_job" 2>/dev/null || true
    if [ -n "$coordination_file" ]; then
      echo "Signaling the paired validator to clean up after restart failure" >&2
      failure_marker_command="$(
        coordination_remote_command "failed:$commit"
      )"
      ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
        "$failure_marker_command" >/dev/null 2>&1 &
      failure_marker_job="$!"
    fi
    for _ in $(seq 1 "$VALIDATOR_FAILURE_CLEANUP_ATTEMPTS"); do
      if ! kill -0 "$validator_job" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$validator_job" 2>/dev/null; then
      kill -KILL "$validator_job" 2>/dev/null || true
    fi
    wait "$validator_job" 2>/dev/null || true
  fi
  if [ -n "$failure_marker_job" ]; then
    for _ in $(seq 1 "$VALIDATOR_FAILURE_MARKER_ATTEMPTS"); do
      if ! kill -0 "$failure_marker_job" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$failure_marker_job" 2>/dev/null; then
      kill -TERM "$failure_marker_job" 2>/dev/null || true
      sleep 1
    fi
    if kill -0 "$failure_marker_job" 2>/dev/null; then
      kill -KILL "$failure_marker_job" 2>/dev/null || true
    fi
    wait "$failure_marker_job" 2>/dev/null || true
    failure_marker_job=""
  fi
  if [ -n "$coordination_file" ]; then
    ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
      "rm -f -- '$coordination_file'" >/dev/null 2>&1 || true
  fi
  if [ -n "$temporary_root" ]; then
    rm -rf -- "$temporary_root"
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while [ "$#" -gt 0 ]; do
  case "$1" in
    --commit)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --commit requires a full 40-character SHA" >&2
        exit 2
      fi
      commit="$2"
      shift 2
      ;;
    --commit=*)
      commit="${1#--commit=}"
      shift
      ;;
    --component)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --component requires all, gateway, or validator" >&2
        exit 2
      fi
      component="$2"
      shift 2
      ;;
    --component=*)
      component="${1#--component=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unsupported attested restart argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: --commit must be a lowercase full 40-character SHA" >&2
  exit 2
fi
case "$component" in
  all|gateway|validator) ;;
  *)
    echo "ERROR: --component must be all, gateway, or validator" >&2
    exit 2
    ;;
esac
for key in "$GATEWAY_KEY" "$VALIDATOR_KEY"; do
  if [ ! -r "$key" ]; then
    echo "ERROR: SSH key is unavailable: $key" >&2
    exit 1
  fi
done
for secret_id in "$GATEWAY_ENV_SECRET_ID" "$VALIDATOR_ENV_SECRET_ID"; do
  if [ -n "$secret_id" ] \
      && ! [[ "$secret_id" =~ ^[A-Za-z0-9/_+=.@-]+$ ]]; then
    echo "ERROR: environment secret id contains unsupported characters" >&2
    exit 2
  fi
done
for remote_path in \
  "$GATEWAY_REPO_ROOT" \
  "$GATEWAY_PYTHON_BIN" \
  "$GATEWAY_DEPLOY_READINESS_PATH"; do
  if ! [[ "$remote_path" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "ERROR: readiness authority path contains unsupported characters" >&2
    exit 2
  fi
done
case "$RELEASE_PREFIX" in
  attested-v2/releases|attested-v2/candidates) ;;
  *)
    echo "ERROR: release prefix is outside the reviewed channels" >&2
    exit 2
    ;;
esac

temporary_root="$(mktemp -d /tmp/leadpoet-attested-restart.XXXXXX)"
helper="$temporary_root/exact_commit_restart_v2.py"
gateway_observation="$temporary_root/gateway-readiness-observation.json"
gateway_evidence="$temporary_root/gateway-readiness-evidence.json"
validator_observation="$temporary_root/validator-readiness-observation.json"
validator_evidence="$temporary_root/validator-readiness-evidence.json"
transition_manifest="$temporary_root/deploy-readiness-transition.json"
final_manifest="$temporary_root/deploy-readiness-v2.json"

echo "Fetching current public V2 compatibility authority"
git -C "$ROOT" fetch origin main
branch_commit="$(git -C "$ROOT" rev-parse --verify origin/main^{commit})"
git -C "$ROOT" cat-file -e "$commit^{commit}"
git -C "$ROOT" show \
  origin/main:Leadpoet/utils/exact_commit_restart_v2.py > "$helper"
python3 "$helper" \
  --repo-root "$ROOT" \
  --selected-commit "$commit" \
  --branch-ref origin/main
echo "Selected release is compatible with current public auditors: $commit"
echo "Current public V2 authority commit: $branch_commit"

ssh_common=(
  -n
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=20
)

install_gateway_readiness_manifest() {
  local action="$1"
  local source="$2"
  local payload=""
  case "$action" in
    readiness-transition|readiness-final) ;;
    *)
      echo "ERROR: unsupported deploy readiness install action" >&2
      return 2
      ;;
  esac
  payload="$(base64 < "$source" | tr -d '\n')"
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "python3 - '$GATEWAY_DEPLOY_READINESS_PATH' '$payload' '$action' <<'PY'
import base64
import json
import os
from pathlib import Path
import sys
import tempfile

destination = Path(sys.argv[1])
document = json.loads(base64.b64decode(sys.argv[2], validate=True))
action = sys.argv[3]
if action not in {'readiness-transition', 'readiness-final'}:
    raise SystemExit('deploy readiness install action is invalid')
if not isinstance(document, dict) or document.get('enforce_resume_block') is not True:
    raise SystemExit('deploy readiness payload is not fail closed')
if action == 'readiness-transition' and document.get('ok') is not False:
    raise SystemExit('deploy readiness transition payload is invalid')
if action == 'readiness-final' and (
    document.get('schema_version') != 'leadpoet.deploy_readiness.v2'
    or document.get('ok') is not True
):
    raise SystemExit('deploy readiness final payload is invalid')
destination.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix='.' + destination.name + '.',
    dir=str(destination.parent),
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(
            (json.dumps(document, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii')
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
PY"
}

invalidate_deploy_readiness() {
  PYTHONPATH="$ROOT" python3 - "$commit" "$transition_manifest" <<'PY'
import sys
from gateway.deploy_readiness import (
    build_deploy_readiness_transition_marker,
    write_deploy_readiness_manifest,
)

write_deploy_readiness_manifest(
    build_deploy_readiness_transition_marker(expected_commit=sys.argv[1]),
    sys.argv[2],
)
PY
  install_gateway_readiness_manifest readiness-transition "$transition_manifest"
  echo "Installed fail-closed deploy readiness transition marker for $commit"
}

coordination_remote_command() {
  local value="$1"
  case "$value" in
    "$commit"|"failed:$commit") ;;
    *)
      echo "ERROR: invalid coordinated restart marker value" >&2
      return 2
      ;;
  esac
  printf '%s\n' \
    "set -Eeuo pipefail
     umask 077
     marker='$coordination_file.tmp'
     printf '%s\\n' '$value' > \"\$marker\"
     mv -f -- \"\$marker\" '$coordination_file'"
}

publish_coordination_value() {
  local remote_command
  remote_command="$(coordination_remote_command "$1")"
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "$remote_command"
}

gateway_active_commit() {
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "curl -fsS --connect-timeout 5 --max-time 15 \
      http://127.0.0.1:8000/build-info \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"git_commit\"])'"
}

validator_active_commit() {
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' \
      leadpoet-validator-main \
      | sed -n 's/^VALIDATOR_V2_DEPLOY_COMMIT=//p'"
}

run_gateway_restart() {
  local secret_environment=""
  if [ -n "$GATEWAY_ENV_SECRET_ID" ]; then
    secret_environment="LEADPOET_GATEWAY_ENV_SECRET_ID='$GATEWAY_ENV_SECRET_ID'"
  fi
  ssh -tt "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "exec env $secret_environment \
      GATEWAY_V2_RELEASE_PREFIX='$RELEASE_PREFIX' \
      bash '$GATEWAY_RESTART' --commit '$commit'"
}

run_validator_restart() {
  local coordination_environment=""
  local expected_forward_commit=""
  local launcher_command=""
  local launcher_command_quoted=""
  local restart_arguments=""
  local command=()
  if [ -n "$coordination_file" ]; then
    coordination_environment="$coordination_file"
  fi
  if [ "$commit" != "$branch_commit" ]; then
    # Historical rollback releases retain the installed controller and use
    # its exact-commit compatibility handoff. A forward release at the frozen
    # public tip uses the normal N-1 -> N pull/re-exec path so the candidate
    # launcher can prepare in parallel and install itself after success.
    restart_arguments="--commit '$commit'"
    launcher_command="exec bash '$VALIDATOR_RESTART' $restart_arguments"
  else
    expected_forward_commit="$commit"
    # On the first N-1 -> N attempt, the installed launcher owns the Git
    # handoff. If a prior attempt already completed that handoff but failed
    # before installing N, retry the exact candidate launcher from its clean
    # Git blob instead of silently falling back to the stale installed copy.
    launcher_command="
      candidate_launcher='$VALIDATOR_REPO_ROOT/validator_restart.sh'
      observed_head=\$(git -C '$VALIDATOR_REPO_ROOT' rev-parse --verify HEAD)
      if [ \"\$observed_head\" = '$commit' ]; then
        if [ ! -r \"\$candidate_launcher\" ] \
            || ! git -C '$VALIDATOR_REPO_ROOT' diff --quiet '$commit' -- validator_restart.sh; then
          echo 'ERROR: selected validator launcher is not the exact candidate Git blob' >&2
          exit 1
        fi
        exec bash \"\$candidate_launcher\"
      fi
      exec bash '$VALIDATOR_RESTART'"
  fi
  printf -v launcher_command_quoted '%q' "$launcher_command"
  command=(
    ssh -tt "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST"
    "exec env \
      VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE='$coordination_environment' \
      VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS='$VALIDATOR_COORDINATION_ATTEMPTS' \
      VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS='$VALIDATOR_COORDINATION_TIMEOUT_SECONDS' \
      VALIDATOR_COORDINATED_EXPECTED_COMMIT='$expected_forward_commit' \
      LEADPOET_VALIDATOR_ENV_SECRET_ID='$VALIDATOR_ENV_SECRET_ID' \
      VALIDATOR_V2_RELEASE_PREFIX='$RELEASE_PREFIX' \
      bash -c $launcher_command_quoted"
  )
  if [ "${VALIDATOR_RESTART_EXEC_SSH:-0}" = "1" ]; then
    exec "${command[@]}"
  fi
  "${command[@]}"
}

verify_gateway_release() {
  local output="$1"
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "cd '$GATEWAY_REPO_ROOT' && PYTHONPATH='$GATEWAY_REPO_ROOT' \
      '$GATEWAY_PYTHON_BIN' - '$commit' <<'PY'
import asyncio
import json
import os
from pathlib import Path
import sys
import urllib.request

expected = sys.argv[1]
# gateway_exact_release_ready: retained as the operator/fault-injection probe label.

def get(path):
    with urllib.request.urlopen(
        'http://127.0.0.1:8000' + path,
        timeout=35,
    ) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise SystemExit('gateway response is not an object: ' + path)
    return value

health = get('/health/v2-authority')
build = get('/build-info')
release = get('/weights/v2/release-evidence/' + expected)
attestation = get('/attest')
if health.get('status') != 'ready':
    raise SystemExit('gateway V2 authority is not ready')
if str(health.get('commit_sha') or '').lower() != expected:
    raise SystemExit('gateway V2 authority commit differs')
if str(build.get('git_commit') or '').lower() != expected:
    raise SystemExit('gateway build-info commit differs')
if release.get('schema_version') != 'leadpoet.auditor_release_evidence.v2':
    raise SystemExit('gateway release evidence schema differs')
if str(release.get('commit_sha') or '').lower() != expected:
    raise SystemExit('gateway release evidence commit differs')
if (
    not isinstance(health.get('enclaves'), dict)
    or health['enclaves'].get('status') != 'ready'
):
    raise SystemExit('gateway authority endpoint lacks ready enclave evidence')

from gateway.deploy_readiness import read_source_commit
from gateway.tee.provider_broker_v2 import (
    measured_retry_policy_hashes,
    provider_registry_hash,
)
from gateway.tee.topology import ROLE_SPECS
from gateway.tee.release_channel_v2 import fetch_release_channel_v2
from gateway.tee.verify_v2_runtime_ready import verify_v2_runtime_ready
from gateway.research_lab.provider_profiles_v2 import (
    verify_required_worker_proxy_profiles_v2,
)
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
)
from gateway.utils.tee_client import TEEClient
from gateway.utils.tee_kms_provision_v2 import (
    load_provider_envelopes,
    provider_reference_hashes_from_envelopes,
)
from gateway.utils.tee_v2_bootstrap import runtime_configuration_documents

processes = []
for candidate in Path('/proc').iterdir():
    if not candidate.name.isdigit():
        continue
    try:
        command = candidate.joinpath('cmdline').read_bytes().split(b'\\0')
    except OSError:
        continue
    if b'-m' in command and b'gateway.main' in command:
        processes.append(candidate)
if len(processes) != 1:
    raise SystemExit('exactly one gateway.main process is required')
runtime_environment = {}
for item in processes[0].joinpath('environ').read_bytes().split(b'\\0'):
    if b'=' not in item:
        continue
    name, value = item.split(b'=', 1)
    runtime_environment[name.decode('utf-8')] = value.decode('utf-8')
os.environ.clear()
os.environ.update(runtime_environment)

config_dir = Path(
    runtime_environment.get('GATEWAY_V2_CONFIG_DIR')
    or '/home/ec2-user/.config/leadpoet/v2'
)
gateway_release_path = Path(
    runtime_environment.get('GATEWAY_V2_RELEASE_MANIFEST')
    or '/home/ec2-user/tee/gateway-v2-release-manifest.json'
)
lineage_path = Path(
    runtime_environment.get('GATEWAY_V2_RELEASE_LINEAGE')
    or '/home/ec2-user/tee/gateway-v2-release-lineage.json'
)
artifact_policy_path = Path(
    runtime_environment.get('GATEWAY_V2_ARTIFACT_POLICY')
    or config_dir / 'encrypted-artifact-policy.json'
)
gateway_root = Path(
    runtime_environment.get('GATEWAY_ROOT')
    or '$GATEWAY_REPO_ROOT/gateway'
)
gateway_release = json.loads(gateway_release_path.read_text(encoding='utf-8'))
lineage = json.loads(lineage_path.read_text(encoding='utf-8'))
channel = fetch_release_channel_v2(
    bucket=(
        runtime_environment.get('GATEWAY_V2_RELEASE_BUCKET')
        or 'leadpoet-attested-v2-artifacts-493765492819'
    ),
    commit_sha=expected,
    prefix=(
        runtime_environment.get('GATEWAY_V2_RELEASE_PREFIX')
        or 'attested-v2/releases'
    ),
)
if gateway_release != channel['gateway_release_manifest']:
    raise SystemExit('active gateway release differs from immutable channel')
validator_release = channel['validator_release_manifest']
envelopes = load_provider_envelopes([
    config_dir / name
    for name in (
        'artifact_master_key.json',
        'openrouter.json',
        'exa.json',
        'scrapingdog.json',
        'deepline.json',
        'supabase_service_role.json',
        'truelist.json',
    )
])
artifact_envelopes = [
    item for item in envelopes
    if item.get('credential_slot') == 'artifact_master_key'
]
if len(artifact_envelopes) != 1:
    raise SystemExit('gateway artifact master-key envelope is not unique')
protected = json.loads(
    (gateway_root / '_attested_runtime/protected_workflows.json').read_text(
        encoding='utf-8'
    )
)
protected_hash = str(protected.get('manifest_hash') or '').lower()
profiles = verify_required_worker_proxy_profiles_v2(config_dir=config_dir)
execution_config = build_research_lab_execution_config(
    environment=runtime_environment
)
documents = runtime_configuration_documents(
    release_manifest=gateway_release,
    gateway_release_lineage=lineage,
    provider_ref_hashes=provider_reference_hashes_from_envelopes(envelopes),
    provider_retry_policy_hashes=measured_retry_policy_hashes(protected_hash),
    provider_registry_hash=provider_registry_hash(
        execution_config=execution_config
    ),
    protected_workflow_manifest_hash=protected_hash,
    encrypted_artifact_policy=json.loads(
        artifact_policy_path.read_text(encoding='utf-8')
    ),
    artifact_master_key_ref_hash=str(
        artifact_envelopes[0]['credential_ref_hash']
    ),
    research_lab_execution_config=execution_config,
    configured_worker_counts=profiles['worker_counts'],
)

async def collect_runtime():
    clients = {
        role: TEEClient(cid=int(spec['cid']))
        for role, spec in ROLE_SPECS.items()
    }
    readiness = await verify_v2_runtime_ready(clients)
    boots = {}
    for role in sorted(ROLE_SPECS):
        boots[role] = await clients[role].v2_get_boot_identity()
    return readiness, boots

runtime_readiness, boots = asyncio.run(collect_runtime())
source_commit, _ = read_source_commit()
observation = {
    'schema_version': 'leadpoet.gateway_deploy_readiness_observation.v2',
    'source_commit': source_commit,
    'build_commit': build.get('git_commit'),
    'gateway_release_manifest': gateway_release,
    'validator_release_manifest': validator_release,
    'compact_lineage': lineage,
    'boot_identities': boots,
    'expected_role_config_hashes': {
        role: documents[role]['configuration_hash']
        for role in sorted(documents)
    },
    'runtime_readiness': runtime_readiness,
    'coordinator_attestation_pcr0': attestation.get('pcr0'),
}
print(json.dumps(observation, sort_keys=True, separators=(',', ':')))
PY" > "$gateway_observation"
  PYTHONPATH="$ROOT" python3 - "$commit" "$gateway_observation" "$output" <<'PY'
import json
from pathlib import Path
import sys
from gateway.deploy_readiness import (
    build_gateway_v2_readiness_evidence_from_observation,
)

observation = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
evidence = build_gateway_v2_readiness_evidence_from_observation(
    expected_commit=sys.argv[1],
    observation=observation,
)
Path(sys.argv[3]).write_text(
    json.dumps(evidence, sort_keys=True, separators=(',', ':')) + '\n',
    encoding='ascii',
)
PY
  echo "Verified fresh Nitro identity and runtime configuration for all gateway roles"
}

verify_validator_release() {
  local output="$1"
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "cd '$VALIDATOR_REPO_ROOT' && PYTHONPATH='$VALIDATOR_REPO_ROOT' \
      python3 - '$commit' <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

expected = sys.argv[1]
# validator_exact_release_ready: retained as the operator/fault-injection probe label.
running = subprocess.check_output(
    ['docker', 'inspect', '-f', '{{.State.Running}}', 'leadpoet-validator-main'],
    text=True,
).strip()
restarts = subprocess.check_output(
    ['docker', 'inspect', '-f', '{{.RestartCount}}', 'leadpoet-validator-main'],
    text=True,
).strip()
environment = subprocess.check_output(
    [
        'docker',
        'inspect',
        '-f',
        '{{range .Config.Env}}{{println .}}{{end}}',
        'leadpoet-validator-main',
    ],
    text=True,
).splitlines()
values = dict(
    line.split('=', 1)
    for line in environment
    if '=' in line
)
if running != 'true' or restarts != '0':
    raise SystemExit('validator coordinator is not cleanly running')
if values.get('VALIDATOR_V2_DEPLOY_COMMIT') != expected:
    raise SystemExit('validator coordinator commit differs')
if values.get('VALIDATOR_WEIGHT_PROTOCOL') != 'authoritative_v2':
    raise SystemExit('validator coordinator is not authoritative V2')

from gateway.tee.release_lineage_v2 import validate_compact_release_lineage_v2
from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.host.runtime_v2_bootstrap import build_runtime_configuration
from validator_tee.host.vsock_client import ValidatorEnclaveClient

host_commit = subprocess.check_output(
    ['git', '-C', '$VALIDATOR_REPO_ROOT', 'rev-parse', 'HEAD'],
    text=True,
).strip().lower()
runtime_environment = dict(os.environ)
runtime_environment.update(values)
os.environ.clear()
os.environ.update(runtime_environment)
gateway_release = json.loads(
    Path('/home/ec2-user/.config/leadpoet/gateway-v2-release-manifest.json').read_text(
        encoding='utf-8'
    )
)
validator_release = json.loads(
    Path('/home/ec2-user/.config/leadpoet/validator-v2-release-manifest.json').read_text(
        encoding='utf-8'
    )
)
lineage = json.loads(
    Path('/home/ec2-user/.config/leadpoet/gateway-v2-release-lineage.json').read_text(
        encoding='utf-8'
    )
)
hotkey_config = json.loads(
    Path('/home/ec2-user/.config/leadpoet/validator-hotkey-config-v2.json').read_text(
        encoding='utf-8'
    )
)
configuration = build_runtime_configuration(
    validator_release=validator_release,
    gateway_release=gateway_release,
    gateway_release_lineage=lineage,
    hotkey_authority_config=hotkey_config,
)
client = ValidatorEnclaveClient()
enclave_health = client.health_check()
if (
    enclave_health.get('status') != 'ok'
    or enclave_health.get('authoritative_v2_configured') is not True
    or enclave_health.get('hotkey_authority_v2_configured') is not True
):
    raise SystemExit('validator enclave authority is not fully ready')
boot = client.get_authoritative_v2_boot_identity()
observation = {
    'schema_version': 'leadpoet.validator_deploy_readiness_observation.v2',
    'host_commit': host_commit,
    'gateway_release_manifest': gateway_release,
    'validator_release_manifest': validator_release,
    'compact_lineage': lineage,
    'boot_identity': boot,
    'expected_config_hash': sha256_json(configuration),
}
print(json.dumps(observation, sort_keys=True, separators=(',', ':')))
PY" > "$validator_observation"
  PYTHONPATH="$ROOT" python3 - "$commit" "$validator_observation" "$output" <<'PY'
import json
from pathlib import Path
import sys
from gateway.deploy_readiness import (
    build_validator_v2_readiness_evidence_from_observation,
)

observation = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
evidence = build_validator_v2_readiness_evidence_from_observation(
    expected_commit=sys.argv[1],
    observation=observation,
)
Path(sys.argv[3]).write_text(
    json.dumps(evidence, sort_keys=True, separators=(',', ':')) + '\n',
    encoding='ascii',
)
PY
  echo "Verified fresh Nitro identity and runtime configuration for validator_weights"
}

finalize_deploy_readiness() {
  PYTHONPATH="$ROOT" python3 - \
    "$commit" "$gateway_evidence" "$validator_evidence" "$final_manifest" <<'PY'
import json
from pathlib import Path
import sys
from gateway.deploy_readiness import (
    build_v2_deploy_readiness_manifest,
    validate_v2_deploy_readiness_manifest,
    write_deploy_readiness_manifest,
)

commit = sys.argv[1]
gateway = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
validator = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
manifest = build_v2_deploy_readiness_manifest(
    expected_commit=commit,
    gateway_evidence=gateway,
    validator_evidence=validator,
)
validate_v2_deploy_readiness_manifest(
    manifest,
    runtime_source_commit=commit,
    runtime_build_commit=commit,
)
write_deploy_readiness_manifest(manifest, sys.argv[4])
PY
  install_gateway_readiness_manifest readiness-final "$final_manifest"
  echo "Installed schema-v2 deploy readiness authority for $commit"
}

case "$component" in
  gateway)
    observed_validator="$(validator_active_commit)"
    if [ "$observed_validator" != "$commit" ]; then
      echo "ERROR: validator is on $observed_validator, not $commit; use --component all" >&2
      exit 1
    fi
    verify_validator_release "$validator_evidence"
    invalidate_deploy_readiness
    run_gateway_restart
    ;;
  validator)
    observed_gateway="$(gateway_active_commit)"
    if [ "$observed_gateway" != "$commit" ]; then
      echo "ERROR: gateway is on $observed_gateway, not $commit; use --component all" >&2
      exit 1
    fi
    verify_gateway_release "$gateway_evidence"
    invalidate_deploy_readiness
    run_validator_restart
    ;;
  all)
    validator_log="$temporary_root/validator-restart.log"
    coordination_file="/tmp/leadpoet-coordinated-restart.$(basename "$temporary_root").ready"
    ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
      "rm -f -- '$coordination_file'"
    invalidate_deploy_readiness
    echo "Starting validator preparation in parallel with the paired gateway restart"
    (
      VALIDATOR_RESTART_EXEC_SSH=1 run_validator_restart
    ) > >(tee "$validator_log") 2>&1 &
    validator_job="$!"

    capture_ready=0
    for _ in $(seq 1 120); do
      if grep -Fq \
          "Acquiring the independently built V2 release channel" \
          "$validator_log"; then
        capture_ready=1
        break
      fi
      if ! kill -0 "$validator_job" 2>/dev/null; then
        wait "$validator_job"
        exit 1
      fi
      sleep 1
    done
    if [ "$capture_ready" != "1" ]; then
      echo "ERROR: validator did not capture a valid restart start" >&2
      exit 1
    fi

    echo "Validator restart start captured; restarting the gateway while validator preparation continues"
    run_gateway_restart
    publish_coordination_value "$commit"
    echo "Gateway restart completed; releasing exact-SHA validator activation"
    wait "$validator_job"
    validator_job=""
    ;;
esac

verify_gateway_release "$gateway_evidence"
verify_validator_release "$validator_evidence"
finalize_deploy_readiness
echo "SUCCESS: gateway and validator are aligned on attested release $commit"
