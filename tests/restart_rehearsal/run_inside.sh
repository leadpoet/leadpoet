#!/bin/bash
set -euo pipefail

FROM_SHA="${REHEARSAL_FROM_SHA:?REHEARSAL_FROM_SHA is required}"
CANDIDATE_SHA="${REHEARSAL_CANDIDATE_SHA:?REHEARSAL_CANDIDATE_SHA is required}"
TRANSITION="${REHEARSAL_TRANSITION:?REHEARSAL_TRANSITION is required}"
COMPONENT="${REHEARSAL_COMPONENT:?REHEARSAL_COMPONENT is required}"
WEIGHT_READINESS_SCENARIO="${REHEARSAL_WEIGHT_READINESS_SCENARIO:-transient_503_recovery}"
REHEARSAL_SCOPE="${REHEARSAL_SCOPE:-exact}"

case "$COMPONENT" in
  gateway|validator) ;;
  *)
    echo "ERROR: REHEARSAL_COMPONENT must be gateway or validator" >&2
    exit 2
    ;;
esac

export REHEARSAL_STATE_ROOT=/rehearsal-state
mkdir -p "$REHEARSAL_STATE_ROOT" /harness/bin /home/ec2-user

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

cat >/home/ec2-user/.config/leadpoet/stateful-epoch-cutover.json <<'JSON'
{
  "schema_version": "leadpoet.subnet_epoch_cutover.v1",
  "mapping_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "status": "stateful_active"
}
JSON
cat >/home/ec2-user/.config/leadpoet/validator-hotkey-config-v2.json <<'JSON'
{"schema_version":"leadpoet.validator_hotkey_config.v2","validator_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"}
JSON
cat >/home/ec2-user/.config/leadpoet/validator-hotkey-envelope-v2.json <<'JSON'
{"schema_version":"leadpoet.validator_hotkey_envelope.v2","ciphertext_b64":"cmVoZWFyc2Fs"}
JSON

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
      bash /home/ec2-user/gw_restart.sh --commit "$CANDIDATE_SHA"
  else
    env \
      HOME=/home/ec2-user \
      GATEWAY_DEPLOY_COMMIT="$CANDIDATE_SHA" \
      LEADPOET_REPO_ROOT=/home/ec2-user/leadpoet_repo \
      GATEWAY_ROOT=/home/ec2-user/leadpoet_repo/gateway \
      GATEWAY_LOG_ROOT=/home/ec2-user/gateway \
      GATEWAY_LOG_FILE=/home/ec2-user/gateway/gateway.log \
      GATEWAY_HOST_RESTART_SCRIPT=/home/ec2-user/gw_restart.sh \
      GATEWAY_TEE_EIF_ROOT=/home/ec2-user/tee \
      GATEWAY_PYTHON_BIN=/home/ec2-user/venv311/bin/python3 \
      GATEWAY_TEE_TOPOLOGY_MODE=full \
      RESEARCH_LAB_TEE_PROTOCOL=v2 \
      bash /home/ec2-user/gw_restart.sh
  fi
  RESTART_STATUS=$?
  set -e

  if [ "$WEIGHT_READINESS_SCENARIO" != "transient_503_recovery" ]; then
    if [ "$RESTART_STATUS" -eq 0 ]; then
      echo "ERROR: gateway restart accepted failure scenario $WEIGHT_READINESS_SCENARIO" >&2
      exit 1
    fi
    /usr/bin/python3.11 /harness/verify_evidence.py \
      "$COMPONENT" "$FROM_SHA" "$CANDIDATE_SHA" "$WEIGHT_READINESS_SCENARIO" "$REHEARSAL_SCOPE"
    echo "TARGETED_RESTART_REGRESSION_EXPECTED_FAILURE component=$COMPONENT candidate=$CANDIDATE_SHA scenario=$WEIGHT_READINESS_SCENARIO"
    exit 0
  fi
  if [ "$RESTART_STATUS" -ne 0 ]; then
    echo "ERROR: gateway restart recovery scenario failed" >&2
    exit "$RESTART_STATUS"
  fi

  test "$(git -C /home/ec2-user/leadpoet_repo rev-parse HEAD)" = "$CANDIDATE_SHA"
  test "$(git -C /home/ec2-user/leadpoet_repo status --porcelain)" = ""
  test "$(git -C /home/ec2-user/leadpoet_repo show "$CANDIDATE_SHA:gw_restart.sh" | sha256sum | cut -d' ' -f1)" = \
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
  mkdir -p /home/ec2-user/leadpoet
  git clone -q /srv/origin.git /home/ec2-user/leadpoet/leadpoet
  git -C /home/ec2-user/leadpoet/leadpoet checkout -q --detach "$FROM_SHA"
  git -C /home/ec2-user/leadpoet/leadpoet branch -f main "$FROM_SHA"
  git -C /home/ec2-user/leadpoet/leadpoet checkout -q main
  git -C /home/ec2-user/leadpoet/leadpoet remote set-url origin /srv/origin.git

  mkdir -p \
    /home/ec2-user/.bittensor/wallets/validator_72/hotkeys \
    /home/ec2-user/.bittensor/wallets/validator_72
  printf '%s\n' '{"ss58Address":"5DummyColdkey"}' \
    >/home/ec2-user/.bittensor/wallets/validator_72/coldkeypub.txt

  echo "REHEARSAL_START component=validator from=$FROM_SHA candidate=$CANDIDATE_SHA transition=$TRANSITION"
  if [ "$TRANSITION" = "rollback" ]; then
    env \
      HOME=/home/ec2-user \
      VALIDATOR_ROOT=/home/ec2-user/leadpoet/leadpoet \
      VALIDATOR_PYTHON_BIN=/home/ec2-user/venv311/bin/python3 \
      VALIDATOR_DOCKER_MIN_FREE_BYTES=1000000000 \
      bash /home/ec2-user/leadpoet/leadpoet/validator_restart.sh \
        --commit "$CANDIDATE_SHA"
  else
    env \
      HOME=/home/ec2-user \
      VALIDATOR_DEPLOY_COMMIT="$CANDIDATE_SHA" \
      VALIDATOR_ROOT=/home/ec2-user/leadpoet/leadpoet \
      VALIDATOR_PYTHON_BIN=/home/ec2-user/venv311/bin/python3 \
      VALIDATOR_DOCKER_MIN_FREE_BYTES=1000000000 \
      bash /home/ec2-user/leadpoet/leadpoet/validator_restart.sh
  fi

  test "$(git -C /home/ec2-user/leadpoet/leadpoet rev-parse HEAD)" = "$CANDIDATE_SHA"
  test "$(
    git -C /home/ec2-user/leadpoet/leadpoet \
      status --porcelain --untracked-files=no
  )" = ""
fi

/usr/bin/python3.11 /harness/verify_evidence.py \
  "$COMPONENT" "$FROM_SHA" "$CANDIDATE_SHA" "$WEIGHT_READINESS_SCENARIO" "$REHEARSAL_SCOPE"
if [ "$REHEARSAL_SCOPE" = "exact" ]; then
  echo "REHEARSAL_SUCCESS component=$COMPONENT candidate=$CANDIDATE_SHA"
else
  echo "TARGETED_RESTART_REGRESSION_SUCCESS component=$COMPONENT candidate=$CANDIDATE_SHA scenario=$WEIGHT_READINESS_SCENARIO"
fi
