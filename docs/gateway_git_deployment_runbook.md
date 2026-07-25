# Gateway Git Deployment Runbook

The gateway runs all host code from the complete checkout at
`/home/ec2-user/leadpoet_repo`. The operator command remains:

```bash
cd /home/ec2-user
bash /home/ec2-user/gw_restart.sh
```

The restart selects one commit from `GITHUB_REPO_URL` and `GITHUB_BRANCH`,
stops the existing processes, fast-forwards the checkout to that exact commit,
and then runs the existing cleanup, PCR0, enclave, dependency, process launch,
and health workflow. Normal unpinned gateway and validator restarts remain
independent; an explicit exact-commit validator restart adds the paired
same-release gate described below.

`--commit <full-sha>` is the canonical operator-only, one-invocation release
selector. The host keeps selector-aware restart controllers and their
pre-selection helpers outside the mutable runtime checkout. A rollback changes
only the attested runtime checkout; it cannot replace the newer controller with
the selected older script. `GATEWAY_DEPLOY_COMMIT` remains accepted for
installed-launcher compatibility during an N-1-to-N handoff. Persistent copies
in Secrets Manager or the cached/runtime environment are ignored and are not
inherited by the relaunched gateway. Normal restarts therefore always follow
the fetched head of `GITHUB_BRANCH`.

## One-Time Cutover

Invoke the cutover operator at or before official SN71 block 300. The operator
must capture that start as its first operational action. The same captured
start remains valid while GitHub attestation, release acquisition, gateway
restart, and validator restart continue after block 300; later stages must not
reapply the deadline. The intended migration commit must already be on the
configured GitHub branch.

```bash
/home/ec2-user/bin/research-lab-admin pause-scoring \
  --reason gateway_restart \
  --actor-ref operator:gateway-restart

/home/ec2-user/bin/research-lab-admin pause-autoresearch \
  --reason gateway_restart \
  --actor-ref operator:gateway-restart

/home/ec2-user/bin/research-lab-admin status
```

On the gateway host, verify and bootstrap the existing full checkout. This is
the only manual Git update; subsequent updates happen inside `gw_restart.sh`.

```bash
set -euo pipefail
cd /home/ec2-user/leadpoet_repo
test "$(git remote get-url origin)" = "https://github.com/leadpoet/leadpoet.git"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git fetch origin
git checkout main
git pull --ff-only origin main
grep -q 'GATEWAY_GIT_DEPLOY_PROTOCOL="1"' gw_restart.sh
test -f scripts/gateway_git_deploy.py

mkdir -p /home/ec2-user/.config/leadpoet/restart-backups
cp -p /home/ec2-user/gw_restart.sh \
  "/home/ec2-user/.config/leadpoet/restart-backups/gw_restart.sh.flat.$(date -u +%Y%m%dT%H%M%SZ)"
install -m 700 gw_restart.sh /home/ec2-user/gw_restart.sh
```

After that checkout update, reinstall the admin wrapper from the operator's
local repository so status and resume commands also import only the canonical
checkout:

```bash
LEADPOET_PROD_WRITE_APPROVED=yes \
  bash scripts/install_research_lab_admin_wrapper.sh leadpoet-gateway
```

Verify the installed wrapper before starting the restart:

```bash
grep -q '/home/ec2-user/leadpoet_repo' /home/ec2-user/bin/research-lab-admin
```

Do not delete `/home/ec2-user/gateway` during this migration. It retains the
existing logs, secrets, and initial emergency-recovery source tree. The Git
checkout must resolve both key paths absolutely; absent overrides default to:

- `/home/ec2-user/gateway/secrets/gateway_private_key.pem`
- `/home/ec2-user/gateway/secrets/arweave_keyfile.json`

## Normal Restart

The checkout must have no visible tracked, staged, or untracked files. Ignored
generated enclave/build artifacts are allowed and are rebuilt by the existing
workflow.

```bash
cd /home/ec2-user
bash /home/ec2-user/gw_restart.sh
```

If GitHub fetch, branch validation, remote validation, or checkout cleanliness
fails, the restart exits before stopping the running gateway. Failures after
process shutdown preserve the existing behavior: the command exits without
automatic rollback or automatic workflow resume.

## Verification

After a successful restart, verify the exact commit and process roots before
resuming protected workflows:

```bash
set -euo pipefail
cd /home/ec2-user/leadpoet_repo
DEPLOYED_SHA="$(git rev-parse HEAD)"
test "$(curl -fsS http://127.0.0.1:8000/build-info | python3 -c 'import json,sys; print(json.load(sys.stdin)["git_commit"])')" = "$DEPLOYED_SHA"
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/attest >/dev/null
/home/ec2-user/bin/research-lab-admin status
```

The latest attempt and last successful deployment records are:

```text
/home/ec2-user/.config/leadpoet/deployments/gateway-current.json
/home/ec2-user/.config/leadpoet/deployments/gateway-last-good.json
```

Resume using the existing operator workflow only after the normal checks pass:

```bash
/home/ec2-user/bin/research-lab-admin recover-stale-candidate-claims \
  --dry-run \
  --actor-ref operator:gateway-restart

/home/ec2-user/bin/research-lab-admin resume-scoring \
  --reason gateway_restart_complete \
  --actor-ref operator:gateway-restart

/home/ec2-user/bin/research-lab-admin resume-autoresearch \
  --reason gateway_restart_complete \
  --actor-ref operator:gateway-restart

/home/ec2-user/bin/research-lab-admin status
```

## Rollback

Rollback is a paired gateway-and-validator deployment. A gateway-only
`gateway-last-good.json` record is not sufficient evidence that the same
validator release completed successfully. Select one full 40-character commit
that:

- Is reachable from `origin/main`.
- Is at or after the stateful V2 compatibility floor, currently
  `94f1c923d092d12cbab95ef8d86317420eede621`.
  The floor includes bounded slow post-rollover allocation readiness and a
  source-consistent protected-workflow identity; older attestations are not
  sufficient rollback evidence.
- Has one immutable release channel containing matching gateway and validator
  manifests.
- Has the same gateway and validator protected V2 symbol contracts, and
  byte-identical protected source files, as current `origin/main`, so auditors
  running current public code recompute and verify the same canonical weights.
- Has passed the exact reverse restart rehearsal from the currently installed
  launcher.

Restart the gateway first with `gw_restart.sh --commit <full-sha>`. Require its
normal build-info, V2 authority, attestation, and authenticated allocation
handoff checks to pass. Only then restart the validator with
`/home/ec2-user/validator_restart.sh --commit <the-same-full-sha>`. The host
validator controller is installed by every successful current release and
remains outside the detached runtime checkout, so the same command remains
available after rollback.

The pinned validator restart fails before shutdown unless the public gateway
reports the same commit through V2 authority health, build-info, and immutable
release evidence. These same-SHA checks retry bounded transient failures but
remain fail closed; a persistent post-start mismatch stops the pinned validator.
A normal unpinned validator restart retains its independent gateway-startup
behavior.

Rollback runs the same enclave rebuild and restart workflow. It does not reuse
newer EIFs or bypass PCR0, attestation, import, or health checks. A commit from
before the stateful V2 compatibility floor is intentionally rejected. A
subsequent roll-forward must run the exact forward rehearsal from the rolled
back commit before production use.
