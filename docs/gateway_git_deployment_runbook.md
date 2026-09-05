# Gateway Git Deployment Runbook

The gateway runs all host code from the complete checkout at
`/home/ec2-user/leadpoet_repo`. The canonical paired operator command is:

```bash
: "${EXPECTED_SHA:?set the full release SHA}"
: "${LOCAL_READINESS_PYTHON:?set an absolute venv bin/python path}"
cd /path/to/the/exact/candidate/checkout
bash scripts/restart_attested_release_local.sh \
  --commit "$EXPECTED_SHA" \
  --local-python "$LOCAL_READINESS_PYTHON" \
  --component all
```

The restart selects the supplied exact commit,
stops the existing processes, fast-forwards the checkout to that exact commit,
builds the local gateway and validator runtime identities, and then runs the
cleanup, PCR0, enclave, dependency, process launch, and health workflow. It
does not wait for GitHub attestation or GitHub Actions. Gateway and validator controllers may run concurrently,
but validator activation is not independent: preparation may overlap the
gateway restart, while every exact-release validator waits at the
application-image boundary for the same gateway release described below.

`--commit <full-sha>` is the canonical operator-only, one-invocation release
selector. The host keeps selector-aware restart controllers and their
pre-selection helpers outside the mutable runtime checkout. A rollback changes
only the exact runtime checkout; it cannot replace the newer controller with
the selected older script. `GATEWAY_DEPLOY_COMMIT` remains accepted for
installed-launcher compatibility during an N-1-to-N handoff. Persistent copies
in Secrets Manager or the cached/runtime environment are ignored and are not
inherited by the relaunched gateway. Normal restarts therefore always follow
the fetched head of `GITHUB_BRANCH`.

## One-Time Cutover

Invoke the cutover operator at or before official SN71 block 300. The operator
must capture that start as its first operational action. The same captured
start remains valid while local release preparation, gateway restart, and
validator restart continue after block 300; later stages must not reapply the
deadline. The intended migration commit must already be on the configured Git
branch.

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

## First Release With Miner-Maintenance Control

The deployed N-1 controller hydrates the gateway secret before it fetches and
materializes the candidate. The first release containing the protected
miner-maintenance helper must therefore use the paired exact-candidate option.
Do not run a separate remote helper or an unpinned host wrapper:

```bash
set -euo pipefail
: "${EXPECTED_ATTESTED_SHA:?set the full attested release SHA}"
: "${LOCAL_READINESS_PYTHON:?set an absolute venv bin/python path}"
[[ "$EXPECTED_ATTESTED_SHA" =~ ^[0-9a-f]{40}$ ]]
cd /path/to/the/exact/candidate/checkout
bash scripts/restart_attested_release_local.sh \
  --commit "$EXPECTED_ATTESTED_SHA" \
  --local-python "$LOCAL_READINESS_PYTHON" \
  --component all \
  --disable-miner-submissions-before-restart
```

The paired operator first proves that its own script and the installed-controller
verifier are the selected candidate's exact Git blobs. It rejects replacement
refs, grafts, alternates, unsafe Git environment overrides, non-production
release prefixes, alternate secret identities, and single-component use. On
the gateway it verifies and seals the complete installed controller bundle
before executing its deployment helper.

The exact candidate archive then acquires the existing canonical gateway lock
on descriptor 9. Under that uninterrupted lock it revalidates the isolated
plan and candidate tree, the singleton actively COMPLIANCE-locked release
channel, the protected source, the fixed EC2 instance-role authority, and the
installed controller. It first invokes the fixed production
`research_lab_source_add_acquire_restart_guard_v2` RPC and exact-reads the
singleton control row. The guard identity is fixed to the canonical production
gateway restart authority, while its owner and actor are invocation-specific.
The acquire RPC compares the exact monotonic generation read under the same
control lock. A fresh canonical retry transfers that same guard to its new
owner and increments the generation immediately; the prior invocation can no
longer renew, prove quiescence, or release it. The existing gateway restart
lock serializes the host side of that transfer. Migration 145 makes the guarded pause
share one transaction-scoped advisory lock with miner SOURCE_ADD admission, so
a concurrent admission is either committed before the pause or rejected after
it. A missing RPC, unavailable readback, resumed/changed row, unavailable
control projection, enabled dispatcher, or explicitly enabled intake fails
before shutdown. The legacy N-1 status predates the separate `intake_enabled`
projection, but its exact protected route checks the same durable control; the
candidate must expose and prove the explicit false projection after startup.

Migration 172 also makes `research_lab_source_add_claim_work` acquire that
same `source-add-control` lock before reading the pause or leasing work. The
restart then polls the fixed
`research_lab_source_add_restart_quiescence_v1` RPC under that lock until the
durable control remains paused, the same guard is active and matches, and the
count of every `work_status='leased'` row is exactly zero, including expired
leases. This orders all pre-pause claims before the zero readback and prevents
a later claim while paused. The poll is bounded; a live, expired, or hung lease
at the deadline aborts the restart and leaves SOURCE_ADD paused and guarded for
an exact retry or deliberate operator recovery. Resume is rejected while a
guard commitment exists, even after its lease expires; recovery explicitly
reacquires the canonical guard with a new owner/generation and exact-releases
it. A missing/invalid
migration-172 RPC also fails closed. Dispatcher status alone is not a
quiescence proof.

Only after SOURCE_ADD is durably paused does the helper change
`RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED` to `false`. The version-stage
transaction uses a private non-secret crash journal and reconstructs the exact
original `AWSCURRENT`/`AWSPREVIOUS`/custom-label topology. After interruption,
a retry either restores that exact prior topology when `AWSCURRENT` never
moved, or completes the exact verified false promotion when it did. Reconciliation
finishes before any new change.

No persistent restart receipt is created. The candidate carries only a sealed,
unlinked invocation proof through the exact installed N-1 wrapper. The proof
binds the candidate/tree/release/controller identities, the final
`AWSCURRENT` VersionId, strict raw-document hash, complete stage-topology hash,
the exact durable SOURCE_ADD control-row commitment, the canonical restart-
guard commitment, its invocation-owner/generation commitment, the exact
active-guard/paused/zero-leased quiescence commitment, and the exact
deterministic gateway-env bytes that frozen
controller `0dd` derives from that document. The N-1 wrapper itself remains
byte-identical to Git; there is no AWS executable, PATH, or wrapper
interposition.

Before shutdown and again after startup, the candidate descriptor-safely reads
the installed env cache and requires its byte hash to equal that expected
rendering. It also rechecks the current raw document and complete topology,
requires the hydrated parent value to be exactly `false`, and requires the live
Research Lab status boolean to be exactly false. It also exact-reads the
proof-bound SOURCE_ADD pause and paused/zero-leased quiescence before shutdown
and after startup, then requires the running candidate to report a present,
available, paused control plus `intake_enabled=false` and
`effective_dispatcher_enabled=false`. A transient
alternate Secrets Manager VersionId is equivalent only when its raw document
and the resulting filtered env bytes are byte-identical and the proof-bound
current VersionId/topology are restored at verification. A differing-document
or SOURCE_ADD-control drift fails closed. Secrets Manager and SOURCE_ADD admin
write authority are trusted and non-adversarial during this canonical locked
restart; this path does not claim to prevent an authorized writer from
deliberately changing runtime configuration.

The candidate renews the exact same owner/generation with the migration's
14,400-second maximum lease, then repeats the same active-guard and zero-leased
RPC after all ancestry, Docker, pairing, and active-release preparation,
immediately before the first shutdown action. That lease covers the canonical
9,300-second paired-coordination deadline plus bounded startup margin. There is
no unbounded work between the final check and the first process stop. The
durable guard prevents an operator resume from racing that boundary. After the
candidate is live, the strict runtime status, durable pause, exact owner and
generation, active guard, and zero leased count are checked once more. Only
then does the candidate compare-and-release that owner/generation and
exact-read the control again. Migration 174 binds the pause state from before
the first guard acquisition to that exact guard generation. Successful release
atomically restores that state: active remains active and paused remains
paused. An explicit operator pause while the guard is held overrides an earlier
active snapshot. Autoresearch and scoring maintenance state are never changed
by this SOURCE_ADD guard.

The proof and all four controller snapshots are closed in every long-lived
runtime child and closed by the restart parent after the post-start check.
After preparation reports success and the false promotion is durable, any
later failure leaves global miner submissions disabled and SOURCE_ADD paused,
and retains no cross-run restart authority; the same exact paired command is
safe to retry. If the prior gateway is no longer running, the retry skips its
loopback status only after the protected `/proc` scan proves that exact absence;
the durable pause, same canonical guard, fresh invocation owner/generation,
and zero-leased RPC proofs remain mandatory. An earlier failure may instead
leave or restore the exact original secret topology, but it does not restore
active SOURCE_ADD until candidate runtime verification succeeds.

## Normal Restart

The checkout must have no visible tracked, staged, or untracked files. Ignored
generated enclave/build artifacts are allowed and are rebuilt by the existing
workflow.

Once the durable gateway source already contains the explicit `false` value,
ordinary exact-SHA restarts need no persistent proof. Candidate preflight still
requires parent hydration to be exactly false, performs an instance-role-only
Secrets Manager readback, verifies the current document and topology are
stable while descriptor-checking the hydrated cache, requires an exact durable
SOURCE_ADD guarded-pause readback and a bounded exact zero-leased quiescence
drain, and revalidates the singleton locked release channel. The same
last-moment and post-start guard checks apply to receiptless restarts. A stale
or direct launcher with a true value, or one with SOURCE_ADD active or leased
work remaining, fails before shutdown. Keep scoring and autoresearch in their
separately recorded invocation-time maintenance states.

```bash
: "${LOCAL_READINESS_PYTHON:?set an absolute venv bin/python path}"
cd /path/to/the/exact/candidate/checkout
bash scripts/restart_attested_release_local.sh \
  --commit "$EXPECTED_ATTESTED_SHA" \
  --local-python "$LOCAL_READINESS_PYTHON" \
  --component all
```

`--local-python` is required. It must select a dependency-complete virtual
environment containing the declared `cbor2` and `cryptography` packages. The
operator executes its local readiness phases with `-I -S`, admits only the
exact candidate root plus that venv's site-packages, and never falls back to an
ambient system interpreter.

If an authorized operation must transition miner submissions from true to
false, use the paired `--disable-miner-submissions-before-restart` command
above. Do not invoke the fixed-purpose apply CLI manually; the paired path owns
the canonical lock, recovery journal, exact controller handoff, and post-start
proof.

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

The restart does not change scoring or autoresearch maintenance state. Its
SOURCE_ADD guard is transient: after successful runtime verification it
atomically restores the exact SOURCE_ADD pause state captured before the
restart. Verify that restoration without issuing another state transition:

```bash
/home/ec2-user/bin/research-lab-admin source-add status
curl -fsS http://127.0.0.1:8000/research-lab/status
/home/ec2-user/bin/research-lab-admin status
```

When SOURCE_ADD was active before restart, require `paused=false`,
`intake_enabled=true`, and `effective_dispatcher_enabled=true` afterward. When
it was paused, require it to remain paused. A failed restart leaves SOURCE_ADD
paused fail-closed and must not be followed by an automatic or unconditional
resume. Resume scoring, autoresearch, or SOURCE_ADD only through their separate
operator workflows when an operator independently intends that state change.
None of those SOURCE_ADD transitions changes the global miner-submission
switch.

## Rollback

Rollback is a paired gateway-and-validator deployment. A gateway-only
`gateway-last-good.json` record is not sufficient evidence that the same
validator release completed successfully. Select one full 40-character commit
that:

- Is reachable from `origin/main`.
- Has one immutable release channel containing matching gateway and validator
  manifests.
- Provides the authoritative V2 endpoints, envelope schemas, receipt formats,
  signing authorization, and canonical weight protocol consumed by current
  public auditors.
- Has passed the exact reverse restart rehearsal from the currently installed
  launcher.

The compatibility check intentionally does not compare protected implementation
hashes or require later reliability fixes. Those differences inform the
operator's rollback choice but do not make an otherwise attested,
auditor-protocol-compatible release categorically ineligible.

When invoking the two lower-level controllers manually, pass the same full SHA
to both. They may be launched concurrently: the validator can complete its
release checks, rebuild, Nitro/runtime/hotkey preparation, and exact
application-image build while `gw_restart.sh --commit <full-sha>` is still
running. The host validator controller is installed by every successful
current release and remains outside the detached runtime checkout, so the same
command remains available after rollback. Prefer the coordinated command
below, which owns the exact-SHA success/failure marker and cleanup contract
rather than relying on two independent terminals.

The canonical operator command coordinates both restarts while preserving one
restart-start decision and one selected release:

```bash
bash scripts/restart_attested_release_local.sh \
  --commit <full-sha> \
  --local-python </absolute/venv/bin/python>
```

Use `--component gateway` or `--component validator` only when the other
component is already running that exact commit. A mismatch fails before the
requested component restarts and instructs the operator to use the default
paired mode. In paired mode the validator captures its official restart start
first, then prepares in parallel with the gateway. The validator may select and
verify Git/release inputs, stop the old runtime, rebuild its EIF, launch and
provision Nitro, start the opaque chain relay, and build the exact application
image before gateway completion. It may not start a validator coordinator or
worker until all of these conditions hold:

- The application image commit label equals the selected full SHA.
- Its immutable image ID has been captured and remains unchanged across the
  wait.
- The unique coordination marker contains that exact SHA.
- Public gateway V2 authority health, build-info, and immutable release
  evidence all report that exact SHA.

The coordinator does not approve a release itself. Both installed restart
controllers independently build and verify the selected local source, runtime
artifacts, PCR0, current-auditor compatibility, and normal readiness checks.
These local checks do not require a GitHub release, attestation job, or test
job. The validator wrapper repeats the three live gateway checks after
coordinator startup.
If a failed attempt already completed the N-1-to-N Git handoff, a retry may
invoke the repository launcher only after proving that its checkout and
launcher blob equal the selected SHA. The coordinated expected SHA is checked
again after the remote pull, before release preparation or shutdown, so a
concurrent branch advance fails while the existing validator is still running.

If the gateway restart fails, the coordinator immediately terminates the
validator SSH job and publishes a commit-bound failure marker concurrently
through a bounded write, so slow marker transport cannot delay signal cleanup
even if the validator already consumed a prior success marker. Candidate
re-execution and secret hydration cannot replace the paired operator's
coordination path or bounded wait policy. The waiting validator exits without
starting a coordinator or worker and removes prepared validator containers,
host validator/relay processes, Nitro enclave, and Docker lock. A
selected historical deployer that predates the image-prepared barrier falls
back to the same exact-SHA gateway check immediately before invoking that
deployer. Because old-runtime shutdown is one of the overlapped preparation
stages, a late gateway failure may leave the validator safely stopped; rerun
the paired command after fixing the gateway.

Rollback runs the same enclave rebuild and restart workflow. It does not reuse
newer EIFs or bypass PCR0, attestation, import, or health checks. A commit from
an earlier implementation generation is rejected only if it lacks the public
protocol required by current auditors or fails the normal exact release gates.
A subsequent roll-forward must run the exact forward rehearsal from the rolled
back commit before production use.
