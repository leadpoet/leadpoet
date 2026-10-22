# Gateway deployment

The gateway serves the public API and forwards competition requests to the
Arena service. The normal validator runs scoring jobs and submits weights
with its local hotkey.

## Release

The gateway dependency preflight requires stable pip 26.2.2 or newer with a
vendored certifi 2026.10.19 or newer bundle inside the pip package. Otherwise it
bootstraps pip from the PSF's SHA256-pinned 26.2.2 wheel before it downloads host
dependencies, including on a host without pip. It uses the selected gateway
interpreter and its existing virtualenv or user install scope. A failed
download, TLS check, checksum, or offline install
stops the preflight before production shutdown. Host dependency downloads use
pip's patched certificate bundle with `--use-deprecated=legacy-certs`; the old
host CA store is not added. Explicit `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, and
`PIP_CERT` overrides remain supported. Keep custom CA bundles current.
This follows the [PSF emSign incident guidance](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576)
and [pip's certificate-store documentation](https://pip.pypa.io/en/stable/topics/https-certificates/).
It does not update the operating system trust store.
The separate [AL2023 host CA blocklist procedure](../deploy/ca-trust/README.md)
records the reviewed OS configuration, purpose limits, rollback, and context
activation requirements. It is an explicit provisioning action.

1. Fetch `origin/main`. Keep local and concurrent changes intact.
2. Run the current tests in [the deployment checklist](v2_deployment_verification_checklist.md)
   from the exact candidate checkout. Run real PostgreSQL checks for changes
   to database contracts.
3. Commit and merge the reviewed change. Fetch again and record the exact full
   SHA to deploy. Apply required committed migrations through the configured
   migration helper before starting code that needs them.
4. Invoke the installed canonical gateway controller:

   ```bash
   /home/ec2-user/gw_restart.sh --commit <full-commit-sha>
   ```

   It owns release verification, claim draining, process replacement,
   runtime measurements, and readiness checks. Do not invoke inner controllers
   or replace a live source tree by hand. Respect a rejected epoch gate and
   retry when the canonical controller permits it.

   Arena continuity uses a stable Caddy listener on direct port `8000` and the
   public domain. It routes Arena requests to one API-only release slot while
   the controller replaces the single gateway backend and enclave. The Arena
   driver and reviewer run in one separate process; do not start a combined
   Arena service on `8792` or bind `8000` by hand. The first migration can have
   a short direct-port handoff gap. Later canonical restarts keep Arena intake
   and reads on the serving slot while they prepare and check the next slot.

   The fixed loopback port allocation is:

   | Component | Slot 0 | Slot 1 |
   | --- | --- | --- |
   | Mainnet Arena API | `8792` | `8794` |
   | Arena continuity edge | `8890` | `8891` |
   | Hosted testnet Arena API | `8793` | — |

   Ports `8890` and `8891` are reserved for the continuity edges. Ports
   `8800` and `8801` are outside this allocation and can belong to other
   services. The controller rejects an occupied candidate API or edge port;
   it does not select another port or stop an unrelated listener.

   If a restart fails after Arena ingress moves, rerun the canonical command
   for the **same full commit SHA**. It verifies the retained release, serving
   process, and held restart guard before it resumes. An unfinished first
   migration must finish its exact SHA before another release can start. Do
   not edit Caddy state, release archives, or the guard to force a retry.
5. If the validator changed, run the exact committed `validator_restart.sh`
   with `VALIDATOR_DEPLOY_COMMIT` set to the same full SHA. See
   [normal validator operations](arena_normal_validator_weights.md).

GitHub workflow status is diagnostic. It does not authorize or block a
canonical restart. Local release identity and cryptographic checks remain
mandatory.

## Repair an installed controller that cannot prepare its replacement

Use this path only for a confirmed fault in the installed controller's
preparation phase. It runs the candidate's **full canonical restart**, including
the normal host locks, epoch gate, release verification, drain, cutover,
attestation, and readiness checks. It does not install a controller in advance
or skip a failed gate.

Fetch `origin/main` in the canonical repository. Record its full source SHA and
the inspected installed controller SHA from
`~/.config/leadpoet/restart-controller/gateway/current`. Materialize only the
repair helper from that exact commit into an owner-only temporary file:

```bash
cd /home/ec2-user/leadpoet_repo
git fetch origin
# Use the full reviewed source SHA and the full inspected installed SHA below.
source_sha="<full-reviewed-source-sha>"
installed_controller_sha="<full-inspected-controller-sha>"
repair_helper="$(mktemp /tmp/gateway-controller-repair.XXXXXX.py)"
chmod 600 "$repair_helper"
repair_status=0
git show "$source_sha:scripts/bootstrap_gateway_restart_controller.py" > "$repair_helper" || repair_status=$?
if [ "$repair_status" -eq 0 ]; then
  python3 "$repair_helper" \
    --source-commit "$source_sha" \
    --expected-controller-commit "$installed_controller_sha" || repair_status=$?
fi
rm -f "$repair_helper"
(exit "$repair_status")
```

The helper verifies its own committed bytes, the official origin, current
fetched main, the installed controller link and four owned controller files,
and a private exact candidate archive. It refuses an occupied restart or
recovery lock. It rechecks main and the installed controller before it releases
its inspection locks and invokes the full candidate controller. The canonical
controller must then acquire its own normal lock. Do not set authority, phase,
or destructive-state environment variables by hand.

The receipt reports the canonical exit code and the observed installed
controller identity: `previous`, `candidate`, or `unknown`. An early failure
leaves the old installed controller in place. A late failure can occur after
the canonical controller installs its successor; preserve that state and use
the normal exact-SHA recovery flow. The helper never rolls back canonical
runtime, ingress, or guard changes. If the observer is interrupted while the
controller still runs, it retains and reports the private authority path.
If the canonical command returns zero but the candidate installed identity
cannot be verified, the helper returns one and preserves the canonical exit
code in its receipt.

For a confirmed first-cutover failure that retained its owned destructive guard
before any ingress switch or process stop, add
`--resume-invocation gateway-<epoch>-<pid>` to the helper command. It requires the
exact owned `0600` timing ledger in the private restart-timings directory, one
consistent committed source, and a final failed pre-shutdown stage. Ingress,
shutdown, or later timing evidence rejects this path. The existing read-only
guard helper must prove the same owner, candidate, gateway scope, destructive
phase, positive generation, and preserved drain. Only the active guard owner
is carried into the full canonical controller; restart telemetry gets a fresh
invocation. The canonical controller still revalidates, retargets, drains, and
checks readiness. Do not reset the guard or reuse old phase variables.
Timing writes are best effort. Independently verify the serving ingress and
exact live process ownership; missing timing stages alone do not prove that
no process stopped.

A controller identity receipt does not verify runtime deployment. Complete the
normal live checks below, including guard release and terminal restart success.
Subsequent restarts use `/home/ec2-user/gw_restart.sh --commit <full-commit-sha>`.

## Verification

Verify the gateway build identity, Arena API, validator installed identity,
and canonical restart results. Check accepted execution and scoring records,
publication, reward state, and finalized chain outcomes. Use a reward-disabled
shadow round for paid diagnostic runs. Do not inject test submissions or
synthetic rewards into the live competition.

A healthy HTTP endpoint alone does not prove scoring or weights are healthy.
Keep the prior release identity and accepted work for recovery, and use the
canonical controllers for any rollback.
