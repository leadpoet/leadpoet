# Normal Arena validators

The primary, Yuma, Rizzo, and other subnet validators use the same
`neurons/validator.py` process with their own local Bittensor hotkey.
Nitro, an enclave, KMS key unwrapping, and the retired auditor client are not
needed to run this process.

## Scoring and authorization

On mainnet, a registered hotkey with an on-chain validator permit has validator
status, including below 75,000 effective subnet stake weight. New Arena
execution/scoring jobs and planned capacity additionally require **at least
75,000 effective subnet stake weight**. Exactly 75,000 qualifies. Activity does
not bypass the scoring minimum or disqualify a permitted validator.
Every qualifying validator can claim automatically; round runner lists do not
grant or deny access. Test networks retain the existing gateway rule: active
or permitted for scoring, without the mainnet stake minimum. Weight retrieval
requires a validator permit on every network. Both checks use the shared
gateway role helpers; validator identity is separate from scoring eligibility.

Arena reads chain `total_stake` (Bittensor `Metagraph.S`) from its finalized
snapshot, with the existing 60-second cache. Stake changes take effect after
finality and cache refresh. Missing or invalid chain data stops new claims.
Registration, stake, and same-coldkey miner exclusions use one snapshot.
Lease ownership, signatures, stage rules, code review, and miner self-dealing
checks remain in place.

The Arena minimum applies when issuing a lease. A stake decrease alone does not
reject an already-issued job's source/image access, provider calls, or result.
Existing identity and lease protections still apply. Below-threshold claims
for new work return `403 runner_stake_below_minimum`; the runner idles and keeps weights
running. No benchmark work or minimum benchmark stake is required to retrieve
the signed weight state, derive weights, or run the weight submission loop.
Subnet registration and a validator permit are still required for retrieval.
An exact signed-request retry can recover its already-issued lease after a
stake-only drop, without allocating or extending work.

`LAB_ARENA_RUNNER_HOTKEYS` is a conservative capacity plan, not an allowlist.
New rounds count only eligible planned runners, without assuming all qualifying
on-chain validators are online. Zero eligible planned capacity stops creation
of a new round, but not processing existing rounds or publishing weight state.
Existing round configurations and admitted submissions are not rewritten.

The validator pulls assigned miner source, runs the existing Arena ICP and
scoring path, and reports results with its local hotkey. Miner provider
credentials remain in the gateway broker. Validators and submitted models do
not receive the raw keys. The existing gVisor/runsc sandbox is still needed for
scoring; removing Nitro does not remove the model sandbox.

On a private ECR scorer cache miss, the runner uses its active lease to get
temporary download links from the gateway. The gateway limits access to the
exact common scorer image frozen into that round. The runner uses the existing
OCI digest, size, platform, and safe extraction checks before caching the image.
Validators need no AWS account or shared registry credential. Public registry
images keep their existing download path.

Download links are bearer capabilities. They are sent with `Cache-Control:
no-store`, kept in memory, and excluded from logs. ECR links last at most one
hour and can remain valid after the issuing lease ends. The gateway issues
them only while the requesting validator's lease is active. Miner provider
credentials are separate and remain in the gateway broker.

Weights start independently, before scoring setup. An empty queue, missing
runsc, scoring setup error, or scoring-loop error causes scoring to wait/retry,
not the weight loop to stop. Scoring works again when its dependency recovers.
Known host failures include a safe reason and the configured local path in
logs. Provider exception text and runtime subprocess output are not logged.
`scoring loop resumed` means polling recovered; an accepted completion is still
needed to prove that actual Arena work succeeded.

## Weight submission

The validator signs its weight-state request with its own local hotkey. The
gateway verifies the request signature, freshness, action, epoch, and chain
scope, then checks registration and the validator permit in its finalized
metagraph. Anonymous requests and hotkeys without permits cannot retrieve the
signed weight state, which includes its signed reward basis. Below-threshold
permitted validators can retrieve this state. The old public reward-basis routes
are removed, and public round responses no longer include the signed reward basis. The signing public key
and normal public competition results
remain public; this access rule does not make revealed on-chain weights private.

The validator verifies the gateway signing-key pin and the signed accepted
Arena state, including the governing reward basis, chain identity, subnet,
epoch window, and burn hotkey. It derives the same canonical winner/burn vector
from finalized UID ownership and signs the time-locked commitment with its own
local hotkey. The existing runtime, nonce, mortal transaction, and exact
transaction checks still apply.

No scoring work is needed to submit weights. The gateway publishes the current
epoch's state from the governing reward basis, even when no new model has been
scored. A missing, invalid, conflicting, or expired accepted state stops that
submission. It never becomes an invented burn-only state.

Signed bytes are persisted before broadcast. Restarts and uncertain results
reuse those exact bytes. A fresh attempt requires proof of expiry, absence of
inclusion, and an unchanged nonce. Earlier reveals and report retries do not
block the current epoch. Success requires finalized commitment/reveal readback,
`LastUpdate`, the exact revealed vector, and unchanged rewarded UID ownership.
Commitment inclusion alone is not success.

The local host can now access its own hotkey. Protect that machine and wallet
file. Gateway signatures authenticate Arena decisions; local signing does not
independently prove that Arena accepted the correct scores. No receipt graphs,
release identity, model ancestry, or legacy Research Lab verification is used
to authorize these weights.

## Run a validator

Install the repository's existing Python dependencies in a Python 3.11 virtual
environment. The pinned Bittensor 10.5 runtime supplies drand 2.x; the old
Bittensor 9/drand 1 host environment cannot submit stateful commitments.
Create a new environment instead of upgrading the old auditor environment in
place. That can leave incompatible legacy SCALE packages installed. Keep the
existing wallet and validator state paths when changing environments.

From the updated repository checkout:

```bash
python3.11 -m venv .venv-arena
. .venv-arena/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`setup.py` uses this same dependency list. A built package also includes the
local signer's Python modules, public chain profiles, and SN71 epoch mapping.
It does not require Nitro or enclave tooling.

Use Linux x86_64 and the existing runsc setup to score models. The current
sandbox uses rootful namespaces. For full scoring, use the supplied systemd
service, which runs this same entry point as root, or run the command below
from a root shell with the validator's wallet path set explicitly. An ordinary
unprivileged shell can submit weights but cannot launch this sandbox as
configured. This is an existing sandbox requirement, not a signing mode.
Root inside a restricted container may still lack the required mount and
namespace capabilities; use the installed-runtime probe below to verify them.

Install the complete matching gVisor package or release bundle. Current releases
can require the companion `gvisor-bin/` directory beside `runsc`; copying only
the executable may leave an incomplete installation. Follow the
[official gVisor installation instructions](https://gvisor.dev/docs/user_guide/install/).
Select its actual absolute path with `LAB_ARENA_RUNSC_PATH` in the service's
configuration, or with `--arena-runsc-path`. The command-line flag takes
precedence, then the environment, then `/usr/local/bin/runsc`. The validator
does not search `PATH` or download a replacement binary. For an installation
at `/usr/bin/runsc`, explicitly configure that path instead.

For Finney SN71, the validator includes the public RPC endpoint, gateway URL,
and trusted gateway signing-key hash. No API keys or public-configuration
exports are needed. Keep existing state and work paths on updates. On a first
installation, choose persistent writable directories:

```bash
export LAB_ARENA_VALIDATOR_STATE_DIR="$PWD/validator-state"
export LAB_ARENA_RUNNER_WORK_DIR="$PWD/arena-runner"
export LAB_ARENA_RUNSC_PATH=/usr/local/bin/runsc

python neurons/validator.py \
  --netuid 71 --subtensor.network finney \
  --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY \
  --wallet.path /absolute/path/to/YOUR_WALLETS_DIRECTORY
```

For a local **Finney node**, keep the network identity and add the RPC endpoint:

```bash
# Add to the validator command above:
--subtensor.chain_endpoint ws://127.0.0.1:9944
```

The endpoint must serve both WebSocket and HTTP JSON-RPC at the same origin.
Use `wss://` for a remote node; plaintext `ws://` is permitted only on loopback.
The node must follow the selected network and support the finalized reads used
by the validator. Finney archive proofs use the official public archive by
default. To use an operator-owned archive, set `LAB_ARENA_ARCHIVE_ENDPOINT` or
pass `--arena-archive-endpoint`; the command-line flag takes precedence. Test
networks use the selected live endpoint as the archive default.

Archive endpoints must be credential-free WebSocket or HTTP origins. The signer
converts `wss://` to `https://` and `ws://` to `http://` for JSON-RPC, so the
same endpoint must accept HTTP POST requests. Remote endpoints require TLS with
normal certificate verification. There is no insecure TLS bypass. Plaintext is
allowed only on a loopback address. Select an archive that you operate and
trust because its raw RPC replies remain a trust input. The existing genesis,
cutover, exact-block, runtime-profile, and signing checks reject inconsistent
chain data; they do not authenticate an arbitrary archive operator.
Do not put an RPC URL in `--subtensor.network`, and do not set the endpoint to
the bare word `finney`. A TLS connection reset means the RPC connection failed;
check the node, TLS proxy, and network path without disabling certificate checks.

An explicit `--subtensor.chain_endpoint` overrides `LAB_ARENA_CHAIN_ENDPOINT`;
otherwise that environment setting overrides the network's public endpoint.
Existing `LAB_ARENA_API_BASE_URL` and `LAB_ARENA_SIGNING_KEY_HASH` overrides
remain supported. A different gateway, network, or subnet requires its explicit
trusted gateway URL and signing-key pin. Finney SN71 defaults are not used to
trust a custom gateway. Use the trusted pin supplied by the subnet operator if
it changes. Do not blindly accept a key fetched from a gateway.
The hotkey must already exist as a private regular wallet file (mode 0600).
The validator does not create or replace wallets.

`scripts/run_arena_validator.py --environment-file ...` loads
`LAB_ARENA_ARCHIVE_ENDPOINT` from its protected environment file. The direct
`neurons/validator.py` entry point reads the current process environment but
does not load a `.env` file by itself.

For example, add the selected origin to the existing private service file:

```dotenv
# /absolute/path/to/arena-validator-service.env (mode 0600)
LAB_ARENA_ARCHIVE_ENDPOINT=wss://archive.operator.example:443
```

Then run the existing launcher and verify readiness before normal service use:

```bash
sudo /absolute/path/to/.venv-arena/bin/python \
  scripts/run_arena_validator.py \
  --environment-file /absolute/path/to/arena-validator-service.env \
  --check-only
```

For the direct `neurons/validator.py` command, pass
`--arena-archive-endpoint wss://archive.operator.example:443` instead.

For a service, the equivalent wallet settings are `LAB_ARENA_WALLET_NAME`,
`LAB_ARENA_HOTKEY`, and `LAB_ARENA_WALLET_PATH`.
`LAB_ARENA_EXPECTED_HOTKEY` can additionally bind deployment to an existing
public identity. Use an absolute wallet path in services.
Test networks require their matching epoch mapping and an explicit
`LAB_ARENA_BURN_HOTKEY`.

Append `--check-only` to verify local signing access, the gateway key pin,
and finalized chain identity without broadcasting or claiming work.
This check does not verify scoring. Keep it as the mandatory service startup
check so a missing scoring dependency cannot stop weight submission.
`--once` runs one scoring poll and one weight cycle; it is not a substitute
for a continuously supervised process or finalized reveal verification.

Run only one process per hotkey. Keep the same weight state directory across
updates and retries. Never put provider keys or wallet seed values in service
configuration, logs, or Git.

## Diagnose scoring setup

Run `--check-scoring-only` with the same interpreter, runtime path, work path,
and account as the validator. It checks Linux x86_64, the selected executable,
rootful service identity, a five-second `runsc --version` invocation, and runner
directory access, including `sandboxes`, `runs`, `images`, `sources`, and the
fixed `/tmp` socket root. It creates missing named runner directories with private
permissions and removes only its own temporary write probes. Existing runner
root permissions, wallets, journals, and cached files are preserved. Existing
`sandboxes` and `runs` directories retain the normal private-mode requirement.
Configured work-directory symlinks, including parent components, are rejected
without following or replacing them. The fixed OS `/tmp` alias is resolved and
probed without changing shared-directory permissions.

For a service installation, from the release checkout:

```bash
sudo /absolute/path/to/.venv-arena/bin/python \
  scripts/run_arena_validator.py \
  --environment-file /absolute/path/to/arena-validator-service.env \
  --check-scoring-only
```

Use your installed interpreter and service environment file. The command reads
no wallet, contacts no gateway or chain, claims no work, and broadcasts nothing.
It exits nonzero with a reason on failure. Success explicitly reports
`sandbox_execution=not_checked`; it does not prove mounts, sandbox execution,
gateway authorization, or accepted scoring. This mode is mutually exclusive
with `--check-only` and `--once`.

| Reason | Required action |
| --- | --- |
| `runsc_path_invalid` | Set the absolute path to the installed executable. |
| `runsc_missing` | Install runsc or correct the configured path. |
| `runsc_not_executable` | Check the file type, execute permissions, parent access, and noexec mounts. |
| `runsc_unusable` | Check executable architecture and the complete matching installation. |
| `runsc_probe_timeout` | Investigate why the installed runtime's version check hangs. |
| `runsc_probe_cleanup_failed` | Inspect the host for a stuck version-check process; forced cleanup could not be confirmed. |
| `unsupported_host` | Use Linux x86_64 for this Arena integration. |
| `root_required` | Use the configured rootful service with the existing wallet path. |
| `unsafe_work_directory` | Inspect the reported path; preserve its contents and correct the configuration. |
| `work_directory_unwritable` | Check ownership, permissions, read-only mounts, and available storage. |
| `sandbox_launch_failed` | Run the installed-runtime probe and investigate sandbox capabilities. |

After host checks pass, exercise the actual installation with the existing
probe from the same release and with equivalent service privileges:

```bash
sudo /absolute/path/to/.venv-arena/bin/python \
  scripts/_lab_arena_runsc_probe_ci.py \
  --runsc-path /usr/local/bin/runsc
```

Substitute the exact runtime path used by the validator. Always pass
`--runsc-path` when diagnosing an installed service: omitting it downloads a
separate runtime for CI and can conceal a broken service installation.
`--dry-run` validates only the probe plan. A real pass ends with
`LAB_ARENA_RUNSC_PROBE_SUCCESS` and covers execution, worker sockets, read-only
filesystem behavior, timeout, output limits, the agent entrypoint, and cleanup.
For networking it checks the `--network=none` command and a failed outbound
connection smoke test. An unreachable destination can also make that connection
fail, so this test alone does not independently prove network isolation.
The probe uses local provider fixtures and no production leases or credentials.

Then verify a real eligible Arena job reaches an accepted completion, and
independently verify finalized weight reveal for that validator. An empty queue
or a live process proves neither. `included_pending_reveal` means the commitment
is included while reveal confirmation remains pending. Preserve all weight
journals through repairs and restarts. Check every affected validator separately.

## Deployment

Apply committed migration `208-lab-arena-validator-scoring-authority.sql`
before restarting the gateway to this code. It removes the duplicate SQL
runner-list gate. SQL claims remain restricted to the gateway service role.
Migrations through 207 remain prerequisites. This is not a change to scores,
promotion, rewards, or provider accounting.

The validator-only weight access correction needs no additional migration.
It does require both the updated gateway and updated normal validator client:
the client now sends a signed POST instead of an unauthenticated GET. Keep
validator journals and wallets unchanged during the paired update. There is no
anonymous compatibility fallback. Operators of external validators must also
update their client; no new wallet or configuration setting is needed.

The 75,000 scoring stake gate itself needs no additional migration.
Before enabling it, verify qualifying planned capacity for every open round.
Use the canonical gateway restart and retain existing runs and validator
journals. Rolling back to a release without this gate reopens low-stake claims;
use the existing operator claim pause if the minimum must remain enforced
while repairing a rollback. Weight-state retrieval remains available.

Use the canonical gateway restart. For the validator, prepare a mode-0600
environment file with the public configuration and local wallet path, then run
the exact pushed main controller:

```bash
git -C /home/ec2-user/leadpoet/leadpoet fetch --no-tags origin main
git -C /home/ec2-user/leadpoet/leadpoet show "$SHA:validator_restart.sh" \
  | VALIDATOR_DEPLOY_COMMIT="$SHA" bash
```

The controller stages exact committed source, preserves the journal/work
directories, and runs local-wallet readiness before draining the active
service. It then switches the service and retains the old release and private
configuration for rollback. It does not export an enclave key or create a new
hotkey. Use an existing owner-held wallet for the transition.

The primary controller defaults to
`/home/ec2-user/arena-validator-venv311/bin/python3`. Install the committed
`requirements.txt` in that dedicated environment before restarting, or set
`VALIDATOR_PYTHON_BIN` to an already prepared compatible environment. Keep the
old environment intact so rollback can still start the previous service.

The gateway retains its existing read-only ECR access and trusted scorer
repository setting. The validator does not need those settings. No additional
database migration or artifact-staging service is needed for scorer downloads.

The sample systemd service supervises the same normal validator. SIGTERM stops
new claims and drains current work. The old enclave need not be terminated to
run the local path; the controller does not manage unrelated enclave services.

## Validation

Run the focused gate in `docs/v2_deployment_verification_checklist.md`.
It covers local-wallet signing, safe restart/retry, finalized reveals, idle
weight submission, scoring with brokered miner credentials, and registered
validator authorization without runner lists, the exact stake boundary,
completion after a stake-only decrease, and below-threshold weight submission.

Report controlled tests, deployment readiness, and live finalized outcomes
separately. Do not claim Yuma or Rizzo is working solely because the primary is
working; each operator must run the updated process with their own wallet.
