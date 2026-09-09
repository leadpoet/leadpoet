# Open Source Agent Competition operator guide

The Arena is a simple agent-bundle competition with a UTC daily cycle:

1. Day 0: generate twenty ICPs, keep them private, and accept model submissions.
2. Day 1: close Day 0 submissions and reveal all twenty of Day 0's ICPs.
   Evaluate the frozen baseline and every accepted model on that same set.
3. Publish the source, final score, and all twenty per-ICP scores as soon as
   evaluation is complete. There is no additional 24-hour source delay.

Day 1 also starts a new hidden set and submission window for Day 2. The existing
two ten-ICP batches are execution details; new rounds do not eliminate models
between them. The round ID names the evaluation day. `icp_set_date` names the
previous submission day's bank, which remains fixed during restart and retry.
Historical rounds retain their actual bank date; they are not relabelled as a
previous-day evaluation.

Code previews and promotion to public Git branches use the same completed
evaluation gate. Queued, incomplete, and cancelled evaluations do not release
submitted code. Provider credentials are never public. The CLI's public-rerun
consent covers source disclosure.

## Competition boundary

A miner submits one local source directory. The helper:

1. requires `harness.py`;
2. creates one sorted, size-limited archive with normalized metadata;
3. gets a private upload target from the Arena;
4. uploads the archive; and
5. signs a final request so the Arena can validate and accept the bytes.

No Dockerfile, public registry, image tag, commit identity, receipt, source
digest, or release manifest is part of miner admission. The service validates
the declared archive size and safe source structure, then uses its own
submission ID for execution and recovery. The private source reference is
write-once. A miner can use any harness, model, prompts, packages, routing, or
orchestration behind the one callable.

The source contract is `harness.run_icp(icp) -> list[dict]`. `harness.py` can
define the function or re-export it from vendored source. The function must be
synchronous and have exactly one positional parameter. It cannot have
keyword-only parameters, `*args`, or `**kwargs`. It returns at most five
company objects. The public baseline README gives the full input and output
example:
[`leadpoet/pydantic-harness`](https://github.com/leadpoet/pydantic-harness).

Vendored Python modules run directly from the read-only source mount. An
optional `requirements.txt` can contain package names and version constraints.
The runner installs binary wheels only into a bounded cache and mounts those
dependencies read-only. It rejects URLs, local paths, nested requirements,
VCS dependencies, and source builds. The common trusted scorer image supplies
Python for every agent; it is not a miner image or a miner identity.

The organizer supplies host provider keys for execution and judging of the
public baseline:

- `LAB_ARENA_OPENROUTER_API_KEY`
- `LAB_ARENA_SCRAPINGDOG_API_KEY`
- `LAB_ARENA_DEEPLINE_API_KEY`

The host keys are used for baseline traffic. A competing model's
OpenRouter runtime key and Deepline key are submitted separately, encrypted in
the gateway vault, and attached to that submission's execution and judge calls. The
matching OpenRouter management key is used for admission validation and then
discarded. The miner funds those upstream calls. The validator receives an
opaque runtime lease and cannot read the credentials; submitted code receives
provider access only through the broker transport.

The broker permits any model in the organizer-fetched OpenRouter catalog that
has usable pricing. It still enforces the fixed call, token, cost, privacy, and
time limits. The trusted judge can use only its configured judge models.

A shared provider account failure, rate limit, or provider server failure is
an infrastructure failure. It does not give a miner a score of zero. A real
caller error, such as invalid request data, is returned to the bundle.

The judge uses bounded retries. If no valid judge result is available after
those retries, the service cancels the incomplete round before publishing a
ranking. It does not exclude one challenger for a judge or provider failure.
A successful accepted retry takes precedence over a failed attempt. A miner's
own credential or budget failure retains its existing ineligibility rule.
Malformed accepted scoring artifacts also cancel the round before scores are
recorded; they are not company-verification failures.

## Required service configuration

Set these values on the Arena service host:

- `LAB_ARENA_MODE`: `shadow` or `live`. Use `off` to disable the Arena.
- `LAB_ARENA_SUPABASE_URL`
- `LAB_ARENA_SUPABASE_ANON_KEY`
- `LAB_ARENA_SERVICE_KEY`: preferred production credential. Use a scoped
  `sb_secret_` API key whose JWT template role is `lab_arena_service`.
  `LAB_ARENA_SERVICE_JWT` remains a legacy fallback for parity environments.
- `LAB_ARENA_BUCKET`
- `LAB_ARENA_CHAIN_ENDPOINT`
- the three host provider keys listed above
- `LAB_ARENA_SCORER_IMAGE`: a public tag or digest; startup resolves it to a
  digest for trusted scoring
- `LAB_ARENA_RUNNER_HOTKEYS`: the runner hotkeys allowed to claim work
- `LAB_ARENA_BASELINE_HOTKEY`: the registered hotkey that owns each daily
  public baseline entry
- `LAB_ARENA_BASELINE_SOURCE_URL`: optional in live mode. The only live daily
  baseline source is the promoted `leadpoet/pydantic-harness` `lab` branch.
  Remove an old `main` override before service startup. Shadow mode can set a
  different public HTTPS candidate archive.

Common optional values are `AWS_REGION`, `LAB_ARENA_NETUID`,
`LAB_ARENA_NETWORK`, `LAB_ARENA_CHAIN_TIMEOUT_SECONDS`,
`LAB_ARENA_DAILY_CUTOFF_UTC` (default `0`),
`LAB_ARENA_MAX_CHALLENGERS` (default `16`, hard limit `256`),
`LAB_ARENA_MAX_IMAGE_BYTES` for the trusted scorer image,
`LAB_ARENA_POOL_PERCENT`, and
`LAB_ARENA_BANNED_HOTKEYS_PATH`. `LAB_ARENA_REWARDS_ENABLED` defaults to
`false` and is frozen into each new round. `LAB_ARENA_SIGNING_KEY_ID` is
needed only when a live, reward-enabled published round is activated.

Apply `scripts/179-lab-arena-v1.sql` and
`scripts/180-lab-arena-daily-competition.sql`, then
`scripts/181-lab-arena-source-submissions.sql` and
`scripts/182-lab-arena-source-execution.sql`,
`scripts/183-lab-arena-miner-reward-basis.sql`, and
`scripts/184-lab-arena-scoring-failure-isolation.sql`,
`scripts/185-lab-arena-miner-credentials.sql`,
`scripts/187-lab-arena-promotion-threshold.sql`, and
`scripts/188-lab-arena-baseline-promotion.sql`,
`scripts/189-lab-arena-round-network-scope.sql`,
`scripts/190-lab-arena-restart-claim-drain.sql`, and
`scripts/193-lab-arena-upload-recovery.sql`,
`scripts/194-lab-arena-open-scorer-refresh.sql`, then
`scripts/197-lab-arena-reward-chain-scope.sql` with the database owner
before service startup. Then check the service wiring:

`scripts/191-lab-arena-upload-recovery.sql` remains byte-identical only because
an earlier production snapshot records that applied path. Do not apply it to a
new database. Migration 193 is the current forward upload-recovery migration.

```bash
python3 scripts/run_lab_arena_service.py --check-only
```

Migration 193 adds safe replacement of unfinished uploads and accurate
`execution_incomplete:stageN:count` / `scoring_incomplete:stageN:count`
cancellation labels. It preserves historical results and source objects.
Deploy its matching service after applying the migration. Source admission
still uses the existing upload MD5 and server-assigned submission ID.

Migration 194 refreshes only the trusted scorer digest and pinned reference
when an existing open round atomically commits its benchmark. This lets a
deployed scorer fix apply before any work is created. The committed scorer
pin and every other round setting remain immutable.

Start the service:

```bash
python3 scripts/run_lab_arena_service.py --host 127.0.0.1 --port 8792
```

## Canonical restart claim drain

Migration 190 installs the durable claim gate used by the canonical gateway
and validator restart. Its first installation takes the rounds and runs table
locks with `NOWAIT`. If live Arena work holds either table, the complete
migration transaction fails without cancelling that work. Retry the same
idempotent migration through the repository migration helper after the writer
finishes.

An already-running schema-189 Arena service can finish its current work while
migration 190 is applied. A runner can continue against the replacement
service, but canonical paired authority still requires both components at the
exact release. After the database reports schema 190, an older schema-189 Arena
service cannot newly start because its startup schema check rejects the
mismatch. Therefore, schema 190 and the matching candidate runtime form one
cutover dependency. Do not use an older Arena service as a claim-capable
rollback after this migration.

The canonical restart pauses new claims after its existing release,
attestation, and maintenance preflight. It then waits for every captured lease
to have an accepted receipt or an authentic terminal failure receipt with
closed accounting. A lease expiry, worker loss, changed lease generation, or
missing receipt stops the restart before shutdown and restores the prior
operator pause state. Reported failures keep the normal retry assignment; the
restart does not convert them to accepted work.

A failed restart keeps the guard after a destructive phase. A normal canonical
retry by the same retained invocation repeats the complete gateway and
validator path. If the exact candidate advances, the same owner can change the
guard target with a generation-checked operation after the new candidate has
passed the normal preflight. The captured leases, operator pause, and
destructive phase stay unchanged. The controller releases claims only after
the joined gateway and validator readiness manifest passes. There is no
separate completion or release-only path.

The service creates a daily round at 00:00 UTC by default. Set
`LAB_ARENA_DAILY_CUTOFF_UTC` to select another hour, or create one manually:

```bash
python3 scripts/lab_arena_admin.py create --cutoff 2026-09-05T00:00:00Z
```

Each round freezes `LAB_ARENA_NETWORK` and `LAB_ARENA_NETUID` in its
configuration. API and driver instances only select rounds in their configured
chain scope. Historical rows without these fields are treated as Finney/netuid
71. This permits a pinned testnet service to use the shared database without
redirecting or advancing a Finney round. Reward activation and published reward
history are independent for each chain scope. Baseline Git publication keeps
its existing serialized ordering against the shared branch.

## Runner configuration

Each runner needs Linux AMD64, root access for the sandbox mounts, and an
executable gVisor `runsc`. It also needs:

- `LAB_ARENA_MODE`: the same `off`, `shadow`, or `live` mode as the service;
  `off` stops the runner
- `LAB_ARENA_API_BASE_URL`
- `LAB_ARENA_WALLET_NAME`
- `LAB_ARENA_HOTKEY_NAME`
- `LAB_ARENA_WALLET_PATH` when the wallet is outside the default wallet path
- `LAB_ARENA_RUNNER_WORK_DIR`
- `LAB_ARENA_RUNSC_PATH`

`LAB_ARENA_MAX_PARALLEL_RUNS` (default `8`, maximum `8`) and
`LAB_ARENA_ROUND_ID` are optional. Provider
keys, database access, source upload access, and the signing key stay on the
service host. The runner needs read access to the organizer's common trusted
Python/scorer image. It downloads source only through the active run lease,
then mounts source, installed wheels, and the host-owned entrypoint read-only
inside gVisor. Start the runner with:

```bash
python3 scripts/run_lab_arena_runner.py
```

## Miner flow

Choose **Submit Model** in `neurons/miner.py`. It reads the local source
directory and the miner's OpenRouter API key, OpenRouter management key, and
Deepline API key from environment variables or masked prompts. It archives,
uploads, signs, and finalizes the source. Runtime API keys are sent separately
and encrypted for the model's runs. The management key is used only to check
admission and is not stored. No Dockerfile or image tag is required. The
same helper can run directly with those credentials in the environment:

```bash
python3 scripts/lab_arena_miner.py submit-model --source ./my-agent \
  --wallet-name default --hotkey-name default
```

See the repository README for example bundles and schema links. An example is
documentation only; it is not part of admission or scoring.

At the first round cutoff, the service automatically admits the configured
public baseline archive through the same source-admission checks. A temporary
download or object-store failure is retried. An invalid baseline prevents the
round from starting. Each daily round resolves the current promoted `lab`
branch once when baseline execution starts, stores those bytes at the round's
private source reference, and uses only that frozen bundle for execution and
recovery. A later `lab` promotion affects the next round snapshot only. The
operator log reports the archive's ordinary Git commit comment when GitHub
provides it. A live round created before this policy can still show its old
creation-time URL, but an unfrozen download uses `lab`; an already stored or
registered bundle is not replaced. A model must score at least **1.0 point**
above the daily baseline mean on the existing 0–100 scale. A tie or a smaller
gain does not crown a new miner. The highest qualifying model wins.

The gateway publishes that winner's accepted source to both `main` and `lab`
with one atomic Git push. The new commit preserves both branches' history and
contains exactly the submitted source files. The gateway never executes source
while publishing it. Source bundles cannot contain Git metadata, export
attributes, or GitHub workflows that could run with repository credentials.

One ordinary promotion plan is saved on the round before the push. A failed
push leaves promotion pending. A lost push response or restart checks the same
plan and remote heads, then completes it without another promotion commit.
Concurrent branch changes fail closed; they are not overwritten or force-pushed.
There is no sequential fallback when the remote does not support atomic pushes.
The next baseline download and the winner's reward activation wait for promotion
to complete. Existing frozen round bundles do not change during recovery.

Configure one repository-scoped host credential: `LAB_ARENA_GIT_SSH_KEY_PATH`
for a write-enabled deploy key, or `LAB_ARENA_GITHUB_TOKEN`. Do not set both.
SSH requires a verified GitHub host key and a private key readable only by its
owner. Existing Git credential helpers can also provide HTTPS access. The
optional `LAB_ARENA_PROMOTION_WORK_DIR` holds a bare Git cache; losing this cache
does not lose the saved promotion plan. Credentials never enter miner runs.

Outside promotion, `main` remains a development branch. A later `main`-only push
does not affect production. The next daily round loads the promoted `lab` code;
the organizer still owns the baseline entry, while the winning miner remains
the reward payee.

## Rewards and independent disable controls

The Arena result is beside the retained reward settlement path. With Arena
rewards off, no Arena champion allocation is added.

If Arena rewards are enabled, the Arena and the reward-basis gateway must use
the same Supabase database. The published Arena basis must be visible to the
gateway/coordinator. Configure the Arena signing public-key hash and the Arena
reward enable flag on both the gateway/coordinator and validator as required by
the existing reward adapter. A missing, invalid, or unreachable governing
basis fails closed; it is not treated as an empty winner.

Competition publication is separate from reward activation. Publication
writes the participants, rankings, winner decision, and publication time
directly to the round row. It does not need KMS, an epoch read, a signed
receipt, a copied result bundle, or a replay. The driver later retries reward
activation for enabled live rounds, oldest first. Shadow rounds and rounds
created with rewards disabled cannot activate rewards later. New crowned rounds
also require completed baseline promotion before activation. Historical published
rounds are not retroactively promoted or rescored. The default champion pool is
25% of total emissions, subject to the existing epoch eligibility and decay rules.
An activated database record is not proof that chain weights were submitted;
verify canonical publication, validator submission, finalization, and readback.

To disable only the competition, set `LAB_ARENA_MODE=off` and stop the Arena
service and runners. To disable only Arena rewards, turn off the Arena reward
flag on the gateway/coordinator and validator. Neither action requires removal
of retained reward history.

## Focused checks

For Arena changes, run the directly affected tests and:

```bash
git diff --check
python3 -m py_compile lab_arena/*.py scripts/lab_arena_*.py scripts/run_lab_arena_*.py
```

Never place provider keys, database tokens, registry passwords, or signing-key
material in source, test fixtures, command output, or public results.
