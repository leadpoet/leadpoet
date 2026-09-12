# Open Source Agent Competition operator guide

The Arena is a simple agent-bundle competition with a UTC daily cycle:

1. Day 0: generate twenty ICPs, keep them private, and accept model submissions.
2. Day 1: close Day 0 submissions and evaluate the frozen baseline and every
   accepted model on the same private set.
3. Publish source, aggregate final scores, promotion, and rewards as soon as
   evaluation is complete. There is no additional source delay.
4. Day 2: for published rounds with `after_scoring_day2_v1`, reveal the main
   benchmark, confirmation benchmark, outputs, run results, and per-ICP scores
   together. A cancelled round can reveal only its valid committed benchmark.
   The boundary is the exact submission cutoff plus 24 hours. Active overruns
   stay private after that time.

Day 1 also starts a new hidden set and submission window for Day 2. The existing
two ten-ICP batches are execution details; new rounds do not eliminate models
between them. The round ID names the evaluation day. `icp_set_date` names the
previous submission day's bank, which remains fixed during restart and retry.
Historical rounds retain their actual bank date; they are not relabelled as a
previous-day evaluation. Rounds without the frozen delayed-disclosure marker
retain their historical disclosure behavior.

The midnight submission cutoff starts the readiness-driven execution and
scoring batches. Completed work moves to the next batch without fixed 00:30 or
11:00 UTC idle periods. Stage start fields remain nominal capacity-budget
boundaries, while stage close fields bound work that is still incomplete.

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

Acceptance queues a full-source review. A separate gateway worker sends every
file and bundled prompt to `anthropic/claude-sonnet-5`, using the OpenRouter
runtime key encrypted for that exact submission and miner. It has no organizer
key fallback. The daily organizer baseline is exempt; a miner submitting the
same baseline code still requires review with the miner's key.

Only `code_review_status=passed` can freeze a miner into evaluation. The source
must be complete UTF-8 text and fit in the 1M-token judge context, including
output headroom. Requests disable context compression. Missing file coverage,
truncated output, malformed verdicts, and unknown charges are incomplete reviews.
The worker retries incomplete reviews at most three times, with a 60-second
backoff. A live review claim expires after ten minutes so a restart can recover.
After the normal submission cutoff, unresolved reviews may finish within the
existing benchmark preparation window. Four reviews can run at once. At the
benchmark deadline, submissions without a pass are excluded with
`code_review_incomplete`; a confirmed rejection uses `code_review_rejected`.
This does not change the daily evaluation schedule or numerical scoring.

Review charges use `openrouter.code_review` in `lab_arena_ledger`, with
`funding_source=miner_key`. They remain separate from sourcing cost eligibility.
The submission status API exposes review status, file/byte counts, and cost.
Only verdicts, counts, and category codes are persisted. Unknown provider charges retain their reservation
as uncertain; neither a worker crash nor a malformed response makes a free pass.
Apply migration `207-lab-arena-code-review.sql` after migration 206 before
restarting the gateway. The gateway preflight checks the review schema.

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

Before a miner wins the live competition, the host keys fund the baseline.
After promotion, daily baseline execution uses the winning submission's stored
keys. The owner is frozen from the completed promotion in the same competition
mode and chain. Later organizer edits to the baseline source do not change that
owner. Baseline judging continues to use the organizer's keys.

An account-related rejection gets the initial attempt plus three retries. The
broker records each attempt, then permanently selects the corresponding host key
for that provider within the daily round. The affected ICP starts again with a
fresh run; previous accepted ICPs remain unchanged. A key is optional until the
model requests that provider. Provider outages and network failures keep the
normal recovery path. The next daily round tries the miner keys again.

Fallback halves the signed champion reward factor once, relative to the current
configured weekly share. Repeated failed days stay at one half. Every baseline
execution and its judgment must succeed without fallback to clear an existing
penalty. A new competition winner starts at the full factor. Partial success or
an older round cannot restore a reduced factor.

Provider costs remain in the existing ledger. An unknown miner charge from a
confirmed account rejection remains uncertain. Once that provider's fallback
is durable, its retired reservation no longer blocks replacement execution
under the existing execution cap. Settled costs and unrelated uncertain charges
still count against that cap. A retry blocked by this reservation is recorded as
an admission refusal, not as another upstream dispatch. Apply migration
`227-lab-arena-champion-funding.sql` before deploying this behavior.

A competing model's
OpenRouter runtime key and Deepline key are submitted separately, encrypted in
the gateway vault, and attached to that submission's execution and judge calls. An
optional miner Scrapingdog key is encrypted in the same vault and used only when
that model calls Scrapingdog. Only submissions with that key receive a fixed
non-secret `SCRAPINGDOG_API_KEY` runtime handle; the actual key stays in the broker. The
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

### Provider costs

The model's twenty-ICP sourcing allowance is $50 across all three providers,
including failed attempts and retries. OpenRouter uses the provider's
`usage.cost`, not the admission-only management key. Scrapingdog uses the
existing endpoint credit map at $0.00005 per credit; approved routes absent
from that map use five credits, and a company profile uses ten. Deepline uses
the per-call billing in a valid completed response at $0.10 per credit. Its
billing history can aggregate multiple requests, so it is a fallback only for
an exact unshared job. The known-free company search and Hunter discovery can
settle a valid completed response at zero. Other missing or ambiguous charges
retain their reservation; they are not free. Billing-history reads are bounded
and private. They never reach the model.

Reservations and settlement share the existing submission lock. Concurrent
ICPs cannot each claim a fresh budget. Dynamically priced Deepline calls
reserve the remaining allowance and run one at a time per submission. Their
upstream API has no per-call dollar cap: one completed call can exceed its
reservation. Record the full actual charge, block further paid calls, and
exclude over-budget challengers from promotion. Do not describe this as an
absolute upstream charge ceiling.

At publication, the gateway reports provider totals and compares sourcing
cost with the smaller of $50 and $0.50 times returned companies. It counts
one accepted output per ICP and unique company domains within that output.
All retry costs still count. Quality scores remain unchanged; a cost-ineligible
challenger cannot win. Independent judge cost has its own existing $50 cap
and is reported separately. Credentials and provider payloads are not part of
the public cost summary.

Migration 206 changes accounting functions, not historical rows. A legacy live
round that is still open adopts the new limits atomically when its benchmark
commits. Already committed rounds and explicit shadow-test limits do not change.

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
- `LAB_ARENA_RUNNER_HOTKEYS`: planned runner capacity. New rounds count only
  validators eligible under the shared gateway rule: on mainnet, registered
  permitted validators with at least 75,000 effective stake weight. Testnet
  retains its active-or-permitted rule without the mainnet stake minimum.
  This is not an access allowlist: any registered permitted validator meeting
  the minimum can claim, including inactive validators. No qualifying planned
  capacity means no new round; existing rounds and weight-state retrieval
  continue. The finalized chain cache refreshes approximately every 60 seconds.
  A stake-only drop stops new claims after refresh and lets existing leases
  finish under their existing rules. Weight retrieval has no benchmark minimum.
- `LAB_ARENA_BASELINE_HOTKEY`: the registered hotkey that owns each daily
  public baseline entry
- `LAB_ARENA_BASELINE_SOURCE_URL`: optional in live mode. The only live daily
  baseline source is the promoted `leadpoet/pydantic-harness` `lab` branch.
  Remove an old `main` override before service startup. Shadow mode can set a
  different public HTTPS candidate archive.

Common optional values are `AWS_REGION`, `LAB_ARENA_NETUID`,
`LAB_ARENA_NETWORK`, `LAB_ARENA_CHAIN_TIMEOUT_SECONDS`,
`LAB_ARENA_DAILY_CUTOFF_UTC` (default `0`),
`LAB_ARENA_MAX_CHALLENGERS` (default `20`, hard schema limit `256`),
`LAB_ARENA_MAX_IMAGE_BYTES` for the trusted scorer image,
`LAB_ARENA_POOL_PERCENT`, and
`LAB_ARENA_BANNED_HOTKEYS_PATH`. `LAB_ARENA_REWARDS_ENABLED` defaults to
`false` and is frozen into each new round. `LAB_ARENA_SIGNING_KEY_ID` is
needed only when a live, reward-enabled published round is activated.

The challenger limit excludes the baseline. Each hotkey can have one accepted
model per daily round, without replacement; different hotkeys can share a
coldkey. The configured admission limit is not reduced by the conservative
runner workload estimate. Worker concurrency, stage deadlines, and spending
limits remain enforced; a full round can require more runner capacity.

`LAB_ARENA_BENCHMARK_DISCLOSURE_FROM` is an optional aware timestamp, normalized
to UTC. For example, `2026-09-13T00:00:00Z` freezes
`benchmark_disclosure_policy=after_scoring_day2_v1` into each newly created
round whose cutoff is on or after that instant. An existing round keeps its
stored configuration. An unknown or null stored policy fails closed. Before
setting this value, deploy this reader to every service that can serve Arena
public routes or create rounds. Do not roll back to a version that ignores the
marker while a marked round still needs privacy. No database migration is
required because the existing immutable `configuration_doc` stores the policy.

For the September 13 rollout, the already-created September 12 round remains
on its stored legacy policy. The intended activation marks only a newly created
round with a cutoff on or after `2026-09-13T00:00:00Z`. Check the narrow update
before its authorized apply:

```bash
python3 scripts/configure_lab_arena_production.py \
  --benchmark-disclosure-from '2026-09-13T00:00:00Z' \
  --ssh-key /protected/path/to/key --allowed-account ACCOUNT_ID --check

LEADPOET_LAB_ARENA_PRODUCTION_APPLY=1 \
python3 scripts/configure_lab_arena_production.py \
  --benchmark-disclosure-from '2026-09-13T00:00:00Z' \
  --ssh-key /protected/path/to/key --allowed-account ACCOUNT_ID --apply
```

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
then `scripts/205-lab-arena-optional-scrapingdog-credential.sql` and
`scripts/206-lab-arena-combined-provider-budget.sql`
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

The canonical gateway restart pauses new claims after its release,
attestation, and maintenance preflight. It then waits for every captured lease
to have a persisted accepted result or authenticated terminal failure with
closed accounting. A lease expiry, worker loss, changed lease generation, or
missing result stops the restart before shutdown and restores the prior
operator pause state. Reported failures keep the normal retry assignment; the
restart does not convert them to accepted work.

A failed gateway restart keeps the guard after a destructive phase. A canonical
retry by the same retained invocation resumes the gateway restart. If the
candidate advances, the same owner can change the
guard target with a generation-checked operation after the new candidate has
passed the normal preflight. The captured leases, operator pause, and
destructive phase stay unchanged. The gateway releases its claim guard after
its runtime and Arena service pass readiness checks.

The paired normal-validator restart checks the local wallet, gateway signing-key
pin, and finalized chain identity before stopping the old service. It drains
scoring work and preserves signed weight bytes and recovery state. Verify that
both hosts run the intended release, then check automatic weight submission and
finalized commitment/reveal readback. See
[Normal Arena validators](../docs/arena_normal_validator_weights.md).

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
Deepline API key from environment variables or masked prompts. It also accepts
an optional Scrapingdog key from `SCRAPINGDOG_API_KEY` or a masked prompt. It
archives, uploads, signs, and finalizes the source. Runtime API keys are sent
separately and encrypted for the model's runs. The management key is used only to check
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

## Rewards and competition controls

The live Arena service persists the published reward basis and accepted weight
state in Supabase. A registered, permitted validator requests that state with
its local hotkey signature. It verifies the gateway signing-key pin, accepted
state, reward basis, and finalized UID ownership before deriving and submitting
weights. The benchmark stake minimum applies to new scoring work, not weight
retrieval. A missing, invalid, conflicting, or unreachable accepted state stops
that weight submission.

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

Keep the live Arena service available for weight-state requests while pausing
new competition work. Set `LAB_ARENA_DAILY_CUTOFF_UTC=disabled` to stop automatic
creation of new rounds; existing rounds continue. Set
`LAB_ARENA_REWARDS_ENABLED=false` to create future rounds with rewards disabled.
This setting is frozen into each round and does not change existing rounds or
their governing reward history. Validators continue their independent weight
loop. Setting `LAB_ARENA_MODE=off` or stopping the Arena service also removes
weight-state availability and is a service shutdown, not a scoring-only pause.

## Focused checks

For Arena changes, run the directly affected tests and:

```bash
git diff --check
python3 -m py_compile lab_arena/*.py scripts/lab_arena_*.py scripts/run_lab_arena_*.py
```

Never place provider keys, database tokens, registry passwords, or signing-key
material in source, test fixtures, command output, or public results.
