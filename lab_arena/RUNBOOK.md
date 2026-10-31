# Open Source Agent Competition operator guide

The Arena is a simple agent-bundle competition with a UTC daily cycle:

1. Day 0: generate twenty ICPs, keep them private, and accept model submissions.
2. Day 1: close Day 0 submissions and evaluate the frozen baseline and every
   accepted model on the same private set.
3. Publish source, aggregate final scores, promotion, and rewards as soon as
   evaluation is complete. There is no additional source delay.
4. Day 2: for published rounds with `after_scoring_day2_v1`, reveal the main
   benchmark, outputs, run results, and per-ICP scores
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

## Private execution trajectories

Migration `365-lab-arena-trajectories.sql` adds one private table,
`lab_arena_trajectory_events`. It contains runtime start, finish, failure,
stdout and stderr events, plus provider request, response and error events for
OpenRouter, Deepline and ScrapingDog. Each row names the round, submission,
assignment, ICP position, run, attempt and validator. `model_role` distinguishes
the baseline from a miner; `run_kind` distinguishes execution from judging.
`icp_identifier` is the stable round ID plus ICP position.

The gateway attaches these identities from the stored run. Validators upload
runtime events to `/arena/v1/runs/{run_id}/trajectory` with their existing lease.
They need no Supabase key, provider key, new environment setting or additional
signature. The gateway records provider events where it makes the real calls.
Anonymous and publishable-key clients cannot read or write this table. Use the
existing private gateway database access for operator reads.

Runtime output is uploaded after the sandbox stops, in ordered chunks. Each
stream retains at most 64 KiB, with a truncation flag when needed. Provider
content is a bounded, redacted preview with operation, model, call sequence,
status, timing and billing metadata. Credentials and hidden reasoning fields
are removed. Events have UUIDs so upload retries do not duplicate them. These
events are diagnostics, not scoring evidence or validator attestations.

Uploads have bounded retries. A logging failure produces a private warning and
does not change model results, provider retries, billing or scoring. A killed
validator can lose its final buffered runtime output; provider events already
stored by the gateway remain available. This first version adds no log viewer
or automatic retention deletion.

## Capture summary reader

Leadpoet's admin Data capture tab reads captured competition model usage for
one submission. Public round data shows verified reasoning only for runs that
pass the training check, so the site needs this private aggregate.

Migration `593-lab-arena-capture-summary-reader.sql` adds the NOLOGIN role
`leadpoet_capture_reader` with a 60 s role statement timeout and no explicit
table grants. Its explicit EXECUTE grant is one SECURITY DEFINER function,
`lab_arena_capture_summary_v1(p_round_id, p_submission_id)`. The function
returns one row of counts for the submission's execute runs: runs, ICP
positions, completed OpenRouter calls, calls whose saved body reports reasoning
tokens, the token sum, and the first and last call times. It returns no hotkeys,
credentials, prompts, bodies or model output. Score runs and other submissions
are excluded. Counting uses the training export's terminal and token rules:
the latest terminal must be a saved 2xx success, and the reasoning count is the
largest non-negative integer with 1 to 19 digits at `usage.reasoning_tokens`,
`usage.completion_tokens_details.reasoning_tokens`
or `usage.output_tokens_details.reasoning_tokens`. A body that cannot be decoded
counts as a call without reported usage. If the token sum exceeds BIGINT's
maximum, the sum is NULL (unknown); the usage call count remains available.
These provider-reported counts do not prove export completeness or training
eligibility. The reader does not run the exporter's privacy or structure
sanitizer.

Existing database `PUBLIC` grants also apply to this role. `NOINHERIT` does not
remove them, and revoking a grant from the reader cannot override `PUBLIC`.
Review and record that effective surface; do not revoke unrelated shared grants
as part of this rollout. The distinct opaque reader key is restricted at nginx
to the canonical `POST /rest/v1/rpc/lab_arena_capture_summary_v1`, with JSON
arguments in the body. Other methods, paths, query strings, non-public schema
profiles and method/path override headers return 403. Explicit `public` profiles
are allowed. Both the normalized URI and original request target must match.
This is an HTTP boundary, not a claim of database-level one-function isolation.
The backend must remain loopback-only and its internal role JWT must remain
private (`deploy/aws/subnet-api/README.md`).

To enable it:

1. Apply 593 through the configured repository migration helper. It is
   idempotent and changes no stored row, table ACL or existing function. It
   grants the role to a role named `authenticator` only if one exists.
2. Review direct, inherited and `PUBLIC` privileges with the read-only queries
   in `deploy/aws/subnet-api/README.md`. Verify the reader-only HTTP restriction,
   private backend and exact summary function before granting the role to the
   reviewed RDS authenticator. A stopped permission review is not permission to
   activate the key. Do not rerun 593 after an unknown commit without readback.

   ```sql
   GRANT leadpoet_capture_reader TO <reviewed_authenticator>
     WITH INHERIT FALSE, SET TRUE;
   ```

3. Generate a new random `sb_secret_` key for the site only. Add it as
   `capture_reader_key` to the private renderer input, render both files and
   restart the pair as an explicit key rotation (README, "Private
   configuration"). The other keys stay unchanged.
4. Store the key as the site's `LEADPOET_ARENA_CAPTURE_KEY` (leadpoet-site
   `docs/runbooks/aws-ecs-cutover.md`, "Read the arena call ledger"). Never
   reuse the Arena, gateway or public key.
5. Verify with the new key: `POST /rest/v1/rpc/lab_arena_capture_summary_v1`
   with `{"p_round_id": ..., "p_submission_id": ...}` returns one row, and a
   table read such as `GET /rest/v1/lab_arena_ledger?limit=1` returns 403.
   Prove other methods, aliases and a catalog-confirmed harmless PUBLIC RPC
   are denied. Never invoke a mutating legacy RPC to test the boundary.

Rollback: remove `capture_reader_key` from the renderer input and restart the
pair, which revokes the key. To also close the database path, run
`REVOKE leadpoet_capture_reader FROM <reviewed_authenticator>` or
`REVOKE EXECUTE ON FUNCTION public.lab_arena_capture_summary_v1(text, text)
FROM leadpoet_capture_reader`. Neither step touches Arena data.

A call reads each completed call's saved body, so cost grows with the
submission's total response size. Local timing was about 2 ms per 20 KB body.
The role timeout ends an unusually large request instead of letting it run on.

## Competition boundary

A miner submits one local source directory. The helper:

1. requires `harness.py`;
2. creates one sorted, size-limited archive with normalized metadata;
3. gets a private upload target from the Arena;
4. uploads the archive; and
5. signs a final request so the Arena can validate and accept the bytes.

Acceptance queues a full-source review. A separate gateway worker sends every
file and bundled prompt to `anthropic/claude-sonnet-5.5`, using the OpenRouter
runtime key encrypted for that exact submission and miner. It has no organizer
key fallback. The daily organizer baseline is exempt; a miner submitting the
same baseline code still requires review with the miner's key.

Only `code_review_status=passed` can freeze a miner into evaluation. The source
must be complete UTF-8 text and fit in the 1M-token judge context, including
output headroom. Requests disable context compression. Missing file coverage,
truncated output, malformed verdicts, and unknown charges are incomplete reviews.
Explicit HTTP 408/429/5xx, transport, temporary preparation failures, unusable
reviewer responses, and inconclusive research assessments retry within the
existing admission window. Attempts are an audit count, not a rejection rule.
Backoffs are 60, 120, 240, 480, then 900 seconds after each failure. Each worker
invocation can dispatch one request. A ninety-minute window permits at most
nine starts per submission, even with zero provider latency.
Credential, credit, request, and invalid-source errors do not retry. Unknown
legacy errors and missing charges retain the legacy three-attempt limit and
60-second backoff. Accounting uncertainty is not unfinished review recovery.
A live review claim expires after ten
minutes so a restart can recover. Claims start only after source replacement
closes and before the existing benchmark deadline. A late result can settle
its charge but cannot pass the review or admit a submission.
After the normal submission cutoff, unresolved reviews may finish within the
existing benchmark preparation window. Four reviews can run at once. At the
benchmark deadline, submissions without a pass are excluded with
`code_review_incomplete`; a confirmed rejection uses `code_review_rejected`.
This does not change the daily evaluation schedule or numerical scoring.

Review charges use `openrouter.code_review` in `lab_arena_ledger`, with
`funding_source=miner_key`. They remain separate from sourcing cost eligibility.
The submission status API exposes review status, file/byte counts, and cost.
The public list retains previously admitted review exclusions with no score or
source release; it hides superseded and never-admitted uploads. Review and
per-attempt ledger metadata retain only safe category codes, HTTP status, and
retry flags, never provider prose, credentials, or submitted source.
Unknown provider charges retain their reservation
as uncertain; neither a worker crash nor a malformed response makes a free pass.
Migration `585-lab-arena-review-deadline-recovery.sql` updates the existing
review RPCs and widens the audit counter. It does not rewrite prior reviews,
ledger charges, participants, or scores. Known reviewer failure codes are
reclassified on read, including old diagnostics with a stale retry flag.
The preflight requires the deadline retry capability.

The applied deadline body retains its original 585 filename and exact bytes.
Champion-baseline 585 is a separate applied migration; the numbering gate seals
both full filenames and hashes. Publication 586 and atomic review guard 587
remain separate later migrations. Install deadline 585 before atomic 587, and
follow atomic 587's worker-quiescence prerequisite and bounded lock wait.

Deploy this change as a controlled pair of database and service changes. Drain
and stop the review worker through the canonical deployment workflow. Apply
the exact deadline migration 585 before starting the candidate. The old service cannot
pass its review preflight while the truthful v2 RPC schema is active.
The rollback stays outside forward migration discovery as
`585-lab-arena-review-deadline-recovery-rollback.sql.template`. If rollback is
needed, preserve its exact bytes in a separately reviewed and committed SQL file
with the next unused migration number. Stop and drain the candidate, apply that
exact committed rollback, then restart the old release. This restores all three
exact prior RPC definitions and their existing identities and grants. It keeps the widened audit storage and all review rows.
After atomic 587 changes the dependent RPCs, this deadline rollback template
fails closed on those new hashes. Do not edit its guards to force rollback;
any rollback across that dependency needs its own exact reviewed plan.
The old RPCs fail closed for an unfinished review above six attempts. Reapplying
the exact deadline migration 585 restores recovery for that row while its admission window is open.
Neither direction admits a closed-round or previously excluded submission.
The rollback is a separately selected operator artifact. It is not part of the
forward migration fixture or a bulk migration sequence.
When combined with `585-lab-arena-champion-baseline-reasoning.sql`, the review
rollback leaves that baseline contract installed. The fallback release must
retain champion baseline continuity from PR 872. After a baseline capture
snapshot freezes, do not use the older `a88f94b2` or `4b51d975` releases, which
do not understand that snapshot.

No Dockerfile, public registry, image tag, commit identity, receipt, source
digest, or release manifest is part of miner admission. The service validates
the declared archive size and safe source structure, then uses its own
submission ID for execution and recovery. The private source reference for
each submission is write-once. The service allows one current-submission
replacement before the one-hour replacement freeze; it supersedes the prior
submission and uses a new source reference. After the freeze, replacement is
closed. A miner can use any harness, model, prompts, packages, routing, or
orchestration behind the one callable.

The source contract is `harness.run_icp(icp) -> list[dict]`. `harness.py` can
define the function or re-export it from vendored source. The function must be
synchronous and have exactly one positional parameter. It cannot have
keyword-only parameters, `*args`, or `**kwargs`. It returns at most five
company objects. The public baseline README gives the full input and output
example:
[`leadpoet/leadpoet-sales-agent`](https://github.com/leadpoet/leadpoet-sales-agent).

## Execution checkpoint deadline

New rounds freeze `atomic_checkpoint_45m_v1`, a 2700-second model execution
limit, and a 3600-second lease. Existing rounds without this marker retain
their signed execution limit and lease. The runner reads the duration from the
lease; a validator without the checkpoint capability cannot claim work in a
marked round. Scoring uses the round's separate pinned judge limit.

A submitted harness may import the read-only host module
`lab_arena_checkpoint` and call `write(companies)` whenever it has a complete
list of validated company objects. The helper atomically replaces
`/output/companies.json` with `{"companies": [...]}`. The model chooses its own
work and checkpoint times. The host does not ask it to finalize.

The 45-minute clock starts at the gVisor sandbox creation boundary; runsc
startup has a separate 120-second limit. While the model runs, the host reads
bounded output at 50-ms intervals and validates each changed complete document
against the round's normal output schema. It keeps the last valid bytes it
observed before the monotonic cutoff. A malformed or oversized later file does
not erase an earlier valid checkpoint. At the hard cutoff, the host freezes
those bytes before killing or cleaning the sandbox. A valid frozen checkpoint
can complete a timed-out execution; without one, timeout remains a failure.
Provider, credential, budget, and unsettled-cost checks still govern completion.
No file first observed at or after the cutoff is accepted, including output
written during kill or cleanup. A file published within the final poll interval
may miss observation and is therefore not guaranteed to be accepted.

On a Linux x86_64 root validator host, use
`scripts/probe_arena_checkpoint_gvisor.py` with the exact candidate checkout,
installed trusted Python rootfs, runsc path, and an isolated short workdir.
Its two five-second runs check the helper mount/import, valid checkpoint before
timeout, malformed replacement, and a valid file written only during kill.
This local probe has no providers or database writes; it does not prove a live
paid Arena completion.

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

A proved competing-model credit stop retains its unmeasured score and billing
history. Migration `524-lab-arena-proven-funding-recovery.sql` permits a normal
judge retry after a newer paid native dispatch on the same frozen submission,
provider, and credential. The success must belong to the current scoring
revision and follow every same-key credit denial in that round. Cached, free,
unpriced, host-funded, or different-key traffic is not recovery proof. A newer
credit denial invalidates the old proof; a bound authentication or access fault
cannot be cleared by success on another endpoint. Recovery does not validate
source evidence, promise available quota, or change any verdict or retry budget.

The judge uses bounded retries. A failed review after the scoring window closes
gives zero only to that participant's affected ICP. The other accepted ICPs
remain valid. A successful accepted retry takes precedence over a failed
attempt. If no baseline score remains, the round cannot publish a ranking.
Malformed accepted scoring artifacts also cancel the round before scores are
recorded; they are not company-verification failures.

One narrow host exception applies to score attempt 2. If its lease expires
naturally during scoring before the frozen close time, and a lease-authenticated
`RuntimeHostError` records abandonment before any provider event, ledger entry,
result, or output, expiry creates attempt 3. The original paid or failed
attempts remain unchanged. The retry keeps the frozen scoring input and shared
judgment identity, and the zero-call handoff keeps it off the failed runner.
Attempt 3 is final unless a separate authorized recovery is made.

Failed judge runs can include `result_doc.failure_diagnostic.reason`: one of
`source_blocked`, `malformed_response`, `provider_error`,
`unexpected_verifier_error`, or `unknown`. This field contains no provider
payload, URL, or exception text and does not affect scoring or retries. A reason
saved with a failed completion remains available through `ArenaStore.get_run`
after round cancellation and restart. Cancellation does not invent reasons for
unfinished jobs or backfill historical failures.

### Provider costs

The current sourcing policy gives each ICP a $4.00 sourcing admission limit.
Final cost eligibility allows $4.00 of successful sourcing cost for that ICP,
independent of qualified-company count. New rounds freeze
`successful_calls_flat_per_icp_v1`; existing rounds retain their stored policy.
OpenRouter uses the provider's
`usage.cost`, not the admission-only management key. Scrapingdog uses the
existing endpoint credit map at $0.00005 per credit; approved routes absent
from that map use five credits, and a company profile uses ten. Deepline uses
the per-call billing in a valid completed response at $0.10 per credit. Its
billing history can aggregate multiple requests, so it is a fallback only for
an exact unshared job. The known-free company search and Hunter discovery can
settle a valid completed response at zero. Other missing or ambiguous charges
retain their reservation; they are not free. Billing-history reads are bounded
and private. They never reach the model.

Before dispatch, the broker records a deterministic Deepline billing request ID
and the credential fingerprint. If the response is lost, it reads the provider's
per-request billing ledger without repeating the paid request. Migration 243
allows an exact later settlement. While billing remains unresolved, it defers
pending scores only for the affected submission so retries are not consumed by
the same reservation. Other submissions remain claimable. The existing spending
limits and round deadlines remain in force. An unresolved charge stays reserved
and is never silently treated as zero.

Migrations 245 and 246 also cover process loss before the error handler runs.
An exact reservation and dispatch let the gateway reconcile a database-closed
call after lease expiry, stage close, or cancellation. Only lease expiry defers
live retries. A strictly bound miner-key 401, 402, or 403 lets later scores use
the existing credential-refusal path. When the unknown charge blocks budget
admission, that path needs no new paid request. The unknown charge remains
reserved. Missing evidence, a general rate limit, and a
provider outage do not gain that exception. The gateway release preflight
requires migration 246's private schema capability.

Sourcing reservations are serialized per ICP across its attempts. Different
ICPs have separate budgets and can run in parallel. Admission counts confirmed
spend, including billed failures; pending costs do not reserve money. Multiple
admitted calls may be in flight for an ICP. A call admitted below $4 may finish
above $4; record its full charge
and retain its result, then refuse further paid calls for that ICP. Valid
output is still scored, and the other ICPs continue. The admission limit is
not an absolute upstream charge ceiling.

At publication, the gateway reports provider totals and evaluates each ICP
under its frozen per-ICP policy. Successful paid calls count, including retry
calls and calls whose result the model does not use. Failed calls do not enter
competition sourcing cost; their actual billing and unresolved reservations
remain recorded separately. A successful call with an unknown charge blocks
cost eligibility until resolved. Under the flat policy, an ICP above $4
contributes zero to the final average; its original quality score stays
in the run ledger. The other ICPs retain their own eligibility. These rules
apply equally to baseline and miner models. Existing rounds marked
`successful_calls_per_icp_v1` retain their $0.80-times-qualified-count allowance.
Independent judge cost has its existing $50 cap and is reported separately. Credentials and provider payloads
are not part of the public cost summary.

## Required service configuration

Set these values on the Arena service host:

- `LAB_ARENA_MODE`: `shadow` or `live`. Use `off` to disable the Arena.
- `LAB_ARENA_SUPABASE_URL`
- `LAB_ARENA_SERVICE_KEY`: use a scoped
  `sb_secret_` API key whose JWT template role is `lab_arena_service`.
- `LAB_ARENA_BUCKET`
- `LAB_ARENA_CHAIN_ENDPOINT`
- the three host provider keys listed above
- `LAB_ARENA_SCORER_IMAGE`: a public tag or digest; startup resolves it to a
  digest for trusted scoring
- `LAB_ARENA_COMPANY_JUDGMENT_FROM`: optional timezone-aware timestamp for
  company judgment reuse under the existing integrity grader. It applies to
  company-only rounds whose submission period starts at or after this time.
  At the normal open-to-committed transition, the service freezes the reuse
  policy with the tested scorer image and catalog. An already-open round can
  opt in at that transition; a committed or published round cannot change.
  Reuse requires the same round, ICP, scorer policy, company input, and valid
  authority scope. This setting does not enable company-quality grading.
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
  baseline source is the promoted `leadpoet/leadpoet-sales-agent` `lab` branch.
  Remove an old `main` override before service startup. Shadow mode can set a
  different public HTTPS candidate archive.

Common optional values are `AWS_REGION`, `LAB_ARENA_NETUID`,
`LAB_ARENA_NETWORK`, `LAB_ARENA_CHAIN_TIMEOUT_SECONDS`,
`LAB_ARENA_DAILY_CUTOFF_UTC` (default `0`),
`LAB_ARENA_MAX_CHALLENGERS` (default and hard schema limit `256`),
`LAB_ARENA_MAX_IMAGE_BYTES` for the trusted scorer image,
`LAB_ARENA_POOL_PERCENT` (default `30`), and
`LAB_ARENA_BANNED_HOTKEYS_PATH`. `LAB_ARENA_REWARDS_ENABLED` defaults to
`false` and is frozen into each new round. `LAB_ARENA_SIGNING_KEY_ID` is
needed only when a live, reward-enabled published round is activated.

The challenger limit excludes the baseline. Each hotkey can have one current
accepted model per daily round; one source replacement is allowed before the
replacement freeze. Different hotkeys can share a coldkey. New rounds freeze
the configured challenger limit (default `256`) without a workload estimate.
The database applies this limit atomically at signed finalization. New miners
can submit until the limit is reached or the submission cutoff passes. Later
worker capacity changes do not change the admission limit. Worker concurrency,
stage deadlines, and spending limits still apply to every assignment.

When deploying the per-miner admission default, remove any explicit
`LAB_ARENA_MAX_CHALLENGERS=20` override (or set it to `256`). Apply
`395-lab-arena-per-miner-daily-admission.sql` to raise the former default only
for standard, open Finney 71 rounds whose cutoff has not passed and which have
no frozen participants, benchmark, or runs. It restores each affected hotkey's
latest `capacity.round_full` rejection to `uploading` so the miner can retry
signed finalization with valid credentials before cutoff. It does not accept a
source or restore credentials automatically. Earlier submission history and
credentials stay intact. Started, historical, shadow, and rounds with caps
other than `20` stay unchanged. Inspect matching open rounds before applying it:
an intentional cap of `20` has the same stored value as the former default.
Future rounds use the configured default.

`LAB_ARENA_BENCHMARK_DISCLOSURE_FROM` is an optional aware timestamp, normalized
to UTC. For example, `2026-09-13T00:00:00Z` freezes
`benchmark_disclosure_policy=after_scoring_day2_v1` into each newly created
round whose cutoff is on or after that instant. An existing round keeps its
stored configuration. An unknown or null stored policy fails closed. Before
setting this value, deploy this reader to every service that can serve Arena
public routes or create rounds. Do not roll back to a version that ignores the
marker while a marked round still needs privacy. No database migration is
required because the existing immutable `configuration_doc` stores the policy.

For a release upgrade, apply only the required committed migrations through
the configured repository migration helper before service startup. Then check
the service wiring:

```bash
python3 scripts/run_lab_arena_service.py --check-only
```

Migration 193 adds safe replacement of unfinished uploads and accurate
`execution_incomplete:stageN:count` / `scoring_incomplete:stageN:count`
cancellation labels. It preserves historical results and source objects.
Migrations 422 and 423 supersede whole-round execution cancellation: after
bounded lease drain, unfinished assignments remain explicitly incomplete.
Completed ICP results are judged and persisted normally. An incomplete model
has a null final score and cannot win promotion. The current publication path
requires complete accepted baseline execution and judgment at every configured
position. An incomplete baseline cannot authorize publication, a new champion,
or new rewards; existing weight continuity follows the governing signed reward
basis. No missing ICP is silently assigned a score.
Deploy its matching service after applying the migration. Source admission
still uses the existing upload MD5 and server-assigned submission ID.

Migration 194 refreshes only the trusted scorer digest and pinned reference
when an existing open round atomically commits its benchmark. This lets a
deployed scorer fix apply before any work is created. The committed scorer
pin and every other round setting remain immutable.

### Frozen grading and repairable reviewer runtime

The round's `scorer_image_digest/reference` remains its immutable grading and
cache identity. Never change it to repair unfinished reviews. Grading prompts,
models, evidence rules, scoring code and dependencies remain in that image.

`scripts/build_arena_reviewer_runtime.py --grading-image <digest-reference>`
composes one runtime image on that exact base. It does not rebuild dependencies
or copy current grading modules. The only added code is the dedicated
`arena_reviewer_runtime` package and a checked extraction of the investigator's
admission/transport interface and scorer checkpoint/output IO. Unknown source
shapes fail the build. The tool
checks the actual added image layer and unchanged process configuration.
For already separated graders, the grading file remains byte-identical.

For a frozen grader whose unresolved semantic-review return bypasses its
completed-investigation settlement, explicitly add
`--completed-investigation-finality` during both build and `--verify-image`.
This exact-source correction moves the grader's existing settlement block
before the initial and post-repair semantic exits. It retains the strict
malformed-profile exhaustion proof, source-local failure classification,
identity requirements and provider retries. It imports no current grader.
The same correction preserves a proved terminal insufficient-evidence result
through final receipt assembly. Its internal proof marker is set only after
the existing settlement checks pass and is not copied to public or cached
judgments. A stale homepage failure cannot discard that completed negative
proof; incomplete investigations and unresolved source failures still retry.
Version 2 also permits this settlement when headcount was already verified.
Only the remaining incomplete facts must have completed, typed UNPROVEN
findings. It applies the same settlement after schema repair, including reviews
without a semantic dispute. The cached profile failure is retained for audit;
it is not converted to a successful profile fetch. Real provider failures and
unproved positive findings remain retryable.
Unknown or partially corrected source fails closed. Independently verify the
composed image and its affected paths before registering its runtime mapping.
Only new lease generations select the registered runtime; existing leases and
accepted cache authorities retain their original identity and results.

For a completed semantic review that still has missing model identity fields,
explicitly add `--identity-schema-repair` during build and `--verify-image`.
This separate correction (`completed_semantic_identity_schema_repair_v1`)
applies only the three reviewed control-flow changes from `43a29507f`. It lets
the existing single schema repair fill identity when the semantic investigation
completed without failure and did not investigate rebrand. The unproven semantic
claim remains unproven. Missing or conflicting identity and real provider or
source failures still retry. Unknown or partially corrected source fails closed.
The grading pin, prompts, evidence rules, dependencies and prior correction IDs
stay unchanged. Preserve the prior runtime's explicit correction options when
building its replacement; select this new option in addition to those options.
The release gate can run the focused frozen-source tests with
`ARENA_IDENTITY_FROZEN_SOURCE` and `ARENA_IDENTITY_RUNTIME_SOURCE` set to scorer
files extracted from the pinned base and prior runtime images.
Run `tests/test_arena_reviewer_identity_schema_seam.py` with those variables.
Set `ARENA_IDENTITY_COMPOSED_SOURCE` to the scorer file from the new image to
run all 14 identity controls against that exact composed verifier as well.

For an optional first-party auxiliary whose submitted company name differs
from the verified name, add `--auxiliary-source-name-binding` during build and
`--verify-image`. This exact-function correction
(`first_party_auxiliary_exact_name_binding_v1`) omits only the auxiliary that
the unchanged paragraph reviewer already rejects. Shared source validation,
verified signals and matched company evidence remain unchanged. Unknown source
fails closed; an exactly corrected function is accepted without another change.
Preserve the prior runtime's explicit correction options when adding this option.

Deploy a verified composition through the trusted reviewer-runtime mapping.
The mapping is separate from frozen round configuration. A score lease pins
the selected runtime when its generation starts. A later deployment cannot
change that lease's image; a later unfinished retry can use the repair.
Execution images, accepted judgments, frozen cache keys, partial checkpoints,
provider request identity and score calculation do not change. The normal
judge queue and completion path remain the only scoring path.

Apply migration `495-lab-arena-reviewer-runtime-selection.sql` through the
canonical migration helper. Build with `--push`, then independently verify the
published digest with `--verify-image`. On the reviewed gateway release, run
`scripts/register_arena_reviewer_runtime.py` with `--grading-image`,
`--runtime-image`, and the protected `--environment-file`. Its default is a
read-only plan; activation requires `--apply` and
`LEADPOET_ARENA_REVIEWER_RUNTIME_APPLY=1`. Do not update a frozen round or
replace the default grader with an older round's runtime image. Rollback uses
the same registration command to select the original grading image for future
leases; existing leases and accepted work stay intact.

Before activation, replay the affected failure and completed positive/negative
controls. Compare original and extracted request payloads and grading outputs.
After activation, check both grading and runtime identities on actual leases,
accepted-work hashes, cost settlement, publication and downstream weights.
Successful image construction or a host restart alone does not prove recovery.

For the approved exact-source access-denial correction, build and verify with
`--source-access-denied`. The composer applies only the reviewed Exa response
classification and matching investigation/finality checks to the frozen source.
It records `exact_exa_source_access_denied_v1` in the composition evidence.
Use `--complete-negative-finality` as well when retaining the prior approved
negative-finality correction. Neither option copies the current grader.

An Exa API 200 containing one exact requested URL with `CRAWL_HTTP_401` or
`CRAWL_HTTP_403`, the matching integer status and no result is unavailable source
evidence. It can supersede an earlier failed fetch of that same URL. It does
not prove a company fails qualification: the existing bounded discovery and
validated findings must still complete. API authentication, credit, rate-limit,
service errors and ambiguous or mismatched source responses remain failures.
Verify the affected unfinished judgment through accepted completion and public
publication, and compare its previously accepted outputs before and after.

For a typed LinkedIn `source_blocked` diagnostic that becomes a generic fetch
failure, build and verify with `--source-blocked-alias`. It records
`typed_linkedin_source_blocked_alias_v1` and changes only the frozen investigator's
diagnostic normalization to the existing `source_access_denied` class. Preserve
all prior explicit correction options. The base must already contain the exact
source-access correction above, or select that option first. Outer transport
failures, ambiguous crawl failures, malformed evidence and required-search
failures remain retryable. A completed `UNPROVEN` finding remains unproven.

For the approved pending final-correction repair, also build and verify with
`--pending-final-correction`. It changes only the frozen loop predicate so an
already-pending correction uses the next existing turn, even if the preceding
submission finished after the admission deadline. The correction offers only
`submit_findings`; it adds no research or repeated correction, and preserves
all turn limits, evidence validation and provider-failure handling. Composition
evidence records `pending_submit_only_final_correction_v1`. Keep the prior
approved correction flags when composing from the original frozen image.

For the approved failed attribute-source recovery, also build and verify with
`--failed-attribute-source-recovery`. When another typed investigation is
already planned, it includes the active attribute if its independently fetched
source failed. This uses the same investigation and unchanged turn, search,
fetch, time and cost limits. Existing validators must prove a replacement or
complete a healthy UNPROVEN review; provider failures are never source absence
or qualification. Grounded attributes and attribute-only recovery stay unchanged.
Evidence records `failed_attribute_source_shared_review_v1`. Keep the three
prior correction flags when composing the frozen October 16 grader. Register
the verified runtime image through the existing reviewer mapping; this source
correction does not require a gateway or validator host restart.

### Recover an unusable Firecrawl page response

Miner-funded scoring keeps Firecrawl as its first page transport. A completed,
settled response with the exact `firecrawl_source_content_unavailable` code
can use one alternate `generic_http_request` for the source. If that response
binds the exact original source URL to a validated public HTTPS final URL, the
alternate uses that observed destination. Otherwise it keeps the original URL;
it never guesses a hostname or adds a second alternate attempt. The alternate
uses the existing second provider-attempt identity, the same miner credential,
the frozen tool catalog and all existing cost and call limits. It has only the
time left in the original request. The existing settlement rule still applies
to an attempt admitted before that deadline: a slow successful ledger reservation
can finish with the transport's minimum 1 ms timeout; it receives no new request
budget. Unknown billing, authentication failures, source mismatches and other
malformed responses do not authorize this fallback.

The alternate requests Deepline's `raw-v2` response contract and validates the
provider, method, requested URL, final public HTTPS URL, target status and HTML.
Only the page body, content type and validated source URL reach the scorer.
Provider cookies and other headers are not forwarded. A failed source remains
a failure; existing source absence, retry and Exa fallback rules still apply.
Both attempt receipts and charges remain in the ledger. The exact first-failure
code is retained so a repeated action reuses its settled alternate without
making another provider request. The primary's trusted response-URL header pins
the observed destination. Each new alternate reservation pins its actual URL
and upstream timeout. Recovery reproduces that exact body under its existing
execution key, while the transport deadline stays bounded. Older reservations
without a pinned URL keep the original requested URL and their saved timeout,
even when a redirect is now known. Malformed, private or conflicting saved URL
metadata fails closed. No accepted score or frozen grader is changed.

### Verify a scorer release separately from a host release

A gateway or validator restart does not rebuild the scorer image. The CI image
smoke build does not push an image or update production configuration. A merged
scorer fix is therefore not evidence that any round runs it.

For a change intended to affect scoring:

1. Build and publish the image with `scripts/build_lab_arena_judge_image.sh`
   from the clean, reviewed source. Test the affected path and controls inside
   that exact digest, including the normal runner, broker, cost settlement,
   completion and publication path where relevant.
2. Update only `LAB_ARENA_SCORER_IMAGE` through the protected configuration
   procedure. Use the digest-pinned reference. Run the canonical gateway
   release so both API and the sole round driver load it. Restart validators
   only when their host code also requires deployment.
3. Read back the selected image from both actual API and driver processes.
   Retain the image digest, build source and test results. A host source SHA or
   successful health response does not establish scorer deployment.
4. At normal benchmark commitment, verify the round's persisted image digest
   and reference, then the actual execution/scoring leases. Only then report
   the scorer as active for that round. An open round can still display its
   earlier image until commitment; an already committed round keeps its pin.

These are release acceptance checks, not a new runtime identity gate. Do not
require gateway and scorer Git SHAs to match, add validator attestation, or
change a frozen round to make the checks pass. Report an incomplete scorer
release as incomplete even when the host deployment succeeded.

Start the service:

```bash
python3 scripts/run_lab_arena_service.py --host 127.0.0.1 --port 8792
```

## Canonical restart claim drain

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

`LAB_ARENA_ROUND_ID` is optional. Runner parallelism comes from verified
Webshare proxies plus the native route, and is bounded
by the round's frozen `runner_slot_ceiling`. Provider keys, database access,
source upload access, and the signing key stay on the service host. The runner
needs read access to the organizer's common trusted
Python/scorer image. It downloads source only through the active run lease,
then mounts source, installed wheels, and the host-owned entrypoint read-only
inside gVisor. Start the runner with:

```bash
python3 scripts/run_lab_arena_runner.py
```

If three distinct execute leases on one runner expire in the same stage after
lease-bound `RuntimeHostError` reports and no provider cost or result, the
service stops new execute claims from that runner for one frozen lease-TTL
window. It does not end current leases early or change scoring. Inspect and
repair that host before restarting it; healthy runners can claim the retries.

Three distinct current leases with authenticated `RuntimeHostError` reports
at the runtime stage, no provider ledger, and no result or output stop that
runner from claiming more work of the same kind in the current round, stage,
and generation. The guard counts both active leases and naturally expired
leases within one frozen lease-TTL window. Existing leases keep their normal
deadlines. A fault remains counted during that window even if the expiry tick
has not yet changed its run from `leased` to `failed`. Healthy runners can
continue claiming pending work. The guard ends
when fewer than three matching leases remain in that window; the separate
expired-lease cooldown remains in force under its existing rules.

## Miner flow

Run `python scripts/lab_arena_miner.py interactive`. It reads the local source
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
registered baseline bundle is not replaced. A model must score at least **1.0 point**
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

For full-thinking rounds, the baseline freeze selects the completed promotion's
exact commit and freezes its original capture settings separately from the new
challengers' settings. A capture-policy change cannot substitute the bootstrap
for an existing champion. The bootstrap is used only before the first promoted
champion. Historical frozen rounds are not repinned. Baseline cost limits,
scoring, and reward rules remain those of the current round; the capture
snapshot does not make legacy reasoning eligible for strict training exports.

Apply `585-lab-arena-champion-baseline-reasoning.sql`, verify its function
hashes, then deploy the matching
gateway code before the next baseline freeze. The new code must not freeze a
round against the older database function, which cannot save the capture
snapshot. Verify that both the serving API and daily driver use that release;
an older API ignores the separate capture snapshot. The new database function
rejects an older driver's attempt to substitute the bootstrap before freeze.
After the first round freezes the new snapshot, do not roll back to code that
does not understand it. Use a corrected release that preserves the contract.

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
