# Arena training data: collection contract and audit

Audit date: 2026-10-16. Counts below are a production snapshot, not a count of
training-ready examples. The inspected gateway release was
`408ea1724da56de5792835410ec97fd55189ca37`.

## Decision

There is enough retained material to curate a first narrow **evidence-to-company
qualification** experiment. There is not yet a verified historical corpus for
imitating a complete research agent. Do not train on the diagnostic trajectory
events as if they were full conversations. Do not interpret an accepted execution,
a positive aggregate score, or a billing receipt as an independently verified good
training example.

The first experiment should use a single compatible scorer policy and company-level
labels with retained evidence. Keep negative, skipped, outside-slice and unverified
cases distinct. Start with a manually checked pilot, then measure held-out quality
before expanding it. The audit does not authorize a paid training job.

## Production inventory

| Measure | Count | Meaning |
|---|---:|---|
| All run rows | 38,142 | Includes execution, judging, retries, failures and pending work |
| Execution attempts | 20,275 | Not distinct successful research problems |
| Accepted executions | 14,291 | 10,486 miner; 3,805 baseline |
| Accepted judge runs | 14,943 | Includes repeated and cached judgments |
| Accepted executions in published reward rounds | 8,284 | Mainnet candidate pool |
| Those with a stored ICP score | 8,025 | Must still verify score authority and full breakdown |
| Those with a positive ICP score | 773 | Mixed-quality companies can occur in one positive run |
| Accepted executions with diagnostic events | 7,993 | Events do not prove complete data |
| Published reward executions with events | 6,560 | 6,338 scored; 673 positive |
| Distinct round/stage/ICP slots among those 673 | 107 | Repeated participants and inputs reduce effective diversity |

The ledger occupies 41.27 GB and the trajectory table 5.69 GB, including indexes
and TOAST. Approximate table statistics report 4.97 million ledger rows and
2.79 million trajectory events. These are telemetry counts, not examples.
3,757 accepted executions report direct web connections. Only 4,800 accepted
executions record a known validator source commit. Only 4,756 miner executions
have an existing submission archive hash; baseline objects can have a valid frozen
source reference without a recorded archive hash.

These counts include several policy and runtime generations. Shadow/replay rounds,
testnet, cancelled rounds and current unfinished rounds require separate filters.
A saved-output replay provides new judging evidence but no new research trajectory.
A later output-reference audit found 332 accepted rows pointing outside their own
canonical run/round output path. All belonged to cancelled reward-disabled rounds;
sampled records were archive aliases of older output objects. They are excluded
from complete-example export and are not additional independent demonstrations.
The published reward pool was unaffected. One more execution completed during
the audit; the inventory above remains the original consistent snapshot.

A later cache inventory found 7,049 run-judgment cache records covering 5,774
distinct scoring input hashes and 12,860 company breakdowns. The published reward
subset contained 7,579 breakdowns, including 910 marked `company_qualified=true`,
across 1,867 company identity keys and 33 scorer images. These are candidate labels,
not a reviewed training set or a count of unique legal companies. The separate
company cache contained only 67 rows from 61 source judges. It is not the main
historical label store. Select a compatible policy cohort before counting useful
examples; exclude infrastructure failures and skipped cases, and deduplicate
repeated evidence and cached judgments before splitting the data.

The largest published-reward scorer-image cohort illustrates the duplication:
531 distinct scoring inputs with nonempty breakdowns contained 1,440 company
labels, including 214 qualified labels, but only 236 company identity keys and
35 qualified company keys. The cohort's scorer digest is
`sha256:be2c560cf8fc8b21d5854f5f0509e175a03846c81758901a35e726ccff25a848`.
Its 1,226 nonempty failure reasons require classification before using them as
negative examples. One scorer image is a useful first filter, not proof that
every frozen policy is identical. These counts support a small curation and
evaluation pilot, not a claim of broad generalization or a ready training split.

The October 17 follow-up count had 779 positive accepted executions in published
reward rounds. Of these, 346 explicitly reported zero direct web connections.
Only 92 also had a recorded 40-character source commit and an explicit zero local
refusal count. Another 27 lacked the refusal field; absence is not proof of zero.
These filters identify candidates for further review, not complete trajectories:
historical prompts, responses, label authority, source trust and deduplication
still require verification.

A fresh count at **2026-10-17 03:24 UTC** found 38,377 run rows, 20,439 execution
attempts, 14,335 accepted executions (10,516 miner and 3,819 baseline), and 14,957
accepted judge runs. Published reward rounds contained 8,354 accepted executions,
8,095 with scores and 779 with positive scores. Diagnostic events existed for
8,037 accepted executions; the published-reward event subset remained 6,560 runs,
6,338 scored and 673 positive across 107 positive round/stage/ICP slots. There
were 3,800 accepted executions reporting direct web access. The earlier table and
storage figures remain the original snapshot; these live counts do not establish
a training-ready example count.

## Actual data path

1. A round freezes participants, source object references, the benchmark and ICP
   positions, scorer image and policy, provider catalog, schedule and budgets.
2. A signed eligible runner claims an assignment. The stored run binds the
   assignment, execution attempt, lease generation and claim request hash.
3. The runner builds the final JSON input, including scoring reuse data, policies
   and the frozen tool catalog, then runs the source or scorer in its sandbox.
4. Brokered requests pass operation validation, routing, model/token limits,
   credential selection and accounting admission. The immutable reservation binds
   `run_id`, `call_identity`, `base_call_identity`, action sequence, provider attempt
   and normalized request hash. Dispatch and terminal receipts record the outcome.
5. The worker submits an output. Accepted execution bytes live at the run's frozen
   output reference. A separate score run links back through `scored_run_id`.
6. The scorer retains full breakdowns and verifier evidence in a score object.
   Run-level and company-level caches can refer to earlier authoritative judgments.
   Publication derives the per-ICP score. The qualification summary is not the full
   judgment record.

`lab_arena/runner.py`, `broker.py`, `service.py`, `verify.py`, `scoring.py` and the
store contracts are the authoritative paths. Public dashboards and error monitors
are corroborating views; neither proves collection completeness.

## What was missing, and what the durable path stores

| Data | Before this change | Durable collection/export rule |
|---|---|---|
| Exact worker input | Constructed at runtime; lease alone omits later additions | Private gzip snapshot after the final input is built; bind exact run, lease generation and claim hash |
| Full model/tool request | Hash and short projection; prompt text discarded | Private content-addressed gzip object; immutable reservation holds its pointer |
| Requested vs effective model/parameters | Spread over summaries and routing code | Retain admitted, normalized, hashed and effective parameters plus credential-free outbound body, target and route |
| Full returned response | Usually already in `terminal_response.body_b64`; diagnostic view truncated | Reuse and verify the existing receipt; do not write a duplicate response object |
| Worker-visible failures and local calls | Transport failures, local catalog replies and local refusals can lack a paid receipt | One bounded private artifact per run lease retains exceptional requests and returned outcomes; normal paid successes reuse the ledger |
| Provider identity | Broker provider, response model, route and billing records differ | Preserve each with its role; never infer physical model identity from a requested alias alone |
| ICP/company/final output | Present in separate frozen objects | Exact round/position/submission/output joins; retain company order |
| Labels | Full score objects and cache authorities; run score alone insufficient | Verify existing production evidence rules and recompute per-ICP score; reject ambiguity |
| Cost/runtime | Append-only ledger plus resource summaries | Preserve attempts and terminal accounting; unresolved cost is not zero; competition cost is a separate measure |
| Retries/recovery | Multiple events and receipts look like duplicates | Keep logical actions, provider attempts and run attempts distinct; bind inputs to final lease generation |
| Available intermediate state | Model messages/tool results plus optional diagnostic decision/log excerpts | Full broker-visible state is retained; optional decision/log excerpts are explicitly excluded from full-process proof |

The input capture preserves the exact JSON value, not incidental file whitespace.
Request objects record a **planned** outbound request; dispatch receipts determine
whether the provider was called. A `referenced` pointer is not proof of storage.
The exporter must read and hash-check its object before it accepts it.

Capture uses the existing private object store, immutable objects, deterministic
gzip, size limits and a separate S3 client with bounded timeouts and no SDK retries.
The October 17 production readback confirmed all four bucket public-access blocks,
enabled versioning and default AES256 encryption. The host role could not read the
bucket policy-status API; no policy was changed.
No new database table, queue, background monitor, dependency or lineage service is
required. Input uploads require an active owned lease before storage. Request
object writes occur only after accounting admission. Capture failure cannot change
the provider's accounting or score rules. It produces incomplete training evidence.
The worker uploads its exceptional observations before completion, while its lease
is active. The artifact binds the run, lease generation and claim request hash.
The exporter checks its hash, count and action links. Worker evidence never becomes
a provider receipt or a settled charge. Overflow, lost upload and unfinished socket
handlers remain explicit gaps. The bounded storage calls add some latency. The
final release's fresh live proof below verifies the stored path without changing
execution or score rules.

The full payload limit is explicit. Oversized or unavailable evidence is rejected
as a training source; it is never silently clipped and called complete. Credentials
and opaque provider reasoning (encrypted content, signatures, redacted thinking)
are excluded. Readable reasoning that a provider returns as text is kept: chat
`message.reasoning`, readable `reasoning_details` items and Responses reasoning
summaries/content. Each exported call also lists that text in `readable_reasoning`
and carries the round policy's `reasoning_capture` record when one applied (see
[agent reasoning capture](arena-reasoning-capture.md)). Responses are re-read from
the ledger at export time, so this also restores readable text in historical
receipts; objects redacted at capture time stay redacted. Privacy filtering and
exact replay are separate properties from label validity.

## Remaining boundaries

- Direct HTTPS egress records connection/byte totals, not page contents. Historical
  TLS traffic cannot be reconstructed. Runs using it are excluded from complete
  process training unless their relevant observations are independently retained.
  This change does not introduce a TLS interception proxy.
- Historical local frozen-catalog discovery has no paid ledger reservation. Its
  diagnostic events can be abbreviated. New worker observations retain the full
  returned catalog, while the exporter continues to reject an unexplained count
  mismatch. Use a broker-only subset for a first complete call-trajectory experiment.
- Hidden provider chain of thought is not a collection target. Unlogged internal
  variables or private provider search steps are not recoverable. Explicit model
  messages, returned tool outputs and provider-returned readable reasoning are
  the available training state. Rounds with `agent_reasoning_v1` ask agent calls
  for that readable reasoning. For most hosted models it is a provider summary;
  some open-weight models return their full reasoning text.
- Optional model-reported decision files and stdout/stderr remain diagnostic,
  bounded and potentially incomplete. They are not authoritative demonstrations.
- A matching hash proves linkage and integrity, not factual truth or a good label.
  Training selection still needs company-level review and evaluation.
- Ledger receipts preserve provider payloads and original lifecycle times. They
  do not prove that the worker received or used a particular network response.
  The export states this boundary and does not infer concurrency from array order.
- An accepted result can contain an upstream error, a recovered billing result,
  zero qualified companies, or an incomplete judgment. Keep the explicit proof
  dimensions; do not collapse these cases into a success flag.

## Historical reconstruction

Safely recover frozen ICP/context, accepted final outputs, full retained provider
responses, authoritative company breakdowns, per-ICP scores, costs, timing and
attempt/caching relationships where their exact references validate. Compute a
missing baseline archive hash from the validated immutable frozen source object
and label that origin; do not pretend the hash was recorded at execution time.

Do not recreate missing prompts from today's source, rerun a provider search and
call it the original evidence, expand a truncated diagnostic excerpt, or guess a
judge join by timestamp. Such material can support analysis, not exact replay.
Provider billing APIs can confirm model, timing, token usage and charge but do not
restore the missing prompt text.

Some short historical tool requests are recoverable. A read-only check on October
17 reversed only complete diagnostic list wrappers, then required the exact
operation, action sequence and canonical normalized-parameter hash to match the
original ledger reservation. It recovered 31 of 53 requests in the citation judge
sample and 168 of 199 execution-only requests in the published October 15 sample
below. The latter count excludes that execution's separate judge calls.
All matches were Deepline tool requests; no OpenRouter model request matched.
These are verified normalized parameters, not recovered outbound headers, runtime
input files or full model prompts. No historical rows or objects were changed.
Unmatched projections remain incomplete, even when they look plausible.

A concrete pre-change judge run,
`arena-2026-10-16-citationprobe:baseline-2026-10-16-citationprobe:1:0:score:1`,
had 53 provider calls: 22 OpenRouter and 31 Deepline. All 53 terminal response bodies
were retained. No full-request pointers existed. Diagnostic events explicitly
showed loss for 27 of 53 requests and 51 of 53 responses. The response bodies total
11,572,152 base64 bytes; the projections are not a substitute for those receipts.
This fixture replayed saved research output, so it is only a judging example.

An independent read of OpenRouter generation
`gen-1792187452-JF8ykLOYDbcOS5iqZgFJ` returned Perplexity Sonar and a $0.0079 charge,
which exactly matched ledger entry 4970393. It returned no prompt/response text.
Deepline's exact-request billing endpoint returned a final, posted 0.02-credit
charge for ledger entry 4970376, exactly matching its 2,000 micro-USD charge.
These are bounded samples, not a claim that all historical
charges were independently re-audited.

The retained September 13 Supabase archive manifest covers 20,270,597 rows in
15 retired relations. The main dump is 6,006,835,409 bytes. Its 14,743 attested
execution-result rows are infrastructure operations (allocation, source catalog,
model/weight authority and settlement), **not 14,743 agent research runs**. Receipt
constraints exclude request/response bodies. Encrypted artifact envelopes remain
opaque without the original protected decryption path. No full restore or key
export was performed, and this archive is not counted as recoverable training
examples. Retain it unchanged.

The validator also retained 14 old `input/icp.json` files from September 5 and 8.
They were about 2 KB each and did not identify a run, round, assignment or lease
generation. The normal runner deletes its temporary run directory on completion.
These local files cannot establish complete historical trajectories or replace
the new immutable input snapshot. They were left unchanged.

## Export and first experiment

Use `scripts/export_arena_training.py --run-id ... --environment-file ... --output ...`.
It performs read-only object and database access, creates a new private file, and
reports independent completeness dimensions. An incomplete example exits with
code 2; identity ambiguity or access failure exits with code 1. Never treat the
existence of an export file as proof it is eligible for training.

For per-ICP rounds, `competition_cost_eligibility` is a read-only snapshot from
`lab_arena_icp_cost_eligibility` at export time. Its timestamp and frozen policy
caps distinguish it from the published historical reward decision. Actual paid
cost remains separate; an unknown failed-call fee can leave that accounting
incomplete while successful-sourcing eligibility is final. Missing or unsettled
eligibility authority is marked unavailable, never assumed eligible at zero cost.

Start with evidence-to-qualification, or another single scorer subtask with complete
input and a checked label. Select one stable policy/model generation. Separate
companies, near-duplicate evidence, source-code families and time windows across
training and evaluation. Keep cached copies and saved-output replays in the same
split and count them once. Compare against the unchanged base model on a held-out
set using false-positive rate, qualification accuracy and evidence support, not
training loss alone. Set the pilot size from the verified export and review yield;
raw run count does not establish a universal sample threshold.

Deduplication is also supported by primary research showing that duplicate data
can increase memorization and contaminate evaluation: [Lee et al., ACL 2022](https://aclanthology.org/2022.acl-long.577/).

## Verification record

The integrated candidate includes remote main through `e724c9240`. The combined
local gate passed **500 tests in 49.09 seconds**, with no failures or skips and one
existing multipart deprecation warning. It covered capture, input, export,
lease/billing failures, runner, API, gateway proxy/edge and fresh-round verification.
Python compilation, whitespace checks and AGENTS/CLAUDE parity passed. Independent
review verified the privacy, lease-recovery and prior-attempt fixes. Two broader
pre-existing fixture failures were isolated outside the final focused gate:
an exception fixture cannot set its new transaction attribute, and an auth-evidence
fixture expects the old evidence shape. They are not reported as passing.
The closing exporter changes preserve numeric action order, original receipt times,
funding and reconciliation provenance without duplicating equal response bodies.
They also enforce the execution's own output path and expose round filters. The
final capture/input/export check passed **106 tests in 4.26 seconds**; compilation
and whitespace checks passed again.

The exporter from candidate `cc44ff1c094998247f0fdc185a8611dfbf029479` was executed
in an isolated temporary directory on the gateway against **read-only production
records and objects**. It did not replace a serving runtime or construct an Arena
service. The private files remained on the host.

| Verified historical example | Company labels | Response bodies | Full requests | Result |
|---|---:|---:|---:|---|
| Oct 16 citation replay, ICP 0 | 3 | 53 / 53 | 0 | Labels, context, source and costs valid; process incomplete |
| Oct 15 published miner `sub-f86819cba08b000f0f103feaa77b8909`, stage 2, ICP 3 | 5 | 294 / 296 | 0 | Labels, context and source valid; 2 unresolved calls and direct-web evidence gaps |

The second example recomputed its 49.666667 ICP score. These eight company labels
prove reconstruction works on real records. They are not a validated corpus count
or a manual judgment of label quality. Both exports correctly returned exit 2.

Fresh OnePatch telemetry showed 26 trajectory uploads with no span errors in the
last 30-minute window, plus real provider and billing-reconciliation traffic.
The production service's environment tag was blank; host/release checks supplied
the environment identity. Sentry's production 24-hour query returned restart,
weight-chain and OTLP issues, with no collection-specific issue. The absence of
a collection alert did not detect the missing prompts. No unrelated incident was
changed or marked resolved.

**Verification status: production deployment and a fresh accepted execution-to-
judgment export are verified. Full-process completeness remains false for the
explicit direct-web and privacy boundaries below.** The authenticated GitHub App
API confirms repository push access. The earlier connector-availability concern
was incorrect.
No database migration is required for collection. The integrated exporter also
validates each selected and prior launched judge runtime against its persisted
lease binding, separately from the frozen grading image.

The committed verifier supports a fresh, baseline-only two-ICP proof (the minimum
valid round size) with one worker slot, frozen public ICP selection and rewards
disabled. It never reuses saved outputs. Live acceptance still requires that fresh
execution and its real judgment, then independent readback of the exact private
input/request objects, ledger receipts, output and score. A replay fixture or HTTP
health result alone does not meet this acceptance criterion.

The first canonical deployment of PR #510 was commit
`d8a97d5dc1d6296f8f7f49fc327cf406263b37b3` on both gateway and normal validator.
Its fresh reward-disabled proof exposed a capture bug: the frozen Deepline catalog
contains 64-bit JSON Schema integer bounds, but the input endpoint reused the
operational API's `2**53` limit. The endpoint rejected the input while research
continued. PR #512 removes only the integer magnitude limit for training inputs;
all other routes retain their default bound, and body size, depth, strings and
finite-number checks remain active. The exact 1,781,849-byte live JSON input then
round-tripped through the candidate HTTP endpoint and object store with equal
values, matching hash and no redactions. The merged candidate
`6ab0dc94feddfd9220d4c7b611ab9b94284c1909` passed 361 tests in 48.19 seconds,
plus compilation, whitespace, protected-manifest and AGENTS/CLAUDE checks.

At 01:58:37 UTC on October 17, an independent read of the first proof's 119 request
objects verified every hash, length and ledger identity. The objects contained
754,840 uncompressed bytes in 104,356 compressed bytes. All 119 terminal response
receipts were present. This was an in-progress snapshot, not final trajectory
certification. The first execution's missing input remains an explicit historical
gap; it is not backfilled from the local diagnostic copy.

Both canonical controllers completed for `6ab0dc94f`. The gateway passed runtime
attestation and exact `/build-info` checks; the normal validator passed protected
readiness on the same commit. The operator pause remained false. Finalized-chain
readback matched the existing signed weight journal; that finalization predated
the new process and is not evidence of a new broadcast.

The original proof attempt ended with `provider_error`, and Arena created its
normal retry. The same frozen reward-disabled round resumed on the fixed release.
At 02:07:42 UTC, retry `arena-2026-10-17-trainingproof:baseline-2026-10-17-trainingproof:1:0:2`
had a verified lease-generation-2 input object: 1,781,849 bytes, no redactions,
hash `sha256:009c8eb79d6a52b2355788f23c23c69b91983c9ea8a29760de6a973a14435fe9`.

Independent provider records for that retry matched OpenRouter ledger entry
4988957 at $0.020958 and Deepline entry 4989012 at 0.01 credits ($0.001).
OpenRouter identified `anthropic/claude-4.5-sonnet-20250929` on Amazon Bedrock;
Deepline identified `contextdev_post_news_search` with a final posted charge.
This verifies two real billing joins. It does not establish label quality or
prove that every call has settled.

Privacy review found six retained response receipts containing nonempty HTTPS
`auth` parameters. Their values were not emitted or tested, and the export keeps
redacting them. Separate synthetic checks found false credential flags for empty
credential-named values and diagnostic NUL escaping. The follow-up fix preserves
only null and empty JSON strings/objects/arrays, and leaves NUL escaping to JSON
serialization. Nonempty credentials, false/zero sensitive values, signed access
URLs and exact-secret matching retain their existing protection. Diagnostic logs
keep their original escaping behavior. The focused gate passed 176 tests.

Neither observed fresh proof attempt used free catalog discovery. The first
failed attempt reported 158 worker provider calls but had 157 ledger actions;
one gateway request failed before admission. The exporter reports the mismatch
instead of treating it as a complete process. Free local catalog discovery is a
separate known boundary, not the cause of that mismatch.

The second attempt also ended with `provider_error`: 144 worker calls versus 143
ledger actions, zero local refusals and 72 direct web connections. One failed
call had a captured request but no terminal response receipt because its provider
cost was unavailable. These failures exposed the need to retain exceptional
worker-visible outcomes separately from provider receipts. Neither attempt is
counted as a successful training example. Across both attempts, the observation
at 02:22:53 UTC had 300 ledger calls, no inflight calls, three uncertain costs,
and $0.943890 of confirmed charges. The second ICP remained unstarted.

The worker-observation follow-up passed an integrated gate of **505 tests in
54.44 seconds** (55.15 seconds including the release checks), with no failures
or skips and one existing multipart deprecation warning. It covered capture,
inputs, export, worker observations, runner/API behavior, Responses retries,
refusal contracts and gateway routes. A final operation-registry validation
change passed **113 focused tests in 2.26 seconds**. Compilation, whitespace,
protected-workflow and AGENTS/CLAUDE checks passed. Independent review checked
unchanged worker outcomes/counters, billing authority, privacy, hidden retries,
immutable lease binding and incomplete shutdown snapshots.

At 02:36:35 UTC, the old proof runner had exited and its task-owned API server
was stopped after verifying zero leased runs and zero inflight provider calls.
The original process records and logs were retained. The frozen round and its
unstarted second ICP remain available for the final release proof.

### Final collection release

PR [#514](https://github.com/leadpoet/leadpoet/pull/514) merged at
`6ec7e578a16cd7562cd01014459e9b4e24a28feb` on October 17. Both canonical deployment
controllers completed successfully on that exact commit. The gateway's restart
guard preserved all 20 active leases with durable accepted receipts, with no lost
or expired leases. Runtime attestation and exact build identity passed. The
operator pause remained false and normal claims resumed. The normal validator
passed protected readiness; read-only finalized-chain state matched its signed
weight journal. This verifies weight continuity, not a new broadcast from the new
process.

Independent production readback at 03:10 UTC verified two judge-failure artifacts
on this release. Their 21 and 12 worker calls matched 21 and 12 ledger calls,
full requests and full responses. Each artifact retained six nonstandard worker
responses with exact action links, matching hashes and no observation gaps.
These are successful collection checks, not successful judgments or negative
company labels.

At 03:20 UTC, two normal accepted executions on the final release also verified
92/92 and 110/110 full request/response pairs, all settled, plus five and four
exceptional worker responses. Their exact inputs had no redactions. Both used
direct web access (10 and 35 connections), and both had privacy-filtered provider
payloads. One runner reported a dirty Git checkout; its runtime source cannot be
certified merely because its recorded commit matches the release. These runs
prove production capture works and remain excluded from complete-process proof.

The fresh ICP 1 execution started on the same release. Its exact input was
1,781,748 bytes, with no privacy redactions and hash
`sha256:46b413e5d9c88966e7143bb0170b01628799f7430b4ee4bfe56677c383b5754c`.
Independent billing readback matched OpenRouter ledger entry 5000142 to Gemini
2.5 Flash through Google: $0.0004004, rounded up to 401 micro-USD in the ledger.
Deepline entry 5000261 matched the exact Exa request at 0.1 credits ($0.01), final
and posted. Neither billing API returned request/response bodies.

One real accepted judge on the preceding release also verified the new full-input
and request path: `arena-2026-10-17:baseline-2026-10-17:1:7:score:5` retained its
1,871,821-byte input, all 58 requests and responses, and a 94,083-byte score object
with five company breakdowns. All 28 OpenRouter responses disclosed model and
upstream provider: 25 Luna/Azure and three Sonar/Perplexity. Privacy-filtered
payloads remain explicitly non-exact. This is real judging evidence, not a fresh
complete research example by itself.

### Fresh execution, judgment and export readback

The final-release execution
`arena-2026-10-17-trainingproof:baseline-2026-10-17-trainingproof:1:1:1`
was accepted at 03:24:53 UTC. Its actual judge,
`arena-2026-10-17-trainingproof:baseline-2026-10-17-trainingproof:1:1:score:1`,
was accepted at 03:27:45 UTC. Independent readback and the committed read-only
exporter verified the following at 03:29 UTC:

| Evidence | Result |
|---|---|
| Exact execution input | 1,781,748 bytes; matching hash; no redactions |
| Exact judge input | 1,783,582 bytes; matching hash; no redactions |
| Execution provider calls | 143 worker calls, ledger calls, full requests and full responses; all settled |
| Judge provider calls | 13 worker calls, ledger calls, full requests and full responses; all settled |
| Exceptional worker outcomes | Seven retained, complete, hash-verified and linked; no local refusals or unfinished handlers |
| Final output and judgment | One submitted company; one authoritative scored breakdown; per-ICP score recomputed as zero |
| Source and grading | Frozen source archive verified; both runtime commits match `6ec7e578`; judge image matches frozen grading |
| Actual cost | 641,741 micro-USD ($0.641741), including execution and judging |
| Runtime | Execution sandbox 1,166.78 seconds; judge 127.75 seconds; signed start/finish times retained separately |
| Request storage | 151 unique objects for 156 calls; 2,685,666 raw bytes compressed to 540,609 bytes |
| Private export | 12,935,723 bytes; file mode 0600 |

The example hash is
`sha256:b9ff035d1f6b24b11abb02e2b4f5d285668574e17cfc437b52d8a61397d88cd1`.
The export file hash is
`sha256:f4d27a6029f8b39722a58599de523a2cb4c67da04bc6c2afc846b92e17ad42d1`.
The private payload remains on the gateway; only this evidence summary is in Git.

The export passed `context`, `source_identity`, `exact_inputs`,
`input_output_label` and `costs`. It correctly returned exit 2 with
`research_process=false` for exactly three reasons:
`direct_web_content_not_captured`, `request_privacy_nonexact`, and
`response_privacy_nonexact`. The execution made 117 direct HTTPS connections;
their page bodies are not retained. Credential filtering and hidden-reasoning
exclusion also prevent exact replay. There were no missing broker request or
response payloads, unexplained call-count differences, or unsettled costs for
this execution and judge.

This proves the new collection path works from actual fresh research through a
real judge and independent export. It does not certify a complete full-agent
demonstration or a good positive example. A zero score is retained as a scored
outcome, not silently promoted to a successful company label. The whole two-ICP
probe also retains the first ICP's earlier failures; it is not reported as an
all-successful round. Use broker-only research and review the safe retained
content before selecting future full-process training examples.
