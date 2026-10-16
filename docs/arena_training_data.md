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
No new database table, queue, background monitor, dependency or lineage service is
required. Input uploads require an active owned lease before storage. Request
object writes occur only after accounting admission. Capture failure cannot change
provider execution, scores or billing. It produces incomplete training evidence.

The full payload limit is explicit. Oversized or unavailable evidence is rejected
as a training source; it is never silently clipped and called complete. Credentials
and hidden provider reasoning are excluded. Privacy filtering and exact replay
are separate properties from label validity.

## Remaining boundaries

- Direct HTTPS egress records connection/byte totals, not page contents. Historical
  TLS traffic cannot be reconstructed. Runs using it are excluded from complete
  process training unless their relevant observations are independently retained.
  This change does not introduce a TLS interception proxy.
- Local frozen-catalog discovery has no paid ledger reservation. Its diagnostic
  events can be abbreviated. A count mismatch cannot certify these calls as full.
  Use a broker-only subset for a first complete call-trajectory experiment.
- Hidden provider chain of thought is not a collection target. Unlogged internal
  variables or private provider search steps are not recoverable. Explicit model
  messages and returned tool outputs are the available training state.
- Optional model-reported decision files and stdout/stderr remain diagnostic,
  bounded and potentially incomplete. They are not authoritative demonstrations.
- A matching hash proves linkage and integrity, not factual truth or a good label.
  Training selection still needs company-level review and evaluation.
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
Deepline's exact-request billing endpoint also returned the matching request for
ledger entry 4970376. These are bounded samples, not a claim that all historical
charges were independently re-audited.

The retained September 13 Supabase archive manifest covers 20,270,597 rows in
15 retired relations. The main dump is 6,006,835,409 bytes. Its 14,743 attested
execution-result rows are infrastructure operations (allocation, source catalog,
model/weight authority and settlement), **not 14,743 agent research runs**. Receipt
constraints exclude request/response bodies. Encrypted artifact envelopes remain
opaque without the original protected decryption path. No full restore or key
export was performed, and this archive is not counted as recoverable training
examples. Retain it unchanged.

## Export and first experiment

Use `scripts/export_arena_training.py --run-id ... --environment-file ... --output ...`.
It performs read-only object and database access, creates a new private file, and
reports independent completeness dimensions. An incomplete example exits with
code 2; identity ambiguity or access failure exits with code 1. Never treat the
existence of an export file as proof it is eligible for training.

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

Implementation and release evidence are recorded with the task's final review.
Local capture, input, exporter, proxy and existing runner checks must pass after
integration. Live acceptance requires a fresh reward-disabled execution and its
real judgment, then independent readback of the exact private input/request
objects, ledger receipts, output and score. A replay fixture or HTTP health result
alone does not meet this acceptance criterion.
