# Full generated thinking: draft implementation scope

This change is for review in a draft PR. It does not activate the policy, change
an existing round, or deploy any service. The policy name is
`full_generated_thinking_v1`.

## Admission rule

Allow a model only through a tested provider, API, and reasoning mode that emits
its generated thinking text. A summary, encrypted state, token counter, or
explanation requested after the answer is not a substitute. “Full” means the
provider's unsummarized generated text; it does not mean access to neural state
or a guarantee that the text explains every internal computation.

Country of origin and open/closed weights do not decide admission. Actual route
behavior does. Catalog capability flags alone do not grant approval.

## Reviewed route registry

The runtime source is `lab_arena/reasoning_models.json`. It contains seven exact
model/API routes: six Chat Completions routes and DeepSeek through Fireworks on
Responses. All use ZDR, no provider fallback, and a route price bound. Chat calls
use `exclude: false`; Responses uses its typed `reasoning_text` output. GPT-OSS
keeps 8,192 output tokens. The other reviewed routes use 32,768. Gemma uses
`enabled: true`; the other routes use `effort: high`.

| Model ID | API | Exact provider selector | Output allowance |
| --- | --- | --- | --- |
| `openai/gpt-oss-120b` | Chat | `deepinfra/bf16` | 8,192 |
| `mistralai/mistral-large-4-0` | Chat | `mistral/zdr` | 32,768 |
| `deepseek/deepseek-v4.1-flash-1017` | Chat | `fireworks` | 32,768 |
| `openai/gpt-oss-20b` | Chat | `deepinfra/bf16` | 8,192 |
| `reflection/beam-1.0-501b-a23b` | Chat | `nebius/fp8` | 32,768 |
| `google/gemma-4-31b-it` | Chat | `coreweave/fp4` | 32,768 |
| `deepseek/deepseek-v4.1-flash-1017` | Responses | `fireworks` | 32,768 |

The evidence files in `docs/evidence/full-thinking/` contain only synthetic test
inputs and exact responses. Original 8,192-token evidence stays unchanged and
replays with its original frozen registry. The four larger caps bind the
supplemental [32,768-token capture manifest](evidence/full-thinking/chat-output-cap-32768-20261025.json)
by SHA-256 in the current registry. It retains exact requests, responses, wire
bytes, local PostgreSQL readback, and worker/SDK receipts. The tests validate
returned model and selected endpoint version/provider metadata. The separate
[Responses route manifest](evidence/full-thinking/deepseek-fireworks-responses-20261025.json)
retains three synthetic Responses turns, typed full text, two function
continuations, final strict JSON, and exact wire/readable/receipt proof. No
private native-agent prompts are included.
The requested provider selector is recorded separately: returned metadata does
not independently expose its quantization suffix.

The probe reports say `approval_granted: false`: running a probe never approves
a route. Registry approval is a separate review of provider semantics and the
stored compatibility evidence. It is not approval to deploy or train.

Provider semantics are documented by [DeepInfra](https://deepinfra.com/openai/gpt-oss-120b/api),
[Mistral](https://docs.mistral.ai/studio/conversations/reasoning), [Fireworks](https://docs.fireworks.ai/guides/reasoning),
[OpenAI GPT-OSS](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot),
[Reflection](https://developers.reflection.ai/reasoning), and
[Gemma](https://ai.google.dev/gemma/docs/capabilities/thinking).
Mistral's empty tool-only turns are recorded as empty, not counted as traces.
The DeepSeek route through DeepInfra returned HTTP 429 in probes and is not approved.

`docs/arena-full-thinking-candidates.json` is the broader catalog snapshot and
review queue. It includes untested, summary-only, and unknown models, clearly
classified. It is not a runtime allowlist or a claim that every listed model
exposes full thinking. New routes require the same evidence and tests.

## Implementation and acceptance criteria

| Requirement | Implemented behavior / required proof |
| --- | --- |
| Existing source review | In future full-thinking rounds, the existing paid full-source review also checks reachable model-directed research. Python, custom tools, harnesses, and fixed safety/budget scaffolding remain allowed. Fixed extraction workflows and ignored/post-hoc model calls do not satisfy the requirement. Ambiguous control remains unverified. Store the assessment against the exact source archive and review policy. |
| Champion data eligibility | Read the selected accepted runs from durable storage. Apply the frozen per-ICP missing-token rule and 80% ICP coverage floor below. Require exact source review, route, retained context, receipt, output/score and accounting proof. Verified incomplete ICPs stay out of strict training while all measured scores still count. Wrong routes and integrity failures remain hard. Ineligibility does not cancel the round or change its measured score. No extra LLM call reviews runtime traces. |
| Central collection | Apply policy before hashing, capture, and budget admission to all execution model calls. Miner, baseline, and subagent calls share the same gateway. Judges retain their policy. |
| Exact routes | Freeze the reviewed registry in each new round. Pin provider and version expectations. Reject unsupported routes before dispatch. Record returned route mismatches without stopping scoring; exclude them from strict exports. |
| Tool and JSON continuity | Preserve reasoning fields, signatures, opaque continuation state, tool IDs, ordering, and caller schemas in the live conversation. Validate two tool continuations and strict JSON. Never insert thinking into a JSON answer. |
| Useful training data | Export actual generated thinking beside prior observations and the target action. Verify the saved request and provider result against the existing worker delivery record, the frozen route, and the retained text. Client read acknowledgments and named SDK phase callbacks are optional diagnostics in the new policy. Deduplicate alias fields. |
| Linked evidence | Reuse task/ICP, prompts, tools, page observations, receipts, final output, and judgment links. Retain model/route/settings, source version, retries, cost, and runtime. Missing or unsupported evidence remains an explicit gap. |
| Dataset quality | Reuse secret removal, failure distinctions, deduplication, held-out companies, source references, dataset versions, and existing training-permission status. Labels still require evaluation before training. |
| Honest counts | Separate miner/baseline/judge calls, saved text, and complete trajectories. Report empty, summary-only, opaque, truncated, unavailable, failed capture, and unproved routes separately. No invented historical backfill. |
| $6 cap | Set future execution admission and cost eligibility to $6 per ICP for both miners and baseline. Current-policy admission uses confirmed competition cost, without pending or uncertain monetary holds. Historical $4 behavior stays unchanged. |
| Round continuity | No capture failure or missing text cancels a round or stops publication. Missing activation prerequisites create an ordinary round with a visible `not_ready` reason. No live catalog request is needed to create the round. |
| Safe activation | Default off. Require a compatible baseline route and a tested immutable baseline source plus digest before activation. Apply only to future unopened rounds. |

## Frozen token and ICP rules

Future full-thinking rounds freeze
`training_maximum_missing_reasoning_percent`, provisionally **1%**. This is a
configurable policy choice, not an empirically established optimum. For each
ICP, divide positive reported reasoning tokens on completed, untruncated,
verified calls without admissible full text by all reported reasoning tokens on
completed, untruncated, verified calls in that ICP. The comparison uses integer
arithmetic. Summary-only or opaque-only replies can consume this allowance;
unapproved readable fields, wrong routes, and bad hashes or receipts cannot.

Explicitly reported zero reasoning tokens prove a no-thinking completed turn,
including a note plus a tool call or a direct answer. Missing usage is unknown,
not zero. An unproved no-text call makes the ICP incomplete. Each complete ICP
must still have at least one admissible full-thinking training target.

A verified failed provider attempt has no training target and does not enter
the token numerator or denominator. A verified truncated reply also loses only
its own target and is excluded from both token counts. Other independently
reconstructable calls can remain. The new policy also omits a tool-call target
when its verified output usage exactly reaches its requested cap and its JSON
arguments end unfinished at EOF, even if the provider labels it completed. The
report labels this inference; it never repairs the JSON or waives other evidence
checks. Raw exposed text remains saved and separate from usable-target counts.
Malformed arguments in a prior call may remain exact input history when paired
with the declared local tool's result under the new policy. This preserves a
later valid response after a tool error. It does not repair the failed arguments,
admit them as a training target, or waive identity and stored-evidence checks.
If no full target survives, the ICP is
incomplete. An ICP above its frozen missing-token limit is also incomplete.
At least **80%** of selected ICPs must be complete, rounded up. An incomplete
ICP contributes no strict training records; its measured score still counts.

These rules are selected by the presence of the frozen setting. An explicit
`0` uses the new call rules with zero missing-token tolerance. A historical
round without the setting keeps its original behavior, including its limited
unknown-usage tool-only compatibility; a live default cannot rewrite that round.
Exact failed-attempt omissions also apply only to rounds with the new setting.

The new policy uses the saved provider request/result and the existing worker
socket-write record to verify a training pair. It does not require a client to
send a separate acknowledgment or legacy SDK phase snapshots. It adds no TYCHE
callback, receipt protocol, or recovery state. The host-delivered result must
match the saved call, status, byte count, and content; a later billing-recovery
response cannot replace it. Missing client reports do not prove consumption,
and the export must not claim that the agent read or used the result. Actual
input, output, scoring, privacy, and integrity checks remain required. Historical
rounds keep their existing checks.

This verifies the provider-visible trajectory. It does not prove that Arena
observed every local function or web request inside a miner's agent. Those local
coverage limits remain visible in the raw report. Miners can keep their tools
and harnesses without adding logging or phase callbacks. Any tool result sent
back to the model is retained in the model's captured input; provider tool
results are also retained at the gateway.

The selected research check is shared by champion capture eligibility and strict
thinking exports. It retains the raw export's completeness and privacy flags.
Any permitted judge-only gap is listed in the training record's source reference.
Selected execution requests, observations, receipts, thinking, output, and costs
must still pass their exactness checks. A literal `[REDACTED]` in an unchanged
provider response is source text; an actual capture redaction is an evidence gap.

The output allowance is a tested total-token envelope. An effort setting is not
a hard thinking-token budget and cannot guarantee a final answer fits. The
32,768-token checks used three short synthetic calls per route, two tool
continuations, and a negative qualification from conflicting evidence. They
prove acceptance of the requested envelope and exact capture; they do not
exhaust 32,768 output tokens or prove full ICP quality. Both GPT-OSS cap probes
were blocked by the account's ignored DeepInfra route, so their limits remain
8,192. The separate Responses proof uses the same short conflicting-evidence
scenario and does not exhaust the allowance. Truncated responses are excluded
from training targets. Runtime and answer quality may change; the
user has accepted this tradeoff.

## Limits and release gates

- The current baseline and Tyche's hosted OpenAI route do not qualify for this
  full-text-only registry. No baseline or miner is silently switched. The initial
  registry deliberately has no baseline selection. Before activation, publish and
  test a compatible baseline archive, including every subagent route. The
  activation command requires `--reasoning-baseline-source-url` with a same-repo
  40-character commit archive URL and `--reasoning-baseline-source-sha256`.
  Reused accepted/frozen baselines must also match the stored digest.
  At baseline freeze, a future full-thinking round adopts a completed
  full-thinking champion's source commit, digest, and funding owner together.
  If the current champion is from a legacy round, the first full-thinking round
  keeps its tested bootstrap source while retaining that champion as payer.
  This also covers a round opened before the previous round's promotion.
  Frozen rounds do not change their baseline snapshot.
- Only the exact DeepSeek/Fireworks Responses route is approved. The existing
  broker uses completed upstream responses; this PR does not add arbitrary
  upstream streaming support. Existing buffered Codex SSE behavior is preserved.
  Provider-side Responses web search remains unapproved. The latest native test
  uses TYCHE's exact Codex 0.154.0 version, with the substitutions listed in the
  verification report. It does not prove a scored or published TYCHE workflow.
  Observed native token coverage supports keeping the provisional 1% setting;
  the small sample does not establish the best threshold or production readiness.
- Keeping ZDR worked for all seven approved routes. Relaxing retention does not
  turn provider-withheld thinking into readable text. No privacy relaxation is
  included.
- A provider can return no thinking on an individual successful call. Such calls
  are recorded, not called traces. Capture failures cannot silently enter the
  strict dataset. A full trajectory can include valid tool-only turns without
  counting those turns as thinking samples.
- Synthetic compatibility tests do not prove lead quality, a six-dollar full ICP
  run, or production readiness. Before activation, run representative ICPs,
  verify stored/exported artifacts and publication, and review latency, spend,
  failure rates, and output quality. No production claim is made in this PR.
- Apply the exact combined migration tail in numeric order through
  `574-lab-arena-review-guard-row-read.sql` before activation.
  This PR's migrations are `565-lab-arena-full-thinking-six-dollar-cap.sql`,
  `566-lab-arena-research-review-result.sql`,
  `567-lab-arena-full-thinking-champion-baseline.sql`,
  `568-lab-arena-full-thinking-catalog-commit.sql`,
  `569-lab-arena-training-publication-eligibility.sql`, and
  `570-lab-arena-confirmed-full-thinking-final-cost.sql`. Main also includes
  `571-lab-arena-closed-failed-score-host-recovery.sql`,
  `572-lab-arena-concurrent-baseline-judging.sql`,
  `573-lab-arena-closed-provider-eligible-run-materialization.sql`, and migration
  574 above. The review-guard read is required by the current source reviewer.
  Preserve earlier review migrations 562 and 563 and baseline-deadline migration
  564. Use the normal release workflow and retain the source-hash, privilege and
  idempotency checks. The research-review migration preserves both main's
  context-integrity and source-digest handling.

## Reproducible checks

`scripts/probe_full_thinking_routes.py` runs only explicit synthetic probes. It
does not approve a route, activate a policy, use production miner data, or retry
with another provider. Credentials are supplied through the environment and are
not written to the report.

`scripts/prepare_arena_full_thinking_dataset.py` builds a private strict dataset
from existing exports. Empty datasets return a nonzero result. This is a dataset
preparation tool, not authorization to train on miner submissions.

Tests cover exact evidence hashes, broker routing and reservations, stored live
three-turn probes, malformed history, summary-only/opaque-only/truncated exclusion, empty
tool turns, activation failure, old-round behavior, and local PostgreSQL cap
concurrency and settlement. See the PR verification record for executed commands
and remaining exclusions.

## Capture audit follow-up

The [end-to-end audit](arena-trace-capture-audit.md) records fresh real provider
calls through disposable PostgreSQL, object storage, worker socket delivery,
agent read acknowledgements, and raw/readable export readback for every approved
route. It also records tested routes that did not qualify, without silently
admitting them or changing the provider. These are compatibility approvals, not
a claim that every call returns thinking or that any model has passed a full
sales task evaluation.

## Source review and final stored-data check — 2026-10-24

The new scope keeps the current miner-controlled harness. It does not add a
mandatory shared agent loop, new judge, or confidence-based reward. The review
request adds source-backed research-control findings to the existing review.
The reviewer uses the submitting miner's credential and existing reservation,
settlement, and retry rules. The post-run check makes no provider requests.
Existing organizer baseline funding and its source-review exemption remain;
the baseline is only a measured comparison and is never a contender. Its capture
gaps are reported, but do not prevent a verified challenger from beating its score.

The source review estimates whether the code can use observations to choose
substantive research actions and finish/continue. It is not proof that every run
uses that capability. The stored-data check verifies what was returned, delivered,
and retained; it does not prove written thinking faithfully caused a decision or
that a long trace is good training data. No unseen private reasoning is inferred.
Missing full text is distinct from summaries, opaque state, or token counts.
Opaque continuation state can coexist with eligible full text and is preserved.
Distinct auxiliary summaries also stay separate from full-text training targets.
If identical text appears in both a full-text field and a summary field, the
strict export conservatively excludes that ambiguous alias.

Each finalist's saved publication includes a bounded training eligibility result.
The public round/submission view shows reasons and counts without exposing source
review prose or private traces. Training dataset permissions, splits, deduplication,
and label checks remain separate; they are not new champion score rules.

Only future opted-in rounds use these requirements. This PR stays default off.
The PR-only migrations are numbered 564 through 568, after main's 563.
They must be applied in numeric order. No production migration or activation is part of this PR.

## Review fixes — 2026-10-24

Policy-off Chat requests retain the existing message normalization and request hashes.
Full-thinking history preservation applies only when the frozen policy enables it.
Research citations require an exact source excerpt with at least 12 non-whitespace
characters. This is a context floor, not proof that the reviewer is correct. An
`unverified` assessment uses the existing bounded retry policy; a clearly fixed
workflow still fails eligibility.

Migration568 lets the real publication validator accept a coherent
`training_data_incomplete` result while preserving measured scores and all cost
guards. Missing training data disqualifies that contender; it must not stop the
round, hide other results, or prevent an eligible peer from becoming champion.
The database check validates the publication report. The service separately reads
and verifies the durable raw training evidence.

The current PostgreSQL integration test follows that same published winner through
the real Git promoter into a disposable local repository, verifies both branch
heads and source contents, then freezes the winner's source and funding owner in
the next round. This proves the local publication/promotion path; it does not
publish a production champion or activate the policy.


## Coverage floor and billing correction — 2026-10-25

The frozen round field `training_minimum_icp_coverage_percent` defaults to 80
for newly enabled full-thinking rounds. The gateway setting
`LAB_ARENA_TRAINING_MINIMUM_ICP_COVERAGE_PERCENT` can set a future round's value
from 1 through 100. The public round view exposes the frozen value. Changing the
gateway default never changes an existing round. A historical full-thinking
round without the field retains its original 100% rule.

The required complete count is rounded up: 8 of 10, 16 of 20, and 3 of 3 at 80%.
All ICP scores still contribute. Only complete, verified ICPs enter strict training
exports. Raw audit data and readable views retain incomplete ICPs with their gap
reasons. Route violations, changed source identity, or invalid evidence cannot be
excused by the coverage floor. Coverage establishes availability and integrity,
not the educational value or causal faithfulness of the model's thinking.

A documented unsettled selected-provider charge also consumes one incomplete
ICP when the request, receipt, route, and identity proof is intact. A combined
documented privacy or explicit output-length gap still counts as that same ICP.
These ICPs remain excluded from strict exports. A missing or contradictory
accounting record is not treated as a proven unsettled charge.

Migration565 raises the frozen full-thinking cap to $6 while preserving main's
confirmed competition-cost accounting. It adds no pending or uncertain monetary
holds. Call limits, price ceilings, and output-token limits remain enforced.
For future full-thinking rounds with a valid frozen coverage setting, final
eligibility also counts confirmed settled competition cost. Unsettled charges
remain in the ledger and exclude the affected ICP from strict exports, but do
not hold a fully scored baseline or exclude an otherwise eligible entrant by
cost alone. Active provider calls still block final publication. Confirmed
spend over $6 still fails the cost check, including charges whose success is
unknown; proved infrastructure exclusions remain. The cap is on confirmed
competition cost, so later bills can exceed it. Historical rounds keep their
previous billing rule.

Training reads retry a recognized transient object-store error once. If an
accepted output was already read and validated during the same publication
attempt, its exact cached bytes can supply a later failed training read. All
normal source, input, output-hash, receipt, route, cost and privacy checks still
run. That later availability failure consumes one incomplete ICP; it does not
repair or overwrite the stored object. Other missing artifacts and integrity
errors remain hard failures. A first scoring read without verified output is
still a score-authority problem, not a training-coverage exception.

The enhanced source-review check is still tied to full-thinking activation in
this draft. An independent shadow/enforce review, including the baseline, is not
implemented or enabled: the current baseline is created after intake review and
is excluded from the review claim and miner-key funding path. This is an explicit
remaining scope item, not evidence that current rounds have the new check.
