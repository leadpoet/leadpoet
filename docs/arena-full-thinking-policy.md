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

## Initial compatibility-approved whitelist

The runtime source is `lab_arena/reasoning_models.json`. All six entries use Chat Completions, `exclude: false`, an 8,192-token output
limit, ZDR, no provider fallback, and a route price bound. Five use `effort: high`;
Gemma uses its supported `enabled: true` mode. Exact modes live in the registry.

| Model ID | Exact provider selector | Tested result |
| --- | --- | --- |
| `openai/gpt-oss-120b` | `deepinfra/bf16` | Generated text, two tool continuations, strict JSON |
| `mistralai/mistral-large-4-0` | `mistral/zdr` | Generated text on final turn, two tool continuations, strict JSON |
| `deepseek/deepseek-v4.1-flash-1017` | `fireworks` | Generated text, two tool continuations, strict JSON |
| `openai/gpt-oss-20b` | `deepinfra/bf16` | Generated text, two tool continuations, strict JSON |
| `reflection/beam-1.0-501b-a23b` | `nebius/fp8` | Generated text, two tool continuations, strict JSON |
| `google/gemma-4-31b-it` | `coreweave/fp4` | Generated text, two tool continuations, strict JSON |

The evidence files in `docs/evidence/full-thinking/` contain only synthetic test
inputs and exact responses. Their SHA-256 hashes are frozen in the registry.
The tests validate returned model and selected endpoint version/provider metadata.
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
| Champion data eligibility | After execution, ordinary code reads the selected accepted runs from durable storage. Require complete selected research evidence and readable full-thinking text on the approved route, with exact judge inputs and a verified output/score binding. Raw whole-export completeness remains a separate audit: an attributed judge-only response redaction or unresolved judge charge does not erase exact research data. Missing selected evidence, a stale source assessment, truncation, or a wrong route prevents champion eligibility; it does not cancel the round or change the measured score. No extra LLM call reviews runtime traces. |
| Central collection | Apply policy before hashing, capture, and budget admission to all execution model calls. Miner, baseline, and subagent calls share the same gateway. Judges retain their policy. |
| Exact routes | Freeze the reviewed registry in each new round. Pin provider and version expectations. Reject unsupported routes before dispatch. Record returned route mismatches without stopping scoring; exclude them from strict exports. |
| Tool and JSON continuity | Preserve reasoning fields, signatures, opaque continuation state, tool IDs, ordering, and caller schemas in the live conversation. Validate two tool continuations and strict JSON. Never insert thinking into a JSON answer. |
| Useful training data | Export actual generated thinking beside prior observations and the target action. Require request/response hashes, delivery/receipt evidence, a matching frozen route, and complete provider text. Deduplicate alias fields. |
| Linked evidence | Reuse task/ICP, prompts, tools, page observations, receipts, final output, and judgment links. Retain model/route/settings, source version, retries, cost, and runtime. Missing or unsupported evidence remains an explicit gap. |
| Dataset quality | Reuse secret removal, failure distinctions, deduplication, held-out companies, source references, dataset versions, and existing training-permission status. Labels still require evaluation before training. |
| Honest counts | Separate miner/baseline/judge calls, saved text, and complete trajectories. Report empty, summary-only, opaque, truncated, unavailable, failed capture, and unproved routes separately. No invented historical backfill. |
| $6 cap | Set future execution admission and cost eligibility to $6 per ICP for both miners and baseline. Current-policy admission uses confirmed competition cost, without pending or uncertain monetary holds. Historical $4 behavior stays unchanged. |
| Round continuity | No capture failure or missing text cancels a round or stops publication. Missing activation prerequisites create an ordinary round with a visible `not_ready` reason. No live catalog request is needed to create the round. |
| Safe activation | Default off. Require a compatible baseline route and a tested immutable baseline source plus digest before activation. Apply only to future unopened rounds. |

The selected research check is shared by champion capture eligibility and strict
thinking exports. It retains the raw export's completeness and privacy flags.
Any permitted judge-only gap is listed in the training record's source reference.
Selected execution requests, observations, receipts, thinking, output, and costs
must still pass their exactness checks. A literal `[REDACTED]` in an unchanged
provider response is source text; an actual capture redaction is an evidence gap.

The output allowance is a tested total-token envelope. An effort setting is not
a hard thinking-token budget and cannot guarantee a final answer fits. Truncated
responses are explicitly excluded from the strict dataset. Runtime and answer
quality may change; the user has accepted this tradeoff.

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
- No native Responses route is approved yet. The existing broker uses completed
  upstream responses; this PR does not add arbitrary upstream streaming support.
  Existing buffered Codex SSE behavior is preserved, but does not make a Codex
  model eligible. Hosted Responses web search remains unapproved.
- Keeping ZDR worked for all six approved routes. Relaxing retention does not
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
- Apply current main's `562-lab-arena-review-context-integrity-reason.sql`, then
  `563-lab-arena-review-source-digest-reason.sql`. Apply this PR's exact committed
  migrations `564-lab-arena-full-thinking-six-dollar-cap.sql`,
  `565-lab-arena-research-review-result.sql`,
  `566-lab-arena-full-thinking-champion-baseline.sql`,
  `567-lab-arena-full-thinking-catalog-commit.sql`, and
  `568-lab-arena-training-publication-eligibility.sql` through the normal release
  workflow before activation. They check source hashes and database privileges,
  preserve unrelated behavior, and are idempotent.
  The research review migration extends main's exact review function and
  preserves both context-integrity and source-digest handling.

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
Migration 555 was renumbered from 551 after the latest main branch added its own
551 through 554 migrations. No production migration or activation is part of this PR.

## Review fixes — 2026-10-24

Policy-off Chat requests retain the existing message normalization and request hashes.
Full-thinking history preservation applies only when the frozen policy enables it.
Research citations require an exact source excerpt with at least 12 non-whitespace
characters. This is a context floor, not proof that the reviewer is correct. An
`unverified` assessment uses the existing bounded retry policy; a clearly fixed
workflow still fails eligibility.

Migration563 lets the real publication validator accept a coherent
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
