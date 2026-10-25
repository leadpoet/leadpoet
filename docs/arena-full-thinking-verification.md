# Draft PR verification — 2026-10-23

## Resumed verification — selected research and literal markers

The October 24 resumed audit found two false exclusions in the strict data path.
First, an unrelated judge response redaction or unresolved judge charge could
exclude an otherwise exact research trajectory. Champion eligibility, strict
thinking export, and readiness now share one selected-research check. It requires
exact research inputs, observations, receipts, outputs, costs, and their binding
to the accepted judge input and output. The original whole-export completeness
and privacy flags remain unchanged. Permitted judge-only gaps are disclosed in
each strict record. Selected research gaps still reject eligibility.

Second, literal `[REDACTED]` text supplied by a provider was treated as evidence
that capture had removed a secret. The checks now use capture provenance and
hashes. The sanitizer also keeps complete, already-redacted values byte-exact.
Real credentials attached to marker text are still removed. Independent checks
covered 17 attached-marker credential forms and six safe standalone controls.
Readable thinking and summaries retain unchanged marker text; actual capture
redactions still exclude strict thinking and champion eligibility.

The new positive regression cases failed before their respective fixes. The
resumed focused capture, export, readiness, and eligibility run passed 296 tests.
The real PostgreSQL publication, promotion/next-baseline, schema-tail, $6 limit,
policy-off protocol, and completed-call replay checks passed 37 tests. A later
independent check of literal thinking text and real redaction rejection passed
154 focused tests. These test counts overlap and are not a unique total.

A fresh paid candidate-and-baseline run uses a clean frozen checkout and the
Reflection route. It remains separate from the older DeepSeek evidence below.
Its final outcome must be read from the recorded end-to-end result; the unit
checks in this section do not establish live end-to-end success.

Independent CI comparison checked PR run `40151558411` at `f3443cc4e` against
main run `40153179357` at `2b6c490a0`. Both hit the 60-minute limit near 35% of
the suite. All 225 failures observed in that PR run also failed in the main run;
137 cases reached only in the PR run passed. Three representative failures
reproduced on both code lines, including a legacy database fixture that stops
before migration 474 but calls a function introduced there. No PR-specific
failure was observed in the tests that ran. This does not establish complete CI
readiness: the remaining tests were not reached. The relevant publication and
current-schema cases were run separately in the local PostgreSQL checks above.

Scope: [implementation policy](arena-full-thinking-policy.md). The base was
`origin/main` at `09bc088ea`. The implementation is isolated from the user's dirty
checkout. No merge, deployment, restart, production SQL, round mutation, or model
substitution was performed.

## Earlier draft live evidence

Three exact routes passed two consecutive tool continuations and a final strict
JSON response with generated thinking, ZDR, no fallback, required parameter
support, and price bounds. Exact evidence is committed under
`docs/evidence/full-thinking/` and bound to the registry by SHA-256.

| Route | Reported cost of final three-call probe |
| --- | ---: |
| GPT-OSS 120B / DeepInfra BF16 | $0.000136238 |
| Mistral Large 4 / Mistral ZDR | $0.001472620 |
| DeepSeek V4.1 Flash / Fireworks | $0.0001920072 |

All synthetic diagnostic probes together reported $0.004003052. Earlier probes
included a tool-argument correction and two DeepInfra 429 results; no route was
silently changed within a test. This is provider-reported usage, not an invoice
reconciliation. Each probe uses synthetic company evidence only.

Saved evidence passes through the real broker with a fake network transport,
then separately through private request capture, ledger response reconstruction,
and strict training export. The tests use the actual committed registry, verify
hashes and route identity, preserve history and JSON schemas, and deduplicate
thinking aliases. The probe's ASCII JSON hash format and Arena's UTF-8 JSON hash
format are explicitly checked on the Unicode DeepSeek response.

## Earlier draft local checks

Earlier combined result: **1,762 passed, 21 skipped**, one dependency deprecation
warning, in 154.43 seconds. Nineteen skips require the real Codex CLI package or
Linux root/runsc; two are existing declared-field operation cases. None of the
new full-thinking route, export, round, or PostgreSQL tests was skipped.

The final combined run covers these files:

```sh
python3 -m pytest -q -rs \
  tests/lab_arena/reasoning_capture_test.py \
  tests/lab_arena/full_thinking_protocol_test.py \
  tests/lab_arena/full_thinking_round_test.py \
  tests/lab_arena/full_thinking_saved_probe_test.py \
  tests/lab_arena/training_thinking_test.py \
  tests/lab_arena/training_capture_test.py \
  tests/lab_arena/training_export_test.py \
  tests/lab_arena/training_steps_test.py \
  tests/lab_arena/training_readiness_test.py \
  tests/lab_arena/training_dataset_test.py \
  tests/lab_arena/training_export_receipts_test.py \
  tests/lab_arena/training_capture_transport_test.py \
  tests/lab_arena/flat_per_icp_cost_test.py \
  tests/lab_arena/codex_protocol_validation_test.py \
  tests/lab_arena/codex_runtime_test.py \
  tests/test_configure_lab_arena_production.py \
  tests/test_probe_full_thinking_routes.py \
  tests/lab_arena/test_lab_arena_broker.py \
  tests/lab_arena/test_lab_arena_operations.py \
  tests/lab_arena/test_lab_arena_contracts.py \
  tests/lab_arena/broker_policy_revision_test.py \
  tests/lab_arena/broker_accounting_outage_test.py \
  tests/lab_arena/broker_round_lease_wiring_test.py \
  tests/lab_arena/full_thinking_six_dollar_postgres_test.py
```

The PostgreSQL checks use disposable local databases. They apply migration 555 (previously numbered 551)
twice, verify exact function source hashes/privileges, and test $6 boundaries,
parallel holds, settlement release, uncertain costs, dynamic tools, paid failures,
retries, charged host loss, free calls, and historical $4 behavior.

Independent review found and verified fixes for two issues: charged host loss
must count toward admission even when competition scoring exempts it; reused
accepted/frozen baseline archives must also match the frozen digest. The reviewed
SQL selects one current ledger state per call, so a settled charge replaces its
reservation without double counting.

Python compilation for all changed Python files, `git diff --check`, and byte
equality of unchanged `AGENTS.md` and `CLAUDE.md` passed.

## Existing broad fixture failures

These were reproduced in a clean checkout of unchanged `origin/main`, then
compared with the implementation checkout. They were not changed in this PR:

- `test_lab_arena_service_round.py::test_operator_hold_survives_scoring_deadline_then_normal_round_publishes`:
  the local fixture lacks SQL function `lab_arena_execution_attempt_count`.
  The clean base run passed six earlier tests before this failure.
- `test_lab_arena_baseline.py::test_baseline_download_uses_the_same_source_checks_and_freezes`:
  fake `_Store` lacks `latest_published_day`, producing `baseline_source_not_ready`.
  The broader implementation run reported 31 passed and 10 fixture failures;
  the representative failure above was reproduced on the clean base.

These failures mean there is no claim that the full legacy suite or a complete
new-policy publication run passed. A representative full ICP run, compatible
baseline evaluation, production storage/export verification, and publication
proof remain activation gates. The committed registry has `baseline_route: null`
and the policy defaults off. No production migration was attempted.

## End-to-end capture audit follow-up

See [the audit](arena-trace-capture-audit.md) and its committed machine manifest.
The follow-up ran 85 real OpenRouter attempts across 13 models. Every one of the
six runtime-whitelisted routes passed a fresh three-turn model/tool/JSON test
through disposable PostgreSQL, private object storage, the actual worker socket,
SDK read acknowledgement, and raw/readable readback. Three exact routes were
added. Hosted OpenAI returned and retained a summary with ZDR enabled after the
Responses history-format regression was fixed; it remains excluded from the
full-thinking registry.

An independent readback checked 36 saved exports with 84 calls. All returned
readable reasoning strings and usage objects survived raw-to-export-to-view.
Source paths and value hashes matched. One early error probe lacked a successful
export readback and is marked unverified; its route was not admitted.

Final follow-up test runs:

| Check | Result |
| --- | --- |
| Earlier command above plus live harness and renderer tests | 1,797 passed, 28 skipped, one dependency warning |
| Seven opt-in PostgreSQL/socket checks from the live harness | 7 passed, no skips, one dependency warning |
| Final coverage/renderer/saved-probe/harness checks after two reporting edge-case regressions | 46 passed, 7 opt-in skips already exercised separately |

The final edge cases cover `reasoning.max_tokens`, legacy `include_reasoning`,
explicit `none` controls, and null usage aliases beside valid nested token counts.
They change reporting only. No provider call or request was repeated for this
projection-only change. Saved readable views were regenerated and checked again.

The PostgreSQL harness checks ran with `ARENA_CAPTURE_POSTGRES_TEST=1`. They cover
real SQL readback, durable ledger evidence when export fails, successful and
error-body socket acknowledgements, diagnostic route controls, and conservative
budget holds before dispatch. The remaining 21 skips are the earlier real Codex
CLI/Linux-sandbox and existing declared-field cases described above.

Independent review verified live history order, tool IDs, all six worker receipt
chains, current exact OpenRouter endpoint/ZDR metadata, new evidence hashes,
reader provenance, and unchanged active-round behavior. Review findings about an
old Reflection probe reference, derived marker paths, and misleading local-registry
labels were fixed and rechecked. The original failed probe records were retained.

Python compilation, `git diff --check`, unchanged AGENTS/CLAUDE parity, and a
scoped secret-pattern scan passed. Only synthetic public test evidence is in Git.
Private full test artifacts remain under `Arena_Trace_Audit_20261023` in Downloads.
No merge, deployment, production write, restart, or round mutation occurred.


## Reviewer and champion eligibility follow-up — 2026-10-24

This follow-up uses the existing source reviewer and a deterministic post-run
read of stored exports. It keeps custom Python/tools/harnesses and adds no second
LLM review of runtime data. The code-review assessment is source- and policy-bound.
Only future full-thinking rounds apply the gate. Frozen entrants with a stale
review can finish scoring after a restart; their stale evidence prevents a win.
Missing trace evidence cannot cancel publication or overwrite measured scores.

Live tests used the explicitly authorized gateway OpenRouter key, with a shared
$8 internal cap below the user's $15 allowance. Production source reviews still
use the submitting miner's key. The post-run check performs no paid calls.

- Ten independent synthetic source fixtures were executed under controlled model
  and observation changes. The existing paid source reviewer matched all ten
  expected labels: three model-directed, six fixed/decoy workflows, one unverified.
  These small test cases do not establish general reviewer accuracy.
- All six approved routes were attempted through the real Broker, disposable SQL
  ledger, local worker socket, agent read receipts, and raw/readable readback.
  Mistral, DeepSeek, Reflection, and Gemma passed two continuations and correct JSON.
  GPT-OSS 120B and 20B returned HTTP404 because this gateway account ignores their
  pinned DeepInfra provider. Account restrictions were not changed. Earlier route
  evidence remains; current availability with this key is not claimed.
- Four live runs executed the exact previously reviewed adaptive, fixed-extractor,
  and post-hoc fixture sources using the approved DeepSeek/Fireworks route. The
  adaptive source used two or three calls and changed its executed action when
  evidence changed. Fixed and post-hoc sources each produced real full thinking,
  but correctly failed the independent source research requirement. All seven
  calls retained exact provider receipts and readable thinking. This is an
  injected synthetic runtime, not an admitted or scored production submission.
- Confirmed cost: $0.3881845396. Unpriced failure bounds held: $0.011323.
  Combined: $0.3995075396. No paid source-review or provider calls were made by
  subagents. Raw artifacts remain in a private local validation directory.

Relevant broad suite: 1,424 passed and two existing declared-field cases skipped.
Final focused eligibility, publication, dashboard, SQL, and saved-capture replay checks: 84 passed. The final SQL556 regressions cover
ordinary assessed pass/reject, unverified/parser errors, zero-cost local duplicate
rejection, misconduct rejection, settlement replay, legacy isolation, source
mismatch, and guarded migration replay. Independent review identified and fixed
both discarded database assessment metadata and stale-frozen scoring blockage.

A separate 17-test replay/helper run passed. Saved live ledger rows, provider
bytes, worker receipts and agent acknowledgments passed through the real export
assembler and final gate with explicitly synthetic benchmark, source context,
input/output, judgment and phase records. Source bytes match the reviewed fixture.
The original phase-incomplete diagnostic export failed, as did missing-request
and missing-agent-ack variants. This is a hybrid replay, not a live admitted
production qualification.

The new gate does not prove thinking faithfulness, downstream training gain,
production latency, or full real-ICP quality. Large-round export-read latency has
not been measured. Default-off activation and representative end-to-end release
validation remain prerequisites. No production apply, restart, merge, or deploy
was performed. Migration555 is the renamed former551; migration556 persists the
new bounded review result. Both must pass the normal release workflow.

Machine summary: [reviewer/capture evidence](evidence/full-thinking/reviewer-capture-validation-20261024.json).

## Initial ten-ICP sourcing attempt and review corrections — 2026-10-24

**This initial attempt stopped before completion; later results are recorded below.
The PR remains a draft and activation remains off.** Earlier component and synthetic tests did not prove the actual
database publication boundary or real sourcing/qualification workflow.

The new run used the actual source archive, signed submission review, worker
entrypoints, provider broker, durable PostgreSQL ledger, and private object store.
Both the candidate and baseline completed all ten real ICP executions (20 accepted
executions). An independent raw export readback of candidate ICP0 verified clean
committed source identity, complete worker capture, exact agent receipts, required
phase coverage, and five full-thinking model calls.

Judging did not complete. Configured Scrapingdog credentials returned authorization
errors. The Deepline-managed Firecrawl route reported
`THIRD_PARTY_DATA_ENRICHMENT_NOT_ENABLED`. The test adapter also lacked the scorer signal-cache control. A regression
reproduced the missing-method failure through the real worker socket; the adapter
now forwards that request and quota reads to the actual service. The run was stopped to avoid repeated failed charges.
The local database dump and exact referenced objects were preserved for resume.
The test database fixture was also missing current mainline publication, cache,
and locking migrations. It now applies 35 additional guarded migrations, including
the missing independent branches before migration538. Dated round-repair scripts
535 and542–546 are excluded from a fresh database. Independent checks verify five
function definitions and four indexes. The paid result above
predates that fixture correction and must not be presented as release proof.
No scores were fabricated or substituted. Some raw exports also contain explicit
privacy redactions and remain excluded by the strict training-data gate.

Confirmed review defects now have focused regressions:

- Terminal broker replay accepts and validates saved reasoning-route observations
  without another provider call or charge.
- Champion and dataset checks preserve approved full thinking alongside auxiliary
  summaries/opaque state, while keeping the distinct representations separate.
- Migration561 freezes a newly promoted compatible source and payer for a later
  round. Frozen existing rounds remain unchanged.
- Migration562 permits the six-dollar frozen catalog policy at the actual round
  commit boundary without changing historical four-dollar rounds.
- Migration563 permits coherent training-data ineligibility at the real
  publication SQL boundary. A full service/PostgreSQL regression publishes all
  measured scores, excludes the affected contender, and crowns an eligible peer.
  A second case stops one contender with no output: it retains no measured score,
  while the round publishes and the eligible peer wins.
  This regression uses controlled training outcomes and an isolated scorer; it is
  not a substitute for the incomplete live ten-ICP test.
- Source review rejects one-character evidence quotes. Ambiguous `unverified`
  results use the existing bounded retry path. A real SQL test exposed a missing
  retryable-error allowance in migration556; this is corrected. Its 16-test
  PostgreSQL suite proves persistence, backoff, a second claim, settlement,
  permanent-error rejection, and migration replay.
- Policy-off Chat normalization and hashes retain main's legacy behavior.

The focused protocol, worker/shim, reviewer, and real SQL regression suite passed 854 tests;
two existing declared-field cases were skipped. Additional worker frame and HTTP tests verify that only an execute lease with
the frozen policy can use full-thinking continuation fields. Score leases and
forged sandbox hints cannot enable them. A stale error-text assertion was aligned
with main's existing retryable provider-failure behavior; its charge and replay
assertions remain unchanged. The updated quota suite passed 27 tests; the
provider-provenance suite passed 24 tests. Both changes correct stale test
expectations for existing mainline behavior, without changing production scoring.

After merging main's eight-point promotion change, 67 focused integration tests
passed on Python3.11. The E2E schema fixture also includes main's separate
open-round promotion-margin migration; all four publication/schema tests passed
again against that final fixture.

This test's authorized limits are $10 OpenRouter and $5 combined search/scraping.
At the stopped checkpoint, charges plus unresolved reservations are $1.029009 and
$3.059750 respectively. These are conservative ledger amounts, not final invoices.
No production migration, publication, restart, merge, or deployment was performed.

Machine summary: [real sourcing E2E status](evidence/full-thinking/full-sourcing-e2e-status-20261024.json).

## Subsequent local regression checks — 2026-10-24

A real capture readback found that public prose such as `Basic information` was
being redacted as an authorization value. The narrow correction preserves that
prose while still removing real Basic credentials, Bearer values, explicit
authorization headers, known secrets, and signed URLs. The new regression failed
before the correction; 166 capture/trajectory tests passed afterward. Historical
redacted artifacts were not rewritten or declared exact.

After main added the context-integrity review diagnostic, its guarded migration
and this PR's review migration targeted the same old function. The combined
sequence now applies main's context migration first, then the research extension.
The resulting exact function hash and retry behavior passed 51 PostgreSQL checks.
The final research suite passed 22 tests.

The stopped-contender regression now continues through the actual service Git
promotion function against a local bare repository. It checks the winner's source
on both branches, the database promotion commit, and adoption of that source and
funding owner by the next round. The prior frozen configuration and publication
remain unchanged. This uses controlled scoring and training outcomes; it is not
the paid provider trial or a production repository push.

Independent root checks on the combined code passed 113 publication, capture,
and review SQL tests, plus nine terminal-replay tests. The replay tests verify
that completed calls do not trigger another provider send or settlement.

The broad GitHub Actions run `40141397510` failed and reached its 60-minute limit.
It reported 226 failures in 92 files before stopping. Five representative failures
were reproduced unchanged on clean main `c86511fcb`: a stale checkpoint default,
an old cache-key expectation, and three old PostgreSQL fixtures missing migration
474's execution-attempt-count function. This does not classify the remaining
failures or establish a green broad CI run.

## Resumed scoring and export diagnosis — 2026-10-25

The preserved test completed real judging after provider access was restored.
All 20 executions have accepted, linked scores. The actual PostgreSQL publication
succeeded and both public ranking rows matched stored publication. A failed judge
attempt completed on its normal retry while other work continued. No score or
provider response was substituted.

Independent readback verified 20 raw exports, 20 readable views, all 100 execution
model receipt chains, and all 20 selected scorer output hashes. The original strict
exports were empty. Reasons included historical false Basic-prose redactions,
genuine privacy exclusions in fetched pages, unresolved old provider attempts,
and two inputs from an inconsistent old database/object checkpoint. The original
evidence was preserved; it was not rewritten to claim a complete trajectory.

Readback also exposed a separate parser bug: full-thinking Chat history can
contain `refusal: null` on an assistant tool-call turn. The outbound request
preserved that valid field, but the training exporter rejected it. The correction
admits only assistant null refusals in approved full-thinking history. Non-null
refusals and invalid roles remain rejected. A two-step regression failed before
the correction and produced two strict records afterward. All 95 focused export
tests passed. An offline rebuild of one unchanged, complete saved ICP now yields
five strict records instead of zero; this does not repair the other saved gaps.

Main's subsequent source-digest diagnostic also changes the review SQL function.
The compatible order is context562, source-digest563, then research556. The
combined migration and publication checks passed 74 tests, plus two digest/catalog
checks. Independent root verification on commit `7441b8b7a` passed 131 tests across
the current full E2E schema, actual publication and Git promotion, source review,
strict exports, and completed-call replay. No production migration was applied.

The fresh repeat uses the corrected committed source and the same cumulative
spend ledger. The user approved raising only the combined search/scraping limit
to $8; OpenRouter remains capped at $10. Genuine privacy exclusions remain
exclusions and are not relaxed to obtain a passing test.

## Fresh full workflow and final readback — 2026-10-25

The fresh paid test ran on clean commit `7441b8b7a` with the current migration
fixture. Both candidate and baseline completed all ten ICP executions and all
twenty linked real scores. Two execution attempts failed output validation and
one judge attempt failed malformed-response checks; all three recovered through
normal retries. None canceled the round. The actual PostgreSQL round published.
Both stored and public ranking entries have identical shared fields and the same
king decision. The private `cost_summary` is intentionally absent from the public
projection. Both participants were marked `training_data_incomplete`; their
measured scores remained published.

Independent readback, repeated by the owning agent, verified all twenty raw
exports, twenty readable JSON/Markdown views, 8,973 trace events, 1,633 context
references, 622 provider-call records, all 103 execution model acknowledgment
chains, and all twenty linked accepted scores. The checker reported no hash,
link, view, or ranking-field errors. The original strict datasets contain 19
candidate records (10 train, 9 eval) and 16 baseline records (5 train, 11 eval).
These are stored records, not proof of successful research: all twenty measured
scores are zero, and none of the seventeen judged companies qualified. This
bounded one-company research submission is not a validated competitive baseline.

The strict all-ICP acceptance criterion failed. Nine raw examples are complete;
eleven are incomplete. Nine have privacy exclusions, one of those also has an
unresolved Scrapingdog cost, and two have the failed-attempt phase bug described
below. Candidate ICP4 has a complete visible answer but no returned thinking
despite the verified full-thinking request, so its exclusion is correct. Baseline
ICP9 has inherited redaction markers in its model context and cannot be treated
as exact replay. Some privacy exclusions come only from related judge calls;
selected model thinking is still saved, but the current whole-trajectory policy
excludes that example. No privacy or eligibility check was relaxed to pass.

The fresh evidence exposed a second retry interaction. An earlier failed
execution stopped before admission/final snapshots, while its successful retry
stored every required phase. Requiring successful completion phases from that
failed attempt incorrectly marked the whole export incomplete. Commit
`ba5e28779` requires those phases from the selected accepted execution and retains
all prior-attempt receipt, observation, source, privacy, and cost checks. Its
regression failed before the fix; 183 focused tests passed afterward. Independent
root checks passed 72 export/eligibility/thinking tests. Offline checks against
the untouched candidate ICP0 and baseline ICP4 worker snapshots verify the
correction. Missing selected-run phases and tampered prior receipts still fail.
The earlier `b6cc4212a` fix also prevents strict model targets from selecting
provider calls from a superseded failed attempt.

The original exports and manifests remain unchanged. The optional final full
PostgreSQL backup failed because the private test wrapper used `pg_dump15`
against PostgreSQL17. The local database was then removed. Raw exports, readable
views, the exact referenced object store, the private ledger, and the independent
publication snapshot were preserved, but **no corrected authoritative raw export
was produced after the phase fix**. The wrapper now selects the backup tool from
the test server's installation and validates the archive. An unpaid disposable
database reproduced the old-client version failure and passed the matching-client
backup/readback check. This prevention does not recover the missing old backup.

Final cumulative charges plus unresolved reservations are **$2.751869 OpenRouter**
and **$5.586500 combined search/scraping**, within the authorized $10 and $8 caps.
There was no budget reset. These ledger bounds are not final provider invoices.
No further paid calls, production writes, merge to main, or deployment occurred.

The prior broad CI run reached about 35% of the suite before cancellation, with
226 failures and 67 errors. All 229 reached cases in PR-changed test files passed.
Four more failure categories were independently reproduced on both the isolated
base and PR checkout: company diagnostics, publication judgment binding, cost
eligibility test mocks, and a saved scorer case. Relevant files are unchanged on
current main. The remaining errors and unrun portion are not classified; this is
not a green full CI result.

**Activation remains off and this PR remains a draft.** Publication and capture
readback are verified. A complete ten-ICP strict training pass, a competitive
compatible baseline, and green required checks are still outstanding.

The E2E verifier also now checks complete strict training coverage for **both**
candidate and baseline after saving all original artifacts. Previously the final
assertions applied only to the candidate. Commit `fc7462fa4` shares the ten-ICP,
linked-score, and strict-record checks across both roles. Seven acceptance tests
passed independently, including a valid candidate paired with an incomplete
baseline. This test correction did not trigger another paid run or change the
results above.


## October 25 correction and checkpoint readback

The new frozen training coverage default is **80%**, using integer ceiling.
Existing rounds without this field keep 100%; no historical configuration was
rewritten. Complete ICPs remain subject to the existing strict raw/receipt/route
checks, and incomplete ICPs stay outside the strict dataset. Confirmed monetary
costs still control admission; the PR no longer adds pending or uncertain holds.
The PR migrations now occupy 564–568, after main's 563.

Current reproducible checks:

- 156 focused coverage, frozen configuration, strict export, migration,
  publication, promotion, and next-baseline tests passed.
- A further 98 focused cap, coverage/publication, and exact-source fallback
  tests passed. Counts overlap earlier tests; they are not a full CI pass.
- The actual PostgreSQL coverage test uses synthetic sealed evidence and the
  normal service publication path: 8/10 qualifies; 7/10 does not; all measured
  scores publish; an eligible peer wins and is adopted as the next baseline.
- Three reported PR-only regressions were reproduced before repair: second-turn
  summary diagnostic HTTP400, exact replay timing fields, and telemetry vocabulary.

The private saved paid checkpoint `attempt-09-resume-01` has SHA256
`62490a781439ba244e328e82b9dbfc4aa616c2bd6ce73fac861aeadb109fd018`.
A fresh PG17 restore read 20 accepted research executions and 9 accepted scores.
All 20 raw exports and readable views load. Independent comparison of 148 selected
provider calls, including 84 model calls, found no selected research changes.
This checkpoint remains at `stage1_scoring`, so final cost and judgment bindings
are unavailable and strict eligibility has not been established for either role.
The threshold was not rewritten to turn this incomplete run into a passing test.

The repeated scoring failure was an unbound scraper404 for a citation. A narrow
fix sends that typed transport failure through the existing bounded exact-URL
fallback. It does not treat the proxy404 as proof that the page is absent. Only
an authenticated, exact per-URL crawl receipt can establish absence. Exa documents
per-URL statuses separately from API success in its [Contents reference](https://exa.ai/docs/reference/get-contents).
The live retry on this fix remains required; the offline controls do not prove
provider recovery or a complete sourcing run.

The next paid retry confirmed that Exa uses `CRAWL_NOT_FOUND` with integer HTTP404
for this missing page. The parser had accepted only `CRAWL_HTTP_404`. The narrow
correction admits the observed tag only with the same exact URL, empty results,
single error status, and successful API response checks. The positive captured
shape and finality tests failed before the correction. All 100 focused tests
passed after it; the owning agent separately passed the 69 source-absence cases.
An absent source cannot produce a positive qualification result from this check.

That retry's final private checkpoint was interrupted during readable export.
Its partial evidence remains preserved, but its final database and object
snapshot are unavailable. The next run therefore starts from the last complete
`attempt-09-resume-01` checkpoint, not a reconstructed database. All spending
remains in the cumulative ledger. The private runner now saves the database and
object checkpoint before slow exports and retains its work directory on failure.
An injected export interruption preserved a valid checkpoint in an unpaid test.


## October 25 completed paid run and derived 80% verification

`attempt-09-resume-04` completed all 20 accepted research executions and 20
accepted scoring runs, then published the local PostgreSQL round. Its preserved
final dump has SHA256
`edb2afab19f95fe05e130fefa8fa8f2d7cb9ce006befdb8ce967818d1854e617`;
758 objects were checkpointed before readable export. Both entries returned one
qualified company and averaged 8.5. This demonstrates the workflow, not a
competitive-quality baseline.

The frozen exporter (`de5da476c`) produced 20 raw exports and 20 readable views,
32 candidate and 37 baseline strict records. Independent readback verified
84/84 selected model acknowledgements, 7,581 readable events and 1,468 context
values. All 20 selected research records and 148 selected provider calls matched
the prior checkpoint. Public and stored rankings match after the public API's
intentional private-cost projection. The original round retains its 100% rule
and its original published ineligibility; no historic decision was changed.

The subsequent offline re-export isolates failed-judge evidence from selected
research evidence. It records per-run gap attribution while retaining all raw
aggregate incompleteness. A selected execution must still pass source, input,
output, accepted-score, route, hash, worker receipt and acknowledgement checks.
Judge-only request redactions no longer discard exact research. A documented
selected scrape-to-model privacy cascade consumes an incomplete ICP at 80%,
but remains excluded from strict training.

On the unchanged saved database and provider objects, this correction produced
35 candidate and 44 baseline strict records. The separate 80% assessment passed
with 9/10 candidate and 10/10 baseline ICPs. All 20 raw completeness results
remained unchanged. These are derived offline exports; they are not a new paid
run or a retroactive publication. No API call was required for the re-export.

The cumulative test ledger reserves about $3.751 OpenRouter and $7.462 for
search/scraping, including unresolved reservations. The paid run finished within
the original $10/$8 caps. The later approved $8/$10 reallocation, still $18
total, was not needed for this run.

The E2E runner now requires both stored capture gates as well as strict record
coverage. A sufficient record count cannot hide invalid source or receipt proof.
Checkpoint tests also verify matching PostgreSQL dump tools and preserve the
snapshot before an interrupted readable export.
