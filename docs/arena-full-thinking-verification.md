# Draft PR verification — 2026-10-23

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

## Real ten-ICP sourcing test and review corrections — 2026-10-24

**This end-to-end test is not complete. The PR remains a draft and activation
remains off.** Earlier component and synthetic tests did not prove the actual
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
and locking migrations. It now applies 34 additional guarded migrations, including
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

This test's authorized limits are $10 OpenRouter and $5 combined search/scraping.
At the stopped checkpoint, charges plus unresolved reservations are $1.029009 and
$3.059750 respectively. These are conservative ledger amounts, not final invoices.
No production migration, publication, restart, merge, or deployment was performed.

Machine summary: [real sourcing E2E status](evidence/full-thinking/full-sourcing-e2e-status-20261024.json).
