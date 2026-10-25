# Draft PR verification — 2026-10-23

## Provider options for every approved model, October 25

The draft registry now contains **25 model/API/provider routes**, up from seven,
across the same six models. The eighteen additions have independent live proof:
fifteen Chat routes and three DeepSeek Responses routes. Every model has more
options. Mistral's addition is an EU endpoint from the same provider company.
The original seven rows, their default order, output caps and unset baseline
are unchanged. No policy was activated, merged or deployed.

Miners can select a listed alternative with ordinary `provider.only` on each
request. The existing default remains when that field is absent. Selection is
bound to the captured request, its hash, the approved row and the outbound pin.
ZDR, no fallback and price ceilings remain host-controlled. The README lists
the exact choices and settings in six rows. The bundled Codex helper retains
its existing DeepSeek/Fireworks default; it has no provider-selector parameter.

All **25 routes passed fresh three-turn tests on source `4887173a7`** using the
complete expanded registry through the real Arena broker, local worker socket,
and disposable PostgreSQL. The seven original routes omitted the selector;
all eighteen new routes selected it explicitly. Each task required two linked
tool calls, processing conflicting evidence, and a strict final JSON response.
The first GPT-OSS 20B/DeepInfra attempt received HTTP 429. One fresh test on the
same route passed; the failure and its uncertain charge remain recorded.

Final successful evidence: **75 model calls**, **66 full-thinking calls**, nine
explicit zero-token calls, **11,485 reported reasoning tokens**, and zero
missing tokens or unknown usage. All 75 strict input/output pairs passed.
Independent offline checks covered 1,886 readable events, 207 context values,
50 tool calls and Markdown views. The owner independently compared all 75 wire
responses with SQL storage and the exported/readable representation. Nine
Responses exports omit an empty `summary: []` placeholder; raw storage retains
it exactly. No other response value or nonempty reasoning field was dropped.
There were **zero client callbacks** and no extra miner logging requirements.

The [final route matrix](evidence/full-thinking/expanded-whitelist-final-proof-20261025.json)
records source and artifact hashes, failed and interrupted attempts, catalog
exclusions and budget accounting. Separate
[Chat](evidence/full-thinking/alternate-provider-chat-proof-20261025.json) and
[Responses](evidence/full-thinking/alternate-provider-responses-proof-20261025.json)
evidence supports each new registry row. Blocked, failed, interrupted and
untested routes were not added. Account blocks were not bypassed.

The integrated capture/export/route regression passed **611 tests**. The
operations, broker, proxy, shim, rollback and worker regression passed **1,182
tests**, with two existing declared-field skips. Independent review reproduced
and closed a saved-provider-selection mismatch before these final checks.
Offline replay preserved all **79 prior scored targets** (35 candidate and 44
baseline), 85 native TYCHE targets plus one verified cap/EOF omission, and three
earlier real continuation targets. Source and original artifact hashes stayed
unchanged. Compilation, README/registry alignment and `git diff --check` passed.

This work shares the existing $5 OpenRouter allowance. Cumulative reported
charges are **$0.0337479570**, with **$1.195942** retained for unpriced attempts,
or **$1.2296899570 conservatively accounted for**. No paid search/scraping was
used. These short tasks prove route selection, continuation and capture
compatibility. They do not establish lead quality, load reliability, maximum
context/output behavior, a fresh complete scored workflow, or the best coverage
threshold. The diagnostic runs correctly do not become scored training records
without genuine scoring labels. Remote CI remains a separate required check.

## DeepInfra follow-up and README trim, October 25

The user authorized enabling DeepInfra on the test account. Both GPT-OSS routes
then passed three calls each on `aaf257a3c`, with their unchanged 8,192-token
allowance, ZDR and exact provider route. All **seven routes now have fresh
passes**. The [combined evidence](evidence/full-thinking/final-whitelist-live-review-deepinfra-enabled-20261025.json)
preserves the two earlier blocked attempts and supersedes their availability
status below. The owner independently matched all six new wire responses to SQL
terminal responses and their full reasoning text to the readable trace.

Combined results: 21 successful calls, 19 full-thinking turns, two explicit
zero-token turns, 2,828 reported reasoning tokens and no missing reported tokens.
There were no client callbacks. Reported spend is **$0.0039543468**, plus the
unchanged **$0.011323** reserved for the earlier unpriced failures, below $5.
These remain synthetic compatibility checks, not a fresh complete scoring run.
The source Python, SQL and registry were unchanged. No merge or deployment occurred.

README text was reduced from 3,246 to 1,518 words, preserving setup, routes,
provider settings, examples and public rules. Internal review and recovery
details were removed. Local links, JSON examples, registry alignment and absence
of em dashes were checked. The original test evidence remains unchanged.

## Final handoff review — October 25

The final source checkpoint is `e0509ee82`, including main `93f15930e` and
migration 574. This check found and fixed one false rejection: a recovered call
can retain the same provider response in both its uncertain and settled ledger
entries. The shared route checker now accepts repeated identical, validated
route evidence for that exact response. Missing, conflicting, malformed or
unrelated evidence still fails. No new receipt protocol or model-specific bypass
was added. Two actual-decoder recovery cases failed before the fix; twelve
regressions and independent negative checks passed afterward.

The complete E2E fixture now includes main's required review-guard read. Its
NOSUPERUSER test uses a separate migration role because PostgreSQL 18 forbids
demoting the bootstrap superuser. This changes test setup, not production SQL.
The final integrated checks passed:

- **1,035 passed, 14 skipped** in the exact early CI trace-integrity and reviewer
  repair command. The skips require unavailable frozen release source fixtures.
- **20 passed** in the combined migration-tail, review-guard and ACL checks,
  including numeric/production migration order and actual local SQL publication.
  Separate reviewed publication/promotion tests preserve measured scores for an
  ineligible entrant and adopt an eligible champion as the next baseline.
- Python compilation, workflow parsing, README/registry alignment, guide byte
  equality and `git diff --check` passed.

The new [seven-route live check](evidence/full-thinking/final-whitelist-live-review-20261025.json)
made **17 real provider attempts** through the Arena broker, local worker socket,
disposable PostgreSQL and raw/readable storage. Five exact routes completed two
tool continuations and strict final JSON: Mistral, DeepSeek Chat, DeepSeek
Responses, Reflection and Gemma. Their fifteen replies contained thirteen
full-thinking turns and two explicit zero-token turns. All **1,186 reported
reasoning tokens** had returned text; unknown usage and missing-token counts
were zero. There were **zero client callbacks**. The short tasks do not establish
the best missing-token threshold or exhaust the configured output caps.

Both GPT-OSS routes returned OpenRouter 404, `Filter by Ignored Providers`, on
the gateway test account. They are **not fresh passes**. No provider fallback or
account setting change was used. Earlier successful GPT-OSS evidence still
replays with its original frozen registry and 8,192-token allowance: three 120b
responses and six 20b responses. Those replays do not prove current availability.

Final-source offline verification checked all 17 stored wire/host results,
15 strict input/output pairs, 13 readable full-thinking pairs, 411 events,
54 context values and seven Markdown views with no errors. The diagnostic runs
correctly produce no complete ICP training records because they lack real
scoring labels. Separately, all **79 prior scored training targets** remained
unchanged (35 candidate, 44 baseline); the native TYCHE evidence retained 85
ordinary targets plus one verified cap/EOF omission, and all three earlier paid
continuation replies passed. These replays attempted no network calls and
preserved source and original artifact hashes.

This task's new $5 OpenRouter allowance is separate from earlier experiments.
Reported charges are **$0.0035859388**, with **$0.011323** conservatively reserved
for the two unpriced blocked attempts. A reported subtotal of zero for those
attempts is not proof that they were free. No paid search/scraping was used.

The README now gives the concise whitelist and settings. Miners retain their
harnesses and custom tools; existing broker transport is required, custom logs
and acknowledgments are not. Stored provider-visible evidence can be checked
without an extra LLM review. This does not prove client consumption, all local
agent activity, the faithfulness of written reasoning, or training value.
The PR remains draft and inactive. CI, a compatible baseline, and the existing
activation requirements remain outstanding; these tests are not production
deployment verification or a claim that no possible bugs remain.

## Final October 25 capture verification

The integrated regression at `bfdb5989b` passed **4,415 tests**, with **17
skipped**, across 85 files. Actual Codex 0.154.0 and local PostgreSQL checks were
enabled. The skips require Linux root/runsc or exact frozen release source
fixtures, except two existing declared-field cases. This is not full CI or a
production deployment test.

Three additional failures were compared against a clean checkout of main
`a0e2f9870` and reproduced there: two telemetry cases expect no
`arena.reconcile_deepline_cost` span, and one validator logging fixture supplies
an invalid lease. They remain disclosed failures, not passes or PR fixes.

The final narrow history fix preserves malformed **prior** tool arguments as
exact input context only in a closed, declared local-tool exchange under the
new frozen policy. It never repairs arguments or accepts them as a new target.
Eight positive cases failed before the fix; the new history tests cover both
Chat and Responses, pairing, identity, declaration, type, raw-hash and historical
compatibility failures. Independent review also exercised twelve combined
negative controls. The final integrated history, target, export, eligibility,
readiness, coverage, and continuation regression passed **755 tests**, with no
skips, at `b940251e1`. Python compilation and `git diff --check` also passed.

Three original rejected TYCHE continuations were sent to the real provider
through Arena's bridge, worker and broker. All three returned HTTP 200/completed,
with unchanged history and exact saved wire, PostgreSQL and readable evidence.
There were three host writes and no client callbacks. The replies reported
1,693 reasoning tokens. The test cost was $0.0204798636. The redacted top-level
reasoning setting was restored from uniform earlier requests; this is an
exact-history projection, not an unmodified original frame. Newly returned tools
were not executed. The parser fix subsequently admits all three valid replies
without changing the saved raw export.

Total measured spend for this $5 experiment is $0.2528262620, with $2.142311
still conservatively reserved for unpriced earlier attempts. No additional paid
search/scraping was used. The 1% ceiling remains provisional: no missing text was
observed in these reported reasoning tokens, which does not establish the best
population threshold. The 80% complete-ICP floor remains unchanged.

The final restored paid-data check at `bfdb5989b` used a disposable PostgreSQL
17 database with PR migrations 565–570 and main 571–573. Both the historical
80% case and explicit 1%/80% case reached actual SQL publication. The candidate
had **9/10 complete ICPs and 35 strict records**; the baseline had **10/10 and 44**.
Both retained their measured **8.5 scores**. The tie correctly produced no king;
this replay does not demonstrate promotion. Separate promotion regressions are
part of the integrated test suite.

The new-policy strict checks passed with actual evidence and with optional
client callbacks removed in memory. No replacement host evidence was invented.
All **60 raw/readable/Markdown comparisons**, 22,743 nonderived event values and
4,186 context values matched their saved source paths and hashes, with zero
trace errors. These counts span the repeated variants, not unique provider calls.
The check took 1,055.89 seconds and made no external network requests. Original
source, dump and object-tree fingerprints stayed unchanged.

The original whole-run completeness remains four candidate and five baseline
ICPs. Selected research completeness is nine and ten. This distinction is
intentional: the new policy verifies provider-visible research pairs and does
not claim complete local-agent observability or client consumption. The saved
100% source run and private artifacts remain unchanged. This is a replay of a
real paid sourcing/scoring run under an explicit local policy projection, not a
fresh paid 80% run or a deployed S3 verification.

After the final parser change at `b940251e1`, a separate network-disabled
replay verified all **79 saved scored targets** unchanged: input, target, format,
full-thinking segments, delivery digest, and candidate identity. The native TYCHE
export retained **85 ordinary targets and one verified cap/EOF omission**; all
three paid continuation replies became valid targets. Python-source and input
artifact hashes stayed unchanged. SQL publication was not repeated after this
isolated parser change; its preceding `bfdb5989b` verification is stated above.

## October 25 Responses route and current policy

The seventh reviewed route is DeepSeek through Fireworks on Responses, with a
32,768-token allowance and full text in `output.content.reasoning_text`. The
[public synthetic manifest](evidence/full-thinking/deepseek-fireworks-responses-20261025.json)
contains only the three-turn synthetic test: two function continuations, strict
negative JSON qualification, exact provider wire bodies, PostgreSQL readback,
and local worker/SDK receipts. All three saved calls pass the strict exact
next-response target validator. Empty summary-array normalization is explicit;
the original wire bodies retain those arrays.

The latest private TYCHE check uses unchanged source
`dcaeb4bacbb864bc10575fe1758cf487a3466324` and its exact Codex **0.154.0**
binary, downloaded with owner approval and verified against the npm SHA-512
digest. The fresh run used **no client read acknowledgments or phase callbacks**.
Arena saved 88 calls: 86 real model calls and two recorded-page tool calls. All
88 have exact host-write evidence. The model replies contain 79 full-thinking
turns and seven explicitly zero-reasoning-token turns: **44,373 reasoning
tokens**, **206,141 thinking characters**, and zero reported reasoning tokens
without returned text. Raw wire, PostgreSQL readback, and all 6,269 readable
source-linked events matched. Source hashes stayed unchanged.

This proves capture through the existing host path, not client consumption or
complete local-agent observability. It exposed three rejected continuation
requests: a 32,414-character thinking string and tool arguments of 51,314 and
82,130 characters exceeded the old per-field validation limit. The fix permits
full-policy provider history within the unchanged overall operation/frame caps;
it does not clip strings or expand user/system prompts. Historical policy-off
limits and accepted request hashes remain unchanged.

One provider reply also hit exactly 32,768 output tokens with malformed JSON
arguments, although the provider labeled it completed. This is unusable output;
it must not become a valid training action or erase independently valid targets.
The raw bytes remain available for diagnosis.

**The native workflow is not a complete strict training or scoring run.** It
ended without an approved company after repeated worker non-progress. It ran on
the native host rather than gVisor, selected DeepSeek with TYCHE's documented
non-OpenAI plain-function mode, disabled hosted web, used recorded public pages
with zero fixture billing, and limited the task to one company. The disposable
historical capture SQL fixture used an explicit 1%/80% policy projection and a
200-call quota; worker hold/rate-limit retries were disabled for diagnosis.
These substitutions do not establish live lead quality or activation readiness.
No private prompts or provider response bodies are committed.

The preceding exact-runtime run retained 33 completed model responses with
10,547 reasoning tokens and zero missing text, plus an actual 429 attempt whose
upstream and host-delivered error bytes were verified. Its model callbacks were
present; direct TYCHE research-tool callbacks were absent. The new policy removes
that client prerequisite. Earlier installed-runtime checks retained ten full
responses with 1,759 reasoning tokens. Their unrecoverable missing refusal
request remains a gap, not reconstructed evidence.

The current experiment's provider spend and outstanding reservations are recorded
in the private shared $5 budget ledger. No new paid search/scraping is used.
Test reservations do not change confirmed-cost-only competition billing.

The provisional missing-token ceiling is **1% per ICP**, using reported tokens
from completed, untruncated, verified calls. It is configurable and frozen; it
is not a measured optimum. Explicit zero-token turns are distinct from unknown
usage. Proven failures and truncations lose only their own training targets;
ICPs with no surviving full target or above the ceiling are incomplete. The
**80% ICP floor** remains. Explicit frozen `0` selects the new zero-tolerance
rules; an absent setting retains historical behavior. Current release checks
must include PR migrations **565–570** and main's **571–573** in numeric order;
dated verification entries below retain their original checkpoint context.

## October 25 output-envelope supplement

The current registry raises only Mistral, DeepSeek through Fireworks, Reflection,
and Gemma to 32,768 output tokens. The supplemental
[synthetic capture manifest](evidence/full-thinking/chat-output-cap-32768-20261025.json)
binds each real request, provider response and wire body to disposable PostgreSQL
ledger readback and the local worker socket/SDK read acknowledgement. Replaying
these captures through the current broker preserves the requests, full readable
text, two tool continuations, and final negative JSON qualification from
conflicting evidence. Mistral's zero-token tool turns remain zero-thinking turns.

These were short calls with a 32,768-token allowance, not requests that exhausted
the allowance or full ICP quality tests. The manifest preserves the original
reports' `route_probe_passed: false` flag and states the narrower cap, continuation
and capture checks it proves. It does not claim production activation or a new
Responses route. GPT-OSS 120b and 20b remain at 8,192: their new cap probes hit the
account's ignored DeepInfra route. Original evidence files stay byte-identical;
historical replay uses their original frozen 8,192-token registry.


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
The final PR migrations occupy 565–570, after main's 564. Earlier test checkpoints below predate that final numbering.

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


## Final integration checks on current main

The final code checkpoint is `77f6ffe61`, including main `914ee3c32` and its
baseline-deadline migration 564. PR migrations are 565–570 in numeric order.
Internal SQL guard markers remain unchanged because they identify the exact
patch contents. A regression check detects duplicate numbers in the new tail;
it reproduced two migration-564 files before renumbering.

Checks on the merged checkout passed:

- 254 capture/export/eligibility, E2E acceptance/checkpoint, and real PostgreSQL
  coverage/publication/schema checks.
- 40 PostgreSQL migration checks for the $6 cap, review results, champion
  adoption, catalog binding, and publication eligibility.
- 92 protocol, completed-call replay, saved-provider probe, frozen-round,
  coverage, source-review, and telemetry checks.
- The separate migration-number regression, Python compilation and
  `git diff --check` passed.

These checks use local PostgreSQL and saved provider artifacts. They do not
activate the policy or change production. The prior GitHub Actions run
40161447073 reached its 60-minute timeout; there is still no complete green CI
run. Earlier main/PR comparisons found common suite failures, as documented
above. This PR remains a draft, and no merge or deployment is authorized.


The final pinned offline readback used clean exporter commit
`77f6ffe61686250a339781243e6de06c8210cce7`, the unchanged resume-04 dump and all
758 saved objects. It made zero provider calls. It confirmed 35 candidate and
44 baseline strict records; 9/10 and 10/10 complete selected research ICPs;
20 score output hashes and exact score input bindings; 148 unchanged selected
provider calls; 84/84 model acknowledgements; and 7,581 readable trace events.
All 20 raw aggregate completeness values remain unchanged. Both separate 80%
capture assessments pass, while the saved 100% publication remains unchanged.

One old private scorer checker assumed equal breakdown and supplied-judgment
counts. The actual accepted zero-score intrinsic-skip case has one breakdown,
zero supplied company judgments and one verifier receipt. The production output
validator accepts it, and independent checks verified its exact input/output
hashes. The final private evidence checker reports no comparison errors. Its
legacy `ready_for_final_review=false` still requires all ten candidate ICPs;
it is not the new frozen 80% gate. No checker flag was changed to claim that the
original 100% run passed. The final 80% evidence is an offline assessment plus
separate real-PostgreSQL publication tests, not a fresh paid 80% round.


## October 25 follow-up review fixes

The five reported defects were reproduced and corrected in code checkpoint
`668187f20`, based on main `914ee3c32`:

- Preserve the exact existing source-access repair blocks that frozen-grader
  patch scripts match. Keep authenticated, exact-URL source absence handling in
  adjacent blocks. Those initial controls did not cover an outer Exa API
  404/410 response. The later review below records that missed defect and fix.
- Treat an ICP whose verified thinking replies are all truncated as one
  incomplete ICP under the 80% floor. Wrong routes, missing receipts and changed
  source evidence remain hard failures. Truncated data stays out of strict export.
- Treat only a missing export artifact as a soft coverage gap. Hash, membership
  and unexpected export errors fail the entrant's evidence check. A real
  PostgreSQL publication test proves that this does not cancel the round or stop
  an eligible peer from winning and becoming the next baseline.
- Restore redaction of unpadded Basic credentials in free text. Synthetic test
  credentials reproduced the regression; this is not evidence of an actual
  credential exposure. Ordinary prose and existing header redaction still pass.
- Retain the coverage failure reason and complete/incomplete ICP counts in the
  public dashboard response. Private fields remain excluded. This checks the API
  projection, not a browser rendering.

The combined regression command passed **605 tests with 14 skipped**. Five real
PostgreSQL publication cases also passed. The new failing cases were demonstrated
before their corresponding fixes; preserved behavior is covered by existing
capture, source-access, protocol and publication tests. Python compilation,
`git diff --check`, and guide byte-parity checks passed. The critical combined
command now runs early in existing CI so a later full-suite timeout cannot hide
its result. No new production dependency was added.

The 14 skipped checks require actual frozen-release source fixtures. The three
fixture environment variables are unset, and the exact fixtures were not found
in bounded repository and known test-artifact searches. They remain unverified.
A wider reviewer seam run had eight failures; the same eight failed on clean
main `914ee3c32`. They are in the finality and attribute-source repair suites and are not
fixed by this PR. Full CI is still not established as green.

Independent offline readback pinned clean exporter `668187f20`, restored the
unchanged resume-04 dump and used its 758 saved objects. It made zero provider
calls. Both the actual service eligibility method and the direct capture check
pass at an isolated 80% setting: candidate 9/10, baseline 10/10. Strict export
still contains 35 candidate and 44 baseline records. All 20 accepted score
inputs and outputs remain bound, 84/84 model acknowledgements are present, and
all 20 readable views contain the same 7,581 events. The private checker reports
no integrity errors. Compared with the prior export, only recomputation
timestamps and their derived hashes changed. No new redaction or selected
provider-evidence change was found.

The original 100% publication and ranking remain unchanged. This is offline
verification of stored real-provider evidence plus separate PostgreSQL failure
and promotion tests, not a new paid 80% sourcing round. No new paid calls,
production change, merge or deployment occurred. The PR remains a draft.

## October 25 final review corrections

Implementation checkpoint `79d0a61fb` includes main `15ba0eda0`. The later
documentation/fixture commit adds main's migration 571 to the E2E harness tail;
the complete 565–571 tail passed in local PostgreSQL. No policy was activated.

Three further defects in review of `b21790adc` were reproduced and fixed:

- An ICP containing both verified truncated and empty thinking replies now
  consumes one incomplete ICP, like either condition alone. Wrong routes,
  unproved thinking, source changes and receipt failures remain hard failures.
- Outer Exa API 404/410 responses now use `contents_http_*` errors. The real
  broker, investigator and scorer retain retryable provider failure instead of
  finalizing the target page as absent. Exact target-URL absence controls and
  historical frozen-source repair blocks remain covered.
- Publication training reads use the actual local/S3 optional-read interface
  and retry a recognized transient error once. If the canonical accepted output
  was already read successfully in the same publication attempt, its exact raw
  bytes can supply a later failed training read. The normal exporter still
  checks source, input, output hash, receipts, routes, privacy and costs. The
  availability failure consumes one incomplete ICP; it does not repair stored
  evidence or admit incomplete data into strict training. Without that verified
  earlier read, or for other missing objects, access denial, malformed data or
  integrity failures, this exception does not apply.

The storage change reuses existing publication reads and the existing export
validator. An initial duplicate validation helper was removed after independent
failure testing exposed gaps. The public reason allowlist also includes
`accepted_output_unavailable`; its regression failed before that addition.

### Production-path checks and substitutions

The unchanged real paid-run dump and all 758 saved objects were restored into
disposable PostgreSQL. Actual `ArenaService.publish`, `training_reader.export_run`
and SQL publication readback ran together. No export result was mocked. The
disposable round setting was projected to 80%; fault wrappers affected only
later reads after the first verified scoring read.

| Candidate case | Complete ICPs | Eligible | Round published | Saved score |
| --- | ---: | --- | --- | ---: |
| Healthy stored evidence | 9/10 | Yes | Yes | 8.5 |
| One later output read missing | 8/10 | Yes | Yes | 8.5 |
| One later output read repeatedly times out | 8/10 | Yes | Yes | 8.5 |
| Two later output reads missing | 7/10 | No | Yes | 8.5 |

The baseline remained 10/10 complete, eligible and scored 8.5 in each case.
All 40 accepted research/scoring run outcomes stayed unchanged. A second restore
preserved database owners, applied exact guarded migrations 570 and 571, and
repeated healthy and later-missing publication successfully. Original dump and
object fingerprints did not change. These tied real scores did not produce a
new champion; separate PostgreSQL fixtures prove eligible-peer promotion to a
local Git repository and next-round baseline adoption.

The final raw/readable re-export checked 20 raw trajectories and 20 JSON/Markdown
views, 84/84 model acknowledgements, 7,581 source-linked events and 1,488 context
values. Both strict manifests validated: 35 candidate records and 44 baseline
records. All 148 selected provider calls and all raw completeness flags match
the prior saved export. Only observation timestamps and derived export hashes
changed. The original 100% publication remains unchanged. The old private
checker readiness flag requires all ten candidate ICPs and stays false; it is
not the 80% eligibility rule. The final view/hash check has no errors.

### Regression evidence and remaining limits

- The combined existing and new regression command: **952 passed, 19 failed,
  14 skipped**. All 19 failures reproduce on clean main `15ba0eda0`, with the
  same test IDs in the dashboard suite. None is unique to this PR. The 14 skips
  still require unavailable exact historical release fixtures.
- Final real PostgreSQL publication/coverage/promotion tests with the full
  565–571 tail: **14 passed**. Main's 571 migration suite also passed its
  **89 checks**; these counts describe separate commands, not a unique total.
- Independent review ran 12 actual-artifact storage cases, covering missing and
  timeout reads, truncation and mixed thinking gaps. Separate sealed fixtures
  covered privacy and unsettled-cost combinations. Bad hashes, source, receipts,
  access denial and missing prior proof stayed ineligible. Existing policy-off
  protocol, replay, redaction, export, frozen
  repair and dashboard checks were included in the focused regression run.
- Python compilation, workflow YAML parsing, `git diff --check`, and active
  guide byte equality passed. CI remains required; the prior remote run was
  cancelled and does not establish a green build.

These are local executions of production code with saved real provider evidence,
local PostgreSQL/object storage, test service configuration and injected faults.
They are not a new live 80% sourcing run, a deployed S3 test, or a claim that all
possible paths are bug-free. No new paid calls, production writes, PR merge or
production deployment occurred. The independent shadow/enforce review remains outside this
draft's implemented behavior, as stated in the policy document.

The active AGENTS.md/CLAUDE.md instructions were shortened from 156,257 to 6,372
bytes. Exact historical instructions remain in the linked archive. The concise
new rule requires root-cause fixes, tracing every affected actual code path,
production-path and combined-failure tests, and explicit verification limits.

## October 25 hosted search continuation and ordered capture

Approved Responses routes now accept their returned hosted search items in
later requests. The native Codex adapter changes only the provider-specific
search type name in the transport copy. Raw provider responses remain exact.
Readable views preserve output order, including intermediate messages,
reasoning, searches, exposed results and citations. Hosted export records are
marked as provider-executed trajectories; internal prompts and missing search
results are not invented.

[Focused live evidence](evidence/full-thinking/hosted-search-capture-20261025.json)
records fresh Fireworks and Together hosted-search conversations, followed by a
local tool, its result and a final answer. Together returned valid JSON. Fireworks
returned Markdown despite the strict JSON request; this is retained as a failed
format check. Both captured all exposed fields. Fresh GPT-OSS 120B and Mistral
Chat conversations passed two tool turns and final JSON. These runs used actual
OpenRouter, the worker socket and disposable PostgreSQL. No client callbacks
were sent. They are compatibility tests, not a new production quality benchmark.

Raw SQL response bytes match the independently saved wire bodies. The sanitized
Responses projection omits only empty summary arrays. Source review, the 80%
ICP floor, missing-token policy, privacy rules, cost rules and live scorer image
are unchanged. Provider-internal activity and the quality of generated reasoning
remain outside what capture alone can prove.
