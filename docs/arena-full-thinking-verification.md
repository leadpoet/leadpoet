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

The PostgreSQL checks use disposable local databases. They apply migration 551
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
