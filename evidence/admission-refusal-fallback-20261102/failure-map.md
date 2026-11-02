# Admission refusal fallback: downstream failure map

Target: after an explicit provider refusal, run one complete review on an approved
alternate model through the existing claim/settlement path. Never approve from
failure, repeat a completed identity, truncate source, or change historic scores.

| Failure/path | Required behavior | Independent proof |
| --- | --- | --- |
| Source archive/digest/security failure | No provider call; permanent source error | Existing pure/runtime source checks |
| Primary typed refusal | Settle/retain uncertain original charge; persist dedicated refusal code | Runtime typed response and real SQL ledger test |
| Generic HTTP 4xx or untyped provider prose | No alternate selection | Negative response controls |
| Concurrent stale primary workers | SQL refuses original route after refusal; one alternate reservation only | Two-connection PostgreSQL claims |
| Alternate preparation | Exact same full source, comparisons, research policy/schema and bindings | Wire request equality except approved route |
| Unknown full-context fit | Require fresh direct routing metadata, no transforms/compression and actual prompt-token fit | Large complete-source response controls |
| Alternate incomplete/malformed/refusal/unaccounted response | Error remains incomplete, no new paid claim | Runtime + SQL terminal retry cases |
| Claimed identity replay | No dispatch or second settlement | Existing and new idempotency controls |
| Worker crash before/after send | Old reservation becomes uncertain; alternate reservation prevents another alternate call | Expiry recovery + retry SQL control |
| Late/stale result | Existing claim ownership/deadline/status gates apply; no late pass | Preserved SQL controls |
| Cost unknown | Preserve original conservative uncertainty and accumulate all ledger heads | Real ledger amounts and totals |
| Restart guard, cutoff, closed round, baseline | Existing checks still apply to both routes | Preserved SQL path controls |
| Historical generic request_rejected | No new interpretation or data rewrite | Migration row snapshot check |
| Public retry status, training eligibility | Shared retry policy says explicit primary refusal may retry; failed alternate is terminal | Policy/dashboard controls |

Inspection base: dba3aa5f8. Migration599 will patch exact current function bodies,
retain owner/ACL/search_path metadata, and add no tables or columns. No provider,
production, credential-refresh or historical-state actions are part of this work.

## Implemented boundary and checks

The alternate is `openai/gpt-6.1-sol` on the pinned Azure ZDR route. It always
requires fresh direct router metadata, selected Azure and the exact model
identity, no hidden attempt or transformation, actual prompt tokens <= 922,000,
and input plus capped output within 1,050,000 tokens. Source, research criteria,
comparison contexts, response validation and the miner-owned credential remain
on the existing paths. Reservation reuses the broker's published conservative
Sol Azure regional/long-context/cache price ceiling.

Migration599 retains the existing RPC signatures and security metadata. It
allows exactly one alternate reservation after a dedicated primary-refusal
result. Concurrent/stale primary workers cannot reopen the refused route. An
expired alternate has terminal retry flags in both its result and uncertain
ledger entry. No historical row is rewritten. Ordinary primary transient retries
retain migration585's admission deadline and backoff; they have no numeric cap.

Local validation, Python3.11, no provider calls:

- 284 general source-review/runtime/policy/research/dashboard cases passed.
- 82 preserved diagnostic/restart-guard/deadline PostgreSQL cases passed.
- 21 focused alternate runtime cases passed, including source-preserving
  research+similarity context, primary unknown cost, thrown alternate transport
  error and terminal malformed/refusal/usage/routing failures.
- 23 focused real PostgreSQL cases passed. These include migration regression
  before/after proof, non-superuser migration, failed-upgrade rollback,
  identical reapply, unchanged rows/ledger/security, concurrency, crash recovery,
  one alternate claim, primary585 recovery, and zero dispatch at closed,
  deadline, cutoff and restart-guard gates.
- `git diff --check`, Python compilation and instruction-file equality passed.

The production provider route, exact historical recovery and deployed outputs
remain the coordinating task's release/verification responsibility.
