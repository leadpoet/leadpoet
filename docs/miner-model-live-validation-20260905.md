# Miner submission: live testnet validation

## Result

The real miner submission completed the full workflow on testnet 401:
CLI upload and admission → validator source execution → judging → saved scores
→ public result. The normal driver published the round. Restarting the isolated
API preserved the complete public JSON response, including both participants'
scores and outputs.

- Round: `arena-2026-09-05-e2e2`.
- Miner submission: `sub-d81a1fbe8e4763d2202349e9b64325cd`.
- Baseline submission: `baseline-2026-09-05-e2e2`.
- Registered testnet miner UID: 11.
- Active ICP set: `20260905`, all 20 ICPs. Private inputs are not committed.
- Runtime/scoring code: `5c9702929`. Result membership fix and restart readback:
  `6b3858759`.

## Evidence

| Check | Live result |
| --- | --- |
| Actual CLI | `scripts/lab_arena_miner.py submit-model` uploaded source and finalized with three supplied credentials; exit 0 and accepted submission ID. The interactive miner menu calls the same submission helper. |
| Execution | Real Python/PydanticAI source and installed binary wheels ran in fresh gVisor sandboxes. No canned company results or mock provider responses. |
| Complete score coverage | 20 saved per-ICP slots for each participant; each position occurs once. Published final scores equal the mean of those 20 slots. |
| Actual judging | All 35 scoring jobs for accepted outputs completed. Miner judging made both a Deepline call and an OpenRouter call. |
| Submitted-key billing | Miner execution: 370 OpenRouter and 333 Deepline settled calls. Miner judging: one settled call to each provider. Every one is recorded as `miner_key`; none uses organizer fallback. |
| Credentials | Two KMS ciphertexts only: OpenRouter and Deepline. Decrypted values match the submitted keys. No management-key storage slot. Public round/results responses contain none of the three supplied secret values. |
| Restart/recovery | Published round and both result documents are unchanged after restarting only the isolated API. Scores and outputs were not re-created or re-scored. |
| Result privacy | Results are private before publication. After publication, unknown and other-round submission IDs return 404. Public identity comes from the published participant list. |
| Safety | Shadow mode; rewards disabled; no reward activation or reward basis. No chain writes or production changes. |

Recorded miner upstream spend was $10.351283: $4.556152 OpenRouter execution,
$5.792 Deepline execution, and $0.003131 judging. This is not a per-attempt
combined budget claim. The configured OpenRouter caps and provider call limits
remain separate. The submitted test keys came from the operator-authorized
account; this test did not create a separate miner billing account.

## Model outcome, not a quality endorsement

The miner returned accepted output for 17 of 20 ICPs. Three ICPs exhausted their
allowed attempts and received zero under the existing policy. Its nine failed
attempts remain recorded. The baseline returned accepted output for 18 ICPs.

The miner returned Quix, Cider, Mastek, Teads, and Magnite across four ICPs, but
the judge rejected their fit evidence: identity mismatch, conflicting geography,
or stage mismatch. Empty outputs and exhausted attempts also scored zero.
Final scores: miner **0**, baseline **1.02**. No scores were inserted manually,
failures removed, model limits relaxed, or source tuned during the round.

The miner's OpenRouter company-fit judging path ran live. Because fit gates
rejected the returned companies, this run does not prove a successful downstream
intent-verification path or a positive-scoring miner. Those are not implied by
successful submission and publication.

## Defects fixed and retained failures

- S3 Object Lock required a signed upload checksum. The CLI now sends the
  transport header required by S3; model admission has no new digest contract.
- The trusted scorer image's startup shim intercepted submitted SDK calls.
  Submitted source now starts with Python `-I -u -B`; the scorer is unchanged.
- Completion now retries the specific temporary `accounting_open` response
  within a bounded window. Other rejections remain terminal.
- Public results now require membership in the requested published round.

The first isolated round, `arena-2026-09-05-e2e`, cancelled with
`capacity:stage2:1`. Two baseline attempts lacked terminal completion at close.
The exact prior completion failure was not logged; do not claim the accounting
retry conclusively explains that cancellation. Its database and objects remain
retained. The second round had no abandoned completion or lease-expiry failure.

One first public read in the second round returned HTTP 500 from a closed local
PostgreSQL connection. A subsequent read succeeded without state changes. The
database container did not restart, and its logs did not explain the disconnect.
This transient local-transport failure remains disclosed, not silently labelled
a model error or attributed to production PostgREST.

## Scope and checks

This used isolated services on the actual gateway and validator hosts, real
testnet registration, real provider credentials, KMS, S3, and SQL migrations
179–185. SQL RPCs ran through `PsycopgTransport` under `lab_arena_service`.
It did not exercise production PostgREST HTTP wiring, deploy migration 185 to
production, change the baseline repo, or restart production services.

Focused checks passed: 177 combined miner/runtime checks during integration,
43 runner checks after completion-retry changes, and 35 service/API checks after
the result-membership fix and main-branch integration. Syntax and diff checks
passed. CI run `33946036338` had 7,094 passed and six failures; all six were
independently reproduced on its main-branch base. They were not fixed or bypassed
in this miner-scoped change. Do not describe the full CI suite as green.

Before production enablement, follow the staged operator setup in
[miner-model-submissions.md](miner-model-submissions.md), check the production
PostgREST path, and coordinate with the separate baseline/deployment task.
The current runner supports Python agents, not arbitrary language runtimes.
