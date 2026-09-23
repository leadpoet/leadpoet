# Arena score integrity

The opt-in `arena_integrity_v1` round policy uses the
`qualification_integrity_v2` scorer adapter. The policy is saved with each round;
historical rounds retain their existing scorer and publication rules.

## Scoring behavior

- Score the original first-N company slice, capped at five. A verified company
  gets one scoring slot. Normalize legal suffixes and registrable domains, then
  use independently observed identity aliases to resolve different names.
  The same independently verified LinkedIn company identity also deduplicates
  across corporate domains. Distinct verified subsidiaries retain separate slots;
  a shared parent domain or a submitted parent LinkedIn link is not sufficient
  to merge them.
  Submitted LinkedIn fields cannot create or split a verified identity. Shared
  domains alone do not merge unrelated businesses. Duplicate rows receive zero;
  they do not reject the entire output or trigger replacement from later rows.
- Judge claim support separately from recency. Reject a date only when the
  exact source establishes that the same event falls outside the buyer window.
  A source event date outranks publication metadata. Missing or uncertain dates
  remain eligible for ordinary claim verification. Harmless date mismatches do
  not reject supported evidence. The existing indexing tolerance remains.
- For each requested signal, retain the first three distinct canonical source
  URLs and evaluate their evidence together in one source-grounded judgment.
  Repeated URLs and evidence beyond that limit are ignored before judging.
  The advisory first pass cannot approve the criterion, and the optional
  corroboration and post-verdict evidence-repair re-judging paths are disabled for integrity rounds. Multiple
  articles about one signal do not earn repeated chances or breadth credit. Different
  genuine requested signals may use the same publisher. The required primary
  signal must still pass.
- Give the job-source premium only when job content and its employer/publisher
  relationship are verified. `/jobs` or `/careers` in a URL alone earns no
  premium. Valid unfamiliar sources can still earn ordinary evidence credit.
- Keep the first terminal company verdict across infrastructure retries. A
  recovered company retains its original index and is checked against already
  verified identity aliases. Infrastructure exhaustion is incomplete judging,
  not an invented company zero. An unavailable bonus signal also leaves an
  integrity judgment incomplete. Each signal uses its own buyer freshness cap.

## Qualification and costs

Accepted judgments carry original company indexes, canonical identity keys,
duplicate flags, and qualification flags. The persisted qualification receipt
is immutable with its per-ICP score.

For rounds using `successful_calls_per_icp_v1`, baseline and miner models use
the same $4 sourcing admission limit for each ICP. The existing provider ledger
combines all providers and execution attempts for that ICP. Retries do not reset
spending. An admitted call can finish above $4; its full charge and response are
retained. No further paid sourcing calls start for that ICP after the limit is
reached. Other ICPs continue, and valid saved outputs remain available for scoring.

With migration 321, only confirmed charges, including billed failures, consume
the sourcing admission budget. Pending bills and historical monetary holds do not
block new research. New execute lifecycle reservations have amount zero; this
means no hold, not a confirmed free call. Exact call identities remain tracked
for later settlement and cannot be dispatched twice. Concurrent admitted calls
can take confirmed spend above the cap. Current-policy judge admission also
uses confirmed charges only, with its existing separate submission-wide cap.
Judge charges never enter sourcing spend or qualification cost eligibility.

After normal quality scoring, each ICP has a successful-sourcing cost allowance
of $0.80 multiplied by its verified qualified companies. Five companies
allow $4; two allow $1.60. Ignored, duplicate, irrelevant, excess, or failed companies
add no allowance. A cost-ineligible ICP contributes zero to the competition
aggregate; its raw quality score and costs remain recorded. It does not remove
scores from other ICPs or disqualify the whole submission.

Successful paid calls count even when empty or followed by a model failure.
Failed provider calls do not enter final cost eligibility, even if the provider
billed them. In-flight calls and unresolved successful sourcing costs keep their
existing fail-closed treatment. Actual billing and competition costs are reported
separately; judge costs do not enter sourcing eligibility.

Historical rounds retain their frozen rules. Under `successful_calls_v1`, the
runtime admission limit was $80 for the submission, and final successful sourcing
spend was compared with $0.80 times all qualified pairs across its ICPs. Published
historical results are not rewritten.

## Inputs, admission, and repeated judgments

Project ICPs through an allowlist at agent and scorer boundaries. Keep buyer
criteria, exclusions, and requested signals; remove evaluator examples and
generation metadata, including nested hints. Do not remove company names that
are part of an actual buyer requirement.

One finalized chain coldkey may have one active challenger entry per round.
The registration records its finalized block reference. Repeated requests and
source replacement retain the original ownership record. This limits entries
per chain owner; it does not prove that separate coldkeys belong to different
people.

Judgment sharing uses the adapter's normalized effective input, including company
positions and retained evidence order, the round, chain scope, pinned image,
evaluation date, and policy. Run identity, unused `why_now` text, validated but
unused company prose/state, unused fit lookup URLs, duplicate/capped evidence,
and unscored output padding cannot create a fresh cache entry. Accepted evidence and its original validator
provenance are hash-bound and stored atomically. Failed judgments are not
cached. Every recipient retains its own execution output and sourcing cost.
Validator ownership conflicts must still be excluded when reusing evidence.
Lease-time exclusions include the submissions' frozen owners and remain bound
to accepted evidence after ownership changes. A confirmed miner account failure
allows the next identical output to obtain a judgment with its own credentials.

## Promotion

After all twenty ICPs have complete valid results, rank the baseline and
eligible challengers by their twenty-ICP scores. The best eligible challenger
becomes king only when its score is at least one point above the baseline's
effective score. Under the per-ICP policy, that score includes the zero contribution
from each cost-ineligible ICP. The normal one-point promotion margin still applies.
If no eligible challenger reaches the required margin, no new king is selected
and the existing champion remains.
Qualification receipts, cost eligibility, deterministic tie ordering, and all
other publication guards continue to apply. No extra ICP set or finalist-only
stage can change or veto this result.

## Rollout and verification

Apply repository migrations 211 through 214, 251, and 254 in order, after their
existing prerequisites. Apply migration 254 before restarting the updated
gateway; it prevents an older runtime from publishing a cost-ineligible
baseline with its raw score. Published historical documents remain unchanged.
Migration 251 remains the cutover dependency for the twenty-ICP runtime. A
confirmation-enabled older runtime cannot process new work after migration 251
removes its RPCs; rollback needs a separately reviewed schema migration. The
scorer model and image remain unchanged.
Migration 214 keeps
a proven miner credential refusal distinct from an infrastructure failure when
its uncertain provider charge blocks a later reservation. It retains the charge.
Set `LAB_ARENA_INTEGRITY_FROM` to an explicit UTC cutoff timestamp only after
the matching schema and image are installed. It is unset by default. Startup
and new-policy round creation check the integrity schema capability. Never
change the policy of a round already committed.

Tests cover the pictured score-inflation cases, permissive uncertain dates,
valid independent evidence, ownership races, shared-judgment provenance,
infrastructure retries, and saved full-round publication after restarts. Rounds
that announce `intent_details_policy: "intent_details_v1"` use output v5. They
replace the fit prose, fit URL hints, signal `why_now`, and submitted snippets
with one bounded company-level intent paragraph. The paragraph earns no
additional points. It is a binary postcondition checked against the
independently verified signals. A mismatch gives that company zero. A verifier
outage retains the existing infrastructure retry behavior. Historical v1-v4
outputs and cache identities remain unchanged.
The writing rules follow [Tyche's Intent Details contract](https://github.com/gzaentz/tyche/blob/6bc90cc35a194ef5ed30ec77bb1c3f5df2602593/.agents/skills/lead-sourcing/references/output-contract.md):
combine distinct supported activities into natural prose, retain supported
dates, label commercial implications as conditional, and end with a clear ICP
connection. Arena adds one bounded review using its existing pinned intent
verification model. The review uses saved verifier quotes and authoritative
dates; it neither searches for new facts nor rewrites the submitted paragraph.
They use controlled provider results. An empirical false-negative rate still
requires labeled real evidence; passing synthetic date cases does not measure
that rate.
