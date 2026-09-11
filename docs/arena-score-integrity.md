# Arena score integrity

The opt-in `arena_integrity_v1` round policy uses the
`qualification_integrity_v2` scorer adapter. The policy is saved with each round;
historical rounds retain their existing scorer and publication rules.

## Scoring behavior

- Score the original first-N company slice, capped at five. A verified company
  gets one scoring slot. Normalize legal suffixes and registrable domains, then
  use independently observed identity aliases to resolve different names.
  Submitted LinkedIn fields cannot create or split a verified identity. Shared
  domains alone do not merge unrelated businesses. Duplicate rows receive zero;
  they do not reject the entire output or trigger replacement from later rows.
- Judge claim support separately from recency. Reject a date only when the
  exact source establishes that the same event falls outside the buyer window.
  A source event date outranks publication metadata. Missing or uncertain dates
  remain eligible for ordinary claim verification. Harmless date mismatches do
  not reject supported evidence. The existing indexing tolerance remains.
- Take the strongest accepted contribution for each verified requested signal.
  Multiple articles about one signal do not earn breadth credit. Different
  genuine requested signals may use the same publisher. The required primary
  signal must still pass.
- Give the job-source premium only when job content and its employer/publisher
  relationship are verified. `/jobs` or `/careers` in a URL alone earns no
  premium. Valid unfamiliar sources can still earn ordinary evidence credit.
- Keep the first terminal company verdict across infrastructure retries. A
  recovered company retains its original index and is checked against already
  verified identity aliases. Infrastructure exhaustion is incomplete judging,
  not an invented company zero.

## Qualification and costs

Accepted judgments carry original company indexes, canonical identity keys,
duplicate flags, and qualification flags. The persisted qualification receipt
is immutable with its per-ICP score.

The sourcing allowance is the smaller of the fixed execution cap and the
per-company allowance multiplied by unique qualified company slots. Count
qualification separately for each ICP. Ignored, duplicate, irrelevant, excess,
or failed companies add no allowance. Actual execution spend across retries
continues to count. Judge spend is reported separately. Inflight or uncertain
provider charges do not establish eligibility.

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

Judgment sharing uses the exact effective scoring input, including array order,
the round, chain scope, pinned image, evaluation date, and policy. It excludes
only execution-run identity. Accepted evidence and its original validator
provenance are hash-bound and stored atomically. Failed judgments are not
cached. Every recipient retains its own execution output and sourcing cost.
Validator ownership conflicts must still be excluded when reusing evidence.

## Confirmation before promotion

Before evaluation, commit a salted hash of five separately generated private
ICPs. Keep duplicate checks local; the generator receives a distinct random
draw identifier, not the main private bank. Exact structured requirement
duplicates, including ID or prompt-only changes, cannot enter this bank.

After the main twenty ICPs, freeze one cohort: the baseline plus up to three
eligible challengers whose main score is at least one point above the baseline.
Sort challengers by main score, then submission ID. Run their unchanged frozen
sources on the same five confirmation ICPs. Do not substitute new candidates,
pick the best repeated confirmation result, or retry a completed judgment.

The highest eligible confirmation score can win only if it also beats the
confirmation baseline by at least one point. Original main qualification stays
fixed. When no challenger qualifies, skip execution and publish no new king.
Infrastructure gaps cannot be converted into favorable zeros. The database
guards the cohort, recorded scores, costs, and publication transition.

Publish main and confirmation scores separately. Reveal the salted confirmation
document and its results after publication so observers can check the original
commitment. Preserve the main bank's existing disclosure schedule.

## Rollout and verification

Apply repository migrations 211, 212, and 213 in order, after their existing
prerequisites. Migration 210 is reserved by separate disclosure work; reconcile
that work before release if it has merged. Deploy the matching gateway and
judge image together through the existing release process.

Set `LAB_ARENA_INTEGRITY_FROM` to an explicit UTC cutoff timestamp only after
the matching schema and image are installed. It is unset by default. Startup
and new-policy round creation check the integrity schema capability. Never
change the policy of a round already committed. For rollback, stop admitting
new integrity rounds and finish or explicitly cancel existing ones; do not
downgrade their scoring rules.

Tests cover the pictured score-inflation cases, permissive uncertain dates,
valid independent evidence, ownership races, shared-judgment provenance,
infrastructure retries, and saved full-round publication after restarts.
They use controlled provider results. An empirical false-negative rate still
requires labeled real evidence; passing synthetic date cases does not measure
that rate.
