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
confirmation baseline by at least one point. Accepted main company qualifications
stay fixed; final cost eligibility includes confirmation spend and qualified
slots. When no challenger qualifies, skip execution and publish no new king.
Infrastructure gaps cannot be converted into favorable zeros. The database
guards the cohort, recorded scores, costs, and publication transition.

A selected challenger whose terminal confirmation judgment records a credential
or budget failure is disqualified with a null confirmation score and the reason
`confirmation_account_failure`. It remains visible in the original cohort and
ranking. Other original finalists can finish and win. Its main qualification
slots and all actual execution/judge costs remain reported. Missing judgments,
shared infrastructure failures, and baseline failures do not permit this exception.
Both the service and database derive the exception from the frozen scoring plan
and durable attempts, so it survives restarts without replacing candidates.

Publish main and confirmation scores separately. Reveal the salted confirmation
document and its results after publication so observers can check the original
commitment. Preserve the main bank's existing disclosure schedule.

## Rollout and verification

Apply repository migrations 211 through 214 in order, after their existing
prerequisites. Migration 210 is reserved by separate disclosure work; reconcile
that work before release if it has merged. Deploy the matching gateway and
judge image together through the existing release process. Migration 214 keeps
a proven miner credential refusal distinct from an infrastructure failure when
its uncertain provider charge blocks a later reservation. It retains the charge.
Confirmation generation uses the Arena organizer's OpenRouter credential.

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
