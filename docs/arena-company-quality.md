# Arena company quality

`company_quality_policy: "company_quality_v1"` is an opt-in, frozen round
policy. It requires `arena_integrity_v1` and can run with or without
`contacts_v1`. Historical rounds retain their output contract, arithmetic
aggregation, judgments, budgets, and promotion behavior.

## Company output and verification

Quality rounds use `leadpoet.lab_arena.output.v3`, or `.v4` when contacts are
also required. A harness can still return a plain list: the runner records the
schema assigned by the round. A declared incompatible schema is rejected.
The input passed to `harness.run_icp(icp)` includes the quality policy and
company requirements. The return shape remains the same list of companies.

Every company must supply a LinkedIn **company** page that the verifier can
match to that business. HTTP/HTTPS, regional LinkedIn hosts, tracking
parameters, trailing slashes, and known company-page tabs normalize to one
URL. Personal profiles, arbitrary paths, nonstandard ports, and lookalike
hosts do not qualify. If LinkedIn cannot be fetched, the company website and
independent evidence can establish the match. A URL alone is not proof.

For U.S. headquarters, `state` must identify the actual headquarters state,
not the incorporation state or a branch office. Country aliases such as US,
USA and United States normalize; full state names, USPS abbreviations,
case variations and D.C. are accepted. State is optional outside the U.S.
The independent fit verifier checks the claimed country and U.S. state.

Missing or malformed new fields give **that company** zero credit. Other
valid companies in a parseable output remain eligible. Provider failures
remain retryable infrastructure failures; they are not cached as rejections.
Missing contact claims retain the existing contact-policy behavior.

Signal verification accepts contextual attribution such as “we launched” on
an independently verified company website. Publisher identity alone does
not attribute a customer story or partner announcement to the publisher.
An unclear identity gets one bounded clarification, then must be resolved
before receiving credit. Literal company-name matching is not required.

The verifier preserves independently established company identity between
homepage, fit and signal checks. Independently verified LinkedIn identity
deduplicates companies across domains within one buyer request. Separate
subsidiaries retain credit when their distinct identities are established.
The same company may still qualify for different buyer requests.

## Individual company judgments

The gateway builds cache keys from effective company input, buyer criteria,
evaluation date, frozen judging rules, scorer image and round/network scope.
Contact evidence is included when contacts are required. Submitted list
position and unused prose do not create a new judgment.

Validators receive accepted company judgments and leases for missing ones.
The database reserves missing keys atomically; competing claims cannot start
independent judgments for the same authority slot. Signed claim and completion
provenance remain attached to immutable evidence. An ownership conflict still
requires eligible independent authority; sharing does not waive that rule.
Accepted positive and substantive negative judgments are reused. Infrastructure
failures release their reservation for retry instead of becoming cached zeros.

The cache stores the intrinsic company verdict before list-dependent duplicate
penalties. The gateway recomputes company indexes, identity deduplication and
qualification for every destination list. A company judged as a duplicate in
one output can therefore qualify when it appears alone in another output.
Each participant retains its own sourcing costs and output. Cache material
is private to the gateway and authorized scoring leases.

Quality v1 retains the current `max_scored_companies=0` setting: judge all
eligible companies within the buyer's assigned company count. A smaller
operator scoring cap is rejected when validating the new policy, before a
round opens. This prevents list-dependent unjudged slots from entering the
shared cache as company rejections. Historical policy settings are unchanged.

## Coverage-weighted score

Per-company and per-ICP scoring remain unchanged. For every stage selection,
the main twenty-ICP comparison, and the five-ICP confirmation comparison:

```text
score = (sum(sqrt(per_icp_score)) / number_of_assigned_icps) ** 2
```

Missing model outputs occupy zero slots in the fixed denominator. Aggregate
the main twenty directly from those twenty scores; do not average two
already-aggregated stage totals. The policy uses Decimal precision 50, sorted
inputs, roots rounded half-even to forty decimal places, exact subsequent
arithmetic, and half-even rounding to twelve decimal places for stable comparisons.
Infrastructure failures retain the existing incomplete-judging handling.

| Five ICP scores | Historical mean | Quality score |
| --- | ---: | ---: |
| 40, 40, 40, 40, 40 | 40 | 40 |
| 100, 100, 10, 0, 0 | 42 | 21.459644256269 |
| 40, 40, 40, 40, 0 | 32 | 25.6 |

This rewards useful performance across requests without adding a minimum-output
eligibility gate. The existing one-point promotion margin and confirmation
cohort remain unchanged. This changes score aggregation, not the existing
champion lifecycle or reward rules.

## Activation and rollback

1. Install existing migrations through 216, then
   `scripts/20260911200103_lab_arena_company_judgments.sql`.
2. Deploy matching gateway, validator and scorer code/image through the
   existing release process. The quality capability probe must pass.
3. Set `LAB_ARENA_COMPANY_QUALITY_FROM` to an explicit future timestamp with
   timezone, alongside a compatible `LAB_ARENA_INTEGRITY_FROM`. It applies to
   **submission-open time**, so the contract is announced before intake.
4. Verify a new round publishes the quality marker and matching scorer policy;
   run a complete controlled round through saved results and publication.

The activation variable is unset by default. Removing it stops new quality
rounds; already-created quality rounds must finish with their frozen rules.
Never downgrade an active round or rewrite historical accepted judgments.
No production dependency is added by this change.

Automated cases use controlled provider responses. They verify contract and
flow behavior; they do not estimate a real-world false-negative rate.
