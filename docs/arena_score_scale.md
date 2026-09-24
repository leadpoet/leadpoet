# Arena score scale

New integrity rounds freeze `ARENA_SCORE_NORMALIZATION=available_intent_cap_v1`
in the existing `scorer_policy.env_bindings`. All participants in that round use
the same scale. Existing frozen rounds without the binding keep their original
scores; process environment variables cannot change their scale.

The gateway calculates the existing net ICP score, including its fixed company
denominator and existing deductions, then multiplies it by `100 / available_cap`.
The available cap comes from the judge's existing table and the ICP's full list
of distinct requested intent criteria (including bonus criteria):

| Requested criteria | Raw cap | Perfect net ICP score after normalization |
| --- | --- | --- |
| 1 | 60 | 100 |
| 2 | 80 | 100 |
| 3 | 88 | 100 |
| 4 | 92 | 100 |
| 5 | 96 | 100 |
| 6 or more | 100 | 100 |

Submitted evidence cannot reduce the denominator. Raw company judgments and
qualification remain unchanged. For one criterion, a raw net score of 60 becomes
100 and 30 becomes 50. One perfect company out of five contributes 20 points;
missing companies still occupy their original slots. Existing penalty and
coverage deductions retain their proportion of the attainable score.

Within an ICP, scores retain their order and proportions. Across different ICPs,
each now has the same 100-point maximum, so the aggregate can change relative to
the old scale. The round still averages all configured ICP slots, applies the
same cost eligibility, and uses the same promotion margin and reward rules.
No extra evidence, company, contact, or paid call is required.

Migration 361 opts only the open, unstarted September 25 round into this policy.
Published rounds are not rescored. The policy uses the existing transport format
so older validator hosts can carry it without a new protocol. Deterministic
normalization happens in the gateway after the trusted scorer returns its
unchanged raw evidence and scores.
