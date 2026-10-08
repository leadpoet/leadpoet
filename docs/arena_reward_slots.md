# Arena improvement reward slots

Champion selection is unchanged. After final publication and successful code
promotion, reward activation assigns slots from the original published winner
and baseline scores in the same round. Neither scoring nor promotion is rerun.

| Improvement | Share of the existing 30% miner pot |
| --- | --- |
| At least +10 | 50% |
| At least +5 | 30% |
| At least +1 | 20% |

The only slot policy definition is `REWARD_SLOT_POLICY` in
`lab_arena/rewards.py`. Its `pool_percent: 30` is the miner share of total
emissions. The `assignment_mode: all_qualifying` assigns a promoted win to
every tier whose threshold it clears. A +10 win holds all three slots and
earns the full 30% pot; +5 earns 15% of total emissions through two slots;
+1 earns 6% through one slot. The next reward activation signs the policy.
Policy changes never rewrite already accepted epoch states.
For v2 slots, change the pot only in `REWARD_SLOT_POLICY`. The historical
`LAB_ARENA_POOL_PERCENT` setting remains legacy signed metadata and does not
set the v2 slot pot.

A newer qualifying promoted winner replaces a slot holder, regardless of the
holder's earlier score. Smaller wins do not remove larger-improvement holders.
One miner can hold several slots through one or more achievements. There is no
lock period or expiry tied to the age of a holder's achievement, and no weekly
decay. A promoted improvement below +1 changes the champion as usual but earns
no slot. Unpromoted, ineligible, incomplete, cancelled and other-network
results cannot create slot achievements.

Activation uses published, successfully promoted history in the same chain
scope, including eligible historical winners. Each signed slot binds its
source round, winning submission, miner hotkey, baseline submission and both
original scores. Selection uses the exact decimal score difference; rounded
dashboard values are not an input. A new signed basis carries all three slots.
The activation transaction checks the snapshot against the same historical
records before accepting it.

Empty slots and shares belonging to unregistered hotkeys go to the existing
burn hotkey. Shares held by the same registered hotkey are added together.
Unused shares are not redistributed to other holders.

The existing champion-funded baseline behavior is unchanged. If its existing
funding-fallback rule reduces the current champion's reward factor, that factor
applies only to slots held by that champion. It never reduces another holder's
allocation. The existing freshness limit for the governing reward basis also
remains; this is separate from how long an achievement owns a slot.

## Signed-state compatibility

New activations use reward-basis v2 through the existing gateway, accepted
weight state, normal validator and local signer. The signed `slot_policy`
contains the 30% miner pot and its tier percentages. Historical v2 policies
without `pool_percent` keep their original total-emissions arithmetic. The
frozen `reward_constants` and weekly schedule do not rescale new slot payouts.

Validators must support the signed `pool_percent` field before the new policy
becomes effective.
Version 1 bases keep their original hashes, signatures and arithmetic. The
accepted-weight-state envelope and transaction format do not change. Existing
signed transactions, pending reveals and immutable accepted epoch states stay
valid across deployment. Activation starts at a future settlement epoch; it
does not rewrite previous payments or published competition results.
