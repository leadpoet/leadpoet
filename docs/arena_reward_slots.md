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
lock period or expiry tied to the age of a holder's achievement. A promoted
improvement below +1 changes the champion as usual but earns
no slot. Unpromoted, ineligible, incomplete, cancelled and other-network
results cannot create slot achievements.

Each slot's reward halves after every completed reward week, up to four
halvings. A reward week uses the existing 140 settlement epochs. The multiplier
is 100% initially, 50% after one week, 25% after two, 12.5% after three, and
6.25% after four or more. Removed allocation goes to burn; it is not shared
among other holders. `REWARD_SLOT_POLICY.decay` holds the interval and limit.

Each clock starts at the source winning round's original effective reward
epoch. A daily baseline or a new daily reward document does not reset it.
A new qualifying achievement resets only the slots that it replaces, including
when the same miner wins again. Different slots held by one miner can have
different ages. Empty or unregistered slots do not pause their clocks.

Activation uses published, successfully promoted history in the same chain
scope, including eligible historical winners. Each signed slot binds its
source round, winning submission, miner hotkey, baseline submission and both
original scores. Decay-enabled slots also bind their start epoch. A slot
from the current reward round stores `start_epoch: null`, which means that
basis's effective epoch. Historical slots store their original numeric epoch.
Selection uses the exact decimal score difference; rounded
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
without `pool_percent` keep their original total-emissions arithmetic. Policies
without `decay` retain their original non-decaying payouts. The
frozen `reward_constants` and weekly schedule do not rescale new slot payouts.

Validators must support the signed decay policy and slot start epochs before
the new policy becomes effective.
Version 1 bases keep their original hashes, signatures and arithmetic. The
accepted-weight-state envelope and transaction format do not change. Existing
signed transactions, pending reveals and immutable accepted epoch states stay
valid across deployment. Activation starts at a future settlement epoch; it
does not rewrite previous payments or published competition results.
