# Arena improvement reward slots

Champion selection is unchanged. After final publication and successful code
promotion, reward activation assigns slots from the original published winner
and baseline scores in the same round. Neither scoring nor promotion is rerun.

For evaluation dates October 23, 2026 and later, and pending activations whose
effective reward epoch is at least **25954**:

| Improvement | Share of the 50% miner pot | Share of total emissions |
| --- | --- | --- |
| At least +20 | 40% | 20% |
| At least +15 | 30% | 15% |
| At least +8 | 30% | 15% |

Slot policies are defined in `lab_arena/rewards.py`. The
`assignment_mode: all_qualifying` assigns a promoted win to every tier whose
threshold it clears. Before decay or funding reductions, a +20 win holds all
three slots and earns 50% of total emissions; +15 earns 30% through two slots;
+8 earns 15% through one slot.

Activation selects the policy from the round's validated `evaluation_date`
and its effective reward epoch. The epoch-25954 cutover also applies to delayed
activations of earlier rounds. Before that boundary, October 21 and earlier
use the original 30% pot with +10/+5/+1 thresholds and 50/30/20 pot shares;
October 22 uses a 50% pot with those thresholds and 40/30/30 pot shares.
Existing signed bases and accepted epoch states keep their original rules.
The historical `LAB_ARENA_POOL_PERCENT` setting remains legacy signed metadata
and does not set the v2 slot pot.

Starting at epoch 25954, a newly issued accepted state can carry a newly signed
policy projection of the governing reward basis. This one-time transition uses
only that basis's current signed achievements. It chooses the newest represented
achievement that meets each new threshold; it does not revive replaced owners.
It preserves every source score, submission, original effective epoch and decay
clock. A +6.0454545 holder therefore loses its slots; the retained +11.1 holder
qualifies for +8 only, for 15% miner emissions and 85% burn before reductions.
The original published basis and every previously accepted state stay unchanged,
including any state already issued at the cutover. Later round activations use
the normal verified publication and promotion history.

A newer qualifying promoted winner replaces a slot holder, regardless of the
holder's earlier score. Smaller wins do not remove larger-improvement holders.
One miner can hold several slots through one or more achievements. There is no
lock period or expiry tied to the age of a holder's achievement. A promoted
improvement below +8 changes the champion as usual but earns
no slot. Unpromoted, ineligible, incomplete, cancelled and other-network
results cannot create slot achievements.

With `LAB_ARENA_WEEKLY_SLOT_DECAY_ENABLED=true`, each occupied slot halves
after every 140 settlement epochs (the existing reward week, about seven days),
without a reward floor.
The multiplier is 100% initially, 50% after one week, 25% after two, 12.5%
after three, 6.25% after four, 3.125% after five, and so on. The normal chain
weight precision eventually rounds tiny allocations to zero. Removed allocation
goes to burn; it is not shared among other holders.

Each clock starts at the source winning round's original effective reward
epoch. A daily baseline, restart or new daily reward
document does not reset it. A genuinely new promoted model resets only the
slots that it replaces, including when the same miner wins again. Repackaging
the same normalized source does not reset its age. Signed promotion history
establishes uninterrupted occupation; an intervening different owner or model
starts a new tenure. Different slots held by one miner can have different ages.
Unregistered owners do not pause their clocks.

All validators use the same chain epoch and immutable accepted state, without
local wall clocks or a mid-epoch weight change. Old states without
`weekly_slot_decay` retain their signed four-halving limit.

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
allocation. New accepted weight states using a v2 reward basis include the signed
`slot_eligibility: until_replaced` rule. Their occupied slots continue to pay
after the reward basis is more than 45 epochs old. Each slot still follows its
own signed decay clock and stops paying when a newer qualifying achievement
replaces it. Missing owners and empty slots still route their shares to burn.
Accepted states without this field retain the 45-epoch basis freshness limit.
Version 1 champion rewards keep that limit in every state.

## Signed-state compatibility

New activations use reward-basis v2 through the existing gateway, accepted
weight state, normal validator and local signer. The signed `slot_policy`
contains the selected miner pot and its tier percentages. The epoch-25954
threshold transition uses the existing v2 basis and accepted-state schemas;
it adds no field or transaction format and needs no new validator schema. Historical v2 policies
without `pool_percent` keep their original total-emissions arithmetic. Policies
without `decay` retain their original non-decaying payouts. The
frozen `reward_constants` and weekly schedule do not rescale new slot payouts.

Validators must support the optional, signed `weekly_slot_decay` clock before
the gateway enables it. Roll out support with the switch off, verify primary,
Rizzo and Yuma have the matching shared parser and weight kernel, then enable
the switch with `configure_lab_arena_production.py --weekly-slot-decay enabled`
through its existing check/apply controls and the canonical gateway release. The clock carries the
verified source hash and promotion epoch for each occupied slot. It
uses the existing accepted-state persistence; no new reward table is needed.
Missing source or promotion proof stops new state issuance instead of inventing
tenure. Already issued states remain available during recovery.
Version 1 bases keep their original hashes, signatures and arithmetic. The
accepted-weight-state envelope gains one optional, hash-covered field. The
transaction format does not change. Existing signed transactions, pending
reveals and immutable accepted epoch states stay
valid across deployment. Continuity starts with the first newly accepted state
after deployment; already accepted epochs remain unchanged. It does not rewrite
previous payments or published competition results.
