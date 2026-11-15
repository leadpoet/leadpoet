# Arena improvement reward slots

Champion selection is unchanged. After final publication and successful code
promotion, reward activation assigns slots from the original published winner
and baseline scores in the same round. Neither scoring nor promotion is rerun.

For newly promoted models in evaluation rounds dated November 11, 2026 and later:

| Improvement | Share of total emissions before decay |
| --- | --- |
| At least +15 | 35% |
| At least +10 | 25% |
| At least +5 | 15% |

The base burn is 25%. A promoted winner takes every slot whose threshold it
clears: +5 earns 15%, +10 earns 40%, and +15 earns 75%, before reductions.
New rounds require at least +5 for promotion; already committed rounds keep
their frozen promotion margin. Slot policies are defined in `lab_arena/rewards.py`.
The signed `allocation_basis: total_emissions` makes these exact percentages
of total emissions. Absent that field, historical tier percentages still divide
the signed miner pool.

The transition carries the current signed holders into the corresponding
slot positions: +20 becomes +15, +15 becomes +10, and +8 becomes +5. It keeps
their source scores, submitted models and tenure clocks. The November 3
incumbent keeps its original 20%/15%/15% slot allocations, before decay,
until a new qualifying promoted award replaces each slot, even when the
winning hotkey and normalized source are unchanged.
This exception is bound to the verified owner, normalized source and original
epoch 26183; repackaging alone is not a qualifying award. Each replaced slot
uses the new allocation independently. Unused allocation goes to burn.
Lower thresholds do
not revive old displaced winners or fill empty slots from historical results.
Only a new, verified promoted winner can replace an incumbent. One miner can
hold several slots, and smaller wins do not remove higher-tier holders.

Previously activated bases and accepted weight states remain immutable. New
states may contain a signed allocation projection of the governing basis.
The earlier epoch-26301 trigger was premature and is retained only to validate
already issued states, not to select the new policy. A state already issued for
an epoch is never rewritten. If the source
predates October 23, its currently represented achievements first undergo the
existing October 23 projection; that policy transition is not undone.

Historical policies remain supported: October 21 and earlier used a 30% pool
and +10/+5/+1 thresholds; October 22 used a 50% pool and the same thresholds.
From October 23 or reward epoch 25954, thresholds were +20/+15/+8 with total
shares 20%/15%/15%. The legacy `LAB_ARENA_POOL_PERCENT` setting does not change
v2 slot allocations. Unpromoted, ineligible, incomplete, cancelled and
other-network results cannot create slot achievements.

With `LAB_ARENA_WEEKLY_SLOT_DECAY_ENABLED=true`, each occupied slot halves
after every 140 settlement epochs (the existing reward week, about seven days),
without a reward floor.
The multiplier is 100% initially, 50% after one week, 25% after two, 12.5%
after three, 6.25% after four, 3.125% after five, and so on. The normal chain
weight precision eventually rounds tiny allocations to zero. Removed allocation
goes to burn; it is not shared among other holders.

Each clock starts at the source winning round's original effective reward
epoch. A daily baseline, restart or new daily reward
document does not reset it. A new qualifying promoted award resets only the
slots it wins, including when the hotkey and normalized source are unchanged.
Retaining or replaying the same award preserves its original start epoch.
An incorrect inherited clock can be corrected in a future accepted state only
after verifying that award and its frozen source; past states stay unchanged.
Different slots held by one miner can have different ages.
Unregistered owners do not pause their clocks.

All validators use the same chain epoch and immutable accepted state, without
local wall clocks or a mid-epoch weight change. Old states without
`weekly_slot_decay` retain their signed four-halving limit.
The current-round API includes `weekly_slot_decay` only after a state with
that clock is accepted for the current epoch. This field governs current
decay; the returned frozen slot policy remains historical provenance.

Activation carries the latest activated signed slots in the same chain scope
and checks their original published and promoted source records. A newly
promoted current-round winner replaces only the slots it qualifies for. Each signed slot binds its
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
contains the selected miner pot, allocation basis and tier percentages. The epoch-25954
threshold transition uses the existing v2 basis and accepted-state schemas;
it adds no field or transaction format and needs no new validator schema. Historical v2 policies
without `pool_percent` keep their original total-emissions arithmetic. Policies
without `decay` retain their original non-decaying payouts. The
frozen `reward_constants` and weekly schedule do not rescale new slot payouts.

Validators must support `allocation_basis: total_emissions` before its gateway
cutover. The dashboard must interpret this mode without multiplying the tier
percentages by the pool a second time.

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
