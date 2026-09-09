"""Historical receipt-purpose snapshots used by pre-routing migrations.

The canonical allowlist is intentionally current.  Migrations before 158
must still be checked against the exact allowlist that existed before routing
experiments were introduced; otherwise a later canonical purpose is mistaken
for a missing entry in an older migration.
"""

from leadpoet_canonical.attested_v2 import ROLE_PURPOSES


ROUTING_EXPERIMENT_PURPOSES_V2 = frozenset(
    {
        "research_lab.routing_model_binding_observation.v2",
        "research_lab.routing_experiment.v2",
        "research_lab.routing_provider_evidence.v2",
    }
)

# These purposes were part of the immutable coordinator allowlist before the
# provider-outcome checkpoint machinery was retired. They remain in the
# historical migrations and must be checked there without reintroducing them
# to the active runtime allowlist.
HISTORICAL_PROVIDER_OUTCOME_PURPOSES_V2 = frozenset(
    {
        "research_lab.provider_outcome_snapshot.v2",
        "research_lab.provider_outcome_state.v2",
    }
)

# Migration 126 admitted these exact coordinator inputs. Source Add was later
# retired from the active weight-input contract, but the append-only migration
# must continue to match the historical database constraint it installed.
CHAIN_REALIZED_WEIGHT_INPUT_PURPOSES_V1 = frozenset(
    {
        "research_lab.allocation.v2",
        "research_lab.champion_input.v2",
        "research_lab.reimbursement_input.v2",
        "research_lab.source_add_reward_input.v2",
        "research_lab.sourcing_input.v2",
        "research_lab.fulfillment_input.v2",
        "research_lab.leaderboard_input.v2",
        "research_lab.ban_input.v2",
        "research_lab.anomaly_adjustment_input.v2",
    }
)


def canonical_purposes_before_routing_experiment_v2(role: str) -> set[str]:
    """Return the exact canonical role set before migration 158."""

    purposes = set(ROLE_PURPOSES[role])
    if role == "gateway_coordinator":
        purposes.update(HISTORICAL_PROVIDER_OUTCOME_PURPOSES_V2)
    if role == "gateway_scoring":
        purposes.difference_update(ROUTING_EXPERIMENT_PURPOSES_V2)
    return purposes
