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


def canonical_purposes_before_routing_experiment_v2(role: str) -> set[str]:
    """Return the exact canonical role set before migration 158."""

    purposes = set(ROLE_PURPOSES[role])
    if role == "gateway_scoring":
        purposes.difference_update(ROUTING_EXPERIMENT_PURPOSES_V2)
    return purposes
