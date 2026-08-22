"""Generated release shim for the reviewed Lab routing product.

The production release builder regenerates this module with the exact
protected-workflow manifest hash.  The source checkout intentionally has no
attested authority provider, so startup remains disabled until the release
package supplies the fixed provider module.
"""

from gateway.research_lab.routing_release_builder import (
    RELEASE_AUTHORITY_PROVIDER_MODULE,
    RELEASE_MODULE_SCHEMA_VERSION,
    build_reviewed_routing_release_dependencies,
)


RELEASE_MODULE_SCHEMA = RELEASE_MODULE_SCHEMA_VERSION
# The release generator replaces this sentinel with the attested manifest
# digest.  An ungenerated checkout must remain fail closed.
EXPECTED_PROTECTED_WORKFLOW_MANIFEST_HASH = ""


def load_reviewed_routing_release_dependencies():
    """Load only the fixed authority provider linked by the release builder."""

    from gateway.research_lab.attested_routing_release_authorities import (
        load_reviewed_routing_release_authority_sources,
    )

    return build_reviewed_routing_release_dependencies(
        load_reviewed_routing_release_authority_sources(),
        expected_protected_workflow_manifest_hash=(
            EXPECTED_PROTECTED_WORKFLOW_MANIFEST_HASH
        ),
    )


__all__ = [
    "EXPECTED_PROTECTED_WORKFLOW_MANIFEST_HASH",
    "RELEASE_AUTHORITY_PROVIDER_MODULE",
    "RELEASE_MODULE_SCHEMA",
    "load_reviewed_routing_release_dependencies",
]
