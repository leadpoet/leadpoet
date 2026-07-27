from gateway.research_lab.config import (
    DEFAULT_PRIVATE_BUILD_CMD,
    DEFAULT_PRIVATE_TEST_CMD,
)
from research_lab.auto_research_prompt import coerce_component_registry


def test_private_model_commands_require_routing_adapter_v3() -> None:
    assert "sourcing-model-research-lab-adapter:v3" in DEFAULT_PRIVATE_TEST_CMD
    assert "sourcing-model-components:v2" in DEFAULT_PRIVATE_TEST_CMD
    assert "routing-compiler-v1" in DEFAULT_PRIVATE_TEST_CMD
    assert 'research_lab_adapter.adapter_metadata()' in DEFAULT_PRIVATE_BUILD_CMD
    assert 'runtime_metadata["component_registry_version"]' in DEFAULT_PRIVATE_BUILD_CMD
    assert 'runtime_metadata["adapter_version"]' in DEFAULT_PRIVATE_BUILD_CMD
    assert '"sourcing-model-components:v1"' not in DEFAULT_PRIVATE_BUILD_CMD
    assert '"sourcing-model-research-lab-adapter:v1"' not in DEFAULT_PRIVATE_BUILD_CMD


def test_compact_registry_fallback_matches_current_model_contract() -> None:
    registry = coerce_component_registry(
        {
            "component_registry": {
                "source_router": {
                    "purpose": "Route sourcing.",
                    "allowed_patch_types": ["STRATEGY_SWAP"],
                    "strategy_options": ["news", "company_site"],
                }
            }
        }
    )
    assert registry.manifest_version == "sourcing-model-components:v2"
    assert registry.champion_base == "sourcing-model-research-lab-adapter:v3"
