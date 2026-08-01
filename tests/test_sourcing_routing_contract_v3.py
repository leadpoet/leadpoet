import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gateway.research_lab.config import (
    DEFAULT_PRIVATE_BUILD_CMD,
    DEFAULT_PRIVATE_TEST_CMD,
)
from research_lab.auto_research_prompt import coerce_component_registry
from research_lab.sourcing_model_contract_check import reviewed_consumer_snapshots
from tests.private_model_artifact_fixtures import install_reviewed_consumer_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_private_model_commands_require_routing_adapter_v3() -> None:
    assert "sourcing-model-research-lab-adapter:v3" in DEFAULT_PRIVATE_TEST_CMD
    assert "sourcing-model-components:v2" in DEFAULT_PRIVATE_TEST_CMD
    assert "routing-compiler-v2" in DEFAULT_PRIVATE_TEST_CMD
    assert "SubprocessPrivateModelRunner" in DEFAULT_PRIVATE_BUILD_CMD
    assert "build_local_private_artifact_manifest" in DEFAULT_PRIVATE_BUILD_CMD
    assert "RESEARCH_LAB_RUNTIME_SOURCE_ROOT" in DEFAULT_PRIVATE_BUILD_CMD
    assert "runtime_module_path.relative_to(runtime_source_root)" in DEFAULT_PRIVATE_BUILD_CMD
    assert 'runtime_metadata["component_registry_version"]' in DEFAULT_PRIVATE_BUILD_CMD
    assert 'runtime_metadata["adapter_version"]' in DEFAULT_PRIVATE_BUILD_CMD
    assert "RESEARCH_LAB_PRIVATE_MODEL_KMS_KEY_ID" in DEFAULT_PRIVATE_BUILD_CMD
    assert "RESEARCH_LAB_SCORE_BUNDLE_KMS_KEY_ID" not in DEFAULT_PRIVATE_BUILD_CMD
    assert "research-lab-candidate-manifest.XXXXXX" in DEFAULT_PRIVATE_BUILD_CMD
    assert "scripts/build_research_lab_manifest.py" not in DEFAULT_PRIVATE_BUILD_CMD
    assert "/tmp/research_lab_candidate_manifest_hash.txt" not in DEFAULT_PRIVATE_BUILD_CMD
    assert '"sourcing-model-components:v1"' not in DEFAULT_PRIVATE_BUILD_CMD
    assert '"sourcing-model-research-lab-adapter:v1"' not in DEFAULT_PRIVATE_BUILD_CMD


@pytest.mark.parametrize(
    "contract_id",
    tuple(sorted(reviewed_consumer_snapshots())),
)
def test_default_builder_emits_exact_reviewed_contract_pair(
    tmp_path: Path,
    contract_id: str,
) -> None:
    from tests.test_model_authority_v2 import _ready_adapter_metadata

    source = tmp_path / contract_id
    source.mkdir()
    install_reviewed_consumer_snapshot(source, contract_id=contract_id)
    metadata = _ready_adapter_metadata()
    metadata["scoring_adapter_version"] = "sourcing-model-score-adapter:v2"
    (source / "research_lab_adapter.py").write_text(
        "import json\n"
        f"_METADATA = json.loads({json.dumps(json.dumps(metadata))})\n"
        "def adapter_metadata():\n"
        "    return _METADATA\n"
        "def run_icp(icp, context):\n"
        "    return []\n",
        encoding="utf-8",
    )
    marker = (
        'python3 - "${OUTPUT_PATH}" "${IMAGE_DIGEST}" "${MANIFEST_URI}" '
        '"${SIGNATURE_URI}" "${COMMIT_SHA}" <<\'PY\'\n'
    )
    embedded_builder = DEFAULT_PRIVATE_BUILD_CMD.split(marker, 1)[1].split(
        "\nPY\n\nMANIFEST_HASH=",
        1,
    )[0]
    output = source / ".research_lab" / "candidate_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(output),
            (
                "493765492819.dkr.ecr.us-east-1.amazonaws.com/"
                "leadpoet/sourcing-model@sha256:" + "a" * 64
            ),
            "s3://artifacts/candidate.json",
            "s3://artifacts/candidate.sig.b64",
            "b" * 40,
        ],
        cwd=source,
        env={
            **os.environ,
            "PYTHONSAFEPATH": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(ROOT),
            "RESEARCH_LAB_RUNTIME_SOURCE_ROOT": str(ROOT),
        },
        input=embedded_builder,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    manifest = json.loads(output.read_text(encoding="utf-8"))
    snapshot = reviewed_consumer_snapshots()[contract_id]
    contract = snapshot["contract"]
    assert manifest["compatibility_contract"] == {
        "contract_id": contract_id,
        "path": str(contract["canonical_path"]),
        "sha256": str(snapshot["contract_sha256"]),
    }
    assert manifest["consumer_parity_fixtures"] == {
        "path": str(contract["parity_fixture_path"]),
        "sha256": str(snapshot["parity_sha256"]),
    }


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
