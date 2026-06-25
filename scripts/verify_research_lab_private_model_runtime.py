#!/usr/bin/env python3
"""Verify private model runtime bridge without private repo or network access."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.canonical import sha256_json  # noqa: E402
from research_lab.eval.artifacts import validate_private_model_artifact_manifest  # noqa: E402
from research_lab.eval.evaluator import _normalize_company_output  # noqa: E402
from research_lab.eval.private_runtime import (  # noqa: E402
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    PrivateModelAdapterSpec,
    PrivateModelRuntimeError,
    SubprocessPrivateModelRunner,
    build_local_private_artifact_manifest,
    canonicalize_private_model_icp,
    compute_private_source_tree_hash,
    ensure_private_model_outputs,
    load_private_artifact_manifest,
)
import research_lab.eval.private_runtime as private_runtime_module  # noqa: E402


def main() -> int:
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="research-lab-private-runtime-") as tmp:
        root = Path(tmp)
        adapter = root / "research_lab_adapter.py"
        adapter.write_text(
            """
def run_icp(icp, context):
    required = ("required_attribute", "intent_signal", "intent_category", "employee_count", "geography")
    if not all(icp.get(key) for key in required):
        return []
    if context.get("patch", {}).get("patch_type") == "STRATEGY_SWAP":
        icp = dict(icp)
        icp["intent_source"] = context["patch"].get("patch_doc", {}).get("strategy_option", "news")
    return [{
        "company_name": "Acme AI",
        "company_website": "https://acme.example",
        "company_linkedin": "https://www.linkedin.com/company/acme-ai",
        "industry": icp.get("industry", "Software"),
        "subindustry": "Sales Automation",
        "hq_country": icp.get("geography", "United States"),
        "employee_count": "51-200",
        "description": "AI sales automation platform",
        "intent": {
            "source": icp.get("intent_source", "news"),
            "url": "https://example.com/acme-funding",
            "date": "2026-06-01",
            "signal": "Acme AI raised a Series A to expand sales hiring."
        },
        "score": 82.5
    }]
""".strip()
            + "\n",
            encoding="utf-8",
        )

        runner = SubprocessPrivateModelRunner(PrivateModelAdapterSpec(source_path=root, timeout_seconds=30))
        research_lab_icp = {
            "icp_id": "research_lab_fixture",
            "industry": "Software",
            "sub_industry": "Sales Automation",
            "target_geography": "United States",
            "company_size": "51-200",
            "product_service": "AI sales automation platform",
            "intent_signals": ["Launched or announced a new product"],
        }
        canonical_icp = canonicalize_private_model_icp(research_lab_icp)
        if canonical_icp["geography"] != "United States":
            errors.append("canonical private ICP did not map target_geography to geography")
        if canonical_icp["employee_count"] != "51-200":
            errors.append("canonical private ICP did not map company_size to employee_count")
        if canonical_icp["intent_signal"] != "Launched or announced a new product":
            errors.append("canonical private ICP did not extract intent signal text")
        if canonical_icp["intent_category"] != "PRODUCT_LAUNCH":
            errors.append("canonical private ICP did not infer product-launch intent category")
        if not canonical_icp["required_attribute"].startswith("The company offers or provides"):
            errors.append("canonical private ICP did not derive required_attribute from product_service")

        out = runner(
            research_lab_icp,
            {"patch": {"patch_type": "STRATEGY_SWAP", "patch_doc": {"strategy_option": "job_listing"}}},
        )
        if not out or out[0]["intent"]["source"] != "job_listing":
            errors.append("subprocess private model runner did not return patched output")

        normalized = _normalize_company_output(out[0])
        if normalized["country"] != "United States":
            errors.append("private output hq_country did not normalize to country")
        if normalized["sub_industry"] != "Sales Automation":
            errors.append("private output subindustry did not normalize to sub_industry")
        if normalized["intent_signals"][0]["source"] != "job_board":
            errors.append("private output intent source did not normalize")
        if normalized["intent_signals"][0]["matched_icp_signal"] != 0:
            errors.append("private output intent signal did not default matched ICP signal")

        try:
            ensure_private_model_outputs([], context_label="baseline-test", require_non_empty=True)
            errors.append("empty private baseline output was accepted")
        except PrivateModelRuntimeError:
            pass
        try:
            empty_candidate = ensure_private_model_outputs(
                [],
                context_label="candidate-test",
                require_non_empty=False,
            )
            if empty_candidate != []:
                errors.append("empty candidate output did not round-trip as empty list")
        except PrivateModelRuntimeError:
            errors.append("empty candidate output was rejected")

        tree_hash_a = compute_private_source_tree_hash(root)
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "ignored.pyc").write_bytes(b"ignored")
        tree_hash_b = compute_private_source_tree_hash(root)
        if tree_hash_a != tree_hash_b:
            errors.append("source tree hash included ignored pycache files")

        manifest_payload = {
            "component_registry_version": "sourcing-model-components:v1",
            "scoring_adapter_version": "qualification-company-scorer:v1",
        }
        manifest = build_local_private_artifact_manifest(
            source_path=root,
            git_commit_sha="abcdef1234567890",
            image_digest="123456789012.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:" + "2" * 64,
            manifest_uri="s3://leadpoet-private-model-artifacts/manifests/sourcing-model.json",
            signature_ref="kms-signature:research-lab-eval:test",
            component_registry_version=manifest_payload["component_registry_version"],
            scoring_adapter_version=manifest_payload["scoring_adapter_version"],
            build_id="test-build",
            config_payload=manifest_payload,
        )
        if validate_private_model_artifact_manifest(manifest):
            errors.append("generated private model artifact manifest did not validate")
        if manifest["manifest_hash"] != sha256_json({k: v for k, v in manifest.items() if k != "manifest_hash"}):
            errors.append("manifest hash mismatch")
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        loaded_manifest = load_private_artifact_manifest(str(manifest_path))
        if loaded_manifest["manifest_hash"] != manifest["manifest_hash"]:
            errors.append("local private artifact manifest loader changed the manifest")

        original_run = private_runtime_module.subprocess.run

        class _Completed:
            returncode = 0
            stdout = '[{"raw_secret":"should-fail"}]'
            stderr = ""

        def _fake_run(*_args, **_kwargs):
            return _Completed()

        private_runtime_module.subprocess.run = _fake_run
        try:
            try:
                DockerPrivateModelRunner(
                    DockerPrivateModelSpec(
                        image_digest="123456789012.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:" + "9" * 64,
                        pull_before_run=False,
                        timeout_seconds=30,
                    )
                )(research_lab_icp, {})
                errors.append("docker private model runner accepted raw secret output")
            except PrivateModelRuntimeError:
                pass
        finally:
            private_runtime_module.subprocess.run = original_run

        secret_adapter = root / "secret_adapter.py"
        secret_adapter.write_text(
            "def run_icp(icp, context):\n    return [{'raw_secret': 'sk-or-should-fail'}]\n",
            encoding="utf-8",
        )
        try:
            SubprocessPrivateModelRunner(
                PrivateModelAdapterSpec(source_path=root, module_name="secret_adapter", timeout_seconds=30)
            )(research_lab_icp, {})
            errors.append("subprocess private model runner accepted raw secret output")
        except PrivateModelRuntimeError:
            pass

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Research Lab private model runtime bridge verified: subprocess adapter, normalization, manifest hash, secret rejection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
