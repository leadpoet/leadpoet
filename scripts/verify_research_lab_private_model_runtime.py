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
    PrivateModelAdapterSpec,
    PrivateModelRuntimeError,
    SubprocessPrivateModelRunner,
    build_local_private_artifact_manifest,
    compute_private_source_tree_hash,
)


def main() -> int:
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="research-lab-private-runtime-") as tmp:
        root = Path(tmp)
        adapter = root / "research_lab_adapter.py"
        adapter.write_text(
            """
def run_icp(icp, context):
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
        out = runner(
            {"industry": "Software", "geography": "United States"},
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

        secret_adapter = root / "secret_adapter.py"
        secret_adapter.write_text(
            "def run_icp(icp, context):\n    return [{'raw_secret': 'sk-or-should-fail'}]\n",
            encoding="utf-8",
        )
        try:
            SubprocessPrivateModelRunner(
                PrivateModelAdapterSpec(source_path=root, module_name="secret_adapter", timeout_seconds=30)
            )({}, {})
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
