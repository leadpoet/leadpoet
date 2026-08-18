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
    EXPECTED_CONSUMER_CONTRACT_ID,
    PrivateModelAdapterSpec,
    PrivateModelRuntimeError,
    SubprocessPrivateModelRunner,
    canonicalize_private_model_icp,
    compute_private_source_tree_hash,
    employee_count_buckets_for_icp,
    ensure_private_model_outputs,
    load_private_artifact_manifest,
)
import research_lab.eval.private_runtime as private_runtime_module  # noqa: E402
from tests.private_model_artifact_fixtures import (  # noqa: E402
    build_private_artifact_with_adapted_source_admission,
    install_reviewed_consumer_snapshot,
)


def _routing_receipt_line(runtime_cap_seconds: float) -> str:
    receipt = {
        "runtime_cap_seconds": runtime_cap_seconds,
        "capability_contract": {
            "host_registered": ["deadline", "emit", "probe_origin", "resolve_host"],
        },
        "industry_taxonomy": {"taxonomy_content_hash": "sha256:" + "a" * 64},
        "firmographic_discovery": {"plan": {"target": 5}},
        "branches": [
            {
                "source": "company_site",
                "compiled_source": "company_site",
                "source_override": False,
                "route_tool_ids": ["intent.company_site"],
                "route_sources": ["company_site"],
                "route_plan_sha256": "b" * 64,
                "route_policy_sha256": "c" * 64,
                "route_catalog_sha256": "d" * 64,
                "route_context_sha256": "e" * 64,
            }
        ],
    }
    return "sourcing_branch_receipt " + json.dumps(receipt)


def main() -> int:
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="research-lab-private-runtime-") as tmp:
        root = Path(tmp)
        adapter = root / "research_lab_adapter.py"
        fixture_entrypoint = """
def run_icp(icp, context):
    print("diagnostic line that must not corrupt adapter JSON")
    required = ("required_attribute", "intent_signal", "intent_category", "employee_count", "geography")
    if not all(icp.get(key) for key in required):
        return []
    if context.get("patch", {}).get("patch_type") == "STRATEGY_SWAP":
        icp = dict(icp)
        icp["intent_source"] = context["patch"].get("patch_doc", {}).get("strategy_option", "news")
    compiled_source = "company_site"
    selected_source = icp.get("intent_source", compiled_source)
    receipt = {
        "runtime_cap_seconds": context["runtime_options"]["runtime_cap_seconds"],
        "capability_contract": {
            "host_registered": ["deadline", "emit", "probe_origin", "resolve_host"],
        },
        "industry_taxonomy": {"taxonomy_content_hash": "sha256:" + "a" * 64},
        "firmographic_discovery": {"plan": {"target": 5}},
        "branches": [
            {
                "source": selected_source,
                "compiled_source": compiled_source,
                "source_override": selected_source != compiled_source,
                "route_tool_ids": ["intent.company_site"],
                "route_sources": [compiled_source],
                "route_plan_sha256": "b" * 64,
                "route_policy_sha256": "c" * 64,
                "route_catalog_sha256": "d" * 64,
                "route_context_sha256": "e" * 64,
            }
        ],
    }
    sys.stderr.write("sourcing_branch_receipt " + json.dumps(receipt) + "\\n")
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
""".strip() + "\n"
        adapter.write_text(
            "import json\nimport sys\n\n" + fixture_entrypoint,
            encoding="utf-8",
        )
        install_reviewed_consumer_snapshot(
            root,
            contract_id=EXPECTED_CONSUMER_CONTRACT_ID,
        )

        runner = SubprocessPrivateModelRunner(PrivateModelAdapterSpec(source_path=root, timeout_seconds=30))
        research_lab_icp = {
            "icp_id": "research_lab_fixture",
            "industry": "Software",
            "sub_industry": "Sales Automation",
            "target_geography": "United States",
            "employee_count": ["11-50", "51-200", "201-500"],
            "product_service": "AI sales automation platform",
            "intent_signals": ["Launched or announced a new product"],
        }
        legacy_icp = dict(research_lab_icp)
        legacy_icp["employee_count"] = "51-200"
        canonical_icp = canonicalize_private_model_icp(legacy_icp)
        if canonical_icp["geography"] != "United States":
            errors.append("canonical private ICP did not map target_geography to geography")
        if canonical_icp["employee_count"] != "51-200":
            errors.append("canonical private ICP did not keep the primary employee_count")
        canonical_multi = canonicalize_private_model_icp(research_lab_icp)
        if canonical_multi["employee_count"] != ["11-50", "51-200", "201-500"]:
            errors.append("canonical private ICP did not pass multi-band employee_count to adapter")
        if "employee_count_buckets" in canonical_multi:
            errors.append("canonical private ICP leaked gateway-only employee_count_buckets")
        if employee_count_buckets_for_icp(research_lab_icp) != ["11-50", "51-200", "201-500"]:
            errors.append("list-valued employee_count did not normalize")
        legacy_bucket_icp = {**legacy_icp, "employee_count_buckets": ["11-50", "51-200", "201-500"]}
        if employee_count_buckets_for_icp(legacy_bucket_icp) != ["11-50", "51-200", "201-500"]:
            errors.append("legacy employee_count_buckets did not normalize")
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
        (root / "__pycache__").mkdir(exist_ok=True)
        (root / "__pycache__" / "ignored.pyc").write_bytes(b"ignored")
        tree_hash_b = compute_private_source_tree_hash(root)
        if tree_hash_a != tree_hash_b:
            errors.append("source tree hash included ignored pycache files")

        manifest_payload = {
            "component_registry_version": "sourcing-model-components:v2",
            "scoring_adapter_version": "qualification-company-scorer:v1",
        }
        manifest = build_private_artifact_with_adapted_source_admission(
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
            stdout = 'debug line before JSON\n[{"raw_secret":"should-fail"}]'
            stderr = _routing_receipt_line(27.0)

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

        captured_commands: list[list[str]] = []

        class _CommandCompleted:
            returncode = 0
            stdout = "[]"
            stderr = _routing_receipt_line(27.0)

        def _fake_command_run(command, *_args, **_kwargs):
            captured_commands.append(list(command))
            return _CommandCompleted()

        private_runtime_module.subprocess.run = _fake_command_run
        try:
            DockerPrivateModelRunner(
                DockerPrivateModelSpec(
                    image_digest="123456789012.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:" + "7" * 64,
                    pull_before_run=False,
                    timeout_seconds=30,
                )
            )(research_lab_icp, {})
        finally:
            private_runtime_module.subprocess.run = original_run
        docker_run_command = next(
            (
                command
                for command in captured_commands
                if len(command) >= 2
                and Path(command[0]).name == "docker"
                and command[1] == "run"
            ),
            (),
        )
        if (
            "--platform" not in docker_run_command
            or "linux/amd64" not in docker_run_command
        ):
            errors.append("docker private model runner did not pin linux/amd64 platform")

        class _ProviderErrorCompleted:
            returncode = 0
            stdout = "[]"
            stderr = "\n".join(
                (
                    _routing_receipt_line(27.0),
                    "research_lab_private_runtime_provider_error HTTPError: HTTP Error 401: Unauthorized",
                )
            )

        def _fake_provider_error_run(*_args, **_kwargs):
            return _ProviderErrorCompleted()

        private_runtime_module.subprocess.run = _fake_provider_error_run
        try:
            try:
                DockerPrivateModelRunner(
                    DockerPrivateModelSpec(
                        image_digest="123456789012.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:" + "8" * 64,
                        pull_before_run=False,
                        timeout_seconds=30,
                    )
                )(research_lab_icp, {})
                errors.append("docker private model runner accepted empty output with provider error")
            except PrivateModelRuntimeError as exc:
                if "provider-backed sourcing failed" not in str(exc):
                    errors.append("provider error did not produce a clear runtime failure")
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
