"""Tests for the code_editing / code_build fixes from fableanalysis.md.

Covers: bug #18 (forbidden-term scan on added lines only), bug #19 (value-level
secret redaction), bug #21 prompt/parser side (verdict synonyms), bug #22
(novelty semantic key matches the worker's stored shape), bug #29(a) (real
head sha recorded), bug #30 (infra-vs-candidate build failure classification).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
from types import SimpleNamespace

import pytest

from gateway.research_lab import code_build
from gateway.research_lab.code_build import (
    CodeEditBuildError,
    CodeEditPrivateTestError,
    ParentImageSourceContext,
    validate_private_code_edit_diff_artifact,
)
from research_lab import code_editing
from research_lab.canonical import sha256_json
from research_lab.code_editing import CodeEditDraft


def _draft(**overrides):
    payload = dict(
        failure_mode="weak recall",
        mechanism="widen fan-out",
        expected_improvement="+2 companies",
        risk="slower",
        lane="provider",
        target_files=("sourcing_model.py",),
        unified_diff="--- a/sourcing_model.py\n+++ b/sourcing_model.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
        redacted_summary="widen provider fan-out for recall",
        test_plan="smoke",
        rollback_plan="revert",
    )
    payload.update(overrides)
    return CodeEditDraft(**payload)


class _SourceAddMaterializationConfig:
    def code_edit_allowed_path_prefixes(self):
        return ("sourcing_model/",)

    def code_edit_allowed_exact_paths(self):
        return ()

    def code_edit_allowed_suffixes(self):
        return (".py", ".json")


def _source_add_materialization_fixture(
    tmp_path: Path,
    *,
    omit_evaluator: str = "",
    contract_digest: str | None = None,
    bind_contract_hash: bool = True,
    generator_adds_secret: bool = False,
):
    source_root = tmp_path / "source"
    routing_dir = source_root / "sourcing_model" / "routing"
    routing_dir.mkdir(parents=True)
    (source_root / "sourcing_model" / "__init__.py").write_text("", encoding="utf-8")
    (routing_dir / "__init__.py").write_text("", encoding="utf-8")
    runtime_path = routing_dir / "runtime.py"
    runtime_source = (
        "SOURCE_ADD_ROUTING_REGISTRATIONS = (\n"
        "    SourceAddRoutingRegistration(provider_id='existing'),\n"
        ")\n"
    )
    runtime_path.write_text(runtime_source, encoding="utf-8")
    semantic_registry_path = (
        source_root / code_build._SOURCE_ADD_SEMANTIC_REGISTRY_PATH
    )
    semantic_registry_path.write_text(
        '{"has_builtwith":false}\n',
        encoding="utf-8",
    )
    scripts_dir = source_root / "scripts"
    scripts_dir.mkdir()
    semantic_registry_builder_source = (
        "import argparse\n"
        "import json\n"
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--write', action='store_true')\n"
        "args = parser.parse_args()\n"
        "if not args.write:\n"
        "    raise SystemExit(2)\n"
        "runtime = (ROOT / 'sourcing_model' / 'routing' / 'runtime.py').read_text(encoding='utf-8')\n"
        "document = {'credential': 'absent', 'has_builtwith': 'builtwith_trends' in runtime}\n"
    )
    if generator_adds_secret:
        semantic_registry_builder_source += (
            "if document['has_builtwith']:\n"
            "    document['generated_marker'] = 'sk-' + 'x' * 24\n"
        )
    semantic_registry_builder_source += (
        "(ROOT / 'sourcing_model' / 'production_semantic_registry.json').write_text(\n"
        "    json.dumps(document, sort_keys=True, separators=(',', ':')) + '\\n',\n"
        "    encoding='utf-8',\n"
        ")\n"
    )
    (
        scripts_dir
        / Path(code_build._SOURCE_ADD_SEMANTIC_REGISTRY_BUILDER_PATH).name
    ).write_text(
        semantic_registry_builder_source,
        encoding="utf-8",
    )
    fixture_path = source_root / code_build._SOURCE_ADD_PARITY_FIXTURE_PATH
    custody_metadata = {
        "required_dispatch_vector_kinds": ["start"],
        "kind_ids": {"start": "start-request"},
        "domain": "fixture-custody",
        "self_hash_fields": {"start": "request_sha256"},
        "dispatch_vector_builder_id": "fixture-builder",
        "custody_fields": ["request_sha256"],
    }

    def custody_sha256(kind: str, payload: dict) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(kind.encode("utf-8") + b"\0" + encoded).hexdigest()

    custody_payload = {
        "route_sha256": hashlib.sha256(runtime_source.encode("utf-8")).hexdigest()
    }
    custody_hash = custody_sha256("start", custody_payload)
    custody_envelope = {**custody_payload, "request_sha256": custody_hash}
    custody_bytes = json.dumps(
        custody_envelope,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    custody_vector = {
        "fixture_id": "start-dispatch",
        "kind": "start",
        "kind_id": "start-request",
        "domain": "fixture-custody",
        "domain_terminator_hex": "00",
        "self_hash_field": "request_sha256",
        "builder_id": "fixture-builder",
        "custody_sha256": custody_hash,
        "persisted_json_hex": custody_bytes.hex(),
        "persisted_bytes_sha256": hashlib.sha256(custody_bytes).hexdigest(),
        "persisted_bytes_length": len(custody_bytes),
    }
    fixture_document = {
        "fixture_id": "source-add",
        "adversarial_request": {"access_token": "synthetic-secret"},
        "model_runner_custody_v3": {
            **custody_metadata,
            "dispatch_vector_count": 1,
            "dispatch_vectors": [custody_vector],
        },
        "expected_model_runner_custody_v3_projection": {
            "dispatch_vectors": [
                {
                    "fixture_id": "start-dispatch",
                    "custody_sha256": custody_hash,
                    "persisted_bytes_sha256": hashlib.sha256(custody_bytes).hexdigest(),
                }
            ]
        },
    }
    fixture_path.write_text(
        json.dumps(fixture_document, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    fixture_digest = hashlib.sha256(
        json.dumps(
            fixture_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    evaluator_names = [
        evaluator
        for _, evaluator in code_build._SOURCE_ADD_PARITY_PROJECTION_EVALUATORS
        if evaluator != omit_evaluator
    ]
    expected_pairs = list(code_build._SOURCE_ADD_PARITY_PROJECTION_EVALUATORS)
    parity_path = source_root / code_build._SOURCE_ADD_PARITY_MODULE_PATH
    parity_path.write_text(
        """
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
"""
        + f"{code_build._SOURCE_ADD_PARITY_HASH_CONSTANT} = {fixture_digest!r}\n\n"
        + f"FAKE_CUSTODY_METADATA = {custody_metadata!r}\n\n"
        + """
def _fixture_document_sha256(document):
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

def _projection(name, fixtures):
    runtime = (ROOT / "routing" / "runtime.py").read_text(encoding="utf-8")
    return {"evaluator": name, "has_builtwith": "builtwith_trends" in runtime}

def _runtime_identity_sha256():
    runtime = (ROOT / "routing" / "runtime.py").read_bytes()
    return hashlib.sha256(runtime).hexdigest()

_INTENT_SOURCE_CALL_COUNT = 0

def _evaluate_intent_source_parity_cases(fixtures):
    global _INTENT_SOURCE_CALL_COUNT
    _INTENT_SOURCE_CALL_COUNT += 1
    return [{
        **_projection("intent_source", fixtures),
        "outcome": (
            "accepted" if _INTENT_SOURCE_CALL_COUNT == 1 else "unavailable"
        ),
    }]

def model_runner_custody_metadata():
    return FAKE_CUSTODY_METADATA

def custody_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

def custody_envelope_sha256(kind, payload):
    return hashlib.sha256(
        kind.encode("utf-8") + b"\\0" + custody_json_bytes(payload)
    ).hexdigest()

def model_runner_custody_parity_vectors():
    runtime = (ROOT / "routing" / "runtime.py").read_bytes()
    payload = {"route_sha256": hashlib.sha256(runtime).hexdigest()}
    custody_hash = custody_envelope_sha256("start", payload)
    return [{
        "kind": "start",
        "envelope": {**payload, "request_sha256": custody_hash},
    }]

def evaluate_model_runner_custody_v3_parity(fixtures):
    section = fixtures["model_runner_custody_v3"]
    vector = section["dispatch_vectors"][0]
    generated = model_runner_custody_parity_vectors()[0]["envelope"]
    persisted = custody_json_bytes(generated)
    if bytes.fromhex(vector["persisted_json_hex"]) != persisted:
        raise ValueError("custody persisted envelope differs")
    return {
        "dispatch_vectors": [{
            "fixture_id": vector["fixture_id"],
            "custody_sha256": vector["custody_sha256"],
            "persisted_bytes_sha256": vector["persisted_bytes_sha256"],
        }]
    }

"""
        + "\n".join(
            f"def {name}(fixtures):\n    return _projection({name!r}, fixtures)\n"
            for name in evaluator_names
        )
        + f"""
def verify_expected_projections(fixtures=None):
    if fixtures is None:
        fixtures = json.loads(
            (ROOT / "consumer_parity_fixtures.json").read_text(encoding="utf-8")
        )
    if _fixture_document_sha256(fixtures) != {code_build._SOURCE_ADD_PARITY_HASH_CONSTANT}:
        raise ValueError("fixture document differs")
    for expected_key, evaluator_name in {expected_pairs!r}:
        evaluator = globals()[evaluator_name]
        if fixtures.get(expected_key) != evaluator(fixtures):
            raise ValueError("projection differs: " + expected_key)
    expected_source = {{
        "contract_sha256": "fixture-contract",
        "runtime_intent_routing_release_identity_sha256": _runtime_identity_sha256(),
        "cases": _evaluate_intent_source_parity_cases(fixtures),
    }}
    if fixtures.get("expected_intent_source_evaluation_projection") != expected_source:
        raise ValueError("intent source projection differs")
    custody = evaluate_model_runner_custody_v3_parity(fixtures)
    if fixtures.get("expected_model_runner_custody_v3_projection") != custody:
        raise ValueError("model runner custody projection differs")
    return fixtures["expected_projections"]
""",
        encoding="utf-8",
    )
    contract_path = source_root / code_build._SOURCE_ADD_CONSUMER_CONTRACT_PATH
    contract_path.write_text(
        json.dumps(
            {
                "exact_constants": (
                    {
                        code_build._SOURCE_ADD_PARITY_MODULE_PATH: {
                            code_build._SOURCE_ADD_PARITY_HASH_CONSTANT: (
                                contract_digest or fixture_digest
                            ),
                        }
                    }
                    if bind_contract_hash
                    else {"research_lab_adapter.py": {}}
                )
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (source_root / "sourcing_model" / "intent_source_evaluation.py").write_text(
        "def intent_source_evaluation_contract_identity():\n"
        "    return {'contract_sha256': 'fixture-contract'}\n",
        encoding="utf-8",
    )
    source_context = ParentImageSourceContext(
        source_root=source_root,
        source_mode="test",
        parent_image_digest_hash="sha256:" + "1" * 64,
        source_tree_hash="sha256:" + "2" * 64,
        top_level_paths=("sourcing_model",),
        editable_files=(
            code_build._SOURCE_ADD_CONSUMER_CONTRACT_PATH,
            code_build._SOURCE_ADD_ROUTING_RUNTIME_PATH,
            code_build._SOURCE_ADD_PARITY_FIXTURE_PATH,
            code_build._SOURCE_ADD_PARITY_MODULE_PATH,
            code_build._SOURCE_ADD_SEMANTIC_REGISTRY_PATH,
        ),
        file_previews=(),
    )
    # Real model diffs may omit the enclosing tuple name from the hunk label
    # and context while inserting a registration inside the tuple.
    unified_diff = (
        "diff --git a/sourcing_model/routing/runtime.py b/sourcing_model/routing/runtime.py\n"
        "--- a/sourcing_model/routing/runtime.py\n"
        "+++ b/sourcing_model/routing/runtime.py\n"
        "@@ -2,2 +2,3 @@\n"
        "     SourceAddRoutingRegistration(provider_id='existing'),\n"
        "+    SourceAddRoutingRegistration(provider_id='builtwith_trends'),\n"
        " )\n"
    )
    draft = _draft(
        target_files=(code_build._SOURCE_ADD_ROUTING_RUNTIME_PATH,),
        unified_diff=unified_diff,
    )
    return source_context, draft


def test_source_add_materialization_binds_verified_parity_fixture(
    tmp_path,
    monkeypatch,
):
    source_context, draft = _source_add_materialization_fixture(tmp_path)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "must-not-reach-candidate-tool")
    builder = code_build.CodeEditCandidateBuilder(_SourceAddMaterializationConfig())

    materialized = builder.materialize_source_add_derived_artifacts(
        draft=draft,
        source_context=source_context,
    )

    assert materialized.target_files == (
        code_build._SOURCE_ADD_CONSUMER_CONTRACT_PATH,
        code_build._SOURCE_ADD_PARITY_MODULE_PATH,
        code_build._SOURCE_ADD_PARITY_FIXTURE_PATH,
        code_build._SOURCE_ADD_SEMANTIC_REGISTRY_PATH,
        code_build._SOURCE_ADD_ROUTING_RUNTIME_PATH,
    )
    for path in code_build._SOURCE_ADD_DERIVED_ARTIFACT_PATHS:
        assert path in materialized.unified_diff
    assert "builtwith_trends" in materialized.unified_diff
    added_lines = "\n".join(
        line[1:]
        for line in materialized.unified_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    contract_hash = re.search(
        rf'"{code_build._SOURCE_ADD_PARITY_HASH_CONSTANT}": "([0-9a-f]{{64}})"',
        added_lines,
    )
    assert contract_hash is not None
    assert added_lines.count(contract_hash.group(1)) == 2
    assert materialized == builder.materialize_source_add_derived_artifacts(
        draft=draft,
        source_context=source_context,
    )


def test_source_add_materialization_requires_model_semantic_registry_builder(
    tmp_path,
):
    source_context, draft = _source_add_materialization_fixture(tmp_path)
    (
        source_context.source_root
        / code_build._SOURCE_ADD_SEMANTIC_REGISTRY_BUILDER_PATH
    ).unlink()
    builder = code_build.CodeEditCandidateBuilder(_SourceAddMaterializationConfig())

    with pytest.raises(
        CodeEditPrivateTestError,
        match="parity projection materialization failed",
    ) as exc_info:
        builder.materialize_source_add_derived_artifacts(
            draft=draft,
            source_context=source_context,
        )

    assert exc_info.value.failure_stage == "candidate_derived_artifact_failed"


def test_source_add_materialization_preserves_unbound_consumer_contract(tmp_path):
    source_context, draft = _source_add_materialization_fixture(
        tmp_path,
        bind_contract_hash=False,
    )
    contract_path = (
        source_context.source_root
        / code_build._SOURCE_ADD_CONSUMER_CONTRACT_PATH
    )
    original_contract = contract_path.read_bytes()
    builder = code_build.CodeEditCandidateBuilder(_SourceAddMaterializationConfig())

    materialized = builder.materialize_source_add_derived_artifacts(
        draft=draft,
        source_context=source_context,
    )

    assert code_build._SOURCE_ADD_CONSUMER_CONTRACT_PATH not in (
        materialized.target_files
    )
    assert contract_path.read_bytes() == original_contract
    assert code_build._SOURCE_ADD_REQUIRED_DERIVED_ARTIFACT_PATHS.issubset(
        materialized.target_files
    )


def test_source_add_materialization_rejects_new_generated_secret(tmp_path):
    source_context, draft = _source_add_materialization_fixture(
        tmp_path,
        generator_adds_secret=True,
    )
    builder = code_build.CodeEditCandidateBuilder(_SourceAddMaterializationConfig())

    with pytest.raises(
        CodeEditPrivateTestError,
        match="generated artifacts contain secret-shaped material",
    ) as exc_info:
        builder.materialize_source_add_derived_artifacts(
            draft=draft,
            source_context=source_context,
        )

    assert exc_info.value.failure_stage == "candidate_derived_artifact_failed"


def test_source_add_materialization_fails_closed_when_projection_api_differs(
    tmp_path,
):
    missing = code_build._SOURCE_ADD_PARITY_PROJECTION_EVALUATORS[0][1]
    source_context, draft = _source_add_materialization_fixture(
        tmp_path,
        omit_evaluator=missing,
    )
    builder = code_build.CodeEditCandidateBuilder(_SourceAddMaterializationConfig())

    with pytest.raises(
        CodeEditPrivateTestError,
        match="parity projection materialization failed",
    ) as exc_info:
        builder.materialize_source_add_derived_artifacts(
            draft=draft,
            source_context=source_context,
        )

    assert exc_info.value.failure_stage == "candidate_derived_artifact_failed"


def test_source_add_materialization_fails_closed_on_stale_contract_hash(tmp_path):
    source_context, draft = _source_add_materialization_fixture(
        tmp_path,
        contract_digest="0" * 64,
    )
    builder = code_build.CodeEditCandidateBuilder(_SourceAddMaterializationConfig())

    with pytest.raises(
        CodeEditPrivateTestError,
        match="parity projection materialization failed",
    ) as exc_info:
        builder.materialize_source_add_derived_artifacts(
            draft=draft,
            source_context=source_context,
        )

    assert exc_info.value.failure_stage == "candidate_derived_artifact_failed"


def test_non_source_add_draft_does_not_materialize_parity(tmp_path):
    source_context, _ = _source_add_materialization_fixture(tmp_path)
    draft = _draft()
    builder = code_build.CodeEditCandidateBuilder(_SourceAddMaterializationConfig())

    assert builder.materialize_source_add_derived_artifacts(
        draft=draft,
        source_context=source_context,
    ) is draft


def _private_source_diff_fixture():
    unified_diff = (
        "diff --git a/sourcing_model.py b/sourcing_model.py\n"
        "--- a/sourcing_model.py\n"
        "+++ b/sourcing_model.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    artifact_payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_code_edit_source_diff",
        "run_id": "run-bound-source-diff",
        "candidate_index": 3,
        "parent_artifact_hash": "sha256:" + "1" * 64,
        "parent_manifest_hash": "sha256:" + "2" * 64,
        "source_diff_hash": sha256_json({"unified_diff": unified_diff}),
        "target_files": ["sourcing_model.py"],
        "unified_diff": unified_diff,
        "draft_hash": "sha256:" + "3" * 64,
    }
    artifact = {
        **artifact_payload,
        "artifact_hash": sha256_json(artifact_payload),
    }
    build_payload = {
        "parent_artifact_hash": artifact["parent_artifact_hash"],
        "parent_manifest_hash": artifact["parent_manifest_hash"],
        "source_diff_hash": artifact["source_diff_hash"],
        "source_diff_artifact_hash": artifact["artifact_hash"],
        "changed_files": ["sourcing_model.py"],
    }
    build_doc = {
        **build_payload,
        "build_doc_hash": sha256_json(build_payload),
    }
    return artifact, build_doc


def _rehash_private_source_diff_artifact(document):
    payload = {key: value for key, value in document.items() if key != "artifact_hash"}
    return {**payload, "artifact_hash": sha256_json(payload)}


def _rehash_candidate_build_doc(document):
    payload = {key: value for key, value in document.items() if key != "build_doc_hash"}
    return {**payload, "build_doc_hash": sha256_json(payload)}


def _candidate_patch_manifest_for(
    artifact, build_doc, *, candidate_model_artifact_hash="sha256:" + "4" * 64,
    candidate_model_manifest_hash="sha256:" + "5" * 64,
):
    payload = {
        "candidate_kind": "image_build",
        "patch_type": "IMAGE_BUILD",
        "target_component_id": "private_model_source_tree",
        "parent_artifact_hash": artifact["parent_artifact_hash"],
        "candidate_artifact_hash": candidate_model_artifact_hash,
        "candidate_model_manifest_hash": candidate_model_manifest_hash,
        "patch_payload_hash": artifact["source_diff_hash"],
        "candidate_source_diff_hash": artifact["source_diff_hash"],
        "candidate_build_doc_hash": build_doc["build_doc_hash"],
        "redacted_summary": "fixture",
        "validation_result": "passed",
        "patch_doc": {},
    }
    payload["patch_doc"] = {
        "target_files": list(artifact["target_files"]),
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


def test_private_source_diff_artifact_binds_full_build_authority():
    artifact, build_doc = _private_source_diff_fixture()

    assert validate_private_code_edit_diff_artifact(
        artifact,
        candidate_build_doc=build_doc,
        expected_source_diff_hash=artifact["source_diff_hash"],
        expected_parent_artifact_hash=artifact["parent_artifact_hash"],
        expected_run_id=artifact["run_id"],
    ) == artifact


def test_private_source_diff_rejects_hash_consistent_structural_git_metadata():
    artifact, build_doc = _private_source_diff_fixture()
    structural_diff = artifact["unified_diff"].replace(
        "--- a/sourcing_model.py\n",
        "old mode 100644\nnew mode 100755\n--- a/sourcing_model.py\n",
    )
    source_diff_hash = sha256_json({"unified_diff": structural_diff})
    structural_artifact = _rehash_private_source_diff_artifact(
        {
            **artifact,
            "unified_diff": structural_diff,
            "source_diff_hash": source_diff_hash,
        }
    )
    structural_build = _rehash_candidate_build_doc(
        {
            **build_doc,
            "source_diff_hash": source_diff_hash,
            "source_diff_artifact_hash": structural_artifact["artifact_hash"],
        }
    )

    with pytest.raises(CodeEditBuildError, match="not a content-only Git patch"):
        validate_private_code_edit_diff_artifact(
            structural_artifact,
            candidate_build_doc=structural_build,
            expected_source_diff_hash=source_diff_hash,
            expected_parent_artifact_hash=artifact["parent_artifact_hash"],
            expected_run_id=artifact["run_id"],
        )


def test_private_source_diff_accepts_only_declared_unhashed_worker_annotations():
    artifact, build_doc = _private_source_diff_fixture()
    annotated = {
        **build_doc,
        "loop_direction_plan_hash": "sha256:" + "6" * 64,
        "selected_path_id": "alternate-source-path",
        "plan_alignment": {"passes": True},
        "conditional_validation_policy": {"mode": "on"},
        "loop_node_id": "tree-node:" + "7" * 64,
        "loop_dev_score": 51.5,
        "loop_dev_score_version": "fixture-v1",
        "stale_parent_rebase": {"depth": 1},
    }
    patch_manifest = _candidate_patch_manifest_for(artifact, build_doc)

    assert validate_private_code_edit_diff_artifact(
        artifact,
        candidate_build_doc=annotated,
        candidate_patch_manifest=patch_manifest,
        expected_candidate_patch_hash=sha256_json(patch_manifest),
    ) == artifact

    with pytest.raises(
        CodeEditBuildError, match="candidate build document commitment differs"
    ):
        validate_private_code_edit_diff_artifact(
            artifact,
            candidate_build_doc={**annotated, "unreviewed_worker_annotation": True},
            candidate_patch_manifest=patch_manifest,
            expected_candidate_patch_hash=sha256_json(patch_manifest),
        )


def test_private_source_diff_rejects_build_hash_not_bound_by_patch_manifest():
    artifact, build_doc = _private_source_diff_fixture()
    patch_manifest = _candidate_patch_manifest_for(artifact, build_doc)
    rewritten_build = _rehash_candidate_build_doc(
        {**build_doc, "changed_files": ["different.py"]}
    )

    with pytest.raises(
        CodeEditBuildError, match="differs from patch manifest"
    ):
        validate_private_code_edit_diff_artifact(
            artifact,
            candidate_build_doc=rewritten_build,
            candidate_patch_manifest=patch_manifest,
            expected_candidate_patch_hash=sha256_json(patch_manifest),
        )


def test_private_source_diff_normalizes_authenticated_legacy_depth_two_targets():
    first_patch = (
        "diff --git a/sourcing_model.py b/sourcing_model.py\n"
        "--- a/sourcing_model.py\n"
        "+++ b/sourcing_model.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    incremental_patch = (
        "diff --git a/gateway/module.py b/gateway/module.py\n"
        "--- a/gateway/module.py\n"
        "+++ b/gateway/module.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )
    cumulative_patch = first_patch + incremental_patch
    source_diff_hash = sha256_json({"unified_diff": cumulative_patch})
    incremental_hash = sha256_json({"unified_diff": incremental_patch})
    parent_artifact_hash = "sha256:" + "1" * 64
    candidate_artifact_hash = "sha256:" + "4" * 64
    artifact_payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_code_edit_source_diff",
        "run_id": "run-legacy-depth-two",
        "candidate_index": 2,
        "parent_artifact_hash": parent_artifact_hash,
        "parent_manifest_hash": "sha256:" + "2" * 64,
        "source_diff_hash": source_diff_hash,
        "target_files": ["gateway/module.py"],
        "unified_diff": cumulative_patch,
        "draft_hash": "sha256:" + "3" * 64,
    }
    artifact = {
        **artifact_payload,
        "artifact_hash": sha256_json(artifact_payload),
    }
    composition = {
        "schema_version": "research_lab.git_tree_composition.v1",
        "incremental_source_diff_hash": incremental_hash,
        "cumulative_source_diff_hash": source_diff_hash,
        # This was populated from the incremental Git diff before the fix.
        "cumulative_changed_files": ["gateway/module.py"],
        "child_source_tree_hash": candidate_artifact_hash,
    }
    lineage = {
        "schema_version": "research_lab.git_tree_lineage.v1",
        "depth": 2,
        "root_artifact_hash": parent_artifact_hash,
        "incremental_source_diff_hash": incremental_hash,
        "cumulative_source_diff_hash": source_diff_hash,
        "composition": composition,
    }
    build_payload = {
        "parent_artifact_hash": parent_artifact_hash,
        "parent_manifest_hash": artifact["parent_manifest_hash"],
        "source_diff_hash": source_diff_hash,
        "source_diff_artifact_hash": artifact["artifact_hash"],
        "candidate_model_artifact_hash": candidate_artifact_hash,
        "changed_files": ["gateway/module.py", "sourcing_model.py"],
        "git_tree": lineage,
    }
    build_doc = {
        **build_payload,
        "build_doc_hash": sha256_json(build_payload),
    }
    patch_manifest = _candidate_patch_manifest_for(
        artifact,
        build_doc,
        candidate_model_artifact_hash=candidate_artifact_hash,
    )

    validated = validate_private_code_edit_diff_artifact(
        artifact,
        candidate_build_doc=build_doc,
        candidate_patch_manifest=patch_manifest,
        expected_candidate_patch_hash=sha256_json(patch_manifest),
        expected_source_diff_hash=source_diff_hash,
        expected_parent_artifact_hash=parent_artifact_hash,
        expected_run_id=artifact["run_id"],
    )
    assert validated["target_files"] == [
        "gateway/module.py",
        "sourcing_model.py",
    ]

    rewritten_lineage = {
        **lineage,
        "composition": {
            **composition,
            "incremental_changed_files": ["gateway/module.py"],
        },
    }
    rewritten_build = _rehash_candidate_build_doc(
        {**build_doc, "git_tree": rewritten_lineage}
    )
    rewritten_patch = _candidate_patch_manifest_for(
        artifact,
        rewritten_build,
        candidate_model_artifact_hash=candidate_artifact_hash,
    )
    with pytest.raises(CodeEditBuildError, match="target files differ"):
        validate_private_code_edit_diff_artifact(
            artifact,
            candidate_build_doc=rewritten_build,
            candidate_patch_manifest=rewritten_patch,
            expected_candidate_patch_hash=sha256_json(rewritten_patch),
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"unified_diff": "diff --git a/x.py b/x.py\n"}, "commitment differs"),
        ({"run_id": "another-run"}, "run identity differs"),
        ({"parent_artifact_hash": "sha256:" + "9" * 64}, "parent artifact differs"),
        ({"target_files": ["sourcing_model.py", "extra.py"]}, "target files differ"),
    ],
)
def test_private_source_diff_artifact_rejects_tampered_authority(mutation, error):
    artifact, build_doc = _private_source_diff_fixture()
    tampered = _rehash_private_source_diff_artifact({**artifact, **mutation})
    if mutation.keys() == {"unified_diff"}:
        # Preserve the original advertised artifact hash to model an S3 body
        # replacement at an already committed object URI.
        tampered["artifact_hash"] = artifact["artifact_hash"]
    elif "parent_artifact_hash" in mutation:
        build_doc = _rehash_candidate_build_doc(
            {
                **build_doc,
                "source_diff_artifact_hash": tampered["artifact_hash"],
                "parent_artifact_hash": tampered["parent_artifact_hash"],
            }
        )
    elif "target_files" in mutation:
        build_doc = _rehash_candidate_build_doc(
            {
                **build_doc,
                "source_diff_artifact_hash": tampered["artifact_hash"],
                "changed_files": list(tampered["target_files"]),
            }
        )

    with pytest.raises(CodeEditBuildError, match=error):
        validate_private_code_edit_diff_artifact(
            tampered,
            candidate_build_doc=build_doc,
            expected_source_diff_hash=artifact["source_diff_hash"],
            expected_parent_artifact_hash=artifact["parent_artifact_hash"],
            expected_run_id=artifact["run_id"],
        )


def test_private_source_diff_artifact_rejects_rewritten_build_document():
    artifact, build_doc = _private_source_diff_fixture()
    tampered_build = _rehash_candidate_build_doc(
        {**build_doc, "source_diff_artifact_hash": "sha256:" + "8" * 64}
    )
    with pytest.raises(CodeEditBuildError, match="hash differs from build"):
        validate_private_code_edit_diff_artifact(
            artifact,
            candidate_build_doc=tampered_build,
        )


# --- bug #18: forbidden terms scanned on added lines only ---


def _diff_with(context_line: str, added_line: str) -> str:
    return (
        "--- a/gateway/module.py\n"
        "+++ b/gateway/module.py\n"
        "@@ -1,3 +1,4 @@\n"
        f" {context_line}\n"
        "-old = 1\n"
        f"+{added_line}\n"
        " tail = 2\n"
    )


def test_forbidden_term_in_context_line_passes():
    diff = _diff_with("value = load(judge_prompt)", "new = 2")
    assert code_editing._contains_forbidden_material_diff_aware(diff) is False


def test_forbidden_term_in_added_line_rejects():
    diff = _diff_with("clean = 1", "leak = read('judge_prompt')")
    assert code_editing._contains_forbidden_material_diff_aware(diff) is True


def test_forbidden_policy_prose_in_added_comment_or_string_passes():
    comment_diff = _diff_with("clean = 1", "# do not use hidden ICP data")
    string_diff = _diff_with("clean = 1", 'policy = "do not use hidden ICP data"')
    policy_fields_diff = _diff_with(
        "clean = 1",
        'forbidden_fields = ["service_role", "hidden_icp"]',
    )
    assert code_editing._contains_forbidden_material_diff_aware(comment_diff) is False
    assert code_editing._contains_forbidden_material_diff_aware(string_diff) is False
    assert code_editing._contains_forbidden_material_diff_aware(policy_fields_diff) is False


def test_multiline_sensitive_environment_access_rejects():
    diff = (
        "diff --git a/gateway/module.py b/gateway/module.py\n"
        "--- a/gateway/module.py\n"
        "+++ b/gateway/module.py\n"
        "@@ -1 +1,4 @@\n"
        " clean = 1\n"
        "+value = os.environ[\n"
        "+    \"SUPABASE_SERVICE_ROLE_KEY\"\n"
        "+]\n"
    )
    assert code_editing._contains_forbidden_material_diff_aware(diff) is True


def test_secret_shaped_value_in_added_line_rejects():
    diff = _diff_with("clean = 1", 'token = "sk-or-v1-' + "x" * 24 + '"')
    assert code_editing._contains_forbidden_material_diff_aware(diff) is True


@pytest.mark.parametrize(
    "value",
    [
        "AKIA" + "A" * 16,
        "Bearer " + "token-value-" * 3,
        "eyJ" + "a" * 12 + ".eyJ" + "b" * 12 + "." + "c" * 12,
        "-----BEGIN PRIVATE KEY-----",
        "https://user:password@example.test/path",
    ],
)
def test_secret_shaped_values_reject(value):
    assert code_editing._contains_forbidden_material({"reason": value}) is True


def test_forbidden_term_in_removed_line_passes():
    diff = (
        "--- a/gateway/module.py\n"
        "+++ b/gateway/module.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-token = fetch('service_role')\n"
        "+token = fetch_public()\n"
    )
    assert code_editing._contains_forbidden_material_diff_aware(diff) is False


def test_forbidden_api_and_path_remain_blocked():
    api_draft = _draft(
        target_files=("gateway/module.py",),
        unified_diff=(
            "diff --git a/gateway/module.py b/gateway/module.py\n"
            "--- a/gateway/module.py\n"
            "+++ b/gateway/module.py\n"
            "@@ -1 +1,2 @@\n"
            " value = 1\n"
            "+subprocess.run(['unsafe'])\n"
        ),
    )
    with pytest.raises(ValueError, match="code_edit_disallowed_diff_pattern"):
        code_editing.validate_code_edit_draft(api_draft)

    path_draft = _draft(
        target_files=("gateway/.env",),
        unified_diff=(
            "diff --git a/gateway/.env b/gateway/.env\n"
            "--- a/gateway/.env\n"
            "+++ b/gateway/.env\n"
            "@@ -1 +1 @@\n"
            "-SAFE=1\n"
            "+SAFE=2\n"
        ),
    )
    with pytest.raises(ValueError, match="disallowed_repo_path"):
        code_editing.validate_code_edit_draft(path_draft)


def test_new_and_unread_source_paths_remain_blocked():
    builder = object.__new__(code_build.CodeEditCandidateBuilder)
    source_context = SimpleNamespace(editable_files=("sourcing_model/existing.py",))
    new_file_draft = _draft(
        target_files=("sourcing_model/new_file.py",),
        unified_diff=(
            "diff --git a/sourcing_model/new_file.py b/sourcing_model/new_file.py\n"
            "--- a/sourcing_model/new_file.py\n"
            "+++ b/sourcing_model/new_file.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        ),
    )
    assert builder.validate_draft_against_source_context(
        new_file_draft,
        source_context,
    ) == ["code_edit_path_not_in_extracted_source:sourcing_model/new_file.py"]

    existing_draft = _draft(
        target_files=("sourcing_model/existing.py",),
        unified_diff=(
            "diff --git a/sourcing_model/existing.py b/sourcing_model/existing.py\n"
            "--- a/sourcing_model/existing.py\n"
            "+++ b/sourcing_model/existing.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        ),
    )
    assert builder.validate_draft_against_source_context(
        existing_draft,
        source_context,
        read_paths=(),
        require_read=True,
    ) == ["code_edit_unread_source_file:sourcing_model/existing.py"]


def test_candidate_build_timeout_is_one_total_deadline(monkeypatch):
    builder = object.__new__(code_build.CodeEditCandidateBuilder)
    builder.config = SimpleNamespace(code_edit_build_timeout_seconds=30)
    observed = {}

    def fake_build_under_deadline(**_kwargs):
        observed["first"] = code_build._bounded_command_timeout(120)
        monkeypatch.setattr(
            code_build.time,
            "monotonic",
            lambda: observed["started_at"] + 6,
        )
        observed["second"] = code_build._bounded_command_timeout(120)
        return "built"

    observed["started_at"] = code_build.time.monotonic()
    monkeypatch.setattr(builder, "_build_under_deadline", fake_build_under_deadline)
    result = builder.build(
        draft=object(),
        parent_artifact=object(),
        run_id="run-deadline",
        candidate_index=0,
        timeout_seconds=10,
    )

    assert result == "built"
    assert 1 <= observed["first"] <= 10
    assert 1 <= observed["second"] <= 4
    assert not hasattr(code_build._BUILD_DEADLINE, "value")


def test_candidate_build_deadline_rejects_later_commands(monkeypatch):
    builder = object.__new__(code_build.CodeEditCandidateBuilder)
    builder.config = SimpleNamespace(code_edit_build_timeout_seconds=30)
    started_at = code_build.time.monotonic()

    def fake_build_under_deadline(**_kwargs):
        monkeypatch.setattr(
            code_build.time,
            "monotonic",
            lambda: started_at + 11,
        )
        code_build._bounded_command_timeout(120)

    monkeypatch.setattr(builder, "_build_under_deadline", fake_build_under_deadline)
    with pytest.raises(code_build.CodeEditBuildError, match="deadline exhausted"):
        builder.build(
            draft=object(),
            parent_artifact=object(),
            run_id="run-deadline",
            candidate_index=0,
            timeout_seconds=10,
        )
    assert not hasattr(code_build._BUILD_DEADLINE, "value")


def test_candidate_private_test_imports_exact_workspace_with_safe_path(tmp_path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "research_lab_adapter.py").write_text(
        "WORKSPACE_ID = 'candidate-workspace'\n",
        encoding="utf-8",
    )
    inherited_root = tmp_path / "inherited"
    inherited_root.mkdir()
    (inherited_root / "research_lab_adapter.py").write_text(
        "WORKSPACE_ID = 'wrong-workspace'\n",
        encoding="utf-8",
    )
    env = code_build._candidate_private_test_env(
        {
            **os.environ,
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": str(inherited_root),
        },
        repo_dir=candidate_root,
    )
    command = " ".join(
        (
            shlex.quote(sys.executable),
            "-c",
            shlex.quote(
                "import research_lab_adapter; "
                "assert research_lab_adapter.WORKSPACE_ID == "
                "'candidate-workspace'"
            ),
        )
    )

    code_build._run_shell(
        command,
        cwd=candidate_root,
        env=env,
        timeout_seconds=10,
    )

    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(candidate_root.resolve()),
        str(inherited_root),
    ]
    assert env["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)


def test_candidate_private_test_uses_verified_process_interpreter(
    tmp_path,
    monkeypatch,
):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    interpreter_bin = tmp_path / "verified-venv" / "bin"
    interpreter_bin.mkdir(parents=True)
    monkeypatch.setattr(
        code_build.sys,
        "executable",
        str(interpreter_bin / "python3.11"),
    )

    env = code_build._candidate_private_test_env(
        {
            "PATH": os.pathsep.join(
                ("/usr/bin", str(interpreter_bin), "/bin")
            ),
        },
        repo_dir=candidate_root,
    )

    assert env["PATH"].split(os.pathsep) == [
        str(interpreter_bin),
        "/usr/bin",
        "/bin",
    ]


def test_git_apply_accepts_exact_replacement_hunk_without_trailing_context(tmp_path):
    source_dir = tmp_path / "sourcing_model"
    source_dir.mkdir()
    source_path = source_dir / "discovery.py"
    source_path.write_text(
        "def build_query_variants():\n"
        "    variants = []\n"
        "    variants.append('strict')\n"
        "    return variants\n"
        "\n"
        "def next_function():\n"
        "    return True\n",
        encoding="utf-8",
    )
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text(
        "diff --git a/sourcing_model/discovery.py b/sourcing_model/discovery.py\n"
        "--- a/sourcing_model/discovery.py\n"
        "+++ b/sourcing_model/discovery.py\n"
        "@@ -1,4 +1,3 @@\n"
        " def build_query_variants():\n"
        "-    variants = []\n"
        "-    variants.append('strict')\n"
        "-    return variants\n"
        "+    return ['strict', 'companion']\n",
        encoding="utf-8",
    )

    code_build._run_git_apply(
        diff_path,
        cwd=tmp_path,
        timeout_seconds=10,
        check=True,
    )
    code_build._run_git_apply(
        diff_path,
        cwd=tmp_path,
        timeout_seconds=10,
        check=False,
    )

    assert "return ['strict', 'companion']" in source_path.read_text(encoding="utf-8")


def test_git_apply_context_fallback_rejects_addition_only_hunks():
    addition_only = (
        "diff --git a/sourcing_model/discovery.py b/sourcing_model/discovery.py\n"
        "--- a/sourcing_model/discovery.py\n"
        "+++ b/sourcing_model/discovery.py\n"
        "@@ -1,0 +1,1 @@\n"
        "+unsafe_by_line_number = True\n"
    )

    assert code_build._can_retry_git_apply_without_edge_context(addition_only) is False


def test_diff_added_material_keeps_headers_and_added_lines_only():
    # File headers stay (paths are model-chosen — a smuggling vector); context
    # and removed lines are verbatim parent source and are excluded.
    diff = _diff_with("context_material = 1", "also_clean = 1")
    material = code_editing._diff_added_line_material(diff)
    assert "+++ b/gateway/module.py" in material
    assert "also_clean = 1" in material
    assert "context_material" not in material
    assert "old = 1" not in material


def test_forbidden_term_in_model_chosen_path_rejects():
    diff = (
        "--- a/gateway/judge_prompt.py\n"
        "+++ b/gateway/judge_prompt.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    assert code_editing._contains_forbidden_material_diff_aware(diff) is True


# --- bug #19: value-level redaction preserves structure ---


def test_redact_secret_values_preserves_line_count_and_masks_literal():
    source = (
        "import os\n"
        'OPENROUTER_API_KEY = "sk-or-v1-9a8b7c6d5e4f"\n'
        "def fetch():\n"
        "    return 1\n"
    )
    redacted = code_build._redact_secret_values(source)
    assert len(redacted.splitlines()) == len(source.splitlines())
    assert "sk-or-v1-9a8b7c6d5e4f" not in redacted
    assert "def fetch():" in redacted


def test_redact_source_excerpt_value_mode_keeps_keyword_lines(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_REDACT_VALUES_ONLY", raising=False)
    source = "api_key = os.environ.get('EXA_API_KEY')\nplain = 1\n"
    redacted = code_build._redact_source_excerpt(source)
    # The keyword-mentioning line survives (env lookup, no literal secret) —
    # the legacy mode blanked it and produced un-appliable hunks.
    assert "plain = 1" in redacted
    assert len(redacted.splitlines()) == 2


# --- bug #21 (parser side): verdict synonyms ---


@pytest.mark.parametrize("verdict", ["pass", "PASSED", "Approved", "yes", "aligned", "OK"])
def test_pass_verdict_synonyms(verdict):
    raw = f'{{"verdict": "{verdict}", "reason": "looks good", "confidence": 0.8}}'
    parsed = code_editing.parse_plan_alignment_judge_response(raw)
    assert parsed.verdict == "pass"


@pytest.mark.parametrize("verdict", ["fail", "rejected", "misaligned", "unclear-gibberish"])
def test_unrecognized_verdicts_stay_fail(verdict):
    raw = f'{{"verdict": "{verdict}", "reason": "nope"}}'
    parsed = code_editing.parse_plan_alignment_judge_response(raw)
    assert parsed.verdict == "fail"


def test_boolean_passes_field_accepted():
    parsed = code_editing.parse_plan_alignment_judge_response(
        '{"plan_alignment": {"passes": true, "reason": "matches plan"}}'
    )
    assert parsed.verdict == "pass"


# --- bug #22: novelty semantic key matches the worker's stored shape ---


def test_semantic_summary_key_matches_worker_storage_format():
    summary = "Widen the provider query FAN-OUT   to boost recall!" + " pad" * 200
    draft = _draft(redacted_summary=summary)
    # worker.py stores semantic_edit_summary as the raw summary truncated to
    # 500 chars; both sides must normalize identically or the guard is dead.
    worker_stored = summary[:500]
    assert code_editing._semantic_summary_key(draft) == code_editing._normalize_semantic_summary(
        worker_stored
    )


def test_semantic_summary_key_falls_back_when_summary_empty():
    draft = _draft(redacted_summary="", expected_improvement="boost recall by 2")
    assert code_editing._semantic_summary_key(draft) == code_editing._normalize_semantic_summary(
        "boost recall by 2"
    )


def test_normalize_semantic_summary_is_rewording_stable():
    a = code_editing._normalize_semantic_summary("Widen   provider fan-out.")
    b = code_editing._normalize_semantic_summary("widen provider FAN-OUT")
    assert a == b


# --- bug #30: infra failures classified and retried, not charged to candidate ---


@pytest.mark.parametrize(
    "text",
    [
        "no basic auth credentials",
        "authorization token has expired",
        "toomanyrequests: pull rate limit",
        "dial tcp 10.0.0.1:443: i/o timeout",
        "connection reset by peer",
    ],
)
def test_infra_failure_markers_detected(text):
    assert code_build._is_infra_failure_text(text) is True


def test_candidate_build_failure_not_infra():
    assert code_build._is_infra_failure_text("SyntaxError: invalid syntax in sourcing_model.py") is False
    assert code_build._is_infra_failure_text("assert adapter_version", "test failed") is False


def test_infra_retry_flag_default_on(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_BUILD_INFRA_RETRY_ENABLED", raising=False)
    assert code_build._infra_retry_enabled() is True
    monkeypatch.setenv("RESEARCH_LAB_BUILD_INFRA_RETRY_ENABLED", "false")
    assert code_build._infra_retry_enabled() is False


# --- planner lane regression: do not box all miner focus into provider fallback ---


def test_loop_direction_planner_prompt_allows_source_routing_and_query_construction():
    messages = code_editing.build_loop_direction_planner_messages(
        ticket={
            "ticket_id": "ticket-source-routing",
            "brief_public_summary": "Route to an alternate discovery surface after primary search returns completed-empty.",
        },
        artifact_manifest={"git_commit_sha": "a" * 40},
        component_registry={},
        benchmark_public_summary={"zero_company_icps": 8},
        runtime_source_index={
            "editable_files": [
                "sourcing_model/discovery.py",
                "sourcing_model/clients.py",
                "sourcing_model/core.py",
            ]
        },
        budget_context={"requested_compute_budget_usd": 5.0},
    )
    content = messages[-1]["content"]
    context = json.loads(content.split("Context JSON:\n", 1)[1])

    assert "source_routing" in context["allowed_lanes"]
    assert "query_construction" in context["allowed_lanes"]
    assert "Alternate discovery surface/provider routing" in content
    assert '"required_lane":"query_construction"' in content
    assert '"allowed_lanes":["provider_fallback"]' not in content


def test_loop_direction_planner_binds_approved_source_to_model_registration():
    source_context = {
        "schema_version": "1.0",
        "provider_count": 1,
        "providers": [
            {
                "provider_id": "community_accounts",
                "provider_alias": "community accounts",
                "governance_origin": "source_add",
                "manifest_sha256": "a" * 64,
            }
        ],
        "routerverse_source_incorporation": {
            "schema_version": "leadpoet.routerverse_source_suggestions.v2",
            "requests": [
                {
                    "schema_version": (
                        "leadpoet.routerverse_source_incorporation.v2"
                    ),
                    "provider_id": "community_accounts",
                    "stage": "candidate_acquisition",
                    "manifest_sha256": "a" * 64,
                    "intent_categories": [],
                    "best_for": ["icp.structured_eligible"],
                    "avoid_when": [],
                    "best_for_description": (
                        "Approved company-discovery provider for structured ICPs."
                    ),
                    "avoid_when_description": (
                        "Avoid when the runtime binding is unavailable."
                    ),
                    "registration_symbol": (
                        "sourcing_model/routing/runtime.py::"
                        "SOURCE_ADD_ROUTING_REGISTRATIONS"
                    ),
                }
            ],
            "clarifications": [],
        },
    }
    messages = code_editing.build_loop_direction_planner_messages(
        ticket={
            "ticket_id": "ticket-source-add",
            "brief_public_summary": (
                "Use community accounts for company discovery."
            ),
        },
        artifact_manifest={"git_commit_sha": "a" * 40},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_index={
            "editable_files": ["sourcing_model/routing/runtime.py"]
        },
        budget_context={},
        provider_capability_summary=source_context,
    )
    content = messages[-1]["content"]
    context = json.loads(content.split("Context JSON:\n", 1)[1])

    assert context["approved_provider_capabilities"][
        "routerverse_source_incorporation"
    ]["requests"][0]["provider_id"] == "community_accounts"
    assert "select source_routing" in content
    assert "exact SourceAddRoutingRegistration" in content
    assert "best_for_description" in content
    assert "avoid_when_description" in content
    assert "intent_categories" in content
    assert "Never register a provider merely because its name appears" in content


def test_code_edit_prompt_names_source_routing_lane():
    messages = code_editing.build_code_edit_auto_research_messages(
        ticket={"ticket_id": "ticket-source-routing", "brief_public_summary": "try an alternate discovery surface"},
        artifact_manifest={"git_commit_sha": "a" * 40},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_context={"editable_files": ["sourcing_model/discovery.py"]},
        source_inspection_context={"read_files": ["sourcing_model/discovery.py"]},
        budget_context={},
        loop_direction_plan={
            "required_lane": "source_routing",
            "selected_path_id": "alternate_discovery_surface",
        },
        max_candidates=1,
    )
    content = messages[-1]["content"]
    context = json.loads(content.split("Context JSON:\n", 1)[1])

    assert "source_routing" in context["allowed_lanes"]
    assert "source routing" in content


def test_code_edit_prompt_requires_source_add_registration_not_host_wiring():
    source_context = {
        "routerverse_source_incorporation": {
            "requests": [
                {
                    "provider_id": "community_signals",
                    "stage": "intent_evidence",
                    "manifest_sha256": "b" * 64,
                    "registration_symbol": (
                        "sourcing_model/routing/runtime.py::"
                        "SOURCE_ADD_ROUTING_REGISTRATIONS"
                    ),
                }
            ],
            "clarifications": [],
        }
    }
    messages = code_editing.build_code_edit_auto_research_messages(
        ticket={
            "ticket_id": "ticket-source-add",
            "brief_public_summary": (
                "Use community signals for intent discovery."
            ),
        },
        artifact_manifest={"git_commit_sha": "a" * 40},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_context={
            "editable_files": ["sourcing_model/routing/runtime.py"]
        },
        source_inspection_context={
            "read_files": ["sourcing_model/routing/runtime.py"]
        },
        budget_context={},
        loop_direction_plan={
            "required_lane": "source_routing",
            "selected_path_id": "register-community-signals",
        },
        max_candidates=1,
        provider_capability_summary=source_context,
    )
    content = messages[-1]["content"]

    assert "ensure the exact SourceAddRoutingRegistration" in content
    assert "preserve it and do not emit a redundant runtime hunk" in content
    assert "sourcing_model/model_runner.py::_COMMON_SOURCE_ADD_BY_INTENT" in content
    assert "best_for_description" in content
    assert "avoid_when_description" in content
    assert "intent_categories" in content
    assert "hard-coded provider branch" in content
    assert "consumer separately binds and activates" in content


def test_source_add_prompts_bind_v8_constructor_derived_manifest_fields():
    provider_context = {
        "routerverse_source_incorporation": {
            "requests": [
                {
                    "schema_version": (
                        "leadpoet.routerverse_source_incorporation.v3"
                    ),
                    "provider_id": "community_signals",
                    "stage": "intent_evidence",
                    "binding_manifest": {
                        "schema_version": (
                            "leadpoet.intent-source-binding-manifest:v1"
                        ),
                        "tool_id": "intent.source_add.community_signals",
                        "provider_id": "community_signals",
                        "stage": "intent_evidence",
                        "execution_mode": "invoke",
                    },
                    "registration_symbol": (
                        "sourcing_model/routing/runtime.py::"
                        "SOURCE_ADD_ROUTING_REGISTRATIONS"
                    ),
                }
            ],
            "clarifications": [],
        }
    }
    planner_messages = code_editing.build_loop_direction_planner_messages(
        ticket={
            "ticket_id": "ticket-source-add-v8",
            "brief_public_summary": "Register the approved source.",
        },
        artifact_manifest={"git_commit_sha": "a" * 40},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_index={
            "editable_files": ["sourcing_model/routing/runtime.py"]
        },
        budget_context={},
        provider_capability_summary=provider_context,
    )
    planner_content = planner_messages[-1]["content"]

    edit_messages = code_editing.build_code_edit_auto_research_messages(
        ticket={
            "ticket_id": "ticket-source-add-v8",
            "brief_public_summary": "Register the approved source.",
        },
        artifact_manifest={"git_commit_sha": "a" * 40},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_context={
            "editable_files": ["sourcing_model/routing/runtime.py"]
        },
        source_inspection_context={
            "read_files": ["sourcing_model/routing/runtime.py"]
        },
        budget_context={},
        loop_direction_plan={
            "required_lane": "source_routing",
            "selected_path_id": "register-community-signals-v8",
        },
        max_candidates=1,
        provider_capability_summary=provider_context,
    )
    edit_content = edit_messages[-1]["content"]

    assert "binding_manifest is the approved attestation" in planner_content
    assert "revision and manifest_sha256 are constructor-derived" in planner_content
    assert "binding_manifest" in edit_content
    assert "not constructor keywords" in edit_content
    assert "Omit revision and manifest_sha256 in v8" in edit_content
    assert "execution_plan_identity" in edit_content


def test_plan_alignment_judge_does_not_require_v8_manifest_metadata_keywords():
    messages = code_editing.build_plan_alignment_judge_messages(
        loop_direction_plan={
            "required_lane": "source_routing",
            "selected_path_id": "register-community-signals-v8",
            "required_mechanism": "approved source registration",
        },
        draft=_draft(
            lane="source_routing",
            plan_path_id="register-community-signals-v8",
            target_files=("sourcing_model/routing/runtime.py",),
            unified_diff=(
                "diff --git a/sourcing_model/routing/runtime.py "
                "b/sourcing_model/routing/runtime.py\n"
                "--- a/sourcing_model/routing/runtime.py\n"
                "+++ b/sourcing_model/routing/runtime.py\n"
                "@@ -1 +1,2 @@\n"
                " SOURCE_ADD_ROUTING_REGISTRATIONS = (\n"
                "+    SourceAddRoutingRegistration(provider_id='community_signals'),\n"
            ),
        ),
    )
    content = messages[-1]["content"]

    assert "binding_manifest is an approved request-side attestation" in content
    assert "Do not fail a v8 registration merely because" in content
    assert "omits binding_manifest, revision, or manifest_sha256" in content
    assert "execution_plan_identity is an approved manifest-defining" in content
    assert "still fail fixed public-ICP values" in content


def test_code_edit_prompt_requires_direct_git_parent_and_safe_branch_feedback():
    feedback = {
        "schema_version": "research_lab.git_tree_parent_feedback.v1",
        "aggregate_score": 42.0,
        "example_count": 8,
        "examples": [
            {
                "example_number": index,
                "quality_band": "weak" if index == 3 else "adequate",
                "result_count": index,
            }
            for index in range(1, 9)
        ],
        "feedback_hash": "sha256:" + "f" * 64,
    }
    messages = code_editing.build_code_edit_auto_research_messages(
        ticket={"ticket_id": "ticket-git-tree"},
        artifact_manifest={"model_artifact_hash": "sha256:" + "a" * 64},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_context={"editable_files": ["sourcing_model/discovery.py"]},
        source_inspection_context={"read_files": ["sourcing_model/discovery.py"]},
        budget_context={
            "within_run_memory": {
                "git_tree_branch": {
                    "schema_version": "research_lab.git_tree_branch_context.v1",
                    "parent_node_id": "tree-node:" + "1" * 64,
                    "ancestor_node_ids": ["tree-node:" + "1" * 64],
                    "parent_feedback": feedback,
                }
            }
        },
        max_candidates=1,
    )
    content = messages[-1]["content"]
    assert "exact committed parent" in content
    assert "Never recreate the run-start source, merge a sibling" in content
    assert "do not merely optimize the aggregate score" in content
    assert '"example_number":3' in content
    assert "icp_ref" not in content
    assert "company_name" not in content
    assert "provider_output" not in content
    assert "sibling_hypothesis_hashes" not in content


def _v1_1_plan(**overrides):
    path = {
        "path_id": "query-recall",
        "lane": "query_construction",
        "mechanism": "add one bounded query variant",
        "target_behavior": ["recover sparse searches"],
        "must_inspect": ["sourcing_model/discovery.py"],
        "allowed_lanes": ["query_construction"],
        "disallowed_lanes": ["provider_fallback"],
        "must_not_try": ["do not weaken ICP gates"],
        "success_criteria": ["runtime checks pass"],
        "novelty_requirements": ["different from prior attempts"],
        "anti_overfit_checks": ["preserve multiple outputs"],
        "validation_mode": "runtime_checks",
        "validation_paths": [],
    }
    payload = {
        "schema_version": "1.1",
        "miner_focus_interpretation": "improve sparse-query recall",
        "loop_goal": "recover qualified companies",
        "required_lane": path["lane"],
        "required_mechanism": path["mechanism"],
        "target_behavior": path["target_behavior"],
        "must_inspect": path["must_inspect"],
        "allowed_lanes": path["allowed_lanes"],
        "disallowed_lanes": path["disallowed_lanes"],
        "must_not_try": path["must_not_try"],
        "success_criteria": path["success_criteria"],
        "novelty_requirements": path["novelty_requirements"],
        "anti_overfit_checks": path["anti_overfit_checks"],
        "ranked_paths": [path],
        "selected_path_id": path["path_id"],
        "validation_mode": "runtime_checks",
        "validation_paths": [],
    }
    payload.update(overrides)
    return payload


def test_loop_direction_v1_0_checkpoint_remains_compatible():
    plan = code_editing.loop_direction_plan_from_mapping(
        {
            "schema_version": "1.0",
            "required_lane": "query_construction",
            "required_mechanism": "bounded query variant",
            "ranked_paths": [{"path_id": "legacy-path"}],
            "selected_path_id": "legacy-path",
        }
    )
    assert plan.validation_mode == "runtime_checks"
    assert plan.validation_paths == ()
    assert code_editing.loop_direction_plan_contract_errors(plan) == []


def test_loop_direction_v1_1_round_trip_and_contract_validation():
    plan = code_editing.parse_loop_direction_plan_response(json.dumps(_v1_1_plan()))
    first_doc = plan.to_dict()
    reparsed = code_editing.loop_direction_plan_from_mapping(first_doc)
    assert reparsed == plan
    assert reparsed.to_dict()["plan_hash"] == first_doc["plan_hash"]
    assert code_editing.loop_direction_plan_contract_errors(plan) == []


def test_loop_direction_v1_1_selected_path_overrides_duplicate_cover_fields():
    payload = _v1_1_plan(
        required_lane="output_ranking",
        required_mechanism="different top-level mechanism",
        target_behavior=["different top-level behavior"],
        must_inspect=["sourcing_model/other.py"],
        allowed_lanes=["output_ranking"],
        disallowed_lanes=["query_construction"],
        must_not_try=["different top-level safety rule"],
        success_criteria=["different top-level success rule"],
        novelty_requirements=["different top-level novelty rule"],
        anti_overfit_checks=["different top-level overfit rule"],
        validation_mode="existing_test_files",
        validation_paths=["tests/nonexistent.py"],
    )
    plan = code_editing.loop_direction_plan_from_mapping(payload)
    selected = payload["ranked_paths"][0]
    assert plan.required_lane == selected["lane"]
    assert plan.required_mechanism == selected["mechanism"]
    for field in (
        "target_behavior",
        "must_inspect",
        "allowed_lanes",
        "disallowed_lanes",
        "must_not_try",
        "success_criteria",
        "novelty_requirements",
        "anti_overfit_checks",
        "validation_paths",
    ):
        assert getattr(plan, field) == tuple(selected[field])
    assert plan.validation_mode == selected["validation_mode"]
    assert code_editing.loop_direction_plan_contract_errors(plan) == []


def test_loop_direction_v1_1_rejects_inconsistent_selected_path():
    payload = _v1_1_plan()
    payload["ranked_paths"][0]["allowed_lanes"] = ["output_ranking"]
    plan = code_editing.loop_direction_plan_from_mapping(payload)
    assert any(
        error.startswith("ranked_path_lane_not_allowed:")
        for error in code_editing.loop_direction_plan_contract_errors(plan)
    )


def test_loop_direction_v1_1_requires_explicit_path_validation_strategy():
    payload = _v1_1_plan()
    payload["ranked_paths"][0].pop("validation_paths")
    plan = code_editing.loop_direction_plan_from_mapping(payload)
    assert any(
        error.startswith("ranked_path_missing_validation_paths:")
        for error in code_editing.loop_direction_plan_contract_errors(plan)
    )


def test_loop_direction_v1_1_allows_explicit_empty_disallowed_lanes():
    payload = _v1_1_plan(disallowed_lanes=[])
    payload["ranked_paths"][0]["disallowed_lanes"] = []
    plan = code_editing.loop_direction_plan_from_mapping(payload)
    assert code_editing.loop_direction_plan_contract_errors(plan) == []


def test_loop_direction_v1_1_rejects_more_than_eight_ranked_paths():
    payload = _v1_1_plan()
    base_path = payload["ranked_paths"][0]
    payload["ranked_paths"] = [
        {**base_path, "path_id": f"path-{index}"}
        for index in range(9)
    ]
    payload["selected_path_id"] = "path-0"
    plan = code_editing.loop_direction_plan_from_mapping(payload)
    assert (
        "loop_direction_plan_v1_1_allows_at_most_eight_ranked_paths"
        in code_editing.loop_direction_plan_contract_errors(plan)
    )


def test_existing_test_validation_requires_paths():
    payload = _v1_1_plan()
    payload["ranked_paths"][0]["validation_mode"] = "existing_test_files"
    payload["ranked_paths"][0]["validation_paths"] = []
    with pytest.raises(ValueError, match="requires validation_paths"):
        code_editing.loop_direction_plan_from_mapping(payload)


@pytest.mark.parametrize(
    ("reason", "expected_class"),
    [
        (
            "No existing test file appears in runtime_source_context.editable_files and new files are forbidden.",
            "binding_plan_unimplementable",
        ),
        (
            "No existing test file is listed in editable_files for the required coverage.",
            "binding_plan_unimplementable",
        ),
        ("The provider probe refuted this hypothesis.", "provider_probe_refuted_hypothesis"),
    ],
)
def test_legacy_no_viable_refusal_gets_structured_failure_class(reason, expected_class):
    refusal = code_editing.parse_code_edit_no_viable_patch_response(
        json.dumps({"no_viable_patch": True, "reason": reason})
    )
    assert refusal is not None
    assert refusal.failure_class == expected_class


def test_structured_no_viable_refusal_round_trip_and_secret_rejection():
    refusal = code_editing.parse_code_edit_no_viable_patch_response(
        json.dumps(
            {
                "no_viable_patch": True,
                "failure_class": "binding_plan_unimplementable",
                "reason": "required symbol is absent",
                "missing_references": ["discover_companies"],
            }
        )
    )
    assert refusal is not None
    assert refusal.missing_references == ("discover_companies",)
    sanitized = code_editing.parse_code_edit_no_viable_patch_response(
        json.dumps(
            {
                "no_viable_patch": True,
                "failure_class": "no_safe_patch",
                "reason": "no safe patch\n\twithin the current scope",
                "missing_references": [],
            }
        )
    )
    assert sanitized is not None
    assert sanitized.reason == "no safe patch within the current scope"
    policy = code_editing.parse_code_edit_no_viable_patch_response(
        json.dumps({"no_viable_patch": True, "reason": "service_role must not be accessed"})
    )
    assert policy is not None
    with pytest.raises(ValueError, match="forbidden"):
        code_editing.parse_code_edit_no_viable_patch_response(
            json.dumps(
                {
                    "no_viable_patch": True,
                    "reason": "credential unavailable",
                    "service_role_key": "synthetic-secret-value-123456",
                }
            )
        )


def test_sensitive_material_claim_rejects_but_policy_prose_passes():
    verdict = code_editing.parse_plan_alignment_judge_response(
        json.dumps(
            {
                "verdict": "pass",
                "reason": "Do not use hidden ICP data when evaluating this patch.",
            }
        )
    )
    assert verdict.verdict == "pass"

    with pytest.raises(ValueError, match="forbidden"):
        code_editing.parse_plan_alignment_judge_response(
            json.dumps(
                {
                    "verdict": "fail",
                    "reason": "The patch returns hidden_icp data without redaction.",
                }
            )
        )

    with pytest.raises(ValueError, match="forbidden"):
        code_editing.parse_plan_alignment_judge_response(
            json.dumps(
                {
                    "verdict": "fail",
                    "reason": "credential field present",
                    "provider_api_key": "redacted",
                }
            )
        )

    with pytest.raises(ValueError, match="forbidden"):
        code_editing.parse_plan_alignment_judge_response(
            json.dumps(
                {
                    "verdict": "fail",
                    "reason": "The patch mentions hidden_icp material.",
                }
            )
        )


def test_planner_prompt_exposes_safe_validation_capabilities_without_command_text():
    constraints = {
        "new_files_allowed": False,
        "editable_test_path_count": 0,
        "editable_test_paths": [],
        "allowed_validation_modes": ["runtime_checks"],
        "runtime_checks": {"private_test_command_configured": True},
    }
    messages = code_editing.build_loop_direction_planner_messages(
        ticket={"ticket_id": "ticket-validation"},
        artifact_manifest={},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_index={"editable_files": ["sourcing_model/discovery.py"]},
        budget_context={},
        candidate_edit_constraints=constraints,
    )
    content = messages[-1]["content"]
    context = json.loads(content.split("Context JSON:\n", 1)[1])
    assert context["candidate_edit_constraints"] == constraints
    assert "RESEARCH_LAB_PRIVATE_TEST_CMD" not in content
    assert "do not require adding tests" in content
    assert "runtime_checks requires validation_paths=[]" in content


def test_planner_prompt_example_is_internally_consistent_and_source_bound():
    messages = code_editing.build_loop_direction_planner_messages(
        ticket={"ticket_id": "ticket-example"},
        artifact_manifest={},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_index={
            "files": [
                {
                    "path": "sourcing_model/discovery.py",
                    "symbols": [{"qualified_name": "Router.discover_companies"}],
                }
            ]
        },
        budget_context={},
        candidate_edit_constraints={},
    )
    content = messages[-1]["content"]
    example_text = content.split(
        "Required output shape (the selected path and duplicate top-level fields match exactly):\n",
        1,
    )[1].split("\n\nContext JSON:\n", 1)[0]
    example = json.loads(example_text)
    selected = example["ranked_paths"][0]

    assert example["required_lane"] == selected["lane"]
    assert example["required_mechanism"] == selected["mechanism"]
    assert example["must_inspect"] == selected["must_inspect"]
    assert example["must_inspect"] == [
        "sourcing_model/discovery.py::Router.discover_companies"
    ]
    assert code_editing.loop_direction_plan_contract_errors(
        code_editing.loop_direction_plan_from_mapping(example)
    ) == []


# --- bug #29(a): real head sha recorded instead of throwaway git-init sha ---


def _manifest(git_sha="1234567890abcdef1234567890abcdef12345678"):
    from research_lab.eval import PrivateModelArtifactManifest

    return PrivateModelArtifactManifest(
        model_artifact_hash="sha256:" + "a" * 64,
        git_commit_sha=git_sha,
        image_digest="sha256:" + "c" * 64,
        config_hash="sha256:" + "d" * 64,
        component_registry_version="1.0",
        scoring_adapter_version="1.0",
        manifest_uri="s3://bucket/manifest.json",
        manifest_hash="sha256:" + "e" * 64,
        signature_ref="kms://sig",
    )


def _source_admission_receipt(manifest):
    contract = dict(manifest.compatibility_contract or {})
    parity = dict(manifest.consumer_parity_fixtures or {})
    return {
        "decision": "accepted",
        "source_tree_hash": manifest.model_artifact_hash,
        "contract_id": contract.get("contract_id"),
        "contract_hash": contract.get("sha256"),
        "parity_hash": parity.get("sha256"),
    }


def test_recorded_sha_prefers_env_then_parent(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_PRIVATE_SOURCE_HEAD_SHA", "feedbeef" * 5)
    sha, source = code_build._resolve_recorded_commit_sha(
        workspace_sha="0" * 40, parent_artifact=_manifest()
    )
    assert (sha, source) == ("feedbeef" * 5, "env")

    monkeypatch.delenv("RESEARCH_LAB_PRIVATE_SOURCE_HEAD_SHA", raising=False)
    sha, source = code_build._resolve_recorded_commit_sha(
        workspace_sha="0" * 40, parent_artifact=_manifest()
    )
    assert source == "parent_manifest"
    assert sha == "1234567890abcdef1234567890abcdef12345678"


def test_recorded_sha_legacy_flag_restores_workspace(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_BUILD_RECORD_REAL_HEAD_SHA", "false")
    sha, source = code_build._resolve_recorded_commit_sha(
        workspace_sha="9" * 40, parent_artifact=_manifest()
    )
    assert (sha, source) == ("9" * 40, "build_workspace")


def test_recorded_sha_falls_back_to_workspace_when_nothing_valid(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_PRIVATE_SOURCE_HEAD_SHA", raising=False)
    sha, source = code_build._resolve_recorded_commit_sha(
        workspace_sha="9" * 40, parent_artifact=_manifest(git_sha="not-a-sha")
    )
    assert (sha, source) == ("9" * 40, "build_workspace")


def test_built_candidate_artifact_requires_exact_signature_authority(
    monkeypatch,
):
    manifest = _manifest()
    verified = {}
    monkeypatch.setenv(
        "RESEARCH_LAB_PRIVATE_MODEL_KMS_KEY_ID",
        "alias/test-private-artifact",
    )
    monkeypatch.setattr(
        code_build,
        "validate_private_model_artifact_manifest",
        lambda _artifact: [],
    )

    def fake_verify(artifact, *, key_id):
        verified["artifact"] = artifact
        verified["key_id"] = key_id
        return {"verified": True}

    monkeypatch.setattr(
        code_build,
        "verify_private_artifact_manifest_signature",
        fake_verify,
    )
    assert code_build._verify_built_candidate_artifact(
        manifest,
        source_tree_hash=manifest.model_artifact_hash,
        source_compatibility_receipt=_source_admission_receipt(manifest),
    ) == {
        "verified": True
    }
    assert verified == {
        "artifact": manifest,
        "key_id": "alias/test-private-artifact",
    }


def test_built_candidate_artifact_rejects_contract_or_signature_failure(
    monkeypatch,
):
    from research_lab.eval import PrivateModelRuntimeError

    monkeypatch.setattr(
        code_build,
        "validate_private_model_artifact_manifest",
        lambda _artifact: [],
    )
    monkeypatch.setattr(
        code_build,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PrivateModelRuntimeError("consumer contract/parity mismatch")
        ),
    )

    with pytest.raises(
        code_build.CodeEditImageBuildError,
        match="exact contract/signature verification",
    ) as exc_info:
        code_build._verify_built_candidate_artifact(
            _manifest(),
            source_tree_hash="sha256:" + "a" * 64,
            source_compatibility_receipt={},
        )
    assert isinstance(exc_info.value.__cause__, PrivateModelRuntimeError)


def test_built_candidate_artifact_classifies_signature_transport_failure(
    monkeypatch,
):
    from research_lab.eval import PrivateModelRuntimeError

    monkeypatch.setattr(
        code_build,
        "validate_private_model_artifact_manifest",
        lambda _artifact: [],
    )

    def fail_verify(*_args, **_kwargs):
        try:
            raise RuntimeError("connection refused")
        except RuntimeError as cause:
            raise PrivateModelRuntimeError(
                "private artifact manifest KMS signature verification failed"
            ) from cause

    monkeypatch.setattr(
        code_build,
        "verify_private_artifact_manifest_signature",
        fail_verify,
    )

    with pytest.raises(code_build.CodeEditInfraFailureError):
        code_build._verify_built_candidate_artifact(
            _manifest(),
            source_tree_hash="sha256:" + "a" * 64,
            source_compatibility_receipt={},
        )


def test_built_candidate_artifact_rejects_signed_manifest_identity_drift(
    monkeypatch,
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        code_build,
        "validate_private_model_artifact_manifest",
        lambda _artifact: [],
    )
    monkeypatch.setattr(
        code_build,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: {"verified": True},
    )
    with pytest.raises(
        code_build.CodeEditImageBuildError,
        match="differs from its admitted source identity",
    ):
        code_build._verify_built_candidate_artifact(
            manifest,
            source_tree_hash="sha256:" + "b" * 64,
            source_compatibility_receipt=_source_admission_receipt(manifest),
        )


def test_built_legacy_candidate_requires_the_exact_signed_release_identity(
    monkeypatch,
) -> None:
    from dataclasses import replace

    from research_lab.eval import PrivateModelArtifactManifest
    from research_lab.sourcing_model_contract_check import (
        reviewed_consumer_snapshots,
    )

    snapshot = reviewed_consumer_snapshots()[
        "leadpoet-sourcing-wrapper-contract-v7"
    ]
    release = snapshot["release_identities"][0]
    contract = snapshot["contract"]
    manifest = PrivateModelArtifactManifest(
        model_artifact_hash=release["source_tree_hash"],
        git_commit_sha=release["git_commit_sha"],
        image_digest=release["image_digest"],
        config_hash="sha256:" + "d" * 64,
        component_registry_version="sourcing-model-components:v2",
        scoring_adapter_version="qualification-company-scorer:v1",
        compatibility_contract={
            "contract_id": contract["contract_id"],
            "path": contract["canonical_path"],
            "sha256": snapshot["contract_sha256"],
        },
        consumer_parity_fixtures={
            "path": contract["parity_fixture_path"],
            "sha256": snapshot["parity_sha256"],
        },
        manifest_uri="s3://bucket/legacy.json",
        manifest_hash=release["manifest_hash"],
        signature_ref="s3://bucket/legacy.sig.b64",
    )
    receipt = {
        **_source_admission_receipt(manifest),
        "admission_mode": "legacy_exact",
    }
    monkeypatch.setattr(
        code_build,
        "validate_private_model_artifact_manifest",
        lambda _artifact: [],
    )
    monkeypatch.setattr(
        code_build,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: {"verified": True},
    )

    assert code_build._verify_built_candidate_artifact(
        manifest,
        source_tree_hash=manifest.model_artifact_hash,
        source_compatibility_receipt=receipt,
    ) == {"verified": True}
    with pytest.raises(
        code_build.CodeEditImageBuildError,
        match="reviewed signed release",
    ):
        code_build._verify_built_candidate_artifact(
            replace(manifest, git_commit_sha="f" * 40),
            source_tree_hash=manifest.model_artifact_hash,
            source_compatibility_receipt=receipt,
        )


def test_build_scaffold_binds_v2_metadata_to_signed_source_constant(
    tmp_path,
) -> None:
    from gateway.research_lab import code_build

    code_build._write_research_lab_build_scaffold(
        tmp_path,
        base_image_ref=(
            "public.ecr.aws/docker/library/python@sha256:" + "a" * 64
        ),
    )
    dockerfile = (tmp_path / "Dockerfile.research-lab").read_text(
        encoding="utf-8"
    )

    assert (
        "scoring_adapter_version == "
        "research_lab_adapter.SCORING_ADAPTER_VERSION"
        in dockerfile
    )
    assert "signed_scoring_adapter_version" in dockerfile
    assert "company_fit_proof_receipt_contract_identity" in dockerfile
    assert (
        "3efefb93374b8a34c5866374083da556d40c1fb6cf69fd38cf065c177b18d61b"
        in dockerfile
    )
    clear_parent_app = (
        "RUN find /app -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +"
    )
    assert dockerfile.index("WORKDIR /app") < dockerfile.index(clear_parent_app)
    assert dockerfile.index(clear_parent_app) < dockerfile.index("COPY . /app")


def test_candidate_signature_gate_precedes_build_evidence_and_return() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "gateway"
        / "research_lab"
        / "code_build.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    build_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_under_deadline"
    )
    call_lines = {
        node.func.id: node.lineno
        for node in ast.walk(build_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    source_gate_lines = sorted(
        node.lineno
        for node in ast.walk(build_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_sourcing_contract_gate"
    )
    prebuild_gate_line = next(
        node.lineno
        for node in ast.walk(build_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_sourcing_contract_prebuild_gate"
    )
    assert len(source_gate_lines) == 1
    assert (
        prebuild_gate_line
        < call_lines["_run_shell"]
        < call_lines["_run_private_build_under_docker_operation_lock"]
        < source_gate_lines[0]
        < call_lines["_verify_built_candidate_artifact"]
    )
    assert call_lines["_verify_built_candidate_artifact"] < call_lines[
        "_write_private_code_edit_diff_artifact"
    ]
