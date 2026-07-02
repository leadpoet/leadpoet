"""P1 acceptance check (trajectoryimprovements.md): every production
OpenRouter call site is either captured through the shared telemetry layer or
explicitly classified as intentionally uncaptured.

A new file that starts talking to OpenRouter without a classification fails
this test — capture coverage decisions must be deliberate, never silent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# file (repo-relative) → classification
#   captured                 — routes through research_lab/openrouter_telemetry
#                              (transport or record hook) or its own encrypted
#                              raw-trace recorder (hosted worker).
#   uncaptured_by_decision   — dated owner decision 2026-07-02: no fulfillment
#                              trajectory capture (fulfillment gateway paths,
#                              miner production models, subnet validator).
#   not_a_call_site          — proxy/key/url-pattern infrastructure or ops
#                              tooling that mentions the URL without producing
#                              production training data.
CALL_SITE_REGISTRY = {
    # -- captured -----------------------------------------------------------
    "gateway/research_lab/worker.py": "captured",
    "gateway/research_lab/scoring_worker.py": "captured",
    "research_lab/openrouter_telemetry.py": "captured",
    "qualification/scoring/intent_precheck.py": "captured",
    "qualification/scoring/intent_signal_gate.py": "captured",
    "qualification/scoring/intent_verification_three_stage.py": "captured",
    "qualification/scoring/role_batch_check.py": "captured",
    "qualification/scoring/verification_helpers.py": "captured",
    "qualification/validator/hardcoding_detector.py": "captured",
    "validator_models/stage5_verification.py": "captured",
    "gateway/qualification/utils/helpers.py": "captured",
    # -- uncaptured by dated owner decision (2026-07-02): no fulfillment
    # trajectory capture ----------------------------------------------------
    "gateway/fulfillment/icp_checks.py": "uncaptured_by_decision",
    "gateway/fulfillment/intent_details.py": "uncaptured_by_decision",
    "gateway/fulfillment/role_expander.py": "uncaptured_by_decision",
    "gateway/tasks/icp_generator.py": "uncaptured_by_decision",
    "miner_models/Main_fulfillment_model/config.py": "uncaptured_by_decision",
    "miner_models/Main_fulfillment_model/stage4_helpers.py": "uncaptured_by_decision",
    "miner_models/Main_fulfillment_model/web_discovery.py": "uncaptured_by_decision",
    "miner_models/fulfillment_sourcer.py": "uncaptured_by_decision",
    "miner_models/intent_model.py": "uncaptured_by_decision",
    "miner_models/lead_sorcerer_main/src/domain.py": "uncaptured_by_decision",
    "miner_models/qualification_model/_model.py": "uncaptured_by_decision",
    "miner_models/qualification_research_arm_b/_model.py": "uncaptured_by_decision",
    "neurons/validator.py": "uncaptured_by_decision",
    "validator_models/checks_icp.py": "uncaptured_by_decision",
    "validator_models/fulfillment_attribute_verification.py": "uncaptured_by_decision",
    "validator_models/fulfillment_company_verification.py": "uncaptured_by_decision",
    "validator_models/fulfillment_person_verification.py": "uncaptured_by_decision",
    "validator_models/stage4_helpers.py": "uncaptured_by_decision",
    "validator_models/stage4_person_verification.py": "uncaptured_by_decision",
    # -- infrastructure / tooling, not production call sites -----------------
    "gateway/research_lab/key_vault.py": "not_a_call_site",
    "qualification/validator/local_proxy.py": "not_a_call_site",
    "qualification/validator/sandbox_security.py": "not_a_call_site",
}


def _files_mentioning_openrouter() -> set[str]:
    result = subprocess.run(
        [
            "grep",
            "-rl",
            "openrouter.ai/api/v1",
            "--include=*.py",
            "gateway",
            "qualification",
            "research_lab",
            "validator_models",
            "miner_models",
            "neurons",
            "leadpoet_verifier",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "__pycache__" not in line
    }


def test_every_openrouter_call_site_is_classified():
    found = _files_mentioning_openrouter()
    unclassified = sorted(found - set(CALL_SITE_REGISTRY))
    assert not unclassified, (
        "New OpenRouter call sites without a capture classification "
        f"(add them to CALL_SITE_REGISTRY with a deliberate decision): {unclassified}"
    )


def test_captured_sites_actually_reference_the_telemetry_layer():
    for rel_path, classification in CALL_SITE_REGISTRY.items():
        if classification != "captured":
            continue
        path = REPO_ROOT / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert (
            "openrouter_telemetry" in text
            or "_OpenRouterRawTraceRecorder" in text
            or rel_path == "research_lab/openrouter_telemetry.py"
        ), f"{rel_path} is classified 'captured' but references no capture layer"


def test_registry_entries_still_exist():
    stale = [
        rel_path
        for rel_path in CALL_SITE_REGISTRY
        if not (REPO_ROOT / rel_path).exists()
    ]
    assert not stale, f"registry entries for deleted files: {stale}"
