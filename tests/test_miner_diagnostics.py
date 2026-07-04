"""Tests for miner-facing per-candidate diagnostics (sanitized own-run projection).

Security invariants are asserted first: sealed (private) ICPs must never emit
per-ICP data, and diffs / file paths / raw company vectors / secrets must never
appear. Pure module — no DB/network.
"""
from __future__ import annotations

import json

from gateway.research_lab import miner_diagnostics as md


# A distinctive sealed-ICP ref + delta we assert NEVER appear in the output.
PRIVATE_REF = "qualification_private_icp_sets:20260703:icp_SEALEDSECRET_777"
PUBLIC_REF = "qualification_private_icp_sets:20260703:icp_PUBLICOK_001"


def _bundle():
    return {
        "icp_set_hash": "sha256:window1",
        "provider_excluded_icp_ids": ["icp_INFRA_dead"],
        "private_holdout_gate": {
            "gate_type": "public_score_before_private_holdout",
            "decision": "rejected_before_private_holdout",
            "candidate_public_score": 9.0,
            "baseline_public_score": 16.2,
            "baseline_private_score": 40.74,   # sealed info — must NOT surface
        },
        "aggregates": {
            "candidate_score": 5.4, "base_score": 0.0,
            "mean_delta": 5.4, "delta_lcb": -1.83, "icp_count": 5,
            "per_icp_results": [
                # public, helped
                {"icp_ref": PUBLIC_REF, "status": "completed", "delta_vs_base": 32.4,
                 "base_per_icp_score": 0.0, "candidate_per_icp_score": 32.4,
                 "candidate_company_scores": [0.0, 54.0]},
                # public, infra-excluded (provider status)
                {"icp_ref": "icp_pub_infra", "status": "candidate_model_runtime_provider_error",
                 "delta_vs_base": 0.0, "candidate_per_icp_score": 0.0, "candidate_company_scores": []},
                # PRIVATE helped (must only bump pool.helped)
                {"icp_ref": PRIVATE_REF, "status": "completed", "delta_vs_base": 21.6,
                 "base_per_icp_score": 0.0, "candidate_per_icp_score": 21.6,
                 "candidate_company_scores": [54.0]},
                # PRIVATE flat
                {"icp_ref": "icp_priv_flat", "status": "completed", "delta_vs_base": 0.3,
                 "candidate_per_icp_score": 0.3, "candidate_company_scores": []},
                # PRIVATE infra-excluded (via provider_excluded list)
                {"icp_ref": "icp_INFRA_dead", "status": "completed", "delta_vs_base": 0.0,
                 "candidate_per_icp_score": 0.0, "candidate_company_scores": []},
            ],
        },
    }


def _patch_manifest():
    return {
        "target_component_id": "private_model_source_tree",
        "patch_type": "IMAGE_BUILD",
        "candidate_kind": "image_build",
        "redacted_summary": "Add fallback keyword search when primary search returns no candidates.",
        "unified_diff_hash": "sha256:LEAK_DIFF_HASH",           # must NOT appear
        "manifest_hash": "sha256:LEAK_MANIFEST",                # must NOT appear
        "patch_doc": {
            "lane": "provider_fallback",
            "target_files": ["sourcing_model/discovery.py"],    # private path — must NOT appear
            "unified_diff_hash": "sha256:LEAK2",
        },
    }


def _vis():
    return {PUBLIC_REF: "public", "icp_pub_infra": "public"}   # others default → private


def _build():
    return md.build_candidate_diagnostics(
        candidate_id="candidate:abc", bundle_doc=_bundle(),
        patch_manifest=_patch_manifest(), visibility_by_ref=_vis(),
        candidate_status="rejected",
    )


# ───────────────────────── SECURITY INVARIANTS ─────────────────────────

def test_private_icp_ref_never_appears():
    blob = json.dumps(_build())
    assert PRIVATE_REF not in blob
    assert "SEALEDSECRET" not in blob
    # the private helped delta (21.6) must not leak as a per-ICP value either
    # (it may coincide with an aggregate, so assert no per-ICP private row exists)
    for row in _build()["public_icps"]:
        assert row["icp_ref"] != PRIVATE_REF


def test_public_icp_ref_is_exposed():
    doc = _build()
    refs = [r["icp_ref"] for r in doc["public_icps"]]
    assert PUBLIC_REF in refs


def test_no_diff_or_file_paths_leak():
    blob = json.dumps(_build())
    for forbidden in ("LEAK_DIFF_HASH", "LEAK_MANIFEST", "LEAK2",
                      "sourcing_model/discovery.py", "target_files", "unified_diff"):
        assert forbidden not in blob, forbidden


def test_no_raw_company_vectors():
    doc = _build()
    blob = json.dumps(doc)
    assert "candidate_company_scores" not in blob
    # only count + avg exposed
    assert set(doc["scored_companies"].keys()) == {"count", "avg_final_score"}


def test_sealed_baseline_private_score_not_leaked():
    # gate carries baseline_private_score in the raw bundle; the projection
    # must expose only public-score fields.
    gate = _build()["aggregate"]["gate"]
    assert "baseline_private_score" not in gate
    assert set(gate.keys()) <= {"type", "decision", "candidate_public_score", "baseline_public_score"}


def test_unknown_ref_defaults_to_private():
    # an ICP with no visibility entry must be treated as PRIVATE, never public.
    doc = md.build_candidate_diagnostics(
        candidate_id="c", bundle_doc=_bundle(), patch_manifest=_patch_manifest(),
        visibility_by_ref={},  # nothing classified
    )
    assert doc["public_icps"] == []
    assert doc["private_pool"]["icp_count"] == 5


# ───────────────────────── CORRECTNESS ─────────────────────────

def test_emitted_patch_fields():
    p = _build()["emitted_patch"]
    assert p["target_component"] == "private_model_source_tree"
    assert p["patch_type"] == "IMAGE_BUILD"
    assert p["lane"] == "provider_fallback"
    assert p["redacted_summary"].startswith("Add fallback")


def test_private_pool_counts():
    pool = _build()["private_pool"]
    # PRIVATE_REF helped, icp_priv_flat flat, icp_INFRA_dead infra
    assert pool == {"icp_count": 3, "helped": 1, "hurt": 0, "flat": 1, "infra_excluded": 1}


def test_public_icp_values_and_infra_label():
    pub = {r["icp_ref"]: r for r in _build()["public_icps"]}
    assert pub[PUBLIC_REF]["delta"] == 32.4 and pub[PUBLIC_REF]["candidate_score"] == 32.4
    assert pub["icp_pub_infra"]["status"] == "infra_excluded"


def test_infra_aggregate_and_rerun_flag():
    agg = _build()["aggregate"]
    # 2 of 5 ICPs infra-excluded = 0.4 > 0.2 threshold
    assert agg["infra"]["excluded_icps"] == 2
    assert agg["infra"]["rerun_credit_eligible"] is True


def test_scored_companies_avg():
    sc = _build()["scored_companies"]
    # positive scores across all ICPs: 54.0 (public), 54.0 (private) → 2 companies
    assert sc["count"] == 2
    assert sc["avg_final_score"] == 54.0


def test_gate_projection():
    gate = _build()["aggregate"]["gate"]
    assert gate["decision"] == "rejected_before_private_holdout"
    assert gate["candidate_public_score"] == 9.0 and gate["baseline_public_score"] == 16.2


def test_failed_candidate_reduced_doc():
    doc = md.build_candidate_diagnostics(
        candidate_id="c", bundle_doc=None, patch_manifest=_patch_manifest(),
        visibility_by_ref={}, candidate_status="failed",
    )
    assert doc["public_icps"] == [] and doc["private_pool"] == {}
    assert doc["emitted_patch"]["patch_type"] == "IMAGE_BUILD"   # patch summary still shown
    assert doc["scored_companies"] == {"count": 0, "avg_final_score": 0.0}


def test_determinism():
    assert md.build_candidate_diagnostics(
        candidate_id="c", bundle_doc=_bundle(), patch_manifest=_patch_manifest(),
        visibility_by_ref=_vis()) == md.build_candidate_diagnostics(
        candidate_id="c", bundle_doc=_bundle(), patch_manifest=_patch_manifest(),
        visibility_by_ref=_vis())


def test_visibility_map_parse_and_degrade():
    summary = {"visibility_split": {"items": [
        {"icp_ref": "a", "visibility": "public"},
        {"icp_ref": "b", "visibility": "private"},
        {"icp_ref": "", "visibility": "public"},          # skipped
        {"icp_ref": "c", "visibility": "weird"},          # skipped
    ]}}
    m = md.visibility_map_from_benchmark_split(summary)
    assert m == {"a": "public", "b": "private"}
    assert md.visibility_map_from_benchmark_split(None) == {}
    assert md.visibility_map_from_benchmark_split({"visibility_split": {}}) == {}


def test_assert_clean_rejects_secret():
    import pytest
    with pytest.raises(ValueError):
        md._assert_clean({"api_key": "sk-secret"})


# ───────────────────────── SUMMARY SANITIZATION ─────────────────────────

def test_summary_with_secret_marker_is_withheld_not_raised():
    # an LLM summary that happens to contain a secret marker must NOT blow up
    # the whole doc — only that field is withheld.
    pm = dict(_patch_manifest())
    pm["redacted_summary"] = "Rotate the sk-or-abc123 openrouter_api_key in the transport."
    doc = md.build_candidate_diagnostics(
        candidate_id="c", bundle_doc=_bundle(), patch_manifest=pm, visibility_by_ref=_vis())
    blob = json.dumps(doc)
    assert "sk-or-" not in blob and "openrouter_api_key" not in blob
    assert doc["emitted_patch"]["redacted_summary"] == md._WITHHELD
    # the rest of the doc is intact
    assert doc["emitted_patch"]["patch_type"] == "IMAGE_BUILD"
    assert doc["public_icps"]  # still populated


def test_clean_summary_passes_through():
    doc = _build()
    assert doc["emitted_patch"]["redacted_summary"].startswith("Add fallback")
