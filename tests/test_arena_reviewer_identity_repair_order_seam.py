"""Exact b699 one-file correction, offline behavior and cache preservation."""
import ast
import copy
import hashlib
import json
import sys
from types import ModuleType

import pytest

pytestmark = pytest.mark.usefixtures("scorer_process_environment")

from lab_arena import company_judgments, contracts, scorer_entrypoint, scoring
from qualification.scoring import competition
from scripts import build_arena_reviewer_runtime as composer
from scripts import arena_reviewer_identity_repair_order_seam as seam
from tests import test_identity_repair_investigation_order as controls
from tests.lab_arena.company_judgments_test import _input, _miss, _evidence
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.lab_arena import verified_fit_checkpoint_test as checkpoints
from tests.lab_arena import legacy_company_judgments_test as legacy
from tests.reviewer_frozen_sources import frozen_source
from tests.test_arena_company_quality_policy import _positive_breakdown

ORIGINAL = frozen_source("identity_repair_order_b699", seam.FINALITY_FILE)
CORRECTED = seam.convert_identity_repair_order_seam(ORIGINAL)


def _install(monkeypatch, source):
    module = ModuleType("_arena_frozen_b699_identity_repair_order")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, "exact_frozen_b699_lead_scorer.py", "exec"), vars(module))
    monkeypatch.setattr(controls, "scorer", module)
    return module


def test_exact_source_changes_only_the_reviewed_twenty_line_insertion():
    assert hashlib.sha256(ORIGINAL).hexdigest() == seam.ORIGINAL_SHA256
    assert hashlib.sha256(CORRECTED).hexdigest() == seam.REPAIRED_SHA256
    assert seam.convert_identity_repair_order_seam(CORRECTED) == CORRECTED
    assert CORRECTED == ORIGINAL.replace(
        seam._ANCHOR.encode(), (seam._INSERTION + seam._ANCHOR).encode(), 1,
    )
    assert len(seam._INSERTION.splitlines()) == 20
    assert seam._INSERTION in (composer.ROOT / seam.FINALITY_FILE).read_text()
    before, after = ast.parse(ORIGINAL), ast.parse(CORRECTED)
    for tree in (before, after):
        tree.body = [node for node in tree.body if getattr(node, "name", "") != seam.FUNCTION]
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize("source", [
    b"", b"\xff", ORIGINAL + b"\n", ORIGINAL.replace(b"\n", b"\r\n"),
    ORIGINAL + ORIGINAL, CORRECTED + b"# unrelated\n",
    ORIGINAL.replace(b'"rebrand"', b'"unknown"', 1),
    ORIGINAL.replace(seam._ANCHOR.encode(), (seam._INSERTION[:50] + seam._ANCHOR).encode(), 1),
])
def test_unknown_or_partial_sources_fail_closed(source):
    with pytest.raises(seam.IdentityRepairOrderSeamError):
        seam.convert_identity_repair_order_seam(source)


@pytest.mark.parametrize("verify_existing", [False, True])
def test_exclusive_composer_build_and_verification_are_one_file(monkeypatch, capsys, verify_existing):
    base_ref = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    tag = "registry.example/reviewer:reviewer-" + seam.FROZEN_IMAGE_DIGEST[7:19] + "-" + "c" * 12 + "-identity-repair-order"
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else tag
    args = ["build_arena_reviewer_runtime.py", "--grading-image", base_ref,
            "--identity-repair-before-investigation-only"]
    if verify_existing:
        args += ["--verify-image", candidate_ref]
    monkeypatch.setattr(composer.sys, "argv", args)
    monkeypatch.setattr(composer, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda ref: base if ref == base_ref else candidate)

    def extract(ref, _destination, **flags):
        assert ref == base_ref
        assert flags == {"identity_repair_before_investigation_only": True}
        return {seam.FINALITY_FILE: ORIGINAL}

    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay"
        assert files == {seam.FINALITY_FILE: CORRECTED}

    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **_k: builds.append(command))
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"] == [seam.CORRECTION_ID]
    assert proof["original_files"] == {seam.FINALITY_FILE: seam.ORIGINAL_SHA256}
    assert proof["overlay_files"] == {seam.FINALITY_FILE: seam.REPAIRED_SHA256}
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)


@pytest.mark.parametrize("other", [
    "--shim-only", "--empty-domain-negative-finality-only", "--identity-schema-repair",
    "--verified-display-brand", "--complete-negative-finality", "--source-access-denied",
    "--intent-target-denial", "--homepage-profile-alias",
])
def test_exclusive_mode_rejects_mixed_corrections_before_docker(monkeypatch, other):
    monkeypatch.setattr(composer.sys, "argv", [
        "build_arena_reviewer_runtime.py", "--grading-image", "registry.example/base@" + seam.FROZEN_IMAGE_DIGEST,
        "--identity-repair-before-investigation-only", other,
    ])
    with pytest.raises(composer.CompositionError, match="cannot include other corrections"):
        composer.main()


def test_mode_rejects_other_grading_image_before_docker(monkeypatch):
    monkeypatch.setattr(composer.sys, "argv", [
        "build_arena_reviewer_runtime.py", "--grading-image", "registry.example/base@sha256:" + "a" * 64,
        "--identity-repair-before-investigation-only",
    ])
    with pytest.raises(composer.CompositionError, match="exact frozen b699"):
        composer.main()


def test_overlay_rejects_extra_or_changed_files():
    expected = {seam.FINALITY_FILE: CORRECTED}
    composer.verify_overlay(layer([("model/" + seam.FINALITY_FILE, CORRECTED)]), expected)
    for files in (
        [("model/" + seam.FINALITY_FILE, CORRECTED), ("model/unrelated.py", b"pass\n")],
        [("model/" + seam.FINALITY_FILE, CORRECTED + b"\n")],
    ):
        with pytest.raises(composer.CompositionError):
            composer.verify_overlay(layer(files), expected)


@pytest.mark.parametrize("company_quality", [False, True])
def test_actual_b699_reproduces_gap_then_repaired_source_closes_it(monkeypatch, company_quality):
    _install(monkeypatch, ORIGINAL)
    before, events, investigations = controls._exercise(monkeypatch, company_quality=company_quality)
    assert events == ["lead_scorer_reverify", "investigate", "lead_scorer_reverify_schema_repair"]
    assert len(investigations) == 1 and "geography" not in investigations[0]["targets"]
    assert before.details["dimension_decisions"]["geography"] == "unavailable"
    assert before.details.get("failure_class") != "insufficient_fit_evidence"
    _install(monkeypatch, CORRECTED)
    after, events, investigations = controls._exercise(monkeypatch, company_quality=company_quality)
    assert events == ["lead_scorer_reverify", "lead_scorer_reverify_schema_repair", "investigate"]
    assert len(investigations) == 1 and "geography" in investigations[0]["targets"]
    assert after.details["failure_class"] == "insufficient_fit_evidence"


@pytest.mark.parametrize("name,kwargs", [
    ("test_identity_repair_precedes_one_investigation_of_final_gaps", {"geography": "VERIFIED"}),
    ("test_valid_identity_keeps_original_one_investigation_without_schema_repair", {}),
    ("test_all_verified_repaired_dimensions_can_match", {}),
    ("test_existing_rebrand_investigation_is_not_deferred", {}),
    *[("test_unresolved_or_changed_identity_cannot_qualify", {"repair": repair})
      for repair in ("other_entity", "missing", "provider_error", "malformed_response")],
    ("test_post_repair_investigation_keeps_provider_faults_retryable", {"failure": "provider_error", "source_fault": False}),
    ("test_post_repair_investigation_keeps_provider_faults_retryable", {"failure": "", "source_fault": True}),
])
def test_actual_corrected_b699_preserves_native_controls(monkeypatch, name, kwargs):
    _install(monkeypatch, CORRECTED)
    getattr(controls, name)(monkeypatch, **kwargs)


@pytest.mark.parametrize("company_quality", [False, True])
def test_four_completed_company_judgments_reused_only_fifth_is_evaluated(monkeypatch, company_quality):
    _install(monkeypatch, CORRECTED)
    document = _input() if company_quality else legacy._input(("A", "B", "C", "D", "E"))
    refs = company_judgments.build_company_scopes(
        scoring_input=document, round_id="arena-2026-11-05", network_name="finney", netuid=71,
        scorer_image_digest=seam.FROZEN_IMAGE_DIGEST,
        scorer_image_reference="registry/scorer@" + seam.FROZEN_IMAGE_DIGEST,
        integrity_policy="arena_integrity_v1",
        **({"company_quality_policy": "company_quality_v1"} if company_quality
           else {"company_judgment_policy": company_judgments.POLICY}),
    )
    hits = []
    for ref in refs[:4]:
        name = document["companies"][ref["company_index"]]["company_name"]
        positive = _positive_breakdown if company_quality else legacy._positive_breakdown
        row = positive(name, name.lower() + ".example", name.lower())
        if not company_quality:
            row = competition.apply_company_judgment_context(
                [document["companies"][ref["company_index"]]], [row], company_quality=False,
            )[0]
        raw = competition.raw_company_judgment(row)
        if not company_quality:
            raw["company_judgment_policy"] = company_judgments.POLICY
        if ref["company_index"] % 2:
            raw.update(final_score=0.0, failure_reason="company mismatch", intent_signals_detail=[])
            raw["verifier_gate_receipts"][0]["decision"] = "mismatch"
        evidence = _evidence(ref) if company_quality else legacy._snapshot(ref, raw)
        evidence["raw_judgment"] = raw
        hits.append({**_miss(ref), "evidence_hash": contracts.document_hash(evidence), "evidence_doc": evidence})
    document["company_judgment_cache"] = {
        "schema_version": company_judgments.LEASE_SCHEMA_VERSION,
        "hits": hits, "misses": [_miss(refs[4])],
    }
    before = copy.deepcopy(document)
    attempted = []

    def judge(companies, _buyer, _reference):
        assert [company["company_name"] for company in companies] == ["E"]
        attempted.append("E")
        result, events, investigations = controls._exercise(monkeypatch, company_quality=company_quality)
        assert len(investigations) == 1 and "geography" in investigations[0]["targets"]
        assert events == ["lead_scorer_reverify", "lead_scorer_reverify_schema_repair", "investigate"]
        positive = _positive_breakdown if company_quality else legacy._positive_breakdown
        raw = positive("E", "e.example", "e")
        raw.update(final_score=0.0, failure_reason="Company fit unavailable: unproven industry")
        raw["verifier_gate_receipts"][0] = result.receipt("company_fit")
        return competition.apply_company_judgment_context(companies, [raw], company_quality=company_quality)

    judge.company_quality = company_quality
    judge.integrity_policy = True
    judge.contacts_required = False
    monkeypatch.setattr(scoring, "lab_scorer", lambda *_a, **_k: judge)
    monkeypatch.setattr(scoring, "apply_policy_to_environment", lambda *_a, **_k: None)
    output = scorer_entrypoint.score_input(document)
    assert "failure" not in output
    assert attempted == ["E"]
    assert len(output["breakdowns"]) == 5
    assert [row["company_index"] for row in output["company_judgments"]] == [4]
    for row, hit in zip(output["breakdowns"][:4], before["company_judgment_cache"]["hits"]):
        assert row["verifier_gate_receipts"] == hit["evidence_doc"]["raw_judgment"]["verifier_gate_receipts"]
        assert row["intent_signals_detail"] == hit["evidence_doc"]["raw_judgment"]["intent_signals_detail"]
    assert document == before
    assert scoring.scoring_output_from_bytes(json.dumps(output).encode()) == output


def test_runtime_bound_intermediate_fit_checkpoint_remains_a_cache_miss():
    document = checkpoints.legacy._input(("Acme",))
    document["scorer_policy"]["signal_judgment_policy"] = checkpoints.signal_judgments.POLICY
    old = checkpoints.context(document)
    old.update(runtime_image_digest=seam.FROZEN_IMAGE_DIGEST,
               runtime_image_reference="registry/scorer@" + seam.FROZEN_IMAGE_DIGEST)
    new = {**old, "runtime_image_digest": "sha256:" + "f" * 64,
           "runtime_image_reference": "registry/scorer@sha256:" + "f" * 64}
    arguments = dict(scoring_input=document, company_index=0, round_id="arena-2026-11-05",
                     network_name="finney", netuid=71, scorer_image_digest=seam.FROZEN_IMAGE_DIGEST,
                     scorer_image_reference="registry/scorer@" + seam.FROZEN_IMAGE_DIGEST)
    old_scope = checkpoints.signal_judgments.build_fit_scope(**arguments, checkpoint_context=old)
    new_scope = checkpoints.signal_judgments.build_fit_scope(**arguments, checkpoint_context=new)
    assert old_scope["cache_key"] != new_scope["cache_key"]
    assert old_scope["scorer_image_digest"] == new_scope["scorer_image_digest"] == seam.FROZEN_IMAGE_DIGEST
    assert old["scoring_generation_floor"] == new["scoring_generation_floor"] == 0
    # Whole-company evidence is reused above. An unfinished company's old
    # runtime-bound intermediate checkpoint can miss and repeat one bounded
    # native investigation; this correction does not weaken that boundary.
