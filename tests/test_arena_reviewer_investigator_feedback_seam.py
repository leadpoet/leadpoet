"""Frozen50de retains private failure sources and exact acquisition grading."""
import copy
import hashlib
import json

import pytest

from scripts import arena_reviewer_active_acquisition_seam as acquisition
from scripts import arena_reviewer_investigator_feedback_seam as seam
from scripts import arena_reviewer_visible_auxiliary_context_seam as visible
from scripts import build_arena_reviewer_runtime as composer
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.reviewer_frozen_sources import frozen_source

FINALITY = composer.FINALITY_FILE
INVESTIGATOR = composer.GRADER_FILE
FROZEN_FINALITY = frozen_source("submitted_company_context_d657", FINALITY)
CORRECTED_FINALITY = acquisition.convert_active_acquisition_seam(
    visible.convert_submitted_company_context_seam(FROZEN_FINALITY),
)
FLAGS = ["--submitted-company-context-only", "--active-acquisition-target", "--investigator-feedback"]


def original_investigator():
    source = (composer.ROOT / INVESTIGATOR).read_bytes()
    # The source correction can be stacked before this composer. Reverse only
    # its exact reviewed delta, then require the complete historical checksum.
    if hashlib.sha256(source).hexdigest() != seam.ORIGINAL_SHA256:
        for old, new in reversed(seam.DELTAS):
            assert source.count(new) == 1
            source = source.replace(new, old, 1)
    assert hashlib.sha256(source).hexdigest() == seam.ORIGINAL_SHA256
    return source


@pytest.mark.parametrize("flags", [
    ["--investigator-feedback"],
    ["--submitted-company-context-only", "--investigator-feedback"],
    ["--active-acquisition-target", "--investigator-feedback"],
])
def test_feedback_requires_both_existing_corrections_before_docker(monkeypatch, flags):
    monkeypatch.setattr(composer.sys, "argv", ["composer", "--grading-image", "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST, *flags])
    monkeypatch.setattr(composer, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(composer.CompositionError, match="requires submitted-company-context-only and active-acquisition-target"):
        composer.main()


@pytest.mark.parametrize("digest", [visible.SUBMITTED_FROZEN_IMAGE_DIGEST, "sha256:" + "0" * 64])
def test_feedback_rejects_other_frozen_bases_before_docker(monkeypatch, digest):
    monkeypatch.setattr(composer.sys, "argv", ["composer", "--grading-image", "registry.example/reviewer@" + digest, *FLAGS])
    monkeypatch.setattr(composer, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(composer.CompositionError, match="exact frozen50de"):
        composer.main()


@pytest.mark.parametrize("flag", ["--shim-only", "--complete-negative-finality", "--verified-display-brand", "--source-access-denied"])
def test_feedback_cannot_import_unreviewed_corrections(monkeypatch, flag):
    monkeypatch.setattr(composer.sys, "argv", ["composer", "--grading-image", "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST, *FLAGS, flag])
    monkeypatch.setattr(composer, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(composer.CompositionError, match="cannot include other corrections"):
        composer.main()


@pytest.mark.parametrize("verify_existing", [False, True])
def test_composer_preserves_frozen_layers_process_and_acquisition(monkeypatch, capsys, verify_existing):
    frozen = original_investigator()
    corrected = seam.convert_investigator_feedback_seam(frozen)
    base_ref = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    tag = composer.bounded_build_tag("registry.example/reviewer:reviewer-" + seam.FROZEN_IMAGE_DIGEST[7:19] + "-" + "c" * 12 + "-submitted-company-context-active-acquisition-target-investigator-feedback")
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else tag
    argv = ["composer", "--grading-image", base_ref, *FLAGS]
    if verify_existing:
        argv += ["--verify-image", candidate_ref]
    monkeypatch.setattr(composer.sys, "argv", argv)
    monkeypatch.setattr(composer, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {"Env": ["FROZEN=1"], "Entrypoint": ["python", "scorer_entrypoint.py"], "User": "65534:65534"}, "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda ref: base if ref == base_ref else candidate)
    def extract(ref, _destination, **flags):
        assert ref == base_ref and flags == {"submitted_company_context_only": True, "investigator_feedback": True}
        return {FINALITY: FROZEN_FINALITY, INVESTIGATOR: frozen}
    expected = {FINALITY: CORRECTED_FINALITY, INVESTIGATOR: corrected}
    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay"
        assert files == expected
        composer.verify_overlay(layer([("model/" + path, data) for path, data in files.items()]), expected)
    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **_k: builds.append(command))
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"] == [visible.SUBMITTED_CORRECTION_ID, acquisition.CORRECTION_ID, seam.CORRECTION_ID]
    assert proof["original_files"] == {FINALITY: visible.SUBMITTED_ORIGINAL_SHA256, INVESTIGATOR: seam.ORIGINAL_SHA256}
    assert proof["overlay_files"] == {path: hashlib.sha256(data).hexdigest() for path, data in expected.items()}
    assert proof["overlay_files"][FINALITY] == "7994f7574b5e955d56c86d8a3a73a41f30d4dda302e7e0fcfa4a0e197f3d7bcd"
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)


def test_exact_reviewed_delta_hashes_idempotence_and_other_source_bytes():
    import ast
    original = original_investigator()
    corrected = seam.convert_investigator_feedback_seam(original)
    assert hashlib.sha256(corrected).hexdigest() == seam.REPAIRED_SHA256
    assert corrected == (composer.ROOT / INVESTIGATOR).read_bytes()
    assert seam.convert_investigator_feedback_seam(corrected) == corrected
    restored = corrected
    for old, new in reversed(seam.DELTAS):
        assert restored.count(new) == 1
        restored = restored.replace(new, old, 1)
    assert restored == original
    allowed = {"_SYSTEM_PROMPT", "_tools", "_final_submission_feedback", "_semantic_ownership_proof", "_validated_findings", "investigate_company_evidence"}
    def unchanged(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (
            getattr(node, "name", "") in allowed
            or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in allowed for target in node.targets))
        )]
        return ast.dump(tree)
    # Models, caps, security/evidence helpers, imports, and other runtime paths.
    assert unchanged(original) == unchanged(corrected)


def previous_feedback_investigator():
    """Recover the deployed v1 source, before only the six private handoffs."""
    source = original_investigator()
    for old, new in seam.DELTAS[:-7]:
        assert source.count(old) == 1
        source = source.replace(old, new, 1)
    assert hashlib.sha256(source).hexdigest() == "9ddfd54dba12069b74d5993df8900145d6211365416614846a4a180a73e57ef9"
    return source


@pytest.mark.parametrize("final", ["verified", "unproven", "bad_quote", "provider", "research"])
def test_exact_frozen_overlay_keeps_late_correction_bounded(monkeypatch, final):
    from tests import test_company_evidence_late_submission as controls
    corrected = seam.convert_investigator_feedback_seam(original_investigator())
    old, new = seam.DELTAS[-1]
    previous = corrected.replace(new, old, 1)
    assert hashlib.sha256(previous).hexdigest() == "14f2bf1fbe3dbcf384c9fb5760afb04c0c412028226fdca62b94b1ad37e9a37d"
    for source, repaired in ((previous, False), (corrected, True)):
        module = _install(monkeypatch, source, "_frozen_late_correction_" + str(repaired))
        with monkeypatch.context() as scope:
            scope.setattr(controls, "investigator", module)
            scope.setitem(controls._run_industry.__globals__, "investigator", module)
            if repaired:
                controls.test_late_ordinary_submission_gets_only_one_submit_correction(scope, 180.0, final)
            else:
                with pytest.raises(AssertionError):
                    controls.test_late_ordinary_submission_gets_only_one_submit_correction(scope, 180.0, final)


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("fault,reason", [
    ("http", "provider_error"), ("timeout", "provider_error"),
    ("connection", "provider_error"), ("json", "malformed_response"),
    ("unexpected", "unexpected_verifier_error"),
])
def test_exact_frozen_overlay_preserves_fresh_sources_without_finishing_judgment(
    monkeypatch, positive, fault, reason,
):
    from tests import test_company_evidence_failure_source_handoff as controls
    before = _install(monkeypatch, previous_feedback_investigator(), "_frozen_source_handoff_before")
    after = _install(monkeypatch, seam.convert_investigator_feedback_seam(original_investigator()), "_frozen_source_handoff_after")
    for module, fails in ((before, True), (after, False)):
        with monkeypatch.context() as scope:
            scope.setattr(controls, "investigator", module)
            # The shared helper patches its module; use that same exact frozen
            # module for all fetch, clock, model and capture substitutions.
            scope.setitem(controls._investigate.__globals__, "investigator", module)
            if fails:
                with pytest.raises(KeyError, match="_server_fetched_pages"):
                    controls.test_fresh_exact_page_survives_later_failure(scope, positive, fault, reason)
            else:
                controls.test_fresh_exact_page_survives_later_failure(scope, positive, fault, reason)


def test_exact_frozen_overlay_keeps_all_source_handoff_controls(monkeypatch):
    from tests import test_company_evidence_failure_source_handoff as controls
    after = _install(monkeypatch, seam.convert_investigator_feedback_seam(original_investigator()), "_frozen_source_handoff_controls")
    cases = [
        (controls.test_native_industry_advisory_failure_retains_loaded_pages, (False,)),
        (controls.test_native_industry_advisory_failure_retains_loaded_pages, (True,)),
        (controls.test_reasoning_fallthrough_retains_loaded_pages, ()),
        (controls.test_failed_page_cannot_become_a_retained_success, ()),
        (controls.test_failure_handoff_does_not_swallow_cancellation, ()),
        (controls.test_completed_submission_handoff_is_unchanged, ()),
        (controls.test_missing_credentials_still_returns_before_source_initialization, ()),
        *((controls.test_failure_handoff_keeps_existing_prefetch_bounds, (excess,))
          for excess in ("none", "count", "text")),
    ]
    for control, arguments in cases:
        with monkeypatch.context() as scope:
            scope.setattr(controls, "investigator", after)
            scope.setitem(controls._investigate.__globals__, "investigator", after)
            control(scope, *arguments)


@pytest.mark.parametrize("change", ["empty", "encoding", "newline", "crlf", "model", "partial", "old_feedback", "postimage_drift"])
def test_unknown_or_unreviewed_investigator_source_fails_closed(change):
    original = original_investigator()
    corrected = seam.convert_investigator_feedback_seam(original)
    candidates = {
        "empty": b"", "encoding": b"\xff", "newline": original + b"\n",
        "crlf": original.replace(b"\n", b"\r\n"),
        "model": original.replace(b"INVESTIGATOR_MODEL", b"UNREVIEWED_MODEL", 1),
        "partial": original.replace(*seam.DELTAS[0], 1),
        "old_feedback": previous_feedback_investigator(),
        "postimage_drift": corrected + b"\n",
    }
    with pytest.raises(ValueError, match="unrecognized"):
        seam.convert_investigator_feedback_seam(candidates[change])


def test_unreviewed_transform_cannot_pass_approved_postimage(monkeypatch):
    original = original_investigator()
    old, new = seam.DELTAS[0]
    monkeypatch.setattr(seam, "DELTAS", ((old, new + b"\n"), *seam.DELTAS[1:]))
    with pytest.raises(ValueError, match="postimage differs"):
        seam.convert_investigator_feedback_seam(original)


def _install(monkeypatch, source, name):
    import sys
    from types import ModuleType
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(source, name + ".py", "exec"), vars(module))
    return module


def test_exact_frozen_overlay_repairs_native_feedback_and_keeps_verdicts(monkeypatch):
    from tests import test_company_evidence_final_target_guidance as final_controls
    from tests import test_investigator_ownership_contract as ownership_controls
    from tests import test_private_equity_rejection_feedback as feedback_controls
    from tests import test_private_equity_prompt_contract as prompt_controls
    original = original_investigator()
    before = _install(monkeypatch, original, "_frozen_feedback_before")
    after = _install(monkeypatch, seam.convert_investigator_feedback_seam(original), "_frozen_feedback_after")
    # Reproduce the actual forced-final request defect with all I/O mocked.
    with monkeypatch.context() as scope:
        scope.setattr(final_controls, "investigator", before)
        with pytest.raises(AssertionError):
            final_controls.test_ordinary_rejection_becomes_submit_only_on_next_admission(scope)
    cases = [
        ownership_controls.finding(),
        ownership_controls.finding(ownership={**ownership_controls.OWNERSHIP, "control_status": "minority"}),
        ownership_controls.finding(ownership={**ownership_controls.OWNERSHIP, "controlling_owner": "Clear Capital"}),
        ownership_controls.finding(evidence_quote=ownership_controls.CONTROL.replace("Clear Capital", "Other Company")),
        ownership_controls.finding(supporting_evidence_quote_1="Pinecrest Capital invests through equity funds."),
        ownership_controls.finding(evidence_quote="Clear Capital is controlled by Pinecrest Capital, a private-markets firm."),
    ]
    for raw in cases:
        outputs = []
        for module in (before, after):
            monkeypatch.setattr(ownership_controls, "investigator", module)
            outputs.append(ownership_controls.validate(raw, requested_stage="Private Equity"))
        assert outputs[0] == outputs[1]
    monkeypatch.setattr(ownership_controls, "investigator", after)
    monkeypatch.setattr(final_controls, "investigator", after)
    monkeypatch.setattr(feedback_controls, "investigator", after)
    monkeypatch.setattr(prompt_controls, "inv", after)
    final_controls.test_ordinary_rejection_becomes_submit_only_on_next_admission(monkeypatch)
    final_controls.test_captured76_final_admission_removes_stale_fetch_instruction()
    prompt_controls.test_system_and_schema_explain_conditional_literal_grounding()
    prompt_controls.test_literal_private_markets_direct_and_linked_proofs_remain_valid()
    for portfolio_primary in (False, True):
        feedback_controls.test_retained_public_635_quotes_keep_decision_and_explain_failed_links(portfolio_primary)
    for repair_to_positive in (False, True):
        feedback_controls.test_controller_receives_detail_and_can_repair_or_submit_honest_negative(monkeypatch, repair_to_positive)


@pytest.mark.parametrize("change", [None, "unpatched_investigator", "unpatched_grader", "extra", "missing"])
def test_actual_two_file_overlay_rejects_changed_unreviewed_or_missing_source(change):
    original = original_investigator()
    expected = {FINALITY: CORRECTED_FINALITY, INVESTIGATOR: seam.convert_investigator_feedback_seam(original)}
    files = dict(expected)
    if change == "unpatched_investigator":
        files[INVESTIGATOR] = original
    elif change == "unpatched_grader":
        files[FINALITY] = FROZEN_FINALITY
    elif change == "extra":
        files["lab_arena/scorer_entrypoint.py"] = b"unreviewed entrypoint\n"
    elif change == "missing":
        files.pop(FINALITY)
    data = layer([("model/" + path, source) for path, source in files.items()])
    if change is None:
        composer.verify_overlay(data, expected)
    else:
        with pytest.raises(composer.CompositionError):
            composer.verify_overlay(data, expected)
