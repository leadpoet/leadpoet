"""Run native producer regressions against the actual composed frozen functions."""
import pytest

from qualification import scoring
from scripts import arena_reviewer_identity_handoff_seam as seam
from scripts import build_arena_reviewer_runtime as builder
from tests import test_homepage_child_profile_lookup as itw
from tests import test_native_parenthetical_identity_handoff as mec
from tests import test_company_headcount_evidence as headcount
from tests import test_reversed_parenthetical_attribution as attribution
from tests.test_arena_reviewer_identity_handoff_seam import selected_sources
from tests.test_arena_reviewer_investigator_feedback_seam import _install


def install_pair(monkeypatch, sources):
    investigator = _install(monkeypatch, sources[builder.GRADER_FILE],
                            "qualification.scoring.company_evidence_investigator")
    monkeypatch.setattr(scoring, "company_evidence_investigator", investigator)
    scorer = _install(monkeypatch, sources[builder.FINALITY_FILE], "_frozen_identity_scorer")
    for controls, name, value in (
        (itw, "scorer", scorer), (headcount, "scorer", scorer),
        (mec, "lead_scorer", scorer), (mec, "inv", investigator),
        (attribution, "lead_scorer", scorer), (attribution, "inv", investigator),
    ):
        monkeypatch.setattr(controls, name, value)
    assert scorer._complete_verified_first_party_identity is investigator._complete_verified_first_party_identity


@pytest.fixture(autouse=True)
def frozen_identity_pair(monkeypatch):
    install_pair(monkeypatch, seam.convert_identity_handoff_files(selected_sources()))


@pytest.mark.parametrize("case", ["itw_lookup", "itw_caller", "mec_recovery", "mec_investigator"])
def test_selected_runtime_reproduces_native_handoff_failure_before_correction(monkeypatch, case):
    install_pair(monkeypatch, selected_sources())
    with pytest.raises(AssertionError):
        if case == "itw_lookup":
            itw.test_retained_child_homepage_locates_profile_without_approving_identity(monkeypatch)
        elif case == "itw_caller":
            itw.test_actual_fit_caller_uses_one_profile_lookup_and_keeps_unresolved_alias(monkeypatch, None)
        elif case == "mec_recovery":
            mec.test_native_receipt_recovery_keeps_exact_legal_name_after_json_roundtrip()
        else:
            mec.test_native_targeted_investigator_preserves_proof_not_a_bare_acronym(monkeypatch, mec.LEGAL, "VERIFIED")


# Reuse every native producer/consumer case and its parameterized controls;
# their module globals above bind to the complete frozen source, not main.
test_frozen_itw_lookup = itw.test_retained_child_homepage_locates_profile_without_approving_identity
test_frozen_itw_lookup_controls = itw.test_child_lookup_never_uses_conflicting_or_shared_host_identity
test_frozen_itw_binding_controls = itw.test_returned_profile_must_independently_prove_exact_root_company
test_frozen_itw_actual_caller = itw.test_actual_fit_caller_uses_one_profile_lookup_and_keeps_unresolved_alias
test_frozen_mec_recovery = mec.test_native_receipt_recovery_keeps_exact_legal_name_after_json_roundtrip
test_frozen_mec_recovery_controls = mec.test_saved_recovery_rejects_missing_or_conflicting_proof
test_frozen_mec_actual_investigator = mec.test_native_targeted_investigator_preserves_proof_not_a_bare_acronym
test_frozen_no_approximate_count = mec.test_approximate_count_and_derived_band_are_still_invalid
test_frozen_reversed_identity = attribution.test_exact_reversed_legal_initialism_retains_fully_bound_first_party_identity
test_frozen_independent_industry = attribution.test_actual_validation_keeps_quote_and_industry_conditions_separate
test_frozen_attribution_controls = attribution.test_reversal_never_supplies_missing_or_conflicting_identity
test_frozen_exact_source_controls = attribution.test_exact_quote_and_independent_requested_final_domains_still_required
test_frozen_saved_attribution = attribution.test_saved_attribute_recovery_consumer_roundtrip_preserves_identity_metadata
test_frozen_saved_attribution_controls = attribution.test_saved_consumer_never_promotes_an_unavailable_or_conflicting_identity
