"""
Qualification System: Scoring Module

Validator-side scoring for the Lead Qualification Agent competition
(the "model competition").

As of May 2026 the competition is single-path company-mode: miners
submit ``CompanyOutput`` rows (companies surfaced from the open web)
against ``ICPPrompt`` instances; the validator scores each row with
``score_company``.  The historical lead-mode pipeline (DB row
equality + role / seniority / email checks + decision-maker LLM) has
been removed — see lead_scorer.py module docstring for rationale.

Scoring Components (per company):
- ICP Fit:        0-40 points  (industry + product + structural + intent fit)
- Intent Signal:  0-60 points  (per-signal verification + time decay,
                                averaged across all submitted signals)
- Penalties:      cost variability deduction

Max Score per Company: MAX_COMPANY_TOTAL_SCORE = 100 points

This module re-exports a curated public API; submodules
(``lead_scorer``, ``pre_checks``, ``intent_verification``,
``company_verification``, ``champion``, ``emissions``) may also be
imported directly.

NOTE: ``lead_scorer.py`` still contains shared helpers
(``_score_single_intent_signal``, ``_apply_signal_time_decay``,
``_extract_domain``, ``detect_structural_similarity``,
``calculate_age_months``, ``calculate_time_decay_multiplier``,
``extract_score``) that are imported directly by
``gateway/fulfillment/scoring.py`` for fulfillment-side lead ranking,
which still operates on contacts.  Do not move them.
"""

from qualification.scoring.pre_checks import (
    run_company_zero_checks,
    check_industry_match,
    check_sub_industry_match,
    check_country_match,
    check_cost_limit,
    check_time_limit,
    check_hard_time_limit,
    check_duplicate_company,
    ValidationResult,
)

from qualification.scoring.verification_helpers import (
    verify_intent_signal,
    fetch_url_content,
    scrapingdog_linkedin,
    scrapingdog_jobs,
    scrapingdog_generic,
    github_api,
    extract_linkedin_id,
    extract_github_info,
    extract_verification_content,
    llm_verify_claim,
    openrouter_chat,
    compute_cache_key,
    get_cached_verification,
    cache_verification,
    clear_cache,
    get_cache_stats,
    verify_intent_signals_batch,
    is_verification_configured,
    get_verification_config,
    VerificationResult,
    CachedVerification,
)

from qualification.scoring.lead_scorer import (
    score_company,
    score_company_icp_fit,
    score_company_intent_signal,
    calculate_age_months,
    calculate_time_decay_multiplier,
    extract_score,
    detect_structural_similarity,
    MAX_COMPANY_ICP_FIT_SCORE,
    MAX_COMPANY_INTENT_SIGNAL_SCORE,
    MAX_COMPANY_TOTAL_SCORE,
)

from qualification.scoring.company_verification import (
    verify_company_exists,
)

from qualification.scoring.champion import (
    run_champion_selection,
    champion_rebenchmark,
    check_evaluation_set_rotation,
    get_current_champion,
    get_finished_models,
    set_champion,
    dethrone_champion,
    get_model_score,
    create_evaluation,
    log_champion_selected,
    calculate_margin,
    is_valid_dethrone_margin,
    get_champion_history,
    get_current_set_id,
    get_champion_selection_summary,
    reset_champion_state,
    set_mock_champion,
    ChampionInfo,
    ModelScore,
    ChampionSelectionResult,
)

from qualification.scoring.emissions import (
    distribute_emissions,
    is_hotkey_registered,
    log_emissions_event,
    get_champion_weight_allocation,
    get_champion_for_weights,
    calculate_weight_with_champion,
    get_emissions_history,
    get_emissions_summary,
    get_emissions_config,
    reset_emissions_history,
    add_mock_emissions_result,
    EmissionsResult,
    EmissionsSummary,
)

__all__ = [
    # Pre-checks
    "run_company_zero_checks",
    "check_industry_match",
    "check_sub_industry_match",
    "check_country_match",
    "check_cost_limit",
    "check_time_limit",
    "check_hard_time_limit",
    "check_duplicate_company",
    "ValidationResult",
    # Intent verification
    "verify_intent_signal",
    "fetch_url_content",
    "scrapingdog_linkedin",
    "scrapingdog_jobs",
    "scrapingdog_generic",
    "github_api",
    "extract_linkedin_id",
    "extract_github_info",
    "extract_verification_content",
    "llm_verify_claim",
    "openrouter_chat",
    "compute_cache_key",
    "get_cached_verification",
    "cache_verification",
    "clear_cache",
    "get_cache_stats",
    "verify_intent_signals_batch",
    "is_verification_configured",
    "get_verification_config",
    "VerificationResult",
    "CachedVerification",
    # Company scoring
    "score_company",
    "score_company_icp_fit",
    "score_company_intent_signal",
    "calculate_age_months",
    "calculate_time_decay_multiplier",
    "extract_score",
    "detect_structural_similarity",
    "MAX_COMPANY_ICP_FIT_SCORE",
    "MAX_COMPANY_INTENT_SIGNAL_SCORE",
    "MAX_COMPANY_TOTAL_SCORE",
    # Company-existence verification
    "verify_company_exists",
    # Champion selection
    "run_champion_selection",
    "champion_rebenchmark",
    "check_evaluation_set_rotation",
    "get_current_champion",
    "get_finished_models",
    "set_champion",
    "dethrone_champion",
    "get_model_score",
    "create_evaluation",
    "log_champion_selected",
    "calculate_margin",
    "is_valid_dethrone_margin",
    "get_champion_history",
    "get_current_set_id",
    "get_champion_selection_summary",
    "reset_champion_state",
    "set_mock_champion",
    "ChampionInfo",
    "ModelScore",
    "ChampionSelectionResult",
    # Emissions distribution
    "distribute_emissions",
    "is_hotkey_registered",
    "log_emissions_event",
    "get_champion_weight_allocation",
    "get_champion_for_weights",
    "calculate_weight_with_champion",
    "get_emissions_history",
    "get_emissions_summary",
    "get_emissions_config",
    "reset_emissions_history",
    "add_mock_emissions_result",
    "EmissionsResult",
    "EmissionsSummary",
]
