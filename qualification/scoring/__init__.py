"""
Qualification System: Scoring Module

This module implements the lead scoring system for the Lead Qualification
Agent competition. It includes:

- Pre-score validation (automatic zero checks) - Phase 5.0
- Intent signal verification - Phase 5.1
- Lead scoring (ICP fit, decision maker, intent signal) - Phase 5.2
- Champion selection logic - Phase 6.1
- Emissions distribution - Phase 6.2

CRITICAL: This is NEW scoring logic for qualification only.
Do NOT modify any existing validation in validator_models/automated_checks.py.
The qualification scoring is completely separate from the validator scoring.

Scoring Components:
- ICP Fit: 0-20 points
- Decision Maker: 0-30 points
- Intent Signal: 0-50 points (with time decay)
- Penalties: Cost and time deductions

Max Score per Lead: 100 points
"""

from qualification.scoring.pre_checks import (
    run_automatic_zero_checks,
    validate_email,
    check_data_quality,
    check_industry_match,
    check_sub_industry_match,
    check_role_match,
    check_cost_limit,
    check_time_limit,
    check_duplicate_company,
    # Validation result type
    ValidationResult,
)

from qualification.scoring.intent_verification import (
    # Main verification function
    verify_intent_signal,
    # Content fetching
    fetch_url_content,
    scrapingdog_linkedin,
    scrapingdog_jobs,
    scrapingdog_generic,
    github_api,
    # URL parsing
    extract_linkedin_id,
    extract_github_info,
    # Content extraction
    extract_verification_content,
    # LLM verification
    llm_verify_claim,
    openrouter_chat,
    # Cache functions
    compute_cache_key,
    get_cached_verification,
    cache_verification,
    clear_cache,
    get_cache_stats,
    # Batch verification
    verify_intent_signals_batch,
    # Configuration
    is_verification_configured,
    get_verification_config,
    # Types
    VerificationResult,
    CachedVerification,
)

from qualification.scoring.lead_scorer import (
    # Main scoring function
    score_lead,
    # Individual scoring functions
    score_icp_fit,
    score_decision_maker,
    score_intent_signal,
    # Time decay
    calculate_age_months,
    calculate_time_decay_multiplier,
    # Helpers
    extract_score,
    # Batch scoring
    score_leads_batch,
    summarize_scores,
    # Constants
    MAX_ICP_FIT_SCORE,
    MAX_DECISION_MAKER_SCORE,
    MAX_INTENT_SIGNAL_SCORE,
    MAX_TOTAL_SCORE,
)

from qualification.scoring.champion import (
    # Main champion selection
    run_champion_selection,
    champion_rebenchmark,
    check_evaluation_set_rotation,
    # Database operations (placeholders)
    get_current_champion,
    get_finished_models,
    set_champion,
    dethrone_champion,
    get_model_score,
    create_evaluation,
    log_champion_selected,
    # Utilities
    calculate_margin,
    is_valid_dethrone_margin,
    get_champion_history,
    get_current_set_id,
    get_champion_selection_summary,
    # Testing helpers
    reset_champion_state,
    set_mock_champion,
    # Types
    ChampionInfo,
    ModelScore,
    ChampionSelectionResult,
)

from qualification.scoring.emissions import (
    # Main emissions function
    distribute_emissions,
    # Hotkey verification
    is_hotkey_registered,
    # Transparency log
    log_emissions_event,
    # Weight calculation helpers
    get_champion_weight_allocation,
    get_champion_for_weights,
    calculate_weight_with_champion,
    # History and summary
    get_emissions_history,
    get_emissions_summary,
    get_emissions_config,
    # Testing helpers
    reset_emissions_history,
    add_mock_emissions_result,
    # Types
    EmissionsResult,
    EmissionsSummary,
)

__all__ = [
    # Pre-checks (Phase 5.0)
    "run_automatic_zero_checks",
    "validate_email",
    "check_data_quality",
    "check_industry_match",
    "check_sub_industry_match",
    "check_role_match",
    "check_cost_limit",
    "check_time_limit",
    "check_duplicate_company",
    "ValidationResult",
    # Intent verification (Phase 5.1)
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
    # Lead scoring (Phase 5.2)
    "score_lead",
    "score_icp_fit",
    "score_decision_maker",
    "score_intent_signal",
    "calculate_age_months",
    "calculate_time_decay_multiplier",
    "extract_score",
    "score_leads_batch",
    "summarize_scores",
    "MAX_ICP_FIT_SCORE",
    "MAX_DECISION_MAKER_SCORE",
    "MAX_INTENT_SIGNAL_SCORE",
    "MAX_TOTAL_SCORE",
    # Champion selection (Phase 6.1)
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
    # Emissions distribution (Phase 6.2)
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
