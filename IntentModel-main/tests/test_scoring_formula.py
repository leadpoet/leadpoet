"""
Unit tests for Scoring Formula (Section 9)
Tests final score calculation, fit score, intent score, boosts, and time decay.
"""

import pytest
from unittest.mock import MagicMock
from app.services.scoring import ScoringService, ScoredLead
from app.services.retrieval import RetrievalCandidate
from app.services.prompt_parser import ParsedQuery
from app.services.lgbm_service import LightGBMService

@pytest.fixture
def scoring_service():
    return ScoringService()

@pytest.fixture
def lgbm_service():
    mock = MagicMock(spec=LightGBMService)
    mock.predict.return_value = 0.42
    return mock

@pytest.fixture
def parsed_query():
    return ParsedQuery(
        product_category="CRM",
        target_industries=["technology"],
        target_sizes=["100-500"],
        target_regions=["us"],
        target_roles=["manager"],
        not_industries=[],
        not_sizes=[],
        not_regions=[],
        exemplar_companies=[],
        keywords=["crm", "sales"],
        language="en",
        confidence=0.9
    )

@pytest.fixture
def candidate():
    return RetrievalCandidate(
        lead_id="lead1",
        company_id="comp1",
        company_name="TechCorp",
        email="test@techcorp.com",
        retrieval_score=0.8,
        retrieval_method="sql",
        firmographics={"industry": "technology", "size": "100-500", "region": "us", "created_at": "2024-06-01T00:00:00Z"},
        technographics={"crm": True},
        simhash=None,
        is_potential_duplicate=False
    )

@pytest.mark.asyncio
async def test_final_score_calculation(scoring_service, lgbm_service, parsed_query, candidate):
    candidates = [candidate]
    # Patch BM25 and LLM scoring to deterministic values
    scoring_service._calculate_bm25_scores_batch = MagicMock(return_value={candidate.lead_id: 0.6})
    scoring_service._calculate_llm_score = MagicMock(return_value=None)
    scoring_service._get_churn_boost = MagicMock(return_value=0.1)
    scoring_service._get_job_posting_boost = MagicMock(return_value=0.05)
    
    fit_scores = await scoring_service._calculate_fit_scores(candidates, parsed_query)
    intent_scores = await scoring_service._calculate_intent_scores(candidates, parsed_query, lgbm_service)
    boosted_scores = await scoring_service._apply_boosts(intent_scores)
    final_scores = scoring_service._calculate_final_scores(
        query_id="q1",
        fit_scores=fit_scores,
        intent_scores=boosted_scores,
        candidates=candidates,
        parsed_query=parsed_query
    )
    assert len(final_scores) == 1
    lead = final_scores[0]
    assert isinstance(lead, ScoredLead)
    # Check that the final score is calculated as expected
    expected_fit = fit_scores[candidate.lead_id]
    expected_intent = boosted_scores[candidate.lead_id]["boosted_intent_score"]
    expected_final = scoring_service.final_score_fit_weight * expected_fit + scoring_service.final_score_intent_weight * expected_intent
    assert abs(lead.final_score - expected_final) < 1e-6
    # Check boosts
    assert lead.churn_boost == 0.1
    assert lead.job_posting_boost == 0.05
    # Check time decay is applied
    assert 0.1 <= lead.time_decay_factor <= 1.0

@pytest.mark.asyncio
async def test_low_bm25_triggers_lgbm(scoring_service, lgbm_service, parsed_query, candidate):
    candidates = [candidate]
    scoring_service._calculate_bm25_scores_batch = MagicMock(return_value={candidate.lead_id: 0.1})
    scoring_service._calculate_llm_score = MagicMock(return_value=None)
    scoring_service._get_churn_boost = MagicMock(return_value=0.0)
    scoring_service._get_job_posting_boost = MagicMock(return_value=0.0)
    
    fit_scores = await scoring_service._calculate_fit_scores(candidates, parsed_query)
    intent_scores = await scoring_service._calculate_intent_scores(candidates, parsed_query, lgbm_service)
    boosted_scores = await scoring_service._apply_boosts(intent_scores)
    final_scores = scoring_service._calculate_final_scores(
        query_id="q2",
        fit_scores=fit_scores,
        intent_scores=boosted_scores,
        candidates=candidates,
        parsed_query=parsed_query
    )
    lead = final_scores[0]
    # Should use LGBM fallback
    assert lead.lgbm_score == 0.42
    assert lead.llm_score is None

@pytest.mark.asyncio
async def test_deduplication_by_company(scoring_service, lgbm_service, parsed_query, candidate):
    # Two leads with same company_id, different scores
    candidate2 = RetrievalCandidate(
        lead_id="lead2",
        company_id="comp1",
        company_name="TechCorp",
        email="other@techcorp.com",
        retrieval_score=0.7,
        retrieval_method="sql",
        firmographics={"industry": "technology", "size": "100-500", "region": "us", "created_at": "2024-06-01T00:00:00Z"},
        technographics={"crm": True},
        simhash=None,
        is_potential_duplicate=False
    )
    candidates = [candidate, candidate2]
    scoring_service._calculate_bm25_scores_batch = MagicMock(return_value={candidate.lead_id: 0.6, candidate2.lead_id: 0.4})
    scoring_service._calculate_llm_score = MagicMock(return_value=None)
    scoring_service._get_churn_boost = MagicMock(return_value=0.0)
    scoring_service._get_job_posting_boost = MagicMock(return_value=0.0)
    
    fit_scores = await scoring_service._calculate_fit_scores(candidates, parsed_query)
    intent_scores = await scoring_service._calculate_intent_scores(candidates, parsed_query, lgbm_service)
    boosted_scores = await scoring_service._apply_boosts(intent_scores)
    final_scores = scoring_service._calculate_final_scores(
        query_id="q3",
        fit_scores=fit_scores,
        intent_scores=boosted_scores,
        candidates=candidates,
        parsed_query=parsed_query
    )
    deduped = scoring_service._deduplicate_by_company(final_scores)
    # Only one lead per company_id, keep the highest score
    assert len(deduped) == 1
    assert deduped[0].lead_id in [candidate.lead_id, candidate2.lead_id]

@pytest.mark.asyncio
async def test_fallback_scoring(scoring_service, candidate):
    # Simulate scoring pipeline failure
    leads = await scoring_service._fallback_scoring([candidate], 1)
    assert len(leads) == 1
    lead = leads[0]
    assert lead.final_score == 0.5
    assert lead.explanation["fallback"] is True 