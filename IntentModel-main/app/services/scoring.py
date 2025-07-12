"""
Scoring Service - F-3, F-4, F-5, F-6, F-7 Implementation
NumPy fit scoring, BM25, time decay, LLM fallback, boosts, and final ranking.
Target: 10ms P95 for scoring, 4ms for BM25, 120ms for LLM
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math
from datetime import datetime, timedelta

import numpy as np
import openai
from openai import AsyncOpenAI
from whoosh.fields import Schema, TEXT, NUMERIC
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from whoosh.analysis import StandardAnalyzer
import tempfile
import os

from app.core.config import settings
from app.core.redis_client import RedisClient
from app.services.retrieval import RetrievalCandidate
from app.services.prompt_parser import ParsedQuery
from app.services.lgbm_service import LightGBMService
from app.core.circuit_breaker import llm_circuit_breaker
from pybreaker import CircuitBreakerError
from app.core.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ScoredLead:
    """Scored lead with all scoring components."""
    query_id: str
    lead_id: str
    company_id: str
    company_name: str
    email: str
    fit_score: float
    intent_score: float
    final_score: float
    bm25_score: float
    time_decay_factor: float
    churn_boost: float
    job_posting_boost: float
    llm_score: Optional[float]
    lgbm_score: Optional[float]
    explanation: Dict[str, Any]
    firmographics: Dict[str, Any]
    technographics: Dict[str, Any]


class ScoringService:
    """Scoring service with BM25, time decay, LLM fallback, and boosts."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.redis_client = RedisClient()
        
        # Scoring timeouts
        self.scoring_timeout = settings.SCORING_TIMEOUT_MS / 1000.0
        self.bm25_timeout = settings.BM25_SCORING_TIMEOUT_MS / 1000.0
        self.llm_timeout = settings.LLM_SCORING_TIMEOUT_MS / 1000.0
        
        # BM25 configuration
        self.bm25_threshold = settings.BM25_THRESHOLD
        self.time_decay_tau = settings.TIME_DECAY_TAU
        
        # Scoring weights configuration
        self.fit_score_industry_weight = settings.FIT_SCORE_INDUSTRY_WEIGHT
        self.fit_score_size_weight = settings.FIT_SCORE_SIZE_WEIGHT
        self.fit_score_region_weight = settings.FIT_SCORE_REGION_WEIGHT
        self.final_score_fit_weight = settings.FINAL_SCORE_FIT_WEIGHT
        self.final_score_intent_weight = settings.FINAL_SCORE_INTENT_WEIGHT
        
        # Boost values (from BRD Section 9)
        self.churn_boost_value = settings.CHURN_BOOST_VALUE
        self.job_posting_boost_value = settings.JOB_POSTING_BOOST_VALUE
        
        # BM25 index management
        self.bm25_index = None
        self.bm25_temp_dir = None
        self.bm25_searcher = None
        self.bm25_query_parser = None
        self._init_bm25_index()
        
        self.metrics_collector = MetricsCollector()
        
        logger.info(f"Scoring service initialized with weights: fit_weights=[{self.fit_score_industry_weight}, {self.fit_score_size_weight}, {self.fit_score_region_weight}], final_weights=[{self.final_score_fit_weight}, {self.final_score_intent_weight}]")
    
    def _init_bm25_index(self):
        """Initialize BM25 search index."""
        try:
            # Create temporary directory for index
            self.bm25_temp_dir = tempfile.mkdtemp()
            
            # Define schema
            schema = Schema(
                lead_id=TEXT(stored=True),
                company_name=TEXT(stored=True),
                industry=TEXT(stored=True),
                size=TEXT(stored=True),
                region=TEXT(stored=True),
                content=TEXT(stored=True),
                created_at=NUMERIC(stored=True)
            )
            
            # Create index
            self.bm25_index = create_in(self.bm25_temp_dir, schema)
            logger.info("BM25 index initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            self.bm25_index = None
    
    async def _build_bm25_index(self, candidates: List[RetrievalCandidate]):
        """Pre-build BM25 index with all candidate documents."""
        if not self.bm25_index:
            logger.warning("BM25 index not available, skipping index building")
            return
        
        try:
            start_time = time.time()
            
            # Clear existing index
            writer = self.bm25_index.writer()
            
            # Add all candidate documents to index
            for candidate in candidates:
                content = await self._get_candidate_content(candidate)
                
                writer.add_document(
                    lead_id=candidate.lead_id,
                    company_name=candidate.company_name,
                    industry=candidate.firmographics.get("industry", ""),
                    size=candidate.firmographics.get("size", ""),
                    region=candidate.firmographics.get("region", ""),
                    content=content,
                    created_at=int(time.time())
                )
            
            # Commit all documents at once
            writer.commit()
            
            # Create searcher and query parser for reuse
            self.bm25_searcher = self.bm25_index.searcher()
            self.bm25_query_parser = QueryParser("content", self.bm25_index.schema)
            
            build_time = (time.time() - start_time) * 1000
            logger.info(f"BM25 index built with {len(candidates)} documents in {build_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_searcher = None
            self.bm25_query_parser = None
    
    async def score_and_rank(
        self, 
        query_id: str,
        candidates: List[RetrievalCandidate], 
        parsed_query: ParsedQuery, 
        desired_count: int,
        lgbm_service: LightGBMService
    ) -> List[ScoredLead]:
        """
        Score and rank leads using the complete scoring pipeline.
        
        Args:
            query_id: Query ID
            candidates: Retrieved candidates
            parsed_query: Parsed query with filters
            desired_count: Number of leads desired
            lgbm_service: LightGBM service for predictions
            
        Returns:
            List of scored and ranked leads
            
        Target: 10ms P95 for scoring
        """
        start_time = time.time()
        
        try:
            # Step 1: Pre-build BM25 index with all candidates
            await self._build_bm25_index(candidates)
            
            # Step 2: Calculate fit scores (F-3)
            fit_scores = await self._calculate_fit_scores(candidates, parsed_query)
            
            # Step 3: Calculate intent scores (F-5, F-6)
            intent_scores = await self._calculate_intent_scores(
                candidates=candidates, 
                parsed_query=parsed_query,
                lgbm_service=lgbm_service
            )
            
            # Step 4: Apply boosts (F-6)
            boosted_scores = await self._apply_boosts(intent_scores)
            
            # Step 5: Calculate final scores (F-7)
            final_scores = self._calculate_final_scores(
                query_id=query_id, 
                fit_scores=fit_scores, 
                intent_scores=boosted_scores, 
                candidates=candidates,
                parsed_query=parsed_query
            )
            
            # Step 6: Deduplicate by company_id (F-3)
            unique_scores = self._deduplicate_by_company(final_scores)
            
            # Step 7: Trim to desired count (F-4)
            trimmed_scores = unique_scores[:desired_count]
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Scoring completed in {processing_time:.2f}ms: {len(trimmed_scores)} leads")
            
            # Validate latency SLA
            if processing_time > settings.SCORING_TIMEOUT_MS:
                logger.warning(f"Scoring exceeded {settings.SCORING_TIMEOUT_MS}ms target: {processing_time:.2f}ms")
            
            return trimmed_scores
            
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            # Return fallback scores
            return await self._fallback_scoring(candidates, desired_count)
    
    async def _calculate_fit_scores(
        self, 
        candidates: List[RetrievalCandidate], 
        parsed_query: ParsedQuery
    ) -> Dict[str, float]:
        """Calculate fit scores using NumPy (F-3)."""
        fit_scores = {}
        
        for candidate in candidates:
            # Calculate fit based on firmographics match
            industry_match = self._calculate_industry_match(
                candidate.firmographics.get("industry", ""),
                parsed_query.target_industries
            )
            
            size_match = self._calculate_size_match(
                candidate.firmographics.get("size", ""),
                parsed_query.target_sizes
            )
            
            region_match = self._calculate_region_match(
                candidate.firmographics.get("region", ""),
                parsed_query.target_regions
            )
            
            # Weighted average of matches using configurable weights
            fit_score = (
                industry_match * self.fit_score_industry_weight +
                size_match * self.fit_score_size_weight +
                region_match * self.fit_score_region_weight
            )
            
            fit_scores[candidate.lead_id] = fit_score
        
        return fit_scores
    
    def _calculate_industry_match(self, candidate_industry: str, target_industries: List[str]) -> float:
        """Calculate industry match score."""
        if not target_industries:
            return 0.5  # Neutral score if no target specified
        
        if candidate_industry.lower() in [ind.lower() for ind in target_industries]:
            return 1.0
        
        # Partial match logic
        candidate_words = set(candidate_industry.lower().split())
        for target in target_industries:
            target_words = set(target.lower().split())
            if candidate_words.intersection(target_words):
                return 0.7
        
        return 0.0
    
    def _calculate_size_match(self, candidate_size: str, target_sizes: List[str]) -> float:
        """Calculate company size match score."""
        if not target_sizes:
            return 0.5
        
        if candidate_size.lower() in [size.lower() for size in target_sizes]:
            return 1.0
        
        # Size range logic
        candidate_range = self._parse_size_range(candidate_size)
        for target in target_sizes:
            target_range = self._parse_size_range(target)
            if self._size_ranges_overlap(candidate_range, target_range):
                return 0.8
        
        return 0.0
    
    def _calculate_region_match(self, candidate_region: str, target_regions: List[str]) -> float:
        """Calculate region match score."""
        if not target_regions:
            return 0.5
        
        if candidate_region.lower() in [region.lower() for region in target_regions]:
            return 1.0
        
        return 0.0
    
    def _parse_size_range(self, size_str: str) -> Tuple[int, int]:
        """Parse size string to range tuple."""
        try:
            if "-" in size_str:
                parts = size_str.split("-")
                return (int(parts[0]), int(parts[1]))
            elif "+" in size_str:
                min_size = int(size_str.replace("+", ""))
                return (min_size, float('inf'))
            else:
                size = int(size_str)
                return (size, size)
        except:
            return (0, 0)
    
    def _size_ranges_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
        """Check if two size ranges overlap."""
        return not (range1[1] < range2[0] or range2[1] < range1[0])
    
    async def _calculate_intent_scores(
        self, 
        candidates: List[RetrievalCandidate], 
        parsed_query: ParsedQuery,
        lgbm_service: LightGBMService
    ) -> Dict[str, Dict[str, float]]:
        """Calculate intent scores using BM25 and LLM fallback (F-5)."""
        intent_scores = {}
        
        # Calculate BM25 scores for all candidates in batch
        bm25_scores = await self._calculate_bm25_scores_batch(candidates, parsed_query.keywords)
        
        for candidate in candidates:
            # Get BM25 score from batch results
            bm25_score = bm25_scores.get(candidate.lead_id, 0.0)
            
            # Apply time decay
            time_decay_factor = self._calculate_time_decay(candidate.firmographics.get("created_at"))
            
            # Check for cold-start / low-signal scenarios
            llm_score = None
            lgbm_score = None
            base_intent_score = bm25_score

            if bm25_score < self.bm25_threshold:
                # First, try LLM fallback for higher quality signals
                llm_score = await self._calculate_llm_score(candidate, parsed_query)
                if llm_score is not None:
                    base_intent_score = llm_score
                # If LLM fails or is not applicable, use LightGBM as the final fallback (F-10)
                else:
                    lgbm_score = lgbm_service.predict(candidate)
                    base_intent_score = lgbm_score
            
            intent_scores[candidate.lead_id] = {
                "bm25_score": bm25_score,
                "time_decay_factor": time_decay_factor,
                "llm_score": llm_score,
                "lgbm_score": lgbm_score,
                "base_intent_score": base_intent_score
            }
        
        return intent_scores
    
    async def _get_candidate_content(self, candidate: RetrievalCandidate) -> str:
        """Get content for BM25 scoring."""
        # This would fetch intent snippets and other content
        # For now, create mock content
        content_parts = [
            candidate.company_name,
            candidate.firmographics.get("industry", ""),
            candidate.firmographics.get("size", ""),
            candidate.firmographics.get("region", ""),
            str(candidate.technographics)
        ]
        
        return " ".join(filter(None, content_parts))
    
    async def _calculate_bm25_score(self, content: str, keywords: List[str]) -> float:
        """Calculate BM25 score using pre-built index (F-5)."""
        start_time = time.time()
        
        try:
            if not self.bm25_searcher or not self.bm25_query_parser or not keywords:
                return 0.0
            
            # Use pre-built query parser and searcher
            query = self.bm25_query_parser.parse(" ".join(keywords))
            results = self.bm25_searcher.search(query, limit=1)
            
            if results:
                score = results[0].score
            else:
                score = 0.0
            
            processing_time = (time.time() - start_time) * 1000
            if processing_time > settings.BM25_SCORING_TIMEOUT_MS:
                logger.warning(f"BM25 scoring exceeded {settings.BM25_SCORING_TIMEOUT_MS}ms target: {processing_time:.2f}ms")
            
            return score
            
        except Exception as e:
            logger.error(f"BM25 scoring error: {e}")
            return 0.0
    
    async def _calculate_bm25_scores_batch(self, candidates: List[RetrievalCandidate], keywords: List[str]) -> Dict[str, float]:
        """Calculate BM25 scores for all candidates in batch using pre-built index."""
        if not self.bm25_searcher or not self.bm25_query_parser or not keywords:
            return {candidate.lead_id: 0.0 for candidate in candidates}
        
        try:
            start_time = time.time()
            
            # Use pre-built query parser and searcher
            query = self.bm25_query_parser.parse(" ".join(keywords))
            results = self.bm25_searcher.search(query, limit=len(candidates))
            
            # Create score mapping
            scores = {}
            for result in results:
                scores[result["lead_id"]] = result.score
            
            # Fill in missing candidates with 0.0 score
            for candidate in candidates:
                if candidate.lead_id not in scores:
                    scores[candidate.lead_id] = 0.0
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Batch BM25 scoring completed in {processing_time:.2f}ms for {len(candidates)} candidates")
            
            return scores
            
        except Exception as e:
            logger.error(f"Batch BM25 scoring error: {e}")
            return {candidate.lead_id: 0.0 for candidate in candidates}
    
    def _calculate_time_decay(self, created_at: Any) -> float:
        """Calculate time decay factor (F-5)."""
        try:
            if isinstance(created_at, str):
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            elif isinstance(created_at, datetime):
                created_date = created_at
            else:
                return 1.0
            
            days_old = (datetime.now() - created_date).days
            decay_factor = math.exp(-days_old / self.time_decay_tau)
            
            return max(0.1, decay_factor)  # Minimum 0.1
            
        except Exception as e:
            logger.error(f"Time decay calculation error: {e}")
            return 1.0
    
    async def _calculate_llm_score(
        self, 
        candidate: RetrievalCandidate, 
        parsed_query: ParsedQuery
    ) -> Optional[float]:
        """Calculate LLM score for low BM25 candidates, with caching and a circuit breaker."""
        # 1. Check cache first
        cache_key = f"llm_score:{candidate.lead_id}:{parsed_query.product_category}"
        try:
            cached_score = await self.redis_client.get(cache_key)
            if cached_score is not None:
                logger.debug(f"LLM score cache hit for key: {cache_key}")
                self.metrics_collector.record_cache_hit()
                return float(cached_score)
        except Exception as e:
            logger.error(f"Redis cache GET failed for LLM score: {e}")
        
        self.metrics_collector.record_cache_miss()
        logger.debug(f"LLM score cache miss for key: {cache_key}")

        # 2. If not in cache, call LLM via circuit breaker
        try:
            # Create prompt for LLM scoring
            prompt = f"""
            Rate the likelihood (0-100) that this company would be interested in {parsed_query.product_category}.
            
            Company: {candidate.company_name}
            Industry: {candidate.firmographics.get('industry', 'Unknown')}
            Size: {candidate.firmographics.get('size', 'Unknown')}
            Region: {candidate.firmographics.get('region', 'Unknown')}
            Technologies: {candidate.technographics}
            
            Target criteria:
            - Industries: {parsed_query.target_industries}
            - Sizes: {parsed_query.target_sizes}
            - Regions: {parsed_query.target_regions}
            
            Respond with only a number between 0 and 100.
            """
            
            response = await llm_circuit_breaker.call_async(
                self.client.chat.completions.create,
                model=settings.LLM_SCORING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=5,
                timeout=self.llm_timeout # This timeout is for the underlying call
            )
            
            # Parse response
            score_text = response.choices[0].message.content.strip()
            score = float(score_text) / 100.0
            score = max(0.0, min(1.0, score))

            # 3. Store the new score in the cache
            try:
                await self.redis_client.set(cache_key, str(score), ex=3600 * 24) # Cache for 24 hours
            except Exception as e:
                logger.error(f"Redis cache SET failed for LLM score: {e}")
            
            return score
            
        except CircuitBreakerError:
            logger.error("LLM circuit breaker is open. Skipping LLM call.")
            return None
        except asyncio.TimeoutError:
            logger.error("LLM scoring timed out")
            return None
        except Exception as e:
            logger.error(f"LLM scoring error: {e}")
            return None
    
    async def _apply_boosts(self, intent_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Apply churn and job-posting boosts (F-6)."""
        boosted_scores = intent_scores.copy()
        
        for lead_id, scores in boosted_scores.items():
            churn_boost = await self._get_churn_boost(lead_id)
            job_posting_boost = await self._get_job_posting_boost(lead_id)
            
            scores["churn_boost"] = churn_boost
            scores["job_posting_boost"] = job_posting_boost
            
            # Apply boosts to base intent score
            base_score = scores["base_intent_score"]
            boosted_score = base_score + churn_boost + job_posting_boost
            scores["boosted_intent_score"] = max(0.0, min(1.0, boosted_score))
        
        return boosted_scores
    
    async def _get_churn_boost(self, lead_id: str) -> float:
        """Get churn boost for lead (+20)."""
        try:
            # Check Redis cache first
            cache_key = f"churn_boost:{lead_id}"
            cached_boost = await self.redis_client.get(cache_key)
            if cached_boost:
                return float(cached_boost)
            
            # This would query churn detection service
            # For now, return mock boost
            boost = 0.0
            if "churn" in lead_id.lower():
                boost = self.churn_boost_value / 100.0  # Normalize to 0-1
            
            # Cache result
            await self.redis_client.set(cache_key, str(boost), ex=3600)
            
            return boost
            
        except Exception as e:
            logger.error(f"Churn boost error: {e}")
            return 0.0
    
    async def _get_job_posting_boost(self, lead_id: str) -> float:
        """Get job posting boost for lead (+15)."""
        try:
            # Check Redis cache first
            cache_key = f"job_boost:{lead_id}"
            cached_boost = await self.redis_client.get(cache_key)
            if cached_boost:
                return float(cached_boost)
            
            # This would query job posting service
            # For now, return mock boost
            boost = 0.0
            if "job" in lead_id.lower():
                boost = self.job_posting_boost_value / 100.0  # Normalize to 0-1
            
            # Cache result
            await self.redis_client.set(cache_key, str(boost), ex=3600)
            
            return boost
            
        except Exception as e:
            logger.error(f"Job posting boost error: {e}")
            return 0.0
    
    def _calculate_final_scores(
        self, 
        query_id: str,
        fit_scores: Dict[str, float], 
        intent_scores: Dict[str, Dict[str, float]],
        candidates: List[RetrievalCandidate],
        parsed_query: ParsedQuery
    ) -> List[ScoredLead]:
        """Calculate final scores using the formula from BRD Section 9 (F-7)."""
        scored_leads = []
        
        for lead_id, intent_data in intent_scores.items():
            if lead_id not in fit_scores:
                continue
            
            fit_score = fit_scores[lead_id]
            intent_score = intent_data.get("boosted_intent_score", intent_data.get("base_intent_score", 0.0))
            
            # Final score formula using configurable weights
            final_score = self.final_score_fit_weight * fit_score + self.final_score_intent_weight * intent_score
            
            candidate = self._find_candidate_by_id(lead_id, candidates)
            churn_boost = intent_data.get('churn_boost', 0.0)
            job_posting_boost = intent_data.get('job_posting_boost', 0.0)

            # Create ScoredLead object
            scored_lead = ScoredLead(
                query_id=query_id,
                lead_id=lead_id,
                company_id=candidate.company_id if candidate else "",
                company_name=candidate.company_name if candidate else "",
                email=candidate.email if candidate else "",
                fit_score=fit_score,
                intent_score=intent_score,
                final_score=final_score,
                bm25_score=intent_data.get('bm25_score', 0.0),
                time_decay_factor=intent_data.get('time_decay_factor', 0.0),
                churn_boost=churn_boost,
                job_posting_boost=job_posting_boost,
                llm_score=intent_data.get('llm_score'),
                lgbm_score=intent_data.get('lgbm_score'),
                explanation={
                    "fit_score_breakdown": {
                        "industry_match": self._calculate_industry_match(
                            candidate.firmographics.get("industry", "") if candidate else "",
                            parsed_query.target_industries
                        ),
                        "size_match": self._calculate_size_match(
                            candidate.firmographics.get("size", "") if candidate else "",
                            parsed_query.target_sizes
                        ),
                        "region_match": self._calculate_region_match(
                            candidate.firmographics.get("region", "") if candidate else "",
                            parsed_query.target_regions
                        )
                    },
                    "intent_score_breakdown": {
                        "bm2s_score": f"{intent_data.get('bm25_score', 0.0):.2f}",
                        "time_decay": f"{intent_data.get('time_decay_factor', 0.0):.2f}",
                        "llm_score": f"{intent_data.get('llm_score', 'N/A')}",
                        "lgbm_score": f"{intent_data.get('lgbm_score', 'N/A')}",
                        "boosts": f"+{churn_boost:.2f} (churn), +{job_posting_boost:.2f} (jobs)"
                    },
                    "final_score_formula": f"({fit_score:.2f} * {self.final_score_fit_weight}) + ({intent_score:.2f} * {self.final_score_intent_weight})"
                },
                firmographics=candidate.firmographics if candidate else {},
                technographics=candidate.technographics if candidate else {}
            )
            scored_leads.append(scored_lead)
        
        # Sort by final score
        scored_leads.sort(key=lambda x: x.final_score, reverse=True)
        
        return scored_leads
    
    def _find_candidate_by_id(self, lead_id: str, candidates: List[RetrievalCandidate]) -> Optional[RetrievalCandidate]:
        """Helper to find a candidate by lead_id."""
        for c in candidates:
            if c.lead_id == lead_id:
                return c
        return None
    
    def _deduplicate_by_company(self, scored_leads: List[ScoredLead]) -> List[ScoredLead]:
        """Deduplicate scored leads by company_id, keeping the highest score."""
        company_leads = {}
        
        for lead in scored_leads:
            if lead.company_id not in company_leads:
                company_leads[lead.company_id] = lead
            elif lead.final_score > company_leads[lead.company_id].final_score:
                company_leads[lead.company_id] = lead
        
        # Return sorted by final score
        unique_leads = list(company_leads.values())
        unique_leads.sort(key=lambda x: x.final_score, reverse=True)
        
        return unique_leads
    
    async def _fallback_scoring(self, candidates: List[RetrievalCandidate], desired_count: int) -> List[ScoredLead]:
        """Fallback scoring when main pipeline fails."""
        logger.warning("Using fallback scoring")
        
        scored_leads = []
        for i, candidate in enumerate(candidates[:desired_count]):
            scored_lead = ScoredLead(
                query_id="",
                lead_id=candidate.lead_id,
                company_id=candidate.company_id,
                company_name=candidate.company_name,
                email=candidate.email,
                fit_score=0.5,
                intent_score=0.5,
                final_score=0.5,
                bm25_score=0.0,
                time_decay_factor=1.0,
                churn_boost=0.0,
                job_posting_boost=0.0,
                llm_score=None,
                lgbm_score=None,
                explanation={"fallback": True},
                firmographics=candidate.firmographics,
                technographics=candidate.technographics
            )
            scored_leads.append(scored_lead)
        
        return scored_leads
    
    async def cleanup(self):
        """Cleanup resources."""
        # Close BM25 searcher
        if self.bm25_searcher:
            try:
                self.bm25_searcher.close()
                self.bm25_searcher = None
                logger.info("BM25 searcher closed")
            except Exception as e:
                logger.error(f"Error closing BM25 searcher: {e}")
        
        # Close BM25 index
        if self.bm25_index:
            try:
                self.bm25_index.close()
                self.bm25_index = None
                logger.info("BM25 index closed")
            except Exception as e:
                logger.error(f"Error closing BM25 index: {e}")
        
        # Remove temporary directory
        if self.bm25_temp_dir:
            try:
                import shutil
                shutil.rmtree(self.bm25_temp_dir)
                self.bm25_temp_dir = None
                logger.info("BM25 temp directory cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up BM25 temp directory: {e}") 