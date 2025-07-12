"""
Retrieval Service - F-2 and F-2b Implementation
Strict SQL retrieval with progressive relaxation and exemplar ANN search.
Target: 50ms P95 latency for SQL, 18ms P95 for ANN
"""

import asyncio
import time
import logging
import random
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, and_, or_, not_, select, func
from sqlalchemy.orm import selectinload, Session
from sqlalchemy.dialects.postgresql import JSONB
import numpy as np
import json
import hashlib

from app.core.config import settings
from app.services.prompt_parser import ParsedQuery
from app.models.lead import Lead
from app.models.intent_snippet import IntentSnippet
from app.services.simhash_service import SimhashService
from app.core.redis_client import RedisClient

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """Candidate lead from retrieval with metadata."""
    lead_id: str
    company_id: str
    company_name: str
    email: str
    retrieval_score: float
    retrieval_method: str  # "sql" or "ann"
    firmographics: Dict[str, Any]
    technographics: Dict[str, Any]
    simhash: Optional[str] = None
    is_potential_duplicate: Optional[bool] = None
    lead_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass instance to a dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalCandidate':
        """Create a RetrievalCandidate instance from a dictionary."""
        return cls(**data)


class RetrievalService:
    """
    Service for retrieving lead candidates from various sources.
    This service is responsible for the initial fetching of potential leads
    before they are passed to the scoring engine.
    """

    def __init__(self, db: Session):
        self.db = db
        self.simhash_service = SimhashService(db)
        self.redis_client = RedisClient()
        self.min_candidates = settings.MIN_LEADS_PER_QUERY
        self.max_candidates = settings.MAX_LEADS_PER_QUERY
        self.retrieval_timeout = settings.RETRIEVAL_TIMEOUT_MS / 1000.0
        self.ann_timeout = settings.EXEMPLAR_SEARCH_TIMEOUT_MS / 1000.0
        
        # Progressive relaxation levels
        self.relaxation_levels = [
            "strict",      # Exact matches only
            "industry",    # Relax industry constraints
            "size",        # Relax size constraints  
            "region",      # Relax region constraints
            "minimal"      # Minimal constraints
        ]
        
        # Mock data for realistic candidate generation
        self._mock_industries = [
            "Technology", "Healthcare", "Finance", "Manufacturing", "Retail",
            "Education", "Real Estate", "Consulting", "Marketing", "Legal",
            "Transportation", "Energy", "Media", "Non-profit", "Government"
        ]
        
        self._mock_company_sizes = [
            "1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5000+"
        ]
        
        self._mock_regions = [
            "North America", "Europe", "Asia Pacific", "Latin America", "Middle East",
            "United States", "Canada", "United Kingdom", "Germany", "France",
            "Australia", "Japan", "India", "Brazil", "Mexico"
        ]
        
        self._mock_technologies = [
            "Salesforce", "HubSpot", "Marketo", "Pardot", "Zoho",
            "Microsoft Dynamics", "Oracle", "SAP", "Adobe", "Google",
            "AWS", "Azure", "GCP", "Slack", "Zoom", "Teams"
        ]
        
        self._mock_company_suffixes = [
            "Inc.", "LLC", "Corp.", "Ltd.", "Co.", "Group", "Solutions", "Systems",
            "Technologies", "Partners", "Consulting", "Services", "Enterprises"
        ]
        
        self._mock_first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa",
            "James", "Jennifer", "William", "Jessica", "Richard", "Amanda",
            "Thomas", "Nicole", "Christopher", "Stephanie", "Daniel", "Melissa"
        ]
        
        self._mock_last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
        ]

    def _get_query_cache_key(self, parsed_query: 'ParsedQuery') -> str:
        """Creates a deterministic cache key from the parsed query."""
        # Use a stable json representation and hash it
        from dataclasses import asdict
        query_dump = json.dumps(asdict(parsed_query), sort_keys=True)
        return f"retrieval:{hashlib.md5(query_dump.encode()).hexdigest()}"

    async def retrieve_candidates(self, parsed_query: 'ParsedQuery', desired_count: int) -> List[RetrievalCandidate]:
        """
        Retrieves candidates from a data source and performs a Simhash check.
        It now uses Redis to cache the retrieval results.
        """
        # 1. Check cache first
        cache_key = self._get_query_cache_key(parsed_query)
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Retrieval cache hit for key: {cache_key}")
                # Deserialize from JSON
                candidates_data = json.loads(cached_data)
                return [RetrievalCandidate.from_dict(c) for c in candidates_data]
        except Exception as e:
            logger.error(f"Redis cache GET failed for retrieval: {e}")

        logger.info(f"Retrieval cache miss for query: {parsed_query.product_category}")
        
        # --- This is a mock implementation ---
        # In a real system, you would query your primary lead database here.
        mock_candidates = self._get_mock_candidates(desired_count)
        
        # --- Simhash Check Integration ---
        # For each candidate, calculate its Simhash and check for near-duplicates.
        processed_candidates = []
        for candidate_data in mock_candidates:
            # Create a text corpus from the candidate's data
            text_corpus = self.simhash_service.get_lead_text_corpus(candidate_data.to_dict())
            
            # Calculate the Simhash
            candidate_simhash = self.simhash_service.calculate_simhash(text_corpus)
            
            # Check for near-duplicates
            is_duplicate = self.simhash_service.find_near_duplicates(candidate_simhash)
            
            # Create the candidate object with the Simhash and duplicate flag
            processed_candidate = RetrievalCandidate(
                lead_id=candidate_data.lead_id,
                company_id=candidate_data.company_id,
                company_name=candidate_data.company_name,
                email=candidate_data.email,
                retrieval_score=candidate_data.retrieval_score,
                retrieval_method=candidate_data.retrieval_method,
                firmographics=candidate_data.firmographics,
                technographics=candidate_data.technographics,
                simhash=candidate_simhash,
                is_potential_duplicate=is_duplicate,
                lead_metadata=candidate_data.lead_metadata
            )
            
            processed_candidates.append(processed_candidate)

            if is_duplicate:
                logger.warning(f"Candidate {processed_candidate.lead_id} flagged as a potential duplicate.")
        
        # 3. Store the results in the cache
        try:
            # Serialize to JSON
            candidates_data = [c.to_dict() for c in processed_candidates]
            await self.redis_client.set(cache_key, json.dumps(candidates_data), ex=3600 * 4) # Cache for 4 hours
        except Exception as e:
            logger.error(f"Redis cache SET failed for retrieval: {e}")

        return processed_candidates

    def _get_mock_candidates(self, count: int) -> List[RetrievalCandidate]:
        """
        Generate realistic mock candidates for testing and development.
        
        Args:
            count: Number of candidates to generate
            
        Returns:
            List of RetrievalCandidate instances with realistic mock data
        """
        candidates = []
        
        for i in range(count):
            # Generate realistic company data
            company_name = self._generate_company_name()
            company_id = f"comp_{str(uuid.uuid4())[:8]}"
            lead_id = f"lead_{str(uuid.uuid4())[:8]}"
            
            # Generate realistic email
            first_name = random.choice(self._mock_first_names)
            last_name = random.choice(self._mock_last_names)
            email = f"{first_name.lower()}.{last_name.lower()}@{company_name.lower().replace(' ', '').replace('.', '')}.com"
            
            # Generate realistic firmographics
            firmographics = {
                "industry": random.choice(self._mock_industries),
                "size": random.choice(self._mock_company_sizes),
                "region": random.choice(self._mock_regions),
                "revenue": f"${random.randint(1, 1000)}M",
                "founded_year": random.randint(1990, 2023),
                "employee_count": random.randint(10, 10000),
                "headquarters": random.choice(self._mock_regions),
                "website": f"www.{company_name.lower().replace(' ', '').replace('.', '')}.com"
            }
            
            # Generate realistic technographics
            num_technologies = random.randint(2, 8)
            selected_technologies = random.sample(self._mock_technologies, num_technologies)
            technographics = {
                "crm": random.choice([True, False]),
                "marketing_automation": random.choice([True, False]),
                "analytics": random.choice([True, False]),
                "cloud_provider": random.choice(["AWS", "Azure", "GCP", "None"]),
                "technologies": selected_technologies,
                "tech_stack_size": num_technologies,
                "has_mobile_app": random.choice([True, False]),
                "uses_ai_ml": random.choice([True, False])
            }
            
            # Generate realistic retrieval score (0.0 to 1.0)
            retrieval_score = round(random.uniform(0.3, 1.0), 3)
            
            # Randomly choose retrieval method
            retrieval_method = random.choice(["sql", "ann"])
            
            # Create the candidate
            candidate = RetrievalCandidate(
                lead_id=lead_id,
                company_id=company_id,
                company_name=company_name,
                email=email,
                retrieval_score=retrieval_score,
                retrieval_method=retrieval_method,
                firmographics=firmographics,
                technographics=technographics
            )
            
            candidates.append(candidate)
        
        logger.info(f"Generated {len(candidates)} mock candidates")
        return candidates
    
    def _generate_company_name(self) -> str:
        """
        Generate a realistic company name.
        """
        # Company name patterns
        patterns = [
            f"{random.choice(['Tech', 'Data', 'Cloud', 'Digital', 'Smart', 'Next', 'Future', 'Global', 'Advanced', 'Innovative'])} {random.choice(['Solutions', 'Systems', 'Technologies', 'Corp', 'Inc', 'Labs', 'Works', 'Group'])}",
            f"{random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Omega', 'Sigma', 'Nova', 'Zenith', 'Pinnacle', 'Summit'])} {random.choice(self._mock_company_suffixes)}",
            f"{random.choice(['Blue', 'Green', 'Red', 'Yellow', 'Purple', 'Orange', 'Silver', 'Gold', 'Black', 'White'])} {random.choice(['Tech', 'Data', 'Systems', 'Solutions', 'Group'])}",
            f"{random.choice(['North', 'South', 'East', 'West', 'Central', 'Global', 'International', 'Worldwide'])} {random.choice(['Tech', 'Data', 'Systems', 'Solutions', 'Corp'])}",
            f"{random.choice(['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Prime', 'Elite', 'Premium', 'Select', 'Choice'])} {random.choice(['Tech', 'Data', 'Systems', 'Solutions', 'Group'])}"
        ]
        
        return random.choice(patterns)
    
    async def _exemplar_ann_search(
        self, 
        exemplar_companies: List[str], 
        limit: int
    ) -> List[RetrievalCandidate]:
        """
        Exemplar ANN search using pgvector (F-2b).
        
        Target: 18ms P95 latency
        """
        start_time = time.time()
        
        try:
            # This would use pgvector with Instructor-tiny embeddings
            # For now, implement a simplified version
            
            # Get company embeddings for exemplars
            exemplar_embeddings = await self._get_company_embeddings(exemplar_companies)
            
            if not exemplar_embeddings:
                return []
            
            # Calculate average embedding
            avg_embedding = np.mean(exemplar_embeddings, axis=0)
            
            # Search for similar companies using cosine similarity
            similar_companies = await self._vector_similarity_search(
                query_embedding=avg_embedding,
                limit=limit
            )
            
            # Convert to candidates
            candidates = []
            for company_id, similarity_score in similar_companies:
                company_leads = await self._get_company_leads(
                    company_id, 
                    limit=settings.MAX_LEADS_PER_COMPANY
                )
                for lead in company_leads:
                    candidates.append(RetrievalCandidate(
                        lead_id=lead.lead_id,
                        company_id=lead.company_id,
                        company_name=lead.company_name,
                        email=lead.email,
                        retrieval_score=similarity_score,
                        retrieval_method="ann",
                        firmographics=lead.firmographics,
                        technographics=lead.technographics
                    ))
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"ANN search completed in {processing_time:.2f}ms")
            
            # Validate latency SLA
            if processing_time > settings.EXEMPLAR_SEARCH_TIMEOUT_MS:
                logger.warning(f"ANN search exceeded {settings.EXEMPLAR_SEARCH_TIMEOUT_MS}ms target: {processing_time:.2f}ms")
            
            return candidates
            
        except Exception as e:
            logger.error(f"ANN search error: {e}")
            return []
    
    async def _sql_retrieval_with_relaxation(
        self, 
        parsed_query: ParsedQuery, 
        desired_count: int
    ) -> List[RetrievalCandidate]:
        """
        SQL retrieval with progressive relaxation (F-2).
        
        Returns ≥ 4× desired leads or fully relaxed.
        """
        candidates = []
        
        for level in self.relaxation_levels:
            if len(candidates) >= self.min_candidates:
                break
                
            level_candidates = await self._sql_retrieval_at_level(
                parsed_query=parsed_query,
                relaxation_level=level,
                limit=desired_count * 2
            )
            
            candidates.extend(level_candidates)
            logger.info(f"Level '{level}' returned {len(level_candidates)} candidates")
            
            # If we have enough candidates, stop
            if len(candidates) >= self.min_candidates:
                break
        
        return candidates
    
    async def _sql_retrieval_at_level(
        self, 
        parsed_query: ParsedQuery, 
        relaxation_level: str, 
        limit: int
    ) -> List[RetrievalCandidate]:
        """SQL retrieval at specific relaxation level using SQLAlchemy query builder."""
        
        # Build query conditions using SQLAlchemy
        conditions = self._build_sqlalchemy_conditions(parsed_query, relaxation_level)
        
        # Build parameterized query using SQLAlchemy
        query = select(Lead).where(and_(*conditions)).order_by(Lead.created_at.desc()).limit(limit)
        
        try:
            # This would use async database session
            # For now, return mock data
            mock_candidates = []
            for i in range(min(limit, 100)):  # Mock limit
                mock_candidates.append(RetrievalCandidate(
                    lead_id=f"lead_{i}",
                    company_id=f"company_{i}",
                    company_name=f"Company {i}",
                    email=f"user{i}@company{i}.com",
                    retrieval_score=1.0 - (i * 0.01),
                    retrieval_method="sql",
                    firmographics={"industry": "technology", "size": "100-500"},
                    technographics={"crm": True, "marketing": False}
                ))
            
            return mock_candidates
            
        except Exception as e:
            logger.error(f"SQL retrieval error at level {relaxation_level}: {e}")
            return []
    
    def _build_sqlalchemy_conditions(self, parsed_query: ParsedQuery, level: str) -> List:
        """Build SQLAlchemy WHERE conditions based on relaxation level."""
        conditions = [Lead.is_active == True]
        
        if level == "strict":
            # Exact matches only
            if parsed_query.target_industries:
                conditions.append(Lead.firmographics['industry'].astext.in_(parsed_query.target_industries))
            if parsed_query.target_sizes:
                conditions.append(Lead.firmographics['size'].astext.in_(parsed_query.target_sizes))
            if parsed_query.target_regions:
                conditions.append(Lead.firmographics['region'].astext.in_(parsed_query.target_regions))
                
        elif level == "industry":
            # Relax industry constraints
            if parsed_query.target_sizes:
                conditions.append(Lead.firmographics['size'].astext.in_(parsed_query.target_sizes))
            if parsed_query.target_regions:
                conditions.append(Lead.firmographics['region'].astext.in_(parsed_query.target_regions))
                
        elif level == "size":
            # Relax size constraints
            if parsed_query.target_regions:
                conditions.append(Lead.firmographics['region'].astext.in_(parsed_query.target_regions))
                
        elif level == "region":
            # Relax region constraints
            # Only basic filters
            pass
            
        elif level == "minimal":
            # Minimal constraints - just active leads
            pass
        
        # Add NOT filters at all levels
        if parsed_query.not_industries:
            conditions.append(~Lead.firmographics['industry'].astext.in_(parsed_query.not_industries))
        if parsed_query.not_sizes:
            conditions.append(~Lead.firmographics['size'].astext.in_(parsed_query.not_sizes))
        if parsed_query.not_regions:
            conditions.append(~Lead.firmographics['region'].astext.in_(parsed_query.not_regions))
        
        return conditions
    
    async def _get_company_embeddings(self, company_names: List[str]) -> List[np.ndarray]:
        """Get company embeddings for exemplar search."""
        # This would use a pre-computed embedding table
        # For now, return mock embeddings
        embeddings = []
        for company in company_names:
            # Mock 384-dimensional embedding
            embedding = np.random.rand(384)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
        
        return embeddings
    
    async def _vector_similarity_search(
        self, 
        query_embedding: np.ndarray, 
        limit: int
    ) -> List[Tuple[str, float]]:
        """Search for similar companies using vector similarity."""
        # This would use pgvector's cosine similarity
        # For now, return mock results
        results = []
        for i in range(limit):
            company_id = f"company_{i}"
            similarity = 0.9 - (i * 0.01)  # Mock similarity scores
            results.append((company_id, similarity))
        
        return results
    
    async def _get_company_leads(self, company_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get leads for a specific company."""
        # This would query the database
        # For now, return mock data
        leads = []
        for i in range(limit):
            leads.append({
                "lead_id": f"lead_{company_id}_{i}",
                "company_id": company_id,
                "company_name": f"Company {company_id}",
                "email": f"user{i}@company{company_id}.com",
                "firmographics": {"industry": "technology", "size": "100-500"},
                "technographics": {"crm": True, "marketing": False}
            })
        
        return leads
    
    def _deduplicate_candidates(self, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """Deduplicate candidates by company_id, keeping highest score."""
        company_candidates = {}
        
        for candidate in candidates:
            if candidate.company_id not in company_candidates:
                company_candidates[candidate.company_id] = candidate
            elif candidate.retrieval_score > company_candidates[candidate.company_id].retrieval_score:
                company_candidates[candidate.company_id] = candidate
        
        # Sort by retrieval score
        unique_candidates = list(company_candidates.values())
        unique_candidates.sort(key=lambda x: x.retrieval_score, reverse=True)
        
        return unique_candidates
    
    async def _fallback_retrieval(self, desired_count: int) -> List[RetrievalCandidate]:
        """Fallback retrieval when main methods fail."""
        logger.warning("Using fallback retrieval")
        
        candidates = []
        for i in range(min(desired_count * 2, 100)):
            candidates.append(RetrievalCandidate(
                lead_id=f"fallback_lead_{i}",
                company_id=f"fallback_company_{i}",
                company_name=f"Fallback Company {i}",
                email=f"fallback{i}@company{i}.com",
                retrieval_score=0.5,
                retrieval_method="fallback",
                firmographics={"industry": "technology", "size": "100-500"},
                technographics={"crm": True, "marketing": False}
            ))
        
        return candidates
    
    async def cleanup(self):
        """Cleanup resources."""
        # Close any open connections
        pass 