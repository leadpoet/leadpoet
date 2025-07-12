"""
Prompt Parser Service - F-1 Implementation
Parses free-text prompts to JSON including NOT filters and exemplar detection.
Target: 200ms P95 latency
"""

import asyncio
import json
import re
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import logging

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Parsed query structure with extracted components."""
    product_category: str
    target_industries: List[str]
    target_sizes: List[str]
    target_regions: List[str]
    target_roles: List[str]
    not_industries: List[str]
    not_sizes: List[str]
    not_regions: List[str]
    exemplar_companies: List[str]
    keywords: List[str]
    language: str
    confidence: float


class PromptParser:
    """Parses free-text prompts to structured query format."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.cache = {}  # Simple in-memory cache for development
        
        # Predefined industry mappings
        self.industry_mappings = {
            "tech": ["technology", "software", "saas", "it", "information technology"],
            "finance": ["financial", "banking", "insurance", "fintech"],
            "healthcare": ["healthcare", "medical", "pharmaceutical", "biotech"],
            "retail": ["retail", "ecommerce", "consumer goods"],
            "manufacturing": ["manufacturing", "industrial", "automotive"],
            "education": ["education", "edtech", "training"],
            "real_estate": ["real estate", "property", "construction"],
            "consulting": ["consulting", "professional services", "advisory"]
        }
        
        # Predefined size mappings
        self.size_mappings = {
            "startup": ["1-10", "10-50", "startup", "early stage"],
            "sme": ["50-100", "100-500", "small", "medium"],
            "enterprise": ["500-1000", "1000-5000", "5000+", "large", "enterprise"]
        }
        
        # Predefined region mappings
        self.region_mappings = {
            "us": ["united states", "us", "usa", "america"],
            "eu": ["europe", "european union", "eu"],
            "uk": ["united kingdom", "uk", "britain"],
            "ca": ["canada", "canadian"],
            "au": ["australia", "australian"],
            "asia": ["asia", "asian", "apac"]
        }
    
    async def parse(self, prompt: str, filters: Optional[Dict[str, Any]] = None) -> ParsedQuery:
        """
        Parse a free-text prompt to structured query format.
        
        Args:
            prompt: Free-text prompt describing the B2B product and target leads
            filters: Optional pre-defined filters to merge with parsed results
            
        Returns:
            ParsedQuery object with extracted components
            
        Target: 200ms P95 latency
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{prompt}:{json.dumps(filters or {})}"
            if cache_key in self.cache:
                logger.info("Using cached parsed query")
                return self.cache[cache_key]
            
            # Use OpenAI tools for parsing
            parsed_data = await self._parse_with_openai(prompt)
            
            # Apply any pre-defined filters
            if filters:
                parsed_data = self._merge_filters(parsed_data, filters)
            
            # Create ParsedQuery object
            parsed_query = ParsedQuery(
                product_category=parsed_data.get("product_category", ""),
                target_industries=parsed_data.get("target_industries", []),
                target_sizes=parsed_data.get("target_sizes", []),
                target_regions=parsed_data.get("target_regions", []),
                target_roles=parsed_data.get("target_roles", []),
                not_industries=parsed_data.get("not_industries", []),
                not_sizes=parsed_data.get("not_sizes", []),
                not_regions=parsed_data.get("not_regions", []),
                exemplar_companies=parsed_data.get("exemplar_companies", []),
                keywords=parsed_data.get("keywords", []),
                language=parsed_data.get("language", "en"),
                confidence=parsed_data.get("confidence", 0.8)
            )
            
            # Cache the result
            self.cache[cache_key] = parsed_query
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Prompt parsed in {processing_time:.2f}ms")
            
            # Validate latency SLA
            if processing_time > 200:
                logger.warning(f"Prompt parsing exceeded 200ms target: {processing_time:.2f}ms")
            
            return parsed_query
            
        except Exception as e:
            logger.error(f"Error parsing prompt: {e}")
            # Return fallback parsing
            return self._fallback_parse(prompt, filters)
    
    async def _parse_with_openai(self, prompt: str) -> Dict[str, Any]:
        """Parse prompt using OpenAI tools (function calling)."""
        
        # Define the tool schema for structured parsing using the new tools format
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "parse_b2b_query",
                    "description": "Parse a B2B lead generation query into structured components",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_category": {
                                "type": "string",
                                "description": "The main product or service category being evaluated"
                            },
                            "target_industries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Target industries for the leads"
                            },
                            "target_sizes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Target company sizes (e.g., 100-500, enterprise)"
                            },
                            "target_regions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Target geographic regions"
                            },
                            "target_roles": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Target job roles or titles"
                            },
                            "not_industries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Industries to exclude (NOT filters)"
                            },
                            "not_sizes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Company sizes to exclude (NOT filters)"
                            },
                            "not_regions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Regions to exclude (NOT filters)"
                            },
                            "exemplar_companies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Example companies to find similar ones to"
                            },
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Important keywords for intent detection"
                            },
                            "language": {
                                "type": "string",
                                "description": "Language of the prompt (ISO 639-1 code)"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score (0-1) in the parsing"
                            }
                        },
                        "required": ["product_category", "confidence"]
                    }
                }
            }
        ]
        
        # Create the prompt for parsing
        system_prompt = """
        You are a B2B lead generation query parser. Parse the given prompt to extract:
        1. Product category being evaluated
        2. Target industries, company sizes, regions, and roles
        3. NOT filters (exclusions)
        4. Exemplar companies for similarity search
        5. Important keywords for intent detection
        
        Handle multiple languages and various prompt formats. Be precise and extract all relevant information.
        """
        
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=settings.PROMPT_PARSER_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "parse_b2b_query"}},
                    temperature=0.1,
                    max_tokens=500
                ),
                timeout=15.0  # 15 second timeout
            )
            
            # Extract tool call arguments (updated from function_call)
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                if tool_call.function.name == "parse_b2b_query":
                    return json.loads(tool_call.function.arguments)
                else:
                    raise ValueError(f"Unexpected tool call: {tool_call.function.name}")
            else:
                raise ValueError("No tool call returned")
                
        except asyncio.TimeoutError:
            logger.error("OpenAI parsing timed out")
            raise
        except Exception as e:
            logger.error(f"OpenAI parsing failed: {e}")
            raise
    
    def _merge_filters(self, parsed_data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Merge pre-defined filters with parsed data."""
        merged = parsed_data.copy()
        
        # Merge target filters
        for key in ["target_industries", "target_sizes", "target_regions", "target_roles"]:
            if key in filters:
                existing = set(merged.get(key, []))
                new_filters = set(filters[key])
                merged[key] = list(existing.union(new_filters))
        
        # Merge NOT filters
        for key in ["not_industries", "not_sizes", "not_regions"]:
            if key in filters:
                existing = set(merged.get(key, []))
                new_filters = set(filters[key])
                merged[key] = list(existing.union(new_filters))
        
        return merged
    
    def _fallback_parse(self, prompt: str, filters: Optional[Dict[str, Any]] = None) -> ParsedQuery:
        """Fallback parsing using regex patterns when OpenAI fails."""
        logger.warning("Using fallback parsing")
        
        # Simple regex-based extraction
        product_category = self._extract_product_category(prompt)
        target_industries = self._extract_industries(prompt)
        target_sizes = self._extract_sizes(prompt)
        target_regions = self._extract_regions(prompt)
        exemplar_companies = self._extract_exemplar_companies(prompt)
        
        # Merge with filters if provided
        if filters:
            target_industries.extend(filters.get("target_industries", []))
            target_sizes.extend(filters.get("target_sizes", []))
            target_regions.extend(filters.get("target_regions", []))
        
        return ParsedQuery(
            product_category=product_category,
            target_industries=list(set(target_industries)),
            target_sizes=list(set(target_sizes)),
            target_regions=list(set(target_regions)),
            target_roles=[],
            not_industries=[],
            not_sizes=[],
            not_regions=[],
            exemplar_companies=list(set(exemplar_companies)),
            keywords=self._extract_keywords(prompt),
            language="en",
            confidence=0.5
        )
    
    def _extract_product_category(self, prompt: str) -> str:
        """Extract product category using regex patterns."""
        patterns = [
            r"(?:looking for|find|evaluating|considering)\s+([^,\n]+?)(?:\s+software|\s+tool|\s+platform|\s+service)?",
            r"(?:CRM|ERP|HR|marketing|sales|analytics|security)\s+(?:software|tool|platform)",
            r"([^,\n]+?)\s+(?:software|tool|platform|service)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "unknown"
    
    def _extract_industries(self, prompt: str) -> List[str]:
        """Extract target industries using regex patterns."""
        industries = []
        prompt_lower = prompt.lower()
        
        for industry, keywords in self.industry_mappings.items():
            if any(keyword in prompt_lower for keyword in keywords):
                industries.append(industry)
        
        return industries
    
    def _extract_sizes(self, prompt: str) -> List[str]:
        """Extract target company sizes using regex patterns."""
        sizes = []
        prompt_lower = prompt.lower()
        
        # Extract specific size ranges
        size_patterns = [
            r"(\d+-\d+)\s*(?:employees?|people|staff)",
            r"(\d+)\+\s*(?:employees?|people|staff)",
            r"(startup|small|medium|large|enterprise)\s*(?:companies?|businesses?)"
        ]
        
        for pattern in size_patterns:
            matches = re.findall(pattern, prompt_lower)
            sizes.extend(matches)
        
        # Map to standard categories
        for size, keywords in self.size_mappings.items():
            if any(keyword in prompt_lower for keyword in keywords):
                sizes.append(size)
        
        return list(set(sizes))
    
    def _extract_regions(self, prompt: str) -> List[str]:
        """Extract target regions using regex patterns."""
        regions = []
        prompt_lower = prompt.lower()
        
        for region, keywords in self.region_mappings.items():
            if any(keyword in prompt_lower for keyword in keywords):
                regions.append(region)
        
        return regions
    
    def _extract_exemplar_companies(self, prompt: str) -> List[str]:
        """Extract exemplar companies for similarity search."""
        # Look for patterns like "companies like X" or "similar to X"
        patterns = [
            r"companies?\s+like\s+([A-Z][a-zA-Z\s&]+)",
            r"similar\s+to\s+([A-Z][a-zA-Z\s&]+)",
            r"such\s+as\s+([A-Z][a-zA-Z\s&,]+)",
            r"including\s+([A-Z][a-zA-Z\s&,]+)"
        ]
        
        companies = []
        for pattern in patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                # Split by commas and clean up
                company_list = [c.strip() for c in match.split(",")]
                companies.extend(company_list)
        
        return list(set(companies))
    
    def _extract_keywords(self, prompt: str) -> List[str]:
        """Extract important keywords for intent detection."""
        # Simple keyword extraction - in production, use more sophisticated NLP
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        words = re.findall(r'\b\w+\b', prompt.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return list(set(keywords))[:20]  # Limit to top 20 keywords
    
    def clear_cache(self):
        """Clear the parsing cache."""
        self.cache.clear()
        logger.info("Prompt parser cache cleared") 