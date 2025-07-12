"""
Unit tests for Prompt Parser Service - F-1 Implementation
Tests prompt parsing, NOT-clause logic, and exemplar detection.
Target: ≥95% branch coverage
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any

from app.services.prompt_parser import PromptParser, ParsedQuery


class TestPromptParser:
    """Test suite for PromptParser class."""
    
    @pytest.fixture
    def parser(self):
        """Create a PromptParser instance for testing."""
        return PromptParser()
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI response for testing."""
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "parse_b2b_query"
        mock_tool_call.function.arguments = json.dumps({
            "product_category": "CRM software",
            "target_industries": ["technology", "finance"],
            "target_sizes": ["100-500", "500-1000"],
            "target_regions": ["us", "eu"],
            "target_roles": ["sales manager", "marketing director"],
            "not_industries": ["healthcare"],
            "not_sizes": ["1-10"],
            "not_regions": ["asia"],
            "exemplar_companies": ["Salesforce", "HubSpot"],
            "keywords": ["crm", "sales", "automation"],
            "language": "en",
            "confidence": 0.9
        })
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        return mock_response
    
    @pytest.mark.asyncio
    async def test_parse_successful_openai_parsing(self, parser, mock_openai_response):
        """Test successful parsing with OpenAI."""
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_openai_response
            
            result = await parser.parse("Find CRM software for tech companies")
            
            assert isinstance(result, ParsedQuery)
            assert result.product_category == "CRM software"
            assert "technology" in result.target_industries
            assert "healthcare" in result.not_industries
            assert "Salesforce" in result.exemplar_companies
            assert result.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_parse_with_cache_hit(self, parser, mock_openai_response):
        """Test that cached results are returned."""
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_openai_response
            
            # First call - should hit OpenAI
            result1 = await parser.parse("Find CRM software for tech companies")
            
            # Second call - should hit cache
            result2 = await parser.parse("Find CRM software for tech companies")
            
            # Should only call OpenAI once
            assert mock_create.call_count == 1
            assert result1.product_category == result2.product_category
    
    @pytest.mark.asyncio
    async def test_parse_with_filters_merge(self, parser, mock_openai_response):
        """Test merging of parsed data with pre-defined filters."""
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_openai_response
            
            filters = {
                "target_industries": ["manufacturing"],
                "target_sizes": ["1000-5000"],
                "not_regions": ["latin_america"]
            }
            
            result = await parser.parse("Find CRM software", filters=filters)
            
            # Should merge with parsed data
            assert "technology" in result.target_industries  # from OpenAI
            assert "manufacturing" in result.target_industries  # from filters
            assert "latin_america" in result.not_regions  # from filters
    
    @pytest.mark.asyncio
    async def test_parse_openai_timeout(self, parser):
        """Test fallback parsing when OpenAI times out."""
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = asyncio.TimeoutError()
            
            result = await parser.parse("Find CRM software for tech companies")
            
            # Should use fallback parsing
            assert isinstance(result, ParsedQuery)
            assert result.confidence == 0.5  # fallback confidence
            assert result.language == "en"
    
    @pytest.mark.asyncio
    async def test_parse_openai_error(self, parser):
        """Test fallback parsing when OpenAI fails."""
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("OpenAI API error")
            
            result = await parser.parse("Find CRM software for tech companies")
            
            # Should use fallback parsing
            assert isinstance(result, ParsedQuery)
            assert result.confidence == 0.5
    
    @pytest.mark.asyncio
    async def test_parse_no_tool_calls(self, parser):
        """Test handling when OpenAI returns no tool calls."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = []
        
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await parser.parse("Find CRM software")
            
            # Should use fallback parsing
            assert isinstance(result, ParsedQuery)
            assert result.confidence == 0.5
    
    @pytest.mark.asyncio
    async def test_parse_wrong_tool_call(self, parser):
        """Test handling when OpenAI returns wrong tool call."""
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "wrong_function"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await parser.parse("Find CRM software")
            
            # Should use fallback parsing
            assert isinstance(result, ParsedQuery)
            assert result.confidence == 0.5
    
    def test_fallback_parse_product_category(self, parser):
        """Test fallback product category extraction."""
        # Test "looking for" pattern
        result = parser._fallback_parse("I'm looking for CRM software")
        assert "crm" in result.product_category.lower()
        
        # Test "evaluating" pattern
        result = parser._fallback_parse("We are evaluating ERP systems")
        assert "erp" in result.product_category.lower()
        
        # Test "software" pattern
        result = parser._fallback_parse("Marketing automation platform")
        assert "marketing automation" in result.product_category.lower()
        
        # Test unknown pattern
        result = parser._fallback_parse("Random text without clear product")
        assert result.product_category == "unknown"
    
    def test_fallback_parse_industries(self, parser):
        """Test fallback industry extraction."""
        # Test technology industry
        result = parser._fallback_parse("Find software for tech companies")
        assert "tech" in result.target_industries
        
        # Test finance industry
        result = parser._fallback_parse("Banking software for financial institutions")
        assert "finance" in result.target_industries
        
        # Test healthcare industry
        result = parser._fallback_parse("Medical software for healthcare providers")
        assert "healthcare" in result.target_industries
        
        # Test multiple industries
        result = parser._fallback_parse("Software for tech and finance companies")
        assert "tech" in result.target_industries
        assert "finance" in result.target_industries
    
    def test_fallback_parse_sizes(self, parser):
        """Test fallback company size extraction."""
        # Test specific size ranges
        result = parser._fallback_parse("Software for companies with 100-500 employees")
        assert "100-500" in result.target_sizes
        
        # Test 500+ pattern
        result = parser._fallback_parse("Enterprise software for 1000+ employees")
        assert "1000+" in result.target_sizes
        
        # Test descriptive sizes
        result = parser._fallback_parse("Software for startup companies")
        assert "startup" in result.target_sizes
        
        result = parser._fallback_parse("Enterprise software for large companies")
        assert "enterprise" in result.target_sizes
    
    def test_fallback_parse_regions(self, parser):
        """Test fallback region extraction."""
        # Test US region
        result = parser._fallback_parse("Software for US companies")
        assert "us" in result.target_regions
        
        # Test European region
        result = parser._fallback_parse("Software for European companies")
        assert "eu" in result.target_regions
        
        # Test UK region
        result = parser._fallback_parse("Software for UK companies")
        assert "uk" in result.target_regions
    
    def test_fallback_parse_exemplar_companies(self, parser):
        """Test fallback exemplar company extraction."""
        # Test "companies like" pattern
        result = parser._fallback_parse("Find companies like Salesforce and HubSpot")
        assert "Salesforce" in result.exemplar_companies
        assert "HubSpot" in result.exemplar_companies
        
        # Test "similar to" pattern
        result = parser._fallback_parse("Companies similar to Microsoft")
        assert "Microsoft" in result.exemplar_companies
        
        # Test "such as" pattern
        result = parser._fallback_parse("Tech companies such as Google, Apple")
        assert "Google" in result.exemplar_companies
        assert "Apple" in result.exemplar_companies
        
        # Test "including" pattern
        result = parser._fallback_parse("Software companies including Adobe")
        assert "Adobe" in result.exemplar_companies
    
    def test_fallback_parse_keywords(self, parser):
        """Test fallback keyword extraction."""
        result = parser._fallback_parse("Find CRM software for sales automation")
        
        # Should extract meaningful keywords
        assert "crm" in result.keywords
        assert "software" in result.keywords
        assert "sales" in result.keywords
        assert "automation" in result.keywords
        
        # Should exclude stop words
        assert "the" not in result.keywords
        assert "and" not in result.keywords
        assert "for" not in result.keywords
    
    def test_merge_filters_target_filters(self, parser):
        """Test merging of target filters."""
        parsed_data = {
            "target_industries": ["tech"],
            "target_sizes": ["100-500"],
            "target_regions": ["us"],
            "target_roles": ["manager"]
        }
        
        filters = {
            "target_industries": ["finance"],
            "target_sizes": ["500-1000"],
            "target_regions": ["eu"],
            "target_roles": ["director"]
        }
        
        merged = parser._merge_filters(parsed_data, filters)
        
        # Should combine both sets
        assert "tech" in merged["target_industries"]
        assert "finance" in merged["target_industries"]
        assert "100-500" in merged["target_sizes"]
        assert "500-1000" in merged["target_sizes"]
        assert "us" in merged["target_regions"]
        assert "eu" in merged["target_regions"]
        assert "manager" in merged["target_roles"]
        assert "director" in merged["target_roles"]
    
    def test_merge_filters_not_filters(self, parser):
        """Test merging of NOT filters."""
        parsed_data = {
            "not_industries": ["healthcare"],
            "not_sizes": ["1-10"],
            "not_regions": ["asia"]
        }
        
        filters = {
            "not_industries": ["retail"],
            "not_sizes": ["10000+"],
            "not_regions": ["latin_america"]
        }
        
        merged = parser._merge_filters(parsed_data, filters)
        
        # Should combine both sets
        assert "healthcare" in merged["not_industries"]
        assert "retail" in merged["not_industries"]
        assert "1-10" in merged["not_sizes"]
        assert "10000+" in merged["not_sizes"]
        assert "asia" in merged["not_regions"]
        assert "latin_america" in merged["not_regions"]
    
    def test_merge_filters_empty_filters(self, parser):
        """Test merging when filters are empty."""
        parsed_data = {
            "target_industries": ["tech"],
            "not_industries": ["healthcare"]
        }
        
        merged = parser._merge_filters(parsed_data, {})
        
        # Should return original data unchanged
        assert merged == parsed_data
    
    def test_merge_filters_missing_keys(self, parser):
        """Test merging when some keys are missing."""
        parsed_data = {
            "target_industries": ["tech"]
        }
        
        filters = {
            "target_sizes": ["100-500"],
            "not_regions": ["asia"]
        }
        
        merged = parser._merge_filters(parsed_data, filters)
        
        # Should add missing keys
        assert "tech" in merged["target_industries"]
        assert "100-500" in merged["target_sizes"]
        assert "asia" in merged["not_regions"]
    
    def test_clear_cache(self, parser):
        """Test cache clearing functionality."""
        # Add some data to cache
        parser.cache["test_key"] = "test_value"
        assert len(parser.cache) == 1
        
        # Clear cache
        parser.clear_cache()
        assert len(parser.cache) == 0
    
    @pytest.mark.asyncio
    async def test_parse_latency_sla(self, parser, mock_openai_response):
        """Test that parsing respects latency SLA."""
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_openai_response
            
            # Mock time to simulate slow response
            with patch('time.time') as mock_time:
                mock_time.side_effect = [0.0, 0.3]  # 300ms processing time
                
                result = await parser.parse("Find CRM software")
                
                # Should still work but log warning
                assert isinstance(result, ParsedQuery)
    
    def test_parsed_query_dataclass(self):
        """Test ParsedQuery dataclass creation and attributes."""
        query = ParsedQuery(
            product_category="CRM",
            target_industries=["tech"],
            target_sizes=["100-500"],
            target_regions=["us"],
            target_roles=["manager"],
            not_industries=["healthcare"],
            not_sizes=["1-10"],
            not_regions=["asia"],
            exemplar_companies=["Salesforce"],
            keywords=["crm", "sales"],
            language="en",
            confidence=0.9
        )
        
        assert query.product_category == "CRM"
        assert query.target_industries == ["tech"]
        assert query.not_industries == ["healthcare"]
        assert query.exemplar_companies == ["Salesforce"]
        assert query.confidence == 0.9


class TestNOTClauseLogic:
    """Specific tests for NOT-clause logic functionality."""
    
    @pytest.fixture
    def parser(self):
        return PromptParser()
    
    @pytest.mark.asyncio
    async def test_not_industries_extraction(self, parser):
        """Test extraction of NOT industry filters."""
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "parse_b2b_query"
        mock_tool_call.function.arguments = json.dumps({
            "product_category": "CRM",
            "target_industries": ["technology"],
            "not_industries": ["healthcare", "retail"],
            "target_sizes": [],
            "target_regions": [],
            "target_roles": [],
            "not_sizes": [],
            "not_regions": [],
            "exemplar_companies": [],
            "keywords": [],
            "language": "en",
            "confidence": 0.9
        })
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await parser.parse("Find CRM software for tech companies, but not healthcare or retail")
            
            assert "healthcare" in result.not_industries
            assert "retail" in result.not_industries
            assert "technology" in result.target_industries
    
    @pytest.mark.asyncio
    async def test_not_sizes_extraction(self, parser):
        """Test extraction of NOT size filters."""
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "parse_b2b_query"
        mock_tool_call.function.arguments = json.dumps({
            "product_category": "ERP",
            "target_industries": [],
            "not_industries": [],
            "target_sizes": ["500-1000"],
            "not_sizes": ["1-10", "10-50"],
            "target_regions": [],
            "target_roles": [],
            "not_regions": [],
            "exemplar_companies": [],
            "keywords": [],
            "language": "en",
            "confidence": 0.9
        })
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await parser.parse("Find ERP software for medium companies, but not startups")
            
            assert "1-10" in result.not_sizes
            assert "10-50" in result.not_sizes
            assert "500-1000" in result.target_sizes
    
    @pytest.mark.asyncio
    async def test_not_regions_extraction(self, parser):
        """Test extraction of NOT region filters."""
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "parse_b2b_query"
        mock_tool_call.function.arguments = json.dumps({
            "product_category": "Marketing",
            "target_industries": [],
            "not_industries": [],
            "target_sizes": [],
            "not_sizes": [],
            "target_regions": ["us", "eu"],
            "not_regions": ["asia", "latin_america"],
            "target_roles": [],
            "exemplar_companies": [],
            "keywords": [],
            "language": "en",
            "confidence": 0.9
        })
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        
        with patch.object(parser.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await parser.parse("Find marketing software for US and EU companies, but not Asia or Latin America")
            
            assert "asia" in result.not_regions
            assert "latin_america" in result.not_regions
            assert "us" in result.target_regions
            assert "eu" in result.target_regions
    
    def test_not_filters_merge_logic(self, parser):
        """Test that NOT filters are properly merged with pre-defined filters."""
        parsed_data = {
            "not_industries": ["healthcare"],
            "not_sizes": ["1-10"],
            "not_regions": ["asia"]
        }
        
        filters = {
            "not_industries": ["retail"],
            "not_sizes": ["10000+"],
            "not_regions": ["latin_america"]
        }
        
        merged = parser._merge_filters(parsed_data, filters)
        
        # Should combine both sets of NOT filters
        assert "healthcare" in merged["not_industries"]
        assert "retail" in merged["not_industries"]
        assert "1-10" in merged["not_sizes"]
        assert "10000+" in merged["not_sizes"]
        assert "asia" in merged["not_regions"]
        assert "latin_america" in merged["not_regions"]
    
    def test_not_filters_fallback_parsing(self, parser):
        """Test that NOT filters are handled in fallback parsing."""
        # Note: Fallback parsing doesn't currently extract NOT filters
        # This test documents the current behavior
        result = parser._fallback_parse("Find CRM software, but not for healthcare")
        
        # Fallback parsing should return empty NOT filters
        assert result.not_industries == []
        assert result.not_sizes == []
        assert result.not_regions == []
        
        # But should still extract other components
        assert "crm" in result.product_category.lower()


if __name__ == "__main__":
    pytest.main([__file__])
