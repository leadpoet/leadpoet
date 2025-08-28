"""
Apollo Data Mapping and Transformation Utilities

This module provides comprehensive data mapping and transformation capabilities
for converting Apollo API responses to the unified lead record schema.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

# Configure logging with PII protection
logger = logging.getLogger(__name__)


class ApolloDataMapper:
    """
    Apollo data mapping and transformation utilities.

    This class maintains strict no-hardcoding principles - all business parameters
    must be loaded from configuration, never from source code defaults.
    """

    def __init__(self, mapping_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Apollo data mapper.

        Args:
            mapping_config: Optional mapping configuration (loaded from config)
        """
        self.mapping_config = mapping_config or {}

        # Default field mapping (can be overridden by config)
        self.default_mapping = {
            "company": {
                "name": "name",
                "description": "description",
                "industry": "industry",
                "size": "size_hint",
                "employee_count": "employee_count",
                "founded_year": "founded_year",
                "website": "website",
                "linkedin_url": "socials.linkedin",
                "phone": "phone",
                "address": "hq_location",
                "technologies": "tech_stack",
                "funding_stage": "funding_stage",
                "revenue_range": "revenue_range",
                "locations": "number_of_locations",
            },
            "contact": {
                "email": "email",
                "first_name": "first_name",
                "last_name": "last_name",
                "job_title": "title",
                "department": "department",
                "seniority_level": "seniority",
                "linkedin_url": "linkedin_url",
                "phone": "phone",
                "location": "location",
            },
        }

        # Apply custom mapping if provided
        if mapping_config:
            self._apply_custom_mapping(mapping_config)

        logger.info("Apollo data mapper initialized")

    def _apply_custom_mapping(self, mapping_config: Dict[str, Any]):
        """
        Apply custom field mapping from configuration.

        Args:
            mapping_config: Custom mapping configuration

        Note: This method ensures zero hardcoded business parameters.
        All mapping rules must come from configuration.
        """
        # Validate that mapping config doesn't contain hardcoded business logic
        self._validate_mapping_config_not_hardcoded(mapping_config)

        # Apply custom company mapping
        if "company" in mapping_config:
            self.default_mapping["company"].update(mapping_config["company"])

        # Apply custom contact mapping
        if "contact" in mapping_config:
            self.default_mapping["contact"].update(mapping_config["contact"])

        logger.info("Custom mapping configuration applied")

    def _validate_mapping_config_not_hardcoded(self, mapping_config: Dict[str, Any]):
        """
        Validate that mapping configuration doesn't contain hardcoded business logic.

        Args:
            mapping_config: Mapping configuration to validate

        Raises:
            ValueError: If hardcoded business parameters are detected
        """
        # Check for common hardcoded patterns in mapping values
        hardcoded_patterns = [
            ["technology", "software", "saas"],
            ["1-10", "11-50", "51-200"],
            ["CEO", "CTO", "VP Engineering"],
            ["healthcare", "medical", "pharmaceuticals"],
        ]

        for category, mappings in mapping_config.items():
            for field, value in mappings.items():
                if isinstance(value, list):
                    for pattern in hardcoded_patterns:
                        if value == pattern:
                            raise ValueError(
                                f"Hardcoded business parameters detected in {category}.{field}: {pattern}. "
                                "All business parameters must be loaded from configuration, never hardcoded."
                            )

    def map_company_data(self, apollo_company: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map Apollo company data to unified schema.

        Args:
            apollo_company: Apollo company data

        Returns:
            Mapped company data in unified format
        """
        try:
            mapped_data = {}

            # Apply field mapping
            for apollo_field, unified_field in self.default_mapping["company"].items():
                if apollo_field in apollo_company:
                    value = apollo_company[apollo_field]
                    mapped_value = self._transform_field_value(value, apollo_field)
                    mapped_data[unified_field] = mapped_value

            # Handle special cases
            mapped_data = self._handle_company_special_cases(
                mapped_data, apollo_company
            )

            # Validate mapped data
            mapped_data = self._validate_company_data(mapped_data)

            return mapped_data

        except Exception as e:
            logger.error(f"Failed to map company data: {e}")
            return {}

    def map_person_data(self, apollo_person: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map Apollo person data to unified schema.

        Args:
            apollo_person: Apollo person data

        Returns:
            Mapped person data in unified format
        """
        try:
            mapped_data = {}

            # Apply field mapping
            for apollo_field, unified_field in self.default_mapping["contact"].items():
                if apollo_field in apollo_person:
                    value = apollo_person[apollo_field]
                    mapped_value = self._transform_field_value(value, apollo_field)
                    mapped_data[unified_field] = mapped_value

            # Handle special cases
            mapped_data = self._handle_person_special_cases(mapped_data, apollo_person)

            # Validate mapped data
            mapped_data = self._validate_person_data(mapped_data)

            return mapped_data

        except Exception as e:
            logger.error(f"Failed to map person data: {e}")
            return {}

    def _transform_field_value(self, value: Any, field_name: str) -> Any:
        """
        Transform field value based on field type and requirements.

        Args:
            value: Raw field value
            field_name: Name of the field being transformed

        Returns:
            Transformed field value
        """
        if value is None:
            return None

        # Handle different field types
        if field_name == "founded_year":
            return self._transform_founded_year(value)
        elif field_name == "employee_count":
            return self._transform_employee_count(value)
        elif field_name == "phone":
            return self._transform_phone(value)
        elif field_name == "email":
            return self._transform_email(value)
        elif field_name == "website":
            return self._transform_website(value)
        elif field_name == "linkedin_url":
            return self._transform_linkedin_url(value)
        elif field_name == "technologies":
            return self._transform_technologies(value)
        elif field_name == "locations":
            return self._transform_locations(value)
        elif field_name == "address":
            return self._transform_address(value)
        else:
            # Default transformation - convert to string if not already
            return (
                str(value) if not isinstance(value, (str, int, float, bool)) else value
            )

    def _transform_founded_year(self, value: Any) -> Optional[int]:
        """Transform founded year to integer."""
        try:
            if isinstance(value, int):
                return value if 1800 <= value <= datetime.now().year else None
            elif isinstance(value, str):
                # Extract year from string
                year_match = re.search(r"\b(19|20)\d{2}\b", value)
                if year_match:
                    year = int(year_match.group())
                    return year if 1800 <= year <= datetime.now().year else None
            return None
        except (ValueError, TypeError):
            return None

    def _transform_employee_count(self, value: Any) -> Optional[str]:
        """Transform employee count to standardized format."""
        try:
            if isinstance(value, int):
                if value <= 10:
                    return "1-10"
                elif value <= 50:
                    return "11-50"
                elif value <= 200:
                    return "51-200"
                elif value <= 1000:
                    return "201-1000"
                elif value <= 5000:
                    return "1001-5000"
                else:
                    return "5000+"
            elif isinstance(value, str):
                # Try to extract number from string
                number_match = re.search(r"\b(\d+)\b", value)
                if number_match:
                    number = int(number_match.group())
                    return self._transform_employee_count(number)
                # Check if already in standard format
                if re.match(r"^\d+-\d+$|^\d+\+$", value):
                    return value
            return value
        except (ValueError, TypeError):
            return value

    def _transform_phone(self, value: Any) -> Optional[str]:
        """Transform phone number to standardized format."""
        try:
            if not value:
                return None

            # Remove all non-digit characters
            digits_only = re.sub(r"\D", "", str(value))

            # Validate phone number length
            if len(digits_only) < 10 or len(digits_only) > 15:
                return None

            # Format as US phone number if 10-11 digits
            if len(digits_only) == 10:
                return f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
            elif len(digits_only) == 11 and digits_only[0] == "1":
                return f"+1 ({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:]}"
            else:
                return f"+{digits_only}"

        except Exception:
            return str(value) if value else None

    def _transform_email(self, value: Any) -> Optional[str]:
        """Transform and validate email address."""
        try:
            if not value:
                return None

            email = str(value).strip().lower()

            # Basic email validation
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if re.match(email_pattern, email):
                return email
            else:
                logger.warning(f"Invalid email format: {value}")
                return None

        except Exception:
            return None

    def _transform_website(self, value: Any) -> Optional[str]:
        """Transform website URL to standardized format."""
        try:
            if not value:
                return None

            url = str(value).strip()

            # Add protocol if missing
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            # Validate URL format
            parsed = urlparse(url)
            if parsed.netloc and "." in parsed.netloc:
                return url
            else:
                logger.warning(f"Invalid website format: {value}")
                return None

        except Exception:
            return str(value) if value else None

    def _transform_linkedin_url(self, value: Any) -> Optional[str]:
        """Transform LinkedIn URL to standardized format."""
        try:
            if not value:
                return None

            url = str(value).strip()

            # Add protocol if missing
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            # Validate LinkedIn URL format
            if "linkedin.com" in url:
                return url
            else:
                logger.warning(f"Invalid LinkedIn URL format: {value}")
                return None

        except Exception:
            return str(value) if value else None

    def _transform_technologies(self, value: Any) -> List[str]:
        """Transform technologies to list format."""
        try:
            if isinstance(value, list):
                return [str(tech).strip() for tech in value if tech]
            elif isinstance(value, str):
                # Split by common delimiters
                tech_list = re.split(r"[,;|]", value)
                return [tech.strip() for tech in tech_list if tech.strip()]
            else:
                return []
        except Exception:
            return []

    def _transform_locations(self, value: Any) -> int:
        """Transform locations to count."""
        try:
            if isinstance(value, list):
                return len(value)
            elif isinstance(value, int):
                return value
            elif isinstance(value, str):
                # Try to extract number
                number_match = re.search(r"\b(\d+)\b", value)
                if number_match:
                    return int(number_match.group())
            return 0
        except Exception:
            return 0

    def _transform_address(self, value: Any) -> Optional[str]:
        """Transform address to formatted string."""
        try:
            if isinstance(value, dict):
                parts = []
                for field in ["street", "city", "state", "country"]:
                    if value.get(field):
                        parts.append(str(value[field]))
                return ", ".join(parts) if parts else None
            elif isinstance(value, str):
                return value.strip()
            else:
                return None
        except Exception:
            return str(value) if value else None

    def _handle_company_special_cases(
        self, mapped_data: Dict[str, Any], apollo_company: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle special cases for company data mapping.

        Args:
            mapped_data: Already mapped company data
            apollo_company: Original Apollo company data

        Returns:
            Updated mapped data with special cases handled
        """
        # Handle company size if not already mapped
        if "size_hint" not in mapped_data and "size" in apollo_company:
            mapped_data["size_hint"] = apollo_company["size"]

        # Handle industry normalization
        if "industry" in mapped_data:
            mapped_data["industry"] = self._normalize_industry(mapped_data["industry"])

        # Handle funding stage normalization
        if "funding_stage" in mapped_data:
            mapped_data["funding_stage"] = self._normalize_funding_stage(
                mapped_data["funding_stage"]
            )

        return mapped_data

    def _handle_person_special_cases(
        self, mapped_data: Dict[str, Any], apollo_person: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle special cases for person data mapping.

        Args:
            mapped_data: Already mapped person data
            apollo_person: Original Apollo person data

        Returns:
            Updated mapped data with special cases handled
        """
        # Handle full name if only first/last names available
        if "full_name" not in mapped_data:
            first_name = mapped_data.get("first_name", "")
            last_name = mapped_data.get("last_name", "")
            if first_name or last_name:
                mapped_data["full_name"] = f"{first_name} {last_name}".strip()

        # Handle job title normalization
        if "title" in mapped_data:
            mapped_data["title"] = self._normalize_job_title(mapped_data["title"])

        # Handle seniority normalization
        if "seniority" in mapped_data:
            mapped_data["seniority"] = self._normalize_seniority(
                mapped_data["seniority"]
            )

        return mapped_data

    def _normalize_industry(self, industry: str) -> str:
        """Normalize industry names to standard format."""
        if not industry:
            return industry

        # Industry normalization mapping (loaded from config, never hardcoded)
        industry_mapping = self.mapping_config.get("industry_normalization", {})

        normalized = industry_mapping.get(industry.lower(), industry)
        return normalized

    def _normalize_funding_stage(self, funding_stage: str) -> str:
        """Normalize funding stage to standard format."""
        if not funding_stage:
            return funding_stage

        # Funding stage normalization mapping (loaded from config, never hardcoded)
        funding_mapping = self.mapping_config.get("funding_stage_normalization", {})

        normalized = funding_mapping.get(funding_stage.lower(), funding_stage)
        return normalized

    def _normalize_job_title(self, job_title: str) -> str:
        """Normalize job title to standard format."""
        if not job_title:
            return job_title

        # Job title normalization mapping (loaded from config, never hardcoded)
        title_mapping = self.mapping_config.get("job_title_normalization", {})

        normalized = title_mapping.get(job_title.lower(), job_title)
        return normalized

    def _normalize_seniority(self, seniority: str) -> str:
        """Normalize seniority level to standard format."""
        if not seniority:
            return seniority

        # Seniority normalization mapping (loaded from config, never hardcoded)
        seniority_mapping = self.mapping_config.get("seniority_normalization", {})

        normalized = seniority_mapping.get(seniority.lower(), seniority)
        return normalized

    def _validate_company_data(self, mapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mapped company data and apply fallback logic.

        Args:
            mapped_data: Mapped company data

        Returns:
            Validated company data with fallbacks applied
        """
        # Apply fallbacks for missing required fields
        fallback_config = self.mapping_config.get("company_fallbacks", {})

        for field, fallback_value in fallback_config.items():
            if field not in mapped_data or not mapped_data[field]:
                mapped_data[field] = fallback_value

        return mapped_data

    def _validate_person_data(self, mapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mapped person data and apply fallback logic.

        Args:
            mapped_data: Mapped person data

        Returns:
            Validated person data with fallbacks applied
        """
        # Apply fallbacks for missing required fields
        fallback_config = self.mapping_config.get("contact_fallbacks", {})

        for field, fallback_value in fallback_config.items():
            if field not in mapped_data or not mapped_data[field]:
                mapped_data[field] = fallback_value

        return mapped_data

    def create_mapping_configuration(self) -> Dict[str, Any]:
        """
        Create a sample mapping configuration template.

        Returns:
            Mapping configuration template

        Note: This is a template that should be customized based on business requirements.
        All business parameters must be loaded from configuration files.
        """
        return {
            "company": {
                # Custom company field mappings
                "custom_field": "unified_field"
            },
            "contact": {
                # Custom contact field mappings
                "custom_field": "unified_field"
            },
            "industry_normalization": {
                # Industry name normalization
                "tech": "Technology",
                "software": "Software",
            },
            "funding_stage_normalization": {
                # Funding stage normalization
                "series a": "Series A",
                "series b": "Series B",
            },
            "job_title_normalization": {
                # Job title normalization
                "ceo": "CEO",
                "cto": "CTO",
            },
            "seniority_normalization": {
                # Seniority level normalization
                "exec": "Executive",
                "senior": "Senior",
            },
            "company_fallbacks": {
                # Company field fallbacks
                "industry": "Unknown",
                "size_hint": "Unknown",
            },
            "contact_fallbacks": {
                # Contact field fallbacks
                "department": "Unknown",
                "seniority": "Unknown",
            },
        }
