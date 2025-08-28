"""
Search Result Scoring and Ranking for Apollo Lead Generation

This module provides:
- Lead quality scoring based on multiple criteria
- Result ranking and prioritization
- Quality threshold validation
- Scoring algorithm customization
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Weights for different scoring criteria"""

    completeness: float = 0.3
    relevance: float = 0.25
    freshness: float = 0.2
    contact_quality: float = 0.15
    company_quality: float = 0.1


@dataclass
class QualityThresholds:
    """Quality thresholds for lead validation"""

    min_completeness: float = 0.6
    min_relevance: float = 0.7
    min_overall_score: float = 0.65
    min_contact_count: int = 1
    min_company_info: int = 3


class ApolloResultScorer:
    """Scores and ranks Apollo search results"""

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        thresholds: Optional[QualityThresholds] = None,
    ):
        self.weights = weights or ScoringWeights()
        self.thresholds = thresholds or QualityThresholds()

        # Define field importance for completeness scoring
        self.company_important_fields = [
            "name",
            "domain",
            "industry",
            "employee_count",
            "revenue",
            "location",
        ]

        self.contact_important_fields = [
            "first_name",
            "last_name",
            "email",
            "job_title",
            "seniority",
            "company",
        ]

    def score_company_result(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single company result"""
        try:
            # Calculate individual scores
            completeness_score = self._calculate_company_completeness(company_data)
            relevance_score = self._calculate_company_relevance(company_data)
            freshness_score = self._calculate_freshness_score(company_data)
            company_quality_score = self._calculate_company_quality(company_data)

            # Calculate weighted overall score
            overall_score = (
                completeness_score * self.weights.completeness
                + relevance_score * self.weights.relevance
                + freshness_score * self.weights.freshness
                + company_quality_score * self.weights.company_quality
            )

            # Determine quality tier
            quality_tier = self._determine_quality_tier(overall_score)

            return {
                "overall_score": round(overall_score, 3),
                "completeness_score": round(completeness_score, 3),
                "relevance_score": round(relevance_score, 3),
                "freshness_score": round(freshness_score, 3),
                "company_quality_score": round(company_quality_score, 3),
                "quality_tier": quality_tier,
                "meets_thresholds": overall_score >= self.thresholds.min_overall_score,
                "scoring_details": {
                    "weights_used": {
                        "completeness": self.weights.completeness,
                        "relevance": self.weights.relevance,
                        "freshness": self.weights.freshness,
                        "company_quality": self.weights.company_quality,
                    },
                    "thresholds": {
                        "min_completeness": self.thresholds.min_completeness,
                        "min_relevance": self.thresholds.min_relevance,
                        "min_overall": self.thresholds.min_overall_score,
                    },
                },
            }

        except Exception as e:
            logger.error(f"Error scoring company result: {e}")
            return {"overall_score": 0.0, "error": str(e), "meets_thresholds": False}

    def score_person_result(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single person result"""
        try:
            # Calculate individual scores
            completeness_score = self._calculate_person_completeness(person_data)
            relevance_score = self._calculate_person_relevance(person_data)
            freshness_score = self._calculate_freshness_score(person_data)
            contact_quality_score = self._calculate_contact_quality(person_data)

            # Calculate weighted overall score
            overall_score = (
                completeness_score * self.weights.completeness
                + relevance_score * self.weights.relevance
                + freshness_score * self.weights.freshness
                + contact_quality_score * self.weights.contact_quality
            )

            # Determine quality tier
            quality_tier = self._determine_quality_tier(overall_score)

            return {
                "overall_score": round(overall_score, 3),
                "completeness_score": round(completeness_score, 3),
                "relevance_score": round(relevance_score, 3),
                "freshness_score": round(freshness_score, 3),
                "contact_quality_score": round(contact_quality_score, 3),
                "quality_tier": quality_tier,
                "meets_thresholds": overall_score >= self.thresholds.min_overall_score,
                "scoring_details": {
                    "weights_used": {
                        "completeness": self.weights.completeness,
                        "relevance": self.weights.relevance,
                        "freshness": self.weights.freshness,
                        "contact_quality": self.weights.contact_quality,
                    },
                    "thresholds": {
                        "min_completeness": self.thresholds.min_completeness,
                        "min_relevance": self.thresholds.min_relevance,
                        "min_overall": self.thresholds.min_overall_score,
                    },
                },
            }

        except Exception as e:
            logger.error(f"Error scoring person result: {e}")
            return {"overall_score": 0.0, "error": str(e), "meets_thresholds": False}

    def score_lead_result(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score a complete lead result (company + contacts)"""
        try:
            company_data = lead_data.get("company", {})
            contacts = lead_data.get("contacts", [])

            # Score company
            company_score = self.score_company_result(company_data)

            # Score contacts
            contact_scores = []
            for contact in contacts:
                contact_score = self.score_person_result(contact)
                contact_scores.append(contact_score)

            # Calculate lead-level scores
            lead_completeness = self._calculate_lead_completeness(
                company_data, contacts
            )
            lead_relevance = self._calculate_lead_relevance(company_data, contacts)
            lead_freshness = self._calculate_lead_freshness(company_data, contacts)

            # Calculate overall lead score
            overall_score = (
                lead_completeness * self.weights.completeness
                + lead_relevance * self.weights.relevance
                + lead_freshness * self.weights.freshness
                + company_score["overall_score"] * 0.2
                + (
                    sum(cs["overall_score"] for cs in contact_scores)
                    / max(len(contact_scores), 1)
                )
                * 0.1
            )

            # Determine quality tier
            quality_tier = self._determine_quality_tier(overall_score)

            return {
                "overall_score": round(overall_score, 3),
                "company_score": company_score,
                "contact_scores": contact_scores,
                "lead_metrics": {
                    "completeness": round(lead_completeness, 3),
                    "relevance": round(lead_relevance, 3),
                    "freshness": round(lead_freshness, 3),
                    "contact_count": len(contacts),
                    "avg_contact_score": round(
                        sum(cs["overall_score"] for cs in contact_scores)
                        / max(len(contact_scores), 1),
                        3,
                    ),
                },
                "quality_tier": quality_tier,
                "meets_thresholds": (
                    overall_score >= self.thresholds.min_overall_score
                    and len(contacts) >= self.thresholds.min_contact_count
                ),
            }

        except Exception as e:
            logger.error(f"Error scoring lead result: {e}")
            return {"overall_score": 0.0, "error": str(e), "meets_thresholds": False}

    def rank_results(
        self, results: List[Dict[str, Any]], result_type: str = "company"
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Rank results by score (highest first)"""
        try:
            scored_results = []

            for result in results:
                if result_type == "company":
                    score = self.score_company_result(result)
                elif result_type == "person":
                    score = self.score_person_result(result)
                elif result_type == "lead":
                    score = self.score_lead_result(result)
                else:
                    logger.warning(f"Unknown result type: {result_type}")
                    continue

                scored_results.append((result, score))

            # Sort by overall score (highest first)
            scored_results.sort(key=lambda x: x[1]["overall_score"], reverse=True)

            return scored_results

        except Exception as e:
            logger.error(f"Error ranking results: {e}")
            return []

    def filter_by_quality(
        self,
        results: List[Dict[str, Any]],
        result_type: str = "company",
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filter results by quality thresholds"""
        try:
            if min_score is None:
                min_score = self.thresholds.min_overall_score

            scored_results = self.rank_results(results, result_type)

            # Filter by minimum score
            filtered_results = [
                result
                for result, score in scored_results
                if score["overall_score"] >= min_score
            ]

            logger.info(
                f"Filtered {len(results)} results to {len(filtered_results)} high-quality results"
            )

            return filtered_results

        except Exception as e:
            logger.error(f"Error filtering results by quality: {e}")
            return results

    def _calculate_company_completeness(self, company_data: Dict[str, Any]) -> float:
        """Calculate completeness score for company data"""
        if not company_data:
            return 0.0

        total_fields = len(self.company_important_fields)
        filled_fields = 0

        for field in self.company_important_fields:
            value = company_data.get(field)
            if value and self._is_valid_field_value(value):
                filled_fields += 1

        return filled_fields / total_fields if total_fields > 0 else 0.0

    def _calculate_person_completeness(self, person_data: Dict[str, Any]) -> float:
        """Calculate completeness score for person data"""
        if not person_data:
            return 0.0

        total_fields = len(self.contact_important_fields)
        filled_fields = 0

        for field in self.contact_important_fields:
            value = person_data.get(field)
            if value and self._is_valid_field_value(value):
                filled_fields += 1

        return filled_fields / total_fields if total_fields > 0 else 0.0

    def _calculate_company_relevance(self, company_data: Dict[str, Any]) -> float:
        """Calculate relevance score for company data"""
        # This would typically be based on ICP matching
        # For now, return a default score
        return 0.8

    def _calculate_person_relevance(self, person_data: Dict[str, Any]) -> float:
        """Calculate relevance score for person data"""
        # This would typically be based on ICP matching
        # For now, return a default score
        return 0.8

    def _calculate_freshness_score(self, data: Dict[str, Any]) -> float:
        """Calculate freshness score based on data age"""
        # Check for last_updated or similar timestamp fields
        last_updated = (
            data.get("last_updated") or data.get("updated_at") or data.get("created_at")
        )

        if not last_updated:
            return 0.5  # Default score for unknown freshness

        try:
            if isinstance(last_updated, str):
                # Parse timestamp string
                if "T" in last_updated:
                    dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromisoformat(last_updated)
            else:
                dt = last_updated

            # Calculate days since update
            days_old = (datetime.now() - dt).days

            # Score based on age (newer = higher score)
            if days_old <= 30:
                return 1.0
            elif days_old <= 90:
                return 0.8
            elif days_old <= 180:
                return 0.6
            elif days_old <= 365:
                return 0.4
            else:
                return 0.2

        except Exception as e:
            logger.warning(f"Error parsing timestamp {last_updated}: {e}")
            return 0.5

    def _calculate_company_quality(self, company_data: Dict[str, Any]) -> float:
        """Calculate quality score for company data"""
        quality_score = 0.5  # Base score

        # Boost for verified data
        if company_data.get("verified", False):
            quality_score += 0.2

        # Boost for premium data sources
        if company_data.get("data_source") in ["premium", "verified", "enterprise"]:
            quality_score += 0.1

        # Boost for complete contact information
        if company_data.get("phone") and company_data.get("email"):
            quality_score += 0.1

        # Boost for social media presence
        social_fields = ["linkedin", "twitter", "facebook", "website"]
        social_count = sum(1 for field in social_fields if company_data.get(field))
        quality_score += min(0.1, social_count * 0.025)

        return min(1.0, quality_score)

    def _calculate_contact_quality(self, person_data: Dict[str, Any]) -> float:
        """Calculate quality score for contact data"""
        quality_score = 0.5  # Base score

        # Boost for verified data
        if person_data.get("verified", False):
            quality_score += 0.2

        # Boost for premium data sources
        if person_data.get("data_source") in ["premium", "verified", "enterprise"]:
            quality_score += 0.1

        # Boost for complete contact information
        if person_data.get("phone") and person_data.get("email"):
            quality_score += 0.1

        # Boost for social media presence
        social_fields = ["linkedin", "twitter", "facebook"]
        social_count = sum(1 for field in social_fields if person_data.get(field))
        quality_score += min(0.1, social_count * 0.033)

        return min(1.0, quality_score)

    def _calculate_lead_completeness(
        self, company_data: Dict[str, Any], contacts: List[Dict[str, Any]]
    ) -> float:
        """Calculate completeness score for lead data"""
        company_completeness = self._calculate_company_completeness(company_data)

        if not contacts:
            return company_completeness * 0.7  # Penalty for no contacts

        # Average contact completeness
        contact_completeness = sum(
            self._calculate_person_completeness(contact) for contact in contacts
        ) / len(contacts)

        # Weighted average (company 60%, contacts 40%)
        return (company_completeness * 0.6) + (contact_completeness * 0.4)

    def _calculate_lead_relevance(
        self, company_data: Dict[str, Any], contacts: List[Dict[str, Any]]
    ) -> float:
        """Calculate relevance score for lead data"""
        company_relevance = self._calculate_company_relevance(company_data)

        if not contacts:
            return company_relevance * 0.8  # Penalty for no contacts

        # Average contact relevance
        contact_relevance = sum(
            self._calculate_person_relevance(contact) for contact in contacts
        ) / len(contacts)

        # Weighted average (company 60%, contacts 40%)
        return (company_relevance * 0.6) + (contact_relevance * 0.4)

    def _calculate_lead_freshness(
        self, company_data: Dict[str, Any], contacts: List[Dict[str, Any]]
    ) -> float:
        """Calculate freshness score for lead data"""
        company_freshness = self._calculate_freshness_score(company_data)

        if not contacts:
            return company_freshness

        # Average contact freshness
        contact_freshness = sum(
            self._calculate_freshness_score(contact) for contact in contacts
        ) / len(contacts)

        # Weighted average (company 60%, contacts 40%)
        return (company_freshness * 0.6) + (contact_freshness * 0.4)

    def _determine_quality_tier(self, score: float) -> str:
        """Determine quality tier based on score"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "very_good"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "fair"
        elif score >= 0.5:
            return "poor"
        else:
            return "very_poor"

    def _is_valid_field_value(self, value: Any) -> bool:
        """Check if a field value is valid (not empty, null, etc.)"""
        if value is None:
            return False

        if isinstance(value, str):
            return value.strip() != "" and value.lower() not in [
                "null",
                "none",
                "n/a",
                "",
            ]

        if isinstance(value, (list, dict)):
            return len(value) > 0

        return True

    def get_scoring_summary(
        self, results: List[Dict[str, Any]], result_type: str = "company"
    ) -> Dict[str, Any]:
        """Get a summary of scoring results"""
        try:
            scored_results = self.rank_results(results, result_type)

            if not scored_results:
                return {"error": "No results to score"}

            scores = [score["overall_score"] for _, score in scored_results]

            summary = {
                "total_results": len(results),
                "scored_results": len(scored_results),
                "score_statistics": {
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "avg_score": sum(scores) / len(scores),
                    "median_score": sorted(scores)[len(scores) // 2],
                },
                "quality_distribution": {
                    "excellent": len([s for s in scores if s >= 0.9]),
                    "very_good": len([s for s in scores if 0.8 <= s < 0.9]),
                    "good": len([s for s in scores if 0.7 <= s < 0.8]),
                    "fair": len([s for s in scores if 0.6 <= s < 0.7]),
                    "poor": len([s for s in scores if 0.5 <= s < 0.6]),
                    "very_poor": len([s for s in scores if s < 0.5]),
                },
                "threshold_analysis": {
                    "meets_min_threshold": len(
                        [s for s in scores if s >= self.thresholds.min_overall_score]
                    ),
                    "min_threshold": self.thresholds.min_overall_score,
                },
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating scoring summary: {e}")
            return {"error": str(e)}
