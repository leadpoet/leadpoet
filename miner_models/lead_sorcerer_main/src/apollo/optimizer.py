"""
AI-Powered Search Query Optimization for Apollo Lead Generation

This module provides advanced search strategies including:
- AI-powered query optimization
- Dynamic filter generation based on ICP analysis
- Search result scoring and ranking
- Adaptive search strategies
"""

import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class AISearchConfig:
    """Configuration for AI-powered search optimization"""

    enable_ai_optimization: bool = True
    enable_dynamic_filters: bool = True
    enable_result_scoring: bool = True
    enable_adaptive_strategies: bool = True
    max_filter_variations: int = 5
    scoring_weights: Dict[str, float] = None
    learning_rate: float = 0.1
    min_confidence_threshold: float = 0.7


class AISearchQueryOptimizer:
    """AI-powered search query optimizer for Apollo API"""

    def __init__(self, config: AISearchConfig):
        self.config = config
        self.search_history: Dict[str, Dict[str, Any]] = {}
        self.success_patterns: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {}

        # Initialize scoring weights
        if self.config.scoring_weights is None:
            self.config.scoring_weights = {
                "relevance": 0.4,
                "completeness": 0.3,
                "freshness": 0.2,
                "quality_score": 0.1,
            }

    def optimize_search_strategy(
        self, base_filters: Dict[str, Any], search_type: str, icp_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate AI-optimized search strategies"""
        if not self.config.enable_ai_optimization:
            return [base_filters]

        strategies = []

        # Base strategy
        strategies.append(
            {
                "filters": base_filters.copy(),
                "confidence": 1.0,
                "strategy_type": "base",
                "reasoning": "Original search parameters",
            }
        )

        # AI-optimized variations
        if self.config.enable_dynamic_filters:
            ai_variations = self._generate_ai_variations(
                base_filters, search_type, icp_config
            )
            strategies.extend(ai_variations)

        # Sort by confidence score
        strategies.sort(key=lambda x: x["confidence"], reverse=True)

        # Limit to max variations
        return strategies[: self.config.max_filter_variations]

    def _generate_ai_variations(
        self, base_filters: Dict[str, Any], search_type: str, icp_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate AI-powered filter variations"""
        variations = []

        # Industry optimization
        if "industry" in base_filters and "industries" in icp_config:
            industry_variations = self._optimize_industry_filters(
                base_filters, icp_config["industries"], search_type
            )
            variations.extend(industry_variations)

        # Location optimization
        if "country" in base_filters and "countries" in icp_config:
            location_variations = self._optimize_location_filters(
                base_filters, icp_config["countries"], search_type
            )
            variations.extend(location_variations)

        # Job title optimization (for person searches)
        if search_type == "person" and "job_titles" in base_filters:
            job_variations = self._optimize_job_title_filters(
                base_filters, icp_config.get("role_priority", {})
            )
            variations.extend(job_variations)

        # Company size optimization
        if "employee_count" in base_filters:
            size_variations = self._optimize_company_size_filters(base_filters)
            variations.extend(size_variations)

        return variations

    def _optimize_industry_filters(
        self, base_filters: Dict[str, Any], icp_industries: List[str], search_type: str
    ) -> List[Dict[str, Any]]:
        """Optimize industry filters based on ICP and search history"""
        variations = []

        # Analyze industry performance from search history
        industry_performance = self._analyze_industry_performance()

        # Create variations based on performance
        for industry in icp_industries[:3]:  # Top 3 industries
            confidence = self._calculate_industry_confidence(
                industry, industry_performance
            )

            if confidence > self.config.min_confidence_threshold:
                variation = base_filters.copy()
                variation["industry"] = [industry]
                variations.append(
                    {
                        "filters": variation,
                        "confidence": confidence,
                        "strategy_type": "industry_optimized",
                        "reasoning": f"Industry-focused search based on ICP and performance: {industry}",
                    }
                )

        return variations

    def _optimize_location_filters(
        self, base_filters: Dict[str, Any], icp_countries: List[str], search_type: str
    ) -> List[Dict[str, Any]]:
        """Optimize location filters based on ICP and search history"""
        variations = []

        # Analyze location performance from search history
        location_performance = self._analyze_location_performance()

        # Create variations based on performance
        for country in icp_countries[:2]:  # Top 2 countries
            confidence = self._calculate_location_confidence(
                country, location_performance
            )

            if confidence > self.config.min_confidence_threshold:
                variation = base_filters.copy()
                variation["country"] = [country]
                variations.append(
                    {
                        "filters": variation,
                        "confidence": confidence,
                        "strategy_type": "location_optimized",
                        "reasoning": f"Location-focused search based on ICP and performance: {country}",
                    }
                )

        return variations

    def _optimize_job_title_filters(
        self, base_filters: Dict[str, Any], role_priority: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Optimize job title filters based on role priority and search history"""
        variations = []

        # Analyze job title performance from search history
        job_performance = self._analyze_job_title_performance()

        # Create variations based on role priority
        for role, priority in sorted(
            role_priority.items(), key=lambda x: x[1], reverse=True
        )[:3]:
            confidence = self._calculate_job_title_confidence(role, job_performance)

            if confidence > self.config.min_confidence_threshold:
                variation = base_filters.copy()
                variation["job_titles"] = [role]
                variations.append(
                    {
                        "filters": variation,
                        "confidence": confidence,
                        "strategy_type": "role_optimized",
                        "reasoning": f"Role-focused search based on priority and performance: {role}",
                    }
                )

        return variations

    def _optimize_company_size_filters(
        self, base_filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Optimize company size filters based on search history"""
        variations = []

        # Analyze company size performance from search history
        size_performance = self._analyze_company_size_performance()

        # Create variations with different size ranges
        size_ranges = [
            {"min": 1, "max": 50, "label": "startup"},
            {"min": 51, "max": 200, "label": "small"},
            {"min": 201, "max": 1000, "label": "medium"},
            {"min": 1001, "max": 10000, "label": "large"},
        ]

        for size_range in size_ranges:
            confidence = self._calculate_size_confidence(size_range, size_performance)

            if confidence > self.config.min_confidence_threshold:
                variation = base_filters.copy()
                variation["employee_count"] = size_range
                variations.append(
                    {
                        "filters": variation,
                        "confidence": confidence,
                        "strategy_type": "size_optimized",
                        "reasoning": f"Size-focused search: {size_range['label']} companies",
                    }
                )

        return variations

    def _analyze_industry_performance(self) -> Dict[str, float]:
        """Analyze industry performance from search history"""
        performance = {}

        for search_id, search_data in self.search_history.items():
            if "industry" in search_data.get("filters", {}):
                industry = search_data["filters"]["industry"]
                result_count = search_data.get("result_count", 0)
                quality_score = search_data.get("quality_score", 0)

                if industry not in performance:
                    performance[industry] = []

                performance[industry].append(result_count * quality_score)

        # Calculate average performance per industry
        return {
            industry: sum(scores) / len(scores) if scores else 0
            for industry, scores in performance.items()
        }

    def _analyze_location_performance(self) -> Dict[str, float]:
        """Analyze location performance from search history"""
        performance = {}

        for search_id, search_data in self.search_history.items():
            if "country" in search_data.get("filters", {}):
                country = search_data["filters"]["country"]
                result_count = search_data.get("result_count", 0)
                quality_score = search_data.get("quality_score", 0)

                if country not in performance:
                    performance[country] = []

                performance[country].append(result_count * quality_score)

        # Calculate average performance per country
        return {
            country: sum(scores) / len(scores) if scores else 0
            for country, scores in performance.items()
        }

    def _analyze_job_title_performance(self) -> Dict[str, float]:
        """Analyze job title performance from search history"""
        performance = {}

        for search_id, search_data in self.search_history.items():
            if "job_titles" in search_data.get("filters", {}):
                job_titles = search_data["filters"]["job_titles"]
                for job_title in job_titles:
                    result_count = search_data.get("result_count", 0)
                    quality_score = search_data.get("quality_score", 0)

                    if job_title not in performance:
                        performance[job_title] = []

                    performance[job_title].append(result_count * quality_score)

        # Calculate average performance per job title
        return {
            job_title: sum(scores) / len(scores) if scores else 0
            for job_title, scores in performance.items()
        }

    def _analyze_company_size_performance(self) -> Dict[str, float]:
        """Analyze company size performance from search history"""
        performance = {}

        for search_id, search_data in self.search_history.items():
            if "employee_count" in search_data.get("filters", {}):
                emp_count = search_data["filters"]["employee_count"]
                size_label = self._get_size_label(emp_count)
                result_count = search_data.get("result_count", 0)
                quality_score = search_data.get("quality_score", 0)

                if size_label not in performance:
                    performance[size_label] = []

                performance[size_label].append(result_count * quality_score)

        # Calculate average performance per size category
        return {
            size_label: sum(scores) / len(scores) if scores else 0
            for size_label, scores in performance.items()
        }

    def _get_size_label(self, emp_count: Dict[str, int]) -> str:
        """Get size label from employee count range"""
        _ = emp_count.get("min", 0)  # min_emp not used, replaced with _
        max_emp = emp_count.get("max", float("inf"))

        if max_emp <= 50:
            return "startup"
        elif max_emp <= 200:
            return "small"
        elif max_emp <= 1000:
            return "medium"
        else:
            return "large"

    def _calculate_industry_confidence(
        self, industry: str, performance: Dict[str, float]
    ) -> float:
        """Calculate confidence score for industry filter"""
        if industry not in performance:
            return 0.5  # Default confidence for unknown industries

        # Normalize performance score (0-1 range)
        max_performance = max(performance.values()) if performance else 1
        normalized_score = (
            performance[industry] / max_performance if max_performance > 0 else 0
        )

        return min(1.0, normalized_score + 0.3)  # Boost confidence for known industries

    def _calculate_location_confidence(
        self, country: str, performance: Dict[str, float]
    ) -> float:
        """Calculate confidence score for location filter"""
        if country not in performance:
            return 0.5  # Default confidence for unknown countries

        # Normalize performance score (0-1 range)
        max_performance = max(performance.values()) if performance else 1
        normalized_score = (
            performance[country] / max_performance if max_performance > 0 else 0
        )

        return min(1.0, normalized_score + 0.3)  # Boost confidence for known countries

    def _calculate_job_title_confidence(
        self, job_title: str, performance: Dict[str, float]
    ) -> float:
        """Calculate confidence score for job title filter"""
        if job_title not in performance:
            return 0.5  # Default confidence for unknown job titles

        # Normalize performance score (0-1 range)
        max_performance = max(performance.values()) if performance else 1
        normalized_score = (
            performance[job_title] / max_performance if max_performance > 0 else 0
        )

        return min(1.0, normalized_score + 0.3)  # Boost confidence for known job titles

    def _calculate_size_confidence(
        self, size_range: Dict[str, Any], performance: Dict[str, float]
    ) -> float:
        """Calculate confidence score for company size filter"""
        size_label = size_range["label"]

        if size_label not in performance:
            return 0.5  # Default confidence for unknown sizes

        # Normalize performance score (0-1 range)
        max_performance = max(performance.values()) if performance else 1
        normalized_score = (
            performance[size_label] / max_performance if max_performance > 0 else 0
        )

        return min(1.0, normalized_score + 0.3)  # Boost confidence for known sizes

    def record_search_result(
        self,
        search_id: str,
        filters: Dict[str, Any],
        result_count: int,
        quality_score: float,
        response_time: float,
    ):
        """Record search result for learning and optimization"""
        self.search_history[search_id] = {
            "filters": filters,
            "result_count": result_count,
            "quality_score": quality_score,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
        }

        # Keep only recent history (last 100 searches)
        if len(self.search_history) > 100:
            oldest_key = min(
                self.search_history.keys(),
                key=lambda k: self.search_history[k]["timestamp"],
            )
            del self.search_history[oldest_key]

        # Update performance metrics
        self._update_performance_metrics(
            search_id, result_count, quality_score, response_time
        )

    def _update_performance_metrics(
        self,
        search_id: str,
        result_count: int,
        quality_score: float,
        response_time: float,
    ):
        """Update performance metrics for learning"""
        # Create a hash of the search filters for pattern recognition
        filter_hash = hashlib.md5(
            json.dumps(
                self.search_history[search_id]["filters"], sort_keys=True
            ).encode()
        ).hexdigest()

        if filter_hash not in self.performance_metrics:
            self.performance_metrics[filter_hash] = []

        # Store performance metrics
        self.performance_metrics[filter_hash].append(
            {
                "result_count": result_count,
                "quality_score": quality_score,
                "response_time": response_time,
                "timestamp": datetime.now(),
            }
        )

        # Keep only recent metrics (last 20 per pattern)
        if len(self.performance_metrics[filter_hash]) > 20:
            self.performance_metrics[filter_hash] = self.performance_metrics[
                filter_hash
            ][-20:]

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics and insights"""
        return {
            "total_searches": len(self.search_history),
            "unique_patterns": len(self.performance_metrics),
            "success_patterns": len(self.success_patterns),
            "average_confidence": self._calculate_average_confidence(),
            "performance_trends": self._calculate_performance_trends(),
            "recommendations": self._generate_optimization_recommendations(),
        }

    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all search strategies"""
        if not self.search_history:
            return 0.0

        total_confidence = 0
        count = 0

        for search_data in self.search_history.values():
            # Estimate confidence based on result quality and count
            quality_score = search_data.get("quality_score", 0)
            result_count = search_data.get("result_count", 0)

            # Simple confidence calculation
            confidence = min(1.0, (quality_score + result_count / 100) / 2)
            total_confidence += confidence
            count += 1

        return total_confidence / count if count > 0 else 0.0

    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        if not self.search_history:
            return {"trend": "stable", "improvement_rate": 0.0}

        # Group searches by time periods
        recent_searches = []
        older_searches = []

        cutoff_time = datetime.now() - timedelta(days=7)

        for search_data in self.search_history.values():
            timestamp = datetime.fromisoformat(search_data["timestamp"])
            if timestamp > cutoff_time:
                recent_searches.append(search_data)
            else:
                older_searches.append(search_data)

        if not older_searches or not recent_searches:
            return {"trend": "stable", "improvement_rate": 0.0}

        # Calculate average performance for each period
        recent_avg = sum(s.get("quality_score", 0) for s in recent_searches) / len(
            recent_searches
        )
        older_avg = sum(s.get("quality_score", 0) for s in older_searches) / len(
            older_searches
        )

        improvement_rate = (
            ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        )

        if improvement_rate > 5:
            trend = "improving"
        elif improvement_rate < -5:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "improvement_rate": improvement_rate,
            "recent_performance": recent_avg,
            "historical_performance": older_avg,
        }

    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance analysis"""
        recommendations = []

        if not self.search_history:
            recommendations.append(
                "No search history available - start with basic searches to gather data"
            )
            return recommendations

        # Analyze performance patterns
        avg_quality = sum(
            s.get("quality_score", 0) for s in self.search_history.values()
        ) / len(self.search_history)
        avg_results = sum(
            s.get("result_count", 0) for s in self.search_history.values()
        ) / len(self.search_history)

        if avg_quality < 0.6:
            recommendations.append(
                "Overall quality score is low - consider refining search criteria"
            )

        if avg_results < 10:
            recommendations.append(
                "Low result counts - consider broadening search filters"
            )

        if avg_results > 100:
            recommendations.append(
                "High result counts - consider narrowing search filters for better quality"
            )

        # Check for specific patterns
        if len(self.performance_metrics) < 5:
            recommendations.append(
                "Limited search pattern diversity - try different filter combinations"
            )

        # Performance trend recommendations
        trends = self._calculate_performance_trends()
        if trends["trend"] == "declining":
            recommendations.append(
                "Performance is declining - review recent search strategies"
            )
        elif trends["trend"] == "improving":
            recommendations.append(
                "Performance is improving - continue current optimization strategies"
            )

        if not recommendations:
            recommendations.append(
                "Performance is optimal - continue current search strategies"
            )

        return recommendations
