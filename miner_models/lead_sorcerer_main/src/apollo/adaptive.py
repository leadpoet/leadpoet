"""
Adaptive Search Strategies for Apollo Lead Generation

This module provides:
- Dynamic strategy selection based on performance
- Adaptive filter adjustment
- Performance-based strategy evolution
- Multi-strategy execution and comparison
"""

import logging
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for adaptive search strategies"""

    enable_adaptation: bool = True
    adaptation_threshold: float = 0.1  # 10% performance change triggers adaptation
    adaptation_interval: int = 10  # Adapt every N searches
    max_strategies: int = 5
    performance_window: int = 20  # Number of searches to consider for adaptation
    exploration_rate: float = 0.2  # 20% chance to try new strategies


class SearchStrategy:
    """Base class for search strategies"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.performance_history: List[float] = []
        self.success_count = 0
        self.total_count = 0
        self.last_used = None
        self.adaptation_count = 0

    def execute(self, filters: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute the search strategy - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute method")

    def adapt(self, performance_data: Dict[str, Any]) -> bool:
        """Adapt the strategy based on performance data"""
        self.adaptation_count += 1
        return True

    def get_performance_score(self) -> float:
        """Calculate performance score for this strategy"""
        if not self.performance_history:
            return 0.5  # Default score for untested strategies

        recent_performance = self.performance_history[-10:]  # Last 10 results
        return sum(recent_performance) / len(recent_performance)

    def record_performance(self, score: float, result_count: int, quality_score: float):
        """Record performance metrics"""
        self.performance_history.append(score)
        self.total_count += 1

        if score > 0.7:  # Consider successful if score > 70%
            self.success_count += 1

        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]

        self.last_used = datetime.now()

    def get_success_rate(self) -> float:
        """Get success rate for this strategy"""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count


class BroadSearchStrategy(SearchStrategy):
    """Broad search strategy - wide filters for maximum coverage"""

    def __init__(self):
        super().__init__(
            name="broad_search",
            description="Wide-ranging search with minimal filters for maximum coverage",
        )

    def execute(self, filters: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute broad search strategy"""
        import copy

        # Simplify filters to increase coverage
        broad_filters = copy.deepcopy(filters)

        # Remove restrictive filters
        broad_filters.pop("employee_count", None)
        broad_filters.pop("revenue", None)
        broad_filters.pop("seniority", None)

        # Expand industry filters if present
        if "industry" in broad_filters and isinstance(broad_filters["industry"], list):
            # Keep only top 2 industries to avoid over-restriction
            broad_filters["industry"] = broad_filters["industry"][:2]

        # Expand location filters if present
        if "country" in broad_filters and isinstance(broad_filters["country"], list):
            # Keep only top 3 countries
            broad_filters["country"] = broad_filters["country"][:3]

        return {
            "strategy": self.name,
            "filters": broad_filters,
            "expected_coverage": "high",
            "expected_quality": "medium",
        }

    def adapt(self, performance_data: Dict[str, Any]) -> bool:
        """Adapt broad search strategy"""
        super().adapt(performance_data)

        # If quality is too low, make filters slightly more restrictive
        if performance_data.get("avg_quality", 0) < 0.5:
            logger.info(
                "Broad search quality too low - adapting to be more restrictive"
            )
            return True

        return False


class FocusedSearchStrategy(SearchStrategy):
    """Focused search strategy - narrow filters for high quality"""

    def __init__(self):
        super().__init__(
            name="focused_search",
            description="Narrow, focused search with strict filters for high quality",
        )

    def execute(self, filters: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute focused search strategy"""
        import copy

        # Apply strict filters for high quality
        focused_filters = copy.deepcopy(filters)

        # Add quality-focused filters
        focused_filters["verified_only"] = True
        focused_filters["min_employee_count"] = 50

        # Restrict to top industries and locations
        if "industry" in focused_filters and isinstance(
            focused_filters["industry"], list
        ):
            focused_filters["industry"] = focused_filters["industry"][:1]

        if "country" in focused_filters and isinstance(
            focused_filters["country"], list
        ):
            focused_filters["country"] = focused_filters["country"][:1]

        # Add seniority requirements for person searches only if not already specified
        if "job_titles" in focused_filters and not focused_filters.get("seniority"):
            focused_filters["seniority"] = [
                "senior",
                "executive",
                "director",
                "manager",
            ]

        return {
            "strategy": self.name,
            "filters": focused_filters,
            "expected_coverage": "low",
            "expected_quality": "high",
        }

    def adapt(self, performance_data: Dict[str, Any]) -> bool:
        """Adapt focused search strategy"""
        super().adapt(performance_data)

        # If coverage is too low, relax some filters
        if performance_data.get("result_count", 0) < 10:
            logger.info("Focused search coverage too low - relaxing filters")
            return True

        return False


class BalancedSearchStrategy(SearchStrategy):
    """Balanced search strategy - moderate filters for balanced results"""

    def __init__(self):
        super().__init__(
            name="balanced_search",
            description="Moderate filters balancing coverage and quality",
        )

    def execute(self, filters: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute balanced search strategy"""
        import copy

        # Apply moderate filters
        balanced_filters = copy.deepcopy(filters)

        # Moderate industry restriction
        if "industry" in balanced_filters and isinstance(
            balanced_filters["industry"], list
        ):
            balanced_filters["industry"] = balanced_filters["industry"][:2]

        # Moderate location restriction
        if "country" in balanced_filters and isinstance(
            balanced_filters["country"], list
        ):
            balanced_filters["country"] = balanced_filters["country"][:2]

        # Moderate employee count filter
        if "employee_count" in balanced_filters:
            emp_filter = balanced_filters["employee_count"]
            if isinstance(emp_filter, dict):
                # Create a copy of the employee count filter to avoid mutating original
                emp_filter = dict(emp_filter)
                # Adjust to moderate range
                if emp_filter.get("min", 0) < 10:
                    emp_filter["min"] = 10
                if emp_filter.get("max", float("inf")) > 1000:
                    emp_filter["max"] = 1000
                balanced_filters["employee_count"] = emp_filter

        return {
            "strategy": self.name,
            "filters": balanced_filters,
            "expected_coverage": "medium",
            "expected_quality": "medium",
        }

    def adapt(self, performance_data: Dict[str, Any]) -> bool:
        """Adapt balanced search strategy"""
        super().adapt(performance_data)

        # Adapt based on performance balance
        quality = performance_data.get("avg_quality", 0)
        coverage = performance_data.get("result_count", 0)

        if quality < 0.6 and coverage > 50:
            logger.info("Balanced search quality too low - becoming more focused")
            return True
        elif quality > 0.8 and coverage < 20:
            logger.info("Balanced search coverage too low - becoming more broad")
            return True

        return False


class AdaptiveSearchOrchestrator:
    """Orchestrates multiple search strategies with adaptation"""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.strategies: List[SearchStrategy] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.current_strategy_index = 0
        self.adaptation_counter = 0

        # Initialize default strategies
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize default search strategies"""
        self.strategies = [
            BroadSearchStrategy(),
            FocusedSearchStrategy(),
            BalancedSearchStrategy(),
        ]

    def add_strategy(self, strategy: SearchStrategy):
        """Add a custom search strategy"""
        if len(self.strategies) < self.config.max_strategies:
            self.strategies.append(strategy)
            logger.info(f"Added custom strategy: {strategy.name}")
        else:
            logger.warning(
                f"Cannot add strategy {strategy.name}: max strategies reached"
            )

    def select_strategy(self, filters: Dict[str, Any], **kwargs) -> SearchStrategy:
        """Select the best strategy based on performance and context"""
        if not self.strategies:
            raise ValueError("No search strategies available")

        # Check if we should explore new strategies
        if random.random() < self.config.exploration_rate:
            # Randomly select a strategy for exploration
            selected = random.choice(self.strategies)
            logger.info(f"Exploring strategy: {selected.name}")
            return selected

        # Select strategy based on performance
        best_strategy = max(self.strategies, key=lambda s: s.get_performance_score())

        # Check if we should adapt current strategy
        if self._should_adapt():
            self._adapt_strategies()

        return best_strategy

    def _should_adapt(self) -> bool:
        """Determine if strategies should be adapted"""
        if not self.config.enable_adaptation:
            return False

        # Adapt based on interval
        if self.adaptation_counter % self.config.adaptation_interval == 0:
            return True

        # Adapt based on performance threshold
        if len(self.performance_history) >= self.config.performance_window:
            recent_performance = self.performance_history[
                -self.config.performance_window :
            ]
            avg_performance = sum(
                p.get("overall_score", 0) for p in recent_performance
            ) / len(recent_performance)

            if len(self.performance_history) >= self.config.performance_window * 2:
                previous_performance = self.performance_history[
                    -self.config.performance_window
                    * 2 : -self.config.performance_window
                ]
                prev_avg = sum(
                    p.get("overall_score", 0) for p in previous_performance
                ) / len(previous_performance)

                performance_change = abs(avg_performance - prev_avg)
                if performance_change > self.config.adaptation_threshold:
                    logger.info(
                        f"Performance change {performance_change:.2f} exceeds threshold - adapting strategies"
                    )
                    return True

        return False

    def _adapt_strategies(self):
        """Adapt all strategies based on performance data"""
        logger.info("Adapting search strategies")

        for strategy in self.strategies:
            if self.performance_history:
                # Get recent performance data for this strategy
                strategy_performance = [
                    p
                    for p in self.performance_history[-10:]
                    if p.get("strategy") == strategy.name
                ]

                if strategy_performance:
                    # Calculate average performance
                    avg_score = sum(
                        p.get("overall_score", 0) for p in strategy_performance
                    ) / len(strategy_performance)
                    avg_quality = sum(
                        p.get("quality_score", 0) for p in strategy_performance
                    ) / len(strategy_performance)
                    result_count = sum(
                        p.get("result_count", 0) for p in strategy_performance
                    )

                    performance_data = {
                        "avg_score": avg_score,
                        "avg_quality": avg_quality,
                        "result_count": result_count,
                    }

                    # Adapt the strategy
                    strategy.adapt(performance_data)

        self.adaptation_counter += 1

    def execute_search(
        self, filters: Dict[str, Any], search_type: str = "company", **kwargs
    ) -> Dict[str, Any]:
        """Execute search using the best strategy"""
        # Select strategy
        strategy = self.select_strategy(filters, **kwargs)

        # Execute strategy
        strategy_result = strategy.execute(filters, **kwargs)

        # Record execution
        execution_data = {
            "strategy": strategy.name,
            "timestamp": datetime.now().isoformat(),
            "filters": filters,
            "search_type": search_type,
            "strategy_result": strategy_result,
        }

        return execution_data

    def record_performance(
        self,
        strategy_name: str,
        overall_score: float,
        result_count: int,
        quality_score: float,
    ):
        """Record performance metrics for strategy evaluation"""
        performance_data = {
            "strategy": strategy_name,
            "overall_score": overall_score,
            "result_count": result_count,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
        }

        self.performance_history.append(performance_data)

        # Update strategy performance
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.record_performance(overall_score, result_count, quality_score)
                break

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance summary for all strategies"""
        performance_summary = {}

        for strategy in self.strategies:
            performance_summary[strategy.name] = {
                "performance_score": strategy.get_performance_score(),
                "success_rate": strategy.get_success_rate(),
                "total_searches": strategy.total_count,
                "last_used": strategy.last_used.isoformat()
                if strategy.last_used
                else None,
                "adaptation_count": strategy.adaptation_count,
            }

        return {
            "strategies": performance_summary,
            "overall_performance": {
                "total_searches": sum(s.total_count for s in self.strategies),
                "avg_performance": sum(
                    s.get_performance_score() for s in self.strategies
                )
                / len(self.strategies),
                "adaptation_counter": self.adaptation_counter,
            },
            "performance_history_count": len(self.performance_history),
        }

    def get_adaptation_recommendations(self) -> List[str]:
        """Get recommendations for strategy adaptation"""
        recommendations = []

        if not self.performance_history:
            recommendations.append(
                "No performance data available - run searches to gather data"
            )
            return recommendations

        # Analyze strategy performance
        strategy_performance = {}
        for strategy in self.strategies:
            strategy_data = [
                p
                for p in self.performance_history
                if p.get("strategy") == strategy.name
            ]
            if strategy_data:
                avg_score = sum(p.get("overall_score", 0) for p in strategy_data) / len(
                    strategy_data
                )
                strategy_performance[strategy.name] = avg_score

        if strategy_performance:
            best_strategy = max(strategy_performance.items(), key=lambda x: x[1])
            worst_strategy = min(strategy_performance.items(), key=lambda x: x[1])

            if best_strategy[1] - worst_strategy[1] > 0.3:
                recommendations.append(
                    f"Strategy performance gap is large - consider adapting {worst_strategy[0]}"
                )

            if best_strategy[1] < 0.6:
                recommendations.append(
                    "All strategies performing poorly - review base filters and ICP configuration"
                )

        # Check adaptation frequency
        if self.adaptation_counter < 3:
            recommendations.append(
                "Limited strategy adaptation - consider running more searches to enable learning"
            )

        if not recommendations:
            recommendations.append(
                "Strategy performance is balanced - no immediate adaptation needed"
            )

        return recommendations

    def reset_strategies(self):
        """Reset all strategies to initial state"""
        for strategy in self.strategies:
            strategy.performance_history.clear()
            strategy.success_count = 0
            strategy.total_count = 0
            strategy.last_used = None
            strategy.adaptation_count = 0

        self.performance_history.clear()
        self.adaptation_counter = 0

        logger.info("All strategies reset to initial state")
