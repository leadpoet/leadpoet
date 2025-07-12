"""
Cost Telemetry - Track costs per lead and enforce budget constraints.
Provides cost tracking and budget enforcement for the Leadpoet API.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from prometheus_client import Counter, Histogram, Gauge

from app.core.config import settings
from app.core.redis_client import RedisClient

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Breakdown of costs for a query."""
    query_id: str
    timestamp: datetime
    total_cost_usd: float
    llm_cost_usd: float
    api_cost_usd: float
    infrastructure_cost_usd: float
    lead_count: int
    cost_per_lead_usd: float
    llm_hit_ratio: float


class CostTelemetry:
    """Cost telemetry and budget enforcement."""
    
    def __init__(self):
        self.redis_client = RedisClient()
        
        # Cost tracking metrics
        self.total_cost = Counter(
            'leadpoet_total_cost_usd',
            'Total cost in USD',
            ['component']
        )
        
        self.cost_per_lead = Histogram(
            'leadpoet_cost_per_lead_usd',
            'Cost per lead in USD',
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        )
        
        self.llm_hit_ratio = Histogram(
            'leadpoet_llm_hit_ratio',
            'Ratio of leads that used LLM scoring',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        self.daily_cost = Gauge(
            'leadpoet_daily_cost_usd',
            'Daily cost in USD'
        )
        
        # Load cost rates from configuration
        self.cost_rates = {
            'llm_gpt4o_input': settings.COST_LLM_GPT4O_INPUT_RATE / 1000,    # Convert per 1K tokens to per token
            'llm_gpt4o_output': settings.COST_LLM_GPT4O_OUTPUT_RATE / 1000,  # Convert per 1K tokens to per token
            'api_request': settings.COST_API_REQUEST_RATE,
            'infrastructure_per_lead': settings.COST_INFRASTRUCTURE_PER_LEAD_RATE,
        }
        
        # Load budget thresholds from configuration
        self.avg_cost_threshold = settings.COST_AVG_THRESHOLD
        self.p99_cost_threshold = settings.COST_P99_THRESHOLD
        self.daily_budget_usd = settings.COST_DAILY_BUDGET_USD
        
        logger.info(f"Cost telemetry initialized with rates: {self.cost_rates}")
        logger.info(f"Budget thresholds: avg=${self.avg_cost_threshold:.6f}, p99=${self.p99_cost_threshold:.6f}, daily=${self.daily_budget_usd:.2f}")
    
    async def record_query_cost(
        self, 
        query_id: str, 
        lead_count: int, 
        processing_time_ms: float,
        llm_calls: int = 0,
        llm_input_tokens: int = 0,
        llm_output_tokens: int = 0
    ):
        """Record cost for a completed query."""
        try:
            # Calculate cost breakdown
            llm_cost = self._calculate_llm_cost(llm_input_tokens, llm_output_tokens)
            api_cost = self.cost_rates['api_request']
            infrastructure_cost = lead_count * self.cost_rates['infrastructure_per_lead']
            total_cost = llm_cost + api_cost + infrastructure_cost
            
            # Calculate cost per lead
            cost_per_lead = total_cost / lead_count if lead_count > 0 else 0.0
            
            # Calculate LLM hit ratio
            llm_hit_ratio = llm_calls / lead_count if lead_count > 0 else 0.0
            
            # Create cost breakdown
            cost_breakdown = CostBreakdown(
                query_id=query_id,
                timestamp=datetime.now(),
                total_cost_usd=total_cost,
                llm_cost_usd=llm_cost,
                api_cost_usd=api_cost,
                infrastructure_cost_usd=infrastructure_cost,
                lead_count=lead_count,
                cost_per_lead_usd=cost_per_lead,
                llm_hit_ratio=llm_hit_ratio
            )
            
            # Record metrics
            self.total_cost.labels(component='llm').inc(llm_cost)
            self.total_cost.labels(component='api').inc(api_cost)
            self.total_cost.labels(component='infrastructure').inc(infrastructure_cost)
            
            self.cost_per_lead.observe(cost_per_lead)
            self.llm_hit_ratio.observe(llm_hit_ratio)
            
            # Store in Redis for historical tracking
            await self._store_cost_breakdown(cost_breakdown)
            
            # Update daily cost
            await self._update_daily_cost(total_cost)
            
            # Check budget violations
            await self._check_budget_violations(cost_breakdown)
            
            logger.debug(f"Recorded cost for query {query_id}: ${total_cost:.6f} (${cost_per_lead:.6f} per lead)")
            
        except Exception as e:
            logger.error(f"Error recording query cost: {e}")
    
    def _calculate_llm_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate LLM API cost."""
        input_cost = input_tokens * self.cost_rates['llm_gpt4o_input']
        output_cost = output_tokens * self.cost_rates['llm_gpt4o_output']
        return input_cost + output_cost
    
    async def _store_cost_breakdown(self, cost_breakdown: CostBreakdown):
        """Store cost breakdown in Redis for historical tracking."""
        try:
            # Store in Redis sorted set by timestamp
            key = f"costs:{cost_breakdown.timestamp.strftime('%Y-%m-%d')}"
            score = cost_breakdown.timestamp.timestamp()
            value = json.dumps({
                'query_id': cost_breakdown.query_id,
                'total_cost_usd': cost_breakdown.total_cost_usd,
                'llm_cost_usd': cost_breakdown.llm_cost_usd,
                'api_cost_usd': cost_breakdown.api_cost_usd,
                'infrastructure_cost_usd': cost_breakdown.infrastructure_cost_usd,
                'lead_count': cost_breakdown.lead_count,
                'cost_per_lead_usd': cost_breakdown.cost_per_lead_usd,
                'llm_hit_ratio': cost_breakdown.llm_hit_ratio
            })
            
            await self.redis_client.zadd(key, {value: score})
            
            # Keep only last 1000 entries per day
            await self.redis_client.zremrangebyrank(key, 0, -1001)
            
        except Exception as e:
            logger.error(f"Error storing cost breakdown: {e}")
    
    async def _update_daily_cost(self, cost_usd: float):
        """Update daily cost tracking."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            key = f"daily_cost:{today}"
            
            # Get current daily cost
            current_cost = await self.redis_client.get(key)
            current_cost = float(current_cost) if current_cost else 0.0
            
            # Add new cost
            new_cost = current_cost + cost_usd
            await self.redis_client.set(key, str(new_cost), ex=86400)  # 24 hours
            
            # Update Prometheus gauge
            self.daily_cost.set(new_cost)
            
        except Exception as e:
            logger.error(f"Error updating daily cost: {e}")
    
    async def _check_budget_violations(self, cost_breakdown: CostBreakdown):
        """Check for budget violations and alert."""
        try:
            # Check average cost threshold
            if cost_breakdown.cost_per_lead_usd > self.avg_cost_threshold:
                logger.warning(
                    f"Average cost threshold exceeded: ${cost_breakdown.cost_per_lead_usd:.6f} > ${self.avg_cost_threshold:.6f}"
                )
            
            # Check P99 cost threshold
            if cost_breakdown.cost_per_lead_usd > self.p99_cost_threshold:
                logger.warning(
                    f"P99 cost threshold exceeded: ${cost_breakdown.cost_per_lead_usd:.6f} > ${self.p99_cost_threshold:.6f}"
                )
            
            # Check daily budget
            today = datetime.now().strftime('%Y-%m-%d')
            key = f"daily_cost:{today}"
            daily_cost = await self.redis_client.get(key)
            daily_cost = float(daily_cost) if daily_cost else 0.0
            
            if daily_cost > self.daily_budget_usd:
                logger.warning(
                    f"Daily budget exceeded: ${daily_cost:.2f} > ${self.daily_budget_usd:.2f}"
                )
            
        except Exception as e:
            logger.error(f"Error checking budget violations: {e}")
    
    async def get_cost_statistics(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get cost statistics for a specific date."""
        try:
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            key = f"costs:{date}"
            cost_entries = await self.redis_client.zrange(key, 0, -1, withscores=True)
            
            if not cost_entries:
                return {
                    'date': date,
                    'total_queries': 0,
                    'total_cost_usd': 0.0,
                    'avg_cost_per_lead_usd': 0.0,
                    'p95_cost_per_lead_usd': 0.0,
                    'p99_cost_per_lead_usd': 0.0,
                    'total_leads': 0,
                    'avg_llm_hit_ratio': 0.0
                }
            
            # Parse cost entries
            costs_per_lead = []
            total_cost = 0.0
            total_leads = 0
            llm_hit_ratios = []
            
            for entry, _ in cost_entries:
                data = json.loads(entry)
                costs_per_lead.append(data['cost_per_lead_usd'])
                total_cost += data['total_cost_usd']
                total_leads += data['lead_count']
                llm_hit_ratios.append(data['llm_hit_ratio'])
            
            # Calculate statistics
            costs_per_lead.sort()
            total_queries = len(cost_entries)
            
            p95_index = int(0.95 * len(costs_per_lead))
            p99_index = int(0.99 * len(costs_per_lead))
            
            return {
                'date': date,
                'total_queries': total_queries,
                'total_cost_usd': total_cost,
                'avg_cost_per_lead_usd': total_cost / total_leads if total_leads > 0 else 0.0,
                'p95_cost_per_lead_usd': costs_per_lead[p95_index] if p95_index < len(costs_per_lead) else 0.0,
                'p99_cost_per_lead_usd': costs_per_lead[p99_index] if p99_index < len(costs_per_lead) else 0.0,
                'total_leads': total_leads,
                'avg_llm_hit_ratio': sum(llm_hit_ratios) / len(llm_hit_ratios) if llm_hit_ratios else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting cost statistics: {e}")
            return {}
    
    async def get_daily_cost_trend(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily cost trend for the last N days."""
        try:
            trend = []
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                stats = await self.get_cost_statistics(date)
                trend.append(stats)
            
            return list(reversed(trend))  # Return in chronological order
            
        except Exception as e:
            logger.error(f"Error getting daily cost trend: {e}")
            return []
    
    async def get_budget_alerts(self) -> List[Dict[str, Any]]:
        """Get current budget alerts."""
        try:
            alerts = []
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Check daily budget
            key = f"daily_cost:{today}"
            daily_cost = await self.redis_client.get(key)
            daily_cost = float(daily_cost) if daily_cost else 0.0
            
            if daily_cost > self.daily_budget_usd * 0.8:  # 80% threshold
                alerts.append({
                    'type': 'daily_budget_warning',
                    'message': f'Daily budget at ${daily_cost:.2f} (${self.daily_budget_usd:.2f} limit)',
                    'severity': 'warning' if daily_cost < self.daily_budget_usd else 'critical'
                })
            
            # Check recent cost per lead
            recent_stats = await self.get_cost_statistics(today)
            if recent_stats.get('avg_cost_per_lead_usd', 0) > self.avg_cost_threshold:
                alerts.append({
                    'type': 'avg_cost_threshold',
                    'message': f'Average cost per lead ${recent_stats["avg_cost_per_lead_usd"]:.6f} exceeds threshold ${self.avg_cost_threshold:.6f}',
                    'severity': 'warning'
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting budget alerts: {e}")
            return []
    
    async def reset_daily_cost(self):
        """Reset daily cost tracking (for testing)."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            key = f"daily_cost:{today}"
            await self.redis_client.delete(key)
            self.daily_cost.set(0.0)
            logger.info("Daily cost tracking reset")
        except Exception as e:
            logger.error(f"Error resetting daily cost: {e}")
    
    def get_cost_rates(self) -> Dict[str, float]:
        """Get current cost rates for monitoring/debugging."""
        return self.cost_rates.copy()
    
    def get_budget_thresholds(self) -> Dict[str, float]:
        """Get current budget thresholds for monitoring/debugging."""
        return {
            'avg_cost_threshold': self.avg_cost_threshold,
            'p99_cost_threshold': self.p99_cost_threshold,
            'daily_budget_usd': self.daily_budget_usd
        }
    
    def calculate_cost(self, scored_leads: List[Any], llm_call_count: int = 0) -> float:
        """Calculate total cost for a list of scored leads."""
        try:
            lead_count = len(scored_leads)
            if lead_count == 0:
                return 0.0
            
            # Calculate LLM cost
            llm_cost = llm_call_count * (self.cost_rates['llm_gpt4o_input'] * 100 + self.cost_rates['llm_gpt4o_output'] * 50)  # Estimate tokens
            
            # Calculate API cost
            api_cost = self.cost_rates['api_request']
            
            # Calculate infrastructure cost
            infrastructure_cost = lead_count * self.cost_rates['infrastructure_per_lead']
            
            total_cost = llm_cost + api_cost + infrastructure_cost
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0 