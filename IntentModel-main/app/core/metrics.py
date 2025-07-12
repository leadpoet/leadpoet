"""
Metrics Collection - Prometheus metrics for Leadpoet Intent Model v1.1
Provides comprehensive metrics for monitoring performance, costs, and business KPIs.
"""

import time
import logging
import threading
from typing import Dict, Any
from collections import deque

from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)

from app.core.config import settings
from loguru import logger

# Unregister only our specific collectors to avoid duplicates
collectors_to_remove = [
    'leadpoet_queries_total',
    'leadpoet_query_duration_seconds',
    'leadpoet_leads_returned',
    # ... add other metric names
]
for collector in list(REGISTRY._collector_to_names.keys()):
    if any(name in REGISTRY._collector_to_names.get(collector, {}) for name in collectors_to_remove):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass  # Collector might not exist
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Prometheus metrics collector for Leadpoet API."""
    
    def __init__(self):
        # Query metrics
        self.query_counter = Counter(
            'leadpoet_queries_total',
            'Total number of queries',
            ['status', 'method']  # success/error, api/webhook
        )
        
        # Enhanced pipeline stage histograms (Epic 7 Task 1)
        self.query_duration = Histogram(
            'leadpoet_query_duration_seconds',
            'Query processing duration in seconds',
            ['stage'],  # total, parsing, retrieval, scoring, filtering, response
            buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Detailed pipeline stage histograms
        self.pipeline_stage_duration = Histogram(
            'leadpoet_pipeline_stage_duration_seconds',
            'Duration of each pipeline stage in seconds',
            ['stage', 'sub_stage'],  # stage: parsing, retrieval, scoring, filtering; sub_stage: specific component
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        )
        
        # Database operation histograms
        self.db_operation_duration = Histogram(
            'leadpoet_db_operation_duration_seconds',
            'Database operation duration in seconds',
            ['operation', 'table'],  # operation: select, insert, update; table: leads, intent_snippets, queries
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
        )
        
        # Cache operation histograms
        self.cache_operation_duration = Histogram(
            'leadpoet_cache_operation_duration_seconds',
            'Cache operation duration in seconds',
            ['operation', 'cache_type'],  # operation: get, set, delete; cache_type: features, llm_responses, candidates
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1]
        )
        
        # LLM API call histograms
        self.llm_api_duration = Histogram(
            'leadpoet_llm_api_duration_seconds',
            'LLM API call duration in seconds',
            ['model', 'operation'],  # model: gpt-4o, gpt-3.5-turbo; operation: completion, embedding
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        # Scoring component histograms
        self.scoring_component_duration = Histogram(
            'leadpoet_scoring_component_duration_seconds',
            'Individual scoring component duration in seconds',
            ['component', 'method'],  # component: bm25, time_decay, llm, lgbm; method: primary, fallback
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
        )
        
        # Retrieval method histograms
        self.retrieval_method_duration = Histogram(
            'leadpoet_retrieval_method_duration_seconds',
            'Retrieval method duration in seconds',
            ['method', 'phase'],  # method: sql, ann, hybrid; phase: initial, relaxed, final
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        )
        
        self.leads_returned = Histogram(
            'leadpoet_leads_returned',
            'Number of leads returned per query',
            buckets=[10, 50, 100, 200, 500, 1000, 2000]
        )
        
        self.leads_processed = Counter(
            'leadpoet_leads_processed_total',
            'Total leads processed',
            ['stage']  # retrieval, scoring, filtering
        )
        
        # Component-specific metrics
        self.prompt_parsing_duration = Histogram(
            'leadpoet_prompt_parsing_duration_seconds',
            'Prompt parsing duration in seconds',
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        )
        
        self.retrieval_duration = Histogram(
            'leadpoet_retrieval_duration_seconds',
            'Retrieval duration in seconds',
            ['method'],  # sql, ann, hybrid
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        )
        
        self.scoring_duration = Histogram(
            'leadpoet_scoring_duration_seconds',
            'Scoring duration in seconds',
            ['component'],  # bm25, time_decay, llm, final
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'leadpoet_cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses = Counter(
            'leadpoet_cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        
        # Error metrics
        self.error_counter = Counter(
            'leadpoet_errors_total',
            'Total number of errors',
            ['stage', 'error_type']
        )
        
        # Latency SLA metrics
        self.latency_sla_violations = Counter(
            'leadpoet_latency_sla_violations_total',
            'Total latency SLA violations',
            ['threshold']  # p95, p99
        )
        
        # LLM metrics
        self.llm_calls = Counter(
            'leadpoet_llm_calls_total',
            'Total LLM API calls',
            ['model', 'status']
        )
        
        self.llm_tokens = Counter(
            'leadpoet_llm_tokens_total',
            'Total LLM tokens used',
            ['model', 'type']  # input, output
        )
        
        # Business metrics
        self.query_volume = Gauge(
            'leadpoet_query_volume',
            'Current query volume (QPS)'
        )
        
        self.active_queries = Gauge(
            'leadpoet_active_queries',
            'Number of currently active queries'
        )
        
        # Query quality reputation metrics (Epic 7 Task 4)
        self.query_quality_score_gauge = Gauge(
            'leadpoet_query_quality_score',
            'Query quality score (0.0 to 1.0)',
            ['user_id']
        )
        
        self.query_flag_rate_gauge = Gauge(
            'leadpoet_query_flag_rate',
            'Query flag rate percentage (0.0 to 1.0)',
            ['user_id']
        )
        
        self.query_pass_rate_gauge = Gauge(
            'leadpoet_query_pass_rate',
            'Query pass rate percentage (0.0 to 1.0)',
            ['user_id']
        )
        
        self.query_total_queries_gauge = Gauge(
            'leadpoet_query_total_queries',
            'Total queries submitted by user',
            ['user_id']
        )
        
        self.query_submitter_throttled_gauge = Gauge(
            'leadpoet_query_submitter_throttled',
            'Whether user is currently throttled (0 or 1)',
            ['user_id']
        )
        
        self.query_quality_alert_counter = Counter(
            'leadpoet_query_quality_alerts_total',
            'Total query quality alerts',
            ['user_id', 'alert_type']  # alert_type: low_quality, high_flag_rate
        )
        
        # Internal tracking
        self._active_queries = 0
        self._query_times = deque(maxlen=1000)  # For QPS calculation
        self._lock = threading.Lock()
        
        # Cache hit rate tracking - avoid accessing internal Prometheus attributes
        self._cache_hit_counts = {}  # {cache_type: hit_count}
        self._cache_miss_counts = {}  # {cache_type: miss_count}
        self._cache_lock = threading.Lock()
        
        self.requests_total = Counter(
            "requests_total",
            "Total number of processed requests",
            ["endpoint", "method", "http_status"]
        )
        self.request_latency = Histogram(
            "request_latency_seconds",
            "Request latency in seconds",
            ["endpoint", "method"]
        )
        self.lead_cost_usd = Histogram(
            "lead_cost_usd",
            "Estimated cost per lead in USD"
        )
        self.llm_hit_counter = Counter(
            "llm_hit_total",
            "Total number of times the LLM was called as a fallback"
        )
        logger.info("Prometheus metrics collector initialized with enhanced pipeline stage histograms.")
    
    def record_query_metrics(
        self, 
        query_id: str, 
        processing_time_ms: float, 
        lead_count: int, 
        parsed_query: Dict[str, Any]
    ):
        """Record metrics for a completed query."""
        try:
            # Convert to seconds
            processing_time_s = processing_time_ms / 1000.0
            
            # Record query duration
            self.query_duration.labels(stage='total').observe(processing_time_s)
            
            # Record leads returned
            self.leads_returned.observe(lead_count)
            
            # Record query success
            self.query_counter.labels(status='success', method='api').inc()
            
            # Check latency SLA violations
            if processing_time_ms > settings.LATENCY_P95_THRESHOLD:
                self.latency_sla_violations.labels(threshold='p95').inc()
            if processing_time_ms > settings.LATENCY_P99_THRESHOLD:
                self.latency_sla_violations.labels(threshold='p99').inc()
            
            # Update query volume tracking
            with self._lock:
                self._query_times.append(time.time())
                # Calculate QPS over last 60 seconds
                cutoff = time.time() - 60
                self._query_times = deque(
                    [t for t in self._query_times if t > cutoff], 
                    maxlen=1000
                )
                qps = len(self._query_times) / 60.0
                self.query_volume.set(qps)
            
            logger.debug(f"Recorded metrics for query {query_id}: {processing_time_ms:.2f}ms, {lead_count} leads")
            
        except Exception as e:
            logger.error(f"Error recording query metrics: {e}")
    
    def record_prompt_parsing_duration(self, duration_ms: float):
        """Record prompt parsing duration."""
        try:
            duration_s = duration_ms / 1000.0
            self.prompt_parsing_duration.observe(duration_s)
            self.pipeline_stage_duration.labels(stage='parsing', sub_stage='total').observe(duration_s)
        except Exception as e:
            logger.error(f"Error recording prompt parsing duration: {e}")
    
    def record_retrieval_duration(self, duration_ms: float, method: str):
        """Record retrieval duration."""
        try:
            duration_s = duration_ms / 1000.0
            self.retrieval_duration.labels(method=method).observe(duration_s)
            self.pipeline_stage_duration.labels(stage='retrieval', sub_stage=method).observe(duration_s)
        except Exception as e:
            logger.error(f"Error recording retrieval duration: {e}")
    
    def record_scoring_duration(self, duration_ms: float, component: str):
        """Record scoring duration."""
        try:
            duration_s = duration_ms / 1000.0
            self.scoring_duration.labels(component=component).observe(duration_s)
            self.pipeline_stage_duration.labels(stage='scoring', sub_stage=component).observe(duration_s)
        except Exception as e:
            logger.error(f"Error recording scoring duration: {e}")
    
    def record_db_operation_duration(self, duration_ms: float, operation: str, table: str):
        """Record database operation duration."""
        try:
            duration_s = duration_ms / 1000.0
            self.db_operation_duration.labels(operation=operation, table=table).observe(duration_s)
        except Exception as e:
            logger.error(f"Error recording database operation duration: {e}")
    
    def record_cache_operation_duration(self, duration_ms: float, operation: str, cache_type: str):
        """Record cache operation duration."""
        try:
            duration_s = duration_ms / 1000.0
            self.cache_operation_duration.labels(operation=operation, cache_type=cache_type).observe(duration_s)
        except Exception as e:
            logger.error(f"Error recording cache operation duration: {e}")
    
    def record_llm_api_duration(self, duration_ms: float, model: str, operation: str):
        """Record LLM API call duration."""
        try:
            duration_s = duration_ms / 1000.0
            self.llm_api_duration.labels(model=model, operation=operation).observe(duration_s)
        except Exception as e:
            logger.error(f"Error recording LLM API duration: {e}")
    
    def record_scoring_component_duration(self, duration_ms: float, component: str, method: str):
        """Record individual scoring component duration."""
        try:
            duration_s = duration_ms / 1000.0
            self.scoring_component_duration.labels(component=component, method=method).observe(duration_s)
        except Exception as e:
            logger.error(f"Error recording scoring component duration: {e}")
    
    def record_retrieval_method_duration(self, duration_ms: float, method: str, phase: str):
        """Record retrieval method duration."""
        try:
            duration_s = duration_ms / 1000.0
            self.retrieval_method_duration.labels(method=method, phase=phase).observe(duration_s)
        except Exception as e:
            logger.error(f"Error recording retrieval method duration: {e}")
    
    def record_pipeline_stage_duration(self, duration_ms: float, stage: str, sub_stage: str):
        """Record general pipeline stage duration."""
        try:
            duration_s = duration_ms / 1000.0
            self.pipeline_stage_duration.labels(stage=stage, sub_stage=sub_stage).observe(duration_s)
        except Exception as e:
            logger.error(f"Error recording pipeline stage duration: {e}")
    
    def record_leads_processed(self, count: int, stage: str):
        """Record leads processed at a stage."""
        try:
            self.leads_processed.labels(stage=stage).inc(count)
        except Exception as e:
            logger.error(f"Error recording leads processed: {e}")
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        try:
            self.cache_hits.labels(cache_type=cache_type).inc()
            # Track hit count for hit rate calculation
            with self._cache_lock:
                self._cache_hit_counts[cache_type] = self._cache_hit_counts.get(cache_type, 0) + 1
        except Exception as e:
            logger.error(f"Error recording cache hit: {e}")
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        try:
            self.cache_misses.labels(cache_type=cache_type).inc()
            # Track miss count for hit rate calculation
            with self._cache_lock:
                self._cache_miss_counts[cache_type] = self._cache_miss_counts.get(cache_type, 0) + 1
        except Exception as e:
            logger.error(f"Error recording cache miss: {e}")
    
    def record_error(self, stage: str, error_type: str):
        """Record error occurrence."""
        try:
            self.error_counter.labels(stage=stage, error_type=error_type).inc()
        except Exception as e:
            logger.error(f"Error recording error: {e}")
    
    def record_llm_call(self, model: str, status: str, input_tokens: int = 0, output_tokens: int = 0):
        """Record LLM API call."""
        try:
            self.llm_calls.labels(model=model, status=status).inc()
            if input_tokens > 0:
                self.llm_tokens.labels(model=model, type='input').inc(input_tokens)
            if output_tokens > 0:
                self.llm_tokens.labels(model=model, type='output').inc(output_tokens)
        except Exception as e:
            logger.error(f"Error recording LLM call: {e}")
    
    def start_query(self):
        """Mark start of a query."""
        with self._lock:
            self._active_queries += 1
            self.active_queries.set(self._active_queries)
    
    def end_query(self):
        """Mark end of a query."""
        with self._lock:
            self._active_queries = max(0, self._active_queries - 1)
            self.active_queries.set(self._active_queries)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics as string."""
        try:
            return generate_latest()
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return ""
    
    def get_metrics_content_type(self) -> str:
        """Get content type for metrics."""
        return CONTENT_TYPE_LATEST
    
    def get_cache_hit_rate(self, cache_type: str) -> float:
        """Calculate cache hit rate for a cache type using tracked counts."""
        try:
            with self._cache_lock:
                hits = self._cache_hit_counts.get(cache_type, 0)
                misses = self._cache_miss_counts.get(cache_type, 0)
                total = hits + misses
                return hits / total if total > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating cache hit rate: {e}")
            return 0.0
    
    def get_cache_stats(self, cache_type: str) -> Dict[str, int]:
        """Get cache statistics for a cache type."""
        try:
            with self._cache_lock:
                hits = self._cache_hit_counts.get(cache_type, 0)
                misses = self._cache_miss_counts.get(cache_type, 0)
                return {
                    'hits': hits,
                    'misses': misses,
                    'total': hits + misses,
                    'hit_rate': hits / (hits + misses) if (hits + misses) > 0 else 0.0
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'hits': 0, 'misses': 0, 'total': 0, 'hit_rate': 0.0}
    
    def reset_cache_stats(self, cache_type: str = None):
        """Reset cache statistics for a specific type or all types."""
        try:
            with self._cache_lock:
                if cache_type is None:
                    # Reset all cache types
                    self._cache_hit_counts.clear()
                    self._cache_miss_counts.clear()
                else:
                    # Reset specific cache type
                    self._cache_hit_counts.pop(cache_type, None)
                    self._cache_miss_counts.pop(cache_type, None)
            logger.info(f"Reset cache stats for {cache_type or 'all types'}")
        except Exception as e:
            logger.error(f"Error resetting cache stats: {e}")
    
    def get_current_qps(self) -> float:
        """Get current queries per second."""
        with self._lock:
            return len(self._query_times) / 60.0 if self._query_times else 0.0
    
    def get_active_query_count(self) -> int:
        """Get number of currently active queries."""
        with self._lock:
            return self._active_queries
    
    def record_request(self, endpoint: str, method: str, status_code: int):
        self.requests_total.labels(endpoint=endpoint, method=method, http_status=status_code).inc()
    
    def record_latency(self, endpoint: str, method: str, latency: float):
        self.request_latency.labels(endpoint=endpoint, method=method).observe(latency)
    
    def record_lead_cost(self, cost: float):
        if cost > 0:
            self.lead_cost_usd.observe(cost)
    
    def record_llm_hit(self):
        self.llm_hit_counter.inc()
    
    def set_query_quality_score(self, user_id: str, score: float):
        """Set query quality score gauge."""
        try:
            self.query_quality_score_gauge.labels(user_id=user_id).set(score)
        except Exception as e:
            logger.error(f"Error setting query quality score: {e}")
    
    def set_query_flag_rate(self, user_id: str, flag_rate: float):
        """Set query flag rate gauge."""
        try:
            self.query_flag_rate_gauge.labels(user_id=user_id).set(flag_rate)
        except Exception as e:
            logger.error(f"Error setting query flag rate: {e}")
    
    def set_query_pass_rate(self, user_id: str, pass_rate: float):
        """Set query pass rate gauge."""
        try:
            self.query_pass_rate_gauge.labels(user_id=user_id).set(pass_rate)
        except Exception as e:
            logger.error(f"Error setting query pass rate: {e}")
    
    def set_query_total_queries(self, user_id: str, total_queries: int):
        """Set query total queries gauge."""
        try:
            self.query_total_queries_gauge.labels(user_id=user_id).set(total_queries)
        except Exception as e:
            logger.error(f"Error setting query total queries: {e}")
    
    def set_query_submitter_throttled(self, user_id: str, is_throttled: bool):
        """Set query submitter throttled status gauge."""
        try:
            self.query_submitter_throttled_gauge.labels(user_id=user_id).set(1 if is_throttled else 0)
        except Exception as e:
            logger.error(f"Error setting query submitter throttled status: {e}")
    
    def record_query_quality_alert(self, alert_type: str, user_id: str):
        """Record a query quality alert."""
        try:
            self.query_quality_alert_counter.labels(user_id=user_id, alert_type=alert_type).inc()
        except Exception as e:
            logger.error(f"Error recording query quality alert: {e}")
    
    def track_latency(self, endpoint: str, method: str, latency: float):
        """Track latency for an endpoint."""
        self.record_latency(endpoint, method, latency) 