# Monitoring & Alerting for Leadpoet Intent Model v1.1

This document outlines the comprehensive monitoring and alerting strategy for the Leadpoet Intent Model API, including Prometheus metrics, Datadog APM tracing, Grafana dashboards, and query quality reputation alerts.

## Overview

The monitoring system provides:
- **Real-time Performance Monitoring**: Latency, throughput, and error rates
- **Cost Tracking**: Per-lead costs and LLM usage monitoring
- **Pipeline Performance**: Detailed stage-by-stage performance analysis
- **Query Quality Reputation**: Quality tracking and alerting for user performance
- **Distributed Tracing**: End-to-end request tracing with Datadog APM

---

## 1. Prometheus Metrics (Epic 7 Task 1)

### Core API Metrics
- `requests_total`: Total requests by endpoint, method, and status
- `request_latency_seconds`: Response time histograms
- `lead_cost_usd`: Cost per lead distribution
- `llm_hit_total`: LLM fallback usage counter

### Enhanced Pipeline Stage Metrics
- `leadpoet_pipeline_stage_duration_seconds`: Detailed pipeline stage timing
- `leadpoet_db_operation_duration_seconds`: Database operation performance
- `leadpoet_cache_operation_duration_seconds`: Cache operation timing
- `leadpoet_llm_api_duration_seconds`: LLM API call performance
- `leadpoet_scoring_component_duration_seconds`: Individual scoring component timing
- `leadpoet_retrieval_method_duration_seconds`: Retrieval method performance

### Business Metrics
- `leadpoet_queries_total`: Query volume by status and method
- `leadpoet_leads_returned`: Lead count distribution
- `leadpoet_cache_hits_total` / `leadpoet_cache_misses_total`: Cache performance
- `leadpoet_errors_total`: Error tracking by stage and type

### Query Quality Reputation Metrics
- `leadpoet_query_quality_score{user_id="..."}`: Query quality score (0.0 to 1.0)
- `leadpoet_query_flag_rate{user_id="..."}`: Query flag rate percentage (0.0 to 1.0)
- `leadpoet_query_pass_rate{user_id="..."}`: Query pass rate percentage (0.0 to 1.0)
- `leadpoet_query_total_queries{user_id="..."}`: Total queries for a user
- `leadpoet_query_submitter_throttled{user_id="..."}`: Whether a user is throttled (0 or 1)
- `leadpoet_query_quality_alerts_total{user_id="...", alert_type="..."}`: Query quality alert counter by type

---

## 2. Datadog APM Integration (Epic 7 Task 2)

### Configuration
```bash
# Environment variables for Datadog APM
export DD_ENV=production
export DD_SERVICE=leadpoet-intent-model
export DD_VERSION=1.1.0
export DD_AGENT_HOST=localhost
export DD_TRACE_AGENT_PORT=8126
```

### Tracing Features
- **Auto-instrumentation**: FastAPI, SQLAlchemy, Redis, HTTP clients
- **Custom Spans**: Pipeline stages, database operations, cache operations
- **LLM Tracing**: OpenAI API calls with token usage tracking
- **Error Tracking**: Automatic error capture and correlation

### Trace Context Managers
```python
# Pipeline stage tracing
with trace_pipeline_stage("parsing", "prompt_analysis") as span:
    # Your parsing code here
    span.set_tag("query_length", len(query))

# Database operation tracing
with trace_database_operation("select", "leads") as span:
    # Your database query here
    span.set_tag("result_count", len(results))

# LLM call tracing
with trace_llm_call("gpt-4o", "completion", input_tokens=100) as span:
    # Your LLM call here
    span.set_tag("output_tokens", response.usage.completion_tokens)
```

---

## 3. Grafana Dashboards (Epic 7 Task 3)

### Dashboard 1: API Performance
**URL**: `/d/leadpoet-api-performance/leadpoet-api-performance`

**Panels**:
- Request Rate (QPS) with 250 QPS threshold
- Response Time Percentiles (P50, P95, P99) with 400ms/550ms thresholds
- Error Rate (4xx, 5xx) with 5% threshold
- Active Queries and Query Volume

### Dashboard 2: Cost Analysis
**URL**: `/d/leadpoet-cost-analysis/leadpoet-cost-analysis`

**Panels**:
- Cost per Lead (P50, P95) with $0.002/$0.004 thresholds
- LLM Usage Rate with 75/100 calls/sec thresholds
- Total Cost Over Time with $10/$20 hourly thresholds
- Cost Distribution by Query Type

### Dashboard 3: Pipeline Performance
**URL**: `/d/leadpoet-pipeline-performance/leadpoet-pipeline-performance`

**Panels**:
- Pipeline Stage Duration (P95) with 100ms/200ms thresholds
- Database Operation Duration (P95) with 50ms/100ms thresholds
- Cache Hit Rate with 70%/90% thresholds
- Component Performance Breakdown

---

## 4. Query Quality Reputation Alerts (Epic 7 Task 4)

### Alert Rules

#### Low Quality Score Alert
```yaml
- alert: UserLowQualityScore
  expr: leadpoet_query_quality_score < 0.5
  for: 1m
  labels:
    severity: warning
    component: query_quality_reputation
  annotations:
    summary: "User query quality score dropped below threshold"
    description: "User {{ $labels.user_id }} quality score is {{ $value }} (threshold: 0.5)"
```

#### High Flag Rate Alert
```yaml
- alert: UserHighFlagRate
  expr: leadpoet_query_flag_rate > 0.3
  for: 1m
  labels:
    severity: warning
    component: query_quality_reputation
  annotations:
    summary: "User flag rate exceeded threshold"
    description: "User {{ $labels.user_id }} flag rate is {{ $value }} (threshold: 30%)"
```

#### User Throttled Alert
```yaml
- alert: UserThrottled
  expr: leadpoet_query_submitter_throttled > 0
  for: 1m
  labels:
    severity: critical
    component: query_quality_reputation
  annotations:
    summary: "User has been throttled"
    description: "User {{ $labels.user_id }} has been throttled due to poor query quality"
```

### Reputation Calculation
- **EMA Formula**: `new_quality_score = α * current_quality_score + (1-α) * previous_quality_score`
- **Alpha**: 0.1 (10% weight to current performance)
- **Quality Threshold**: 0.5 (50% minimum quality score)
- **Flag Rate Threshold**: 30% (triggers 24-hour throttle)
- **Alert Thresholds**: Quality < 0.3, Flag Rate > 50%

### API Endpoints
- `GET /query-quality/reputation`: Get reputation for specific user or all users
- `GET /query-quality/alerts`: Get active query quality alerts
- `POST /query-quality/clear-expired-throttles`: Clear expired throttles

---

## 5. Alert Rules Summary

### Performance Alerts
```yaml
groups:
  - name: leadpoet-performance-alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m])) > 0.4
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 0.4s)"

      - alert: VeryHighLatency
        expr: histogram_quantile(0.99, rate(request_latency_seconds_bucket[5m])) > 0.55
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Very high API latency detected"
          description: "P99 latency is {{ $value }}s (threshold: 0.55s)"

      - alert: HighErrorRate
        expr: rate(requests_total{http_status=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} (threshold: 5%)"
```

### Cost Alerts
```yaml
      - alert: HighCostPerLead
        expr: histogram_quantile(0.95, rate(lead_cost_usd_bucket[5m])) > 0.004
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High cost per lead detected"
          description: "P95 cost per lead is ${{ $value }} (threshold: $0.004)"

      - alert: HighLLMUsage
        expr: rate(llm_hit_total[5m]) > 100
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High LLM usage detected"
          description: "LLM usage rate is {{ $value }} calls/sec (threshold: 100)"
```

### Pipeline Alerts
```yaml
      - alert: PipelineStageSlow
        expr: histogram_quantile(0.95, rate(leadpoet_pipeline_stage_duration_seconds_bucket[5m])) > 0.2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Pipeline stage performance degraded"
          description: "Stage {{ $labels.stage }}.{{ $labels.sub_stage }} P95 duration is {{ $value }}s"

      - alert: DatabaseSlow
        expr: histogram_quantile(0.95, rate(leadpoet_db_operation_duration_seconds_bucket[5m])) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Database operation performance degraded"
          description: "{{ $labels.operation }} on {{ $labels.table }} P95 duration is {{ $value }}s"
```

---

## 6. Setup Instructions

### 1. Install Dependencies
```bash
pip install ddtrace==1.20.4 prometheus-client==0.19.0
```

### 2. Configure Environment Variables
```bash
# Datadog APM
export DD_ENV=production
export DD_SERVICE=leadpoet-intent-model
export DD_VERSION=1.1.0
export DD_AGENT_HOST=localhost
export DD_TRACE_AGENT_PORT=8126

# Prometheus
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
```

### 3. Import Grafana Dashboards
- Copy JSON configurations from `docs/grafana-dashboards.md`
- Import into Grafana via Dashboard → Import
- Configure Prometheus as data source

### 4. Configure Alert Manager
- Add alert rules to Prometheus configuration
- Configure notification channels (Slack, email, PagerDuty)
- Test alert delivery

### 5. Monitor Key Metrics
- **Latency**: P95 < 400ms, P99 < 550ms
- **Cost**: Average ≤ $0.002, P99 ≤ $0.004
- **Errors**: < 5% error rate
- **Query Quality**: > 0.5 threshold
- **Cache Hit Rate**: > 70%

---

## 7. Troubleshooting

### Common Issues
1. **High Latency**: Check database queries, LLM calls, cache performance
2. **High Costs**: Monitor LLM usage, optimize scoring fallbacks
3. **Low Query Quality**: Review user query quality, adjust thresholds
4. **Cache Misses**: Verify Redis connectivity, check cache keys

### Debug Endpoints
- `/metrics`: Raw Prometheus metrics
- `/healthz`: Service health status
- `/query-quality/reputation`: Query quality reputation data
- `/query-quality/alerts`: Active query quality alerts

### Log Analysis
- Search for "ALERT:" in logs for quality alerts
- Monitor Datadog traces for performance bottlenecks
- Check Grafana dashboards for trend analysis

This monitoring and alerting system provides comprehensive visibility into the Leadpoet Intent Model API performance, costs, and query quality, enabling proactive issue detection and resolution.