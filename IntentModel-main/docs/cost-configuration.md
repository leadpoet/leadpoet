# Cost Configuration Guide

This document explains how to configure cost rates and budgets for the Leadpoet Intent Model v1.1 without requiring code changes.

## Environment Variables

The following environment variables control cost rates and budget thresholds:

### Cost Rates

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `COST_LLM_GPT4O_INPUT_RATE` | 0.005 | LLM GPT-4o input cost per 1K tokens | 0.005 (=$5.00 per 1M tokens) |
| `COST_LLM_GPT4O_OUTPUT_RATE` | 0.015 | LLM GPT-4o output cost per 1K tokens | 0.015 (=$15.00 per 1M tokens) |
| `COST_API_REQUEST_RATE` | 0.0001 | API request cost per request | 0.0001 (=$0.0001 per request) |
| `COST_INFRASTRUCTURE_PER_LEAD_RATE` | 0.0005 | Infrastructure cost per lead processed | 0.0005 (=$0.0005 per lead) |

### Budget Thresholds

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `COST_DAILY_BUDGET_USD` | 100.0 | Daily budget limit in USD | 100.0 (=$100.00 per day) |
| `COST_AVG_THRESHOLD` | 0.002 | Average cost threshold per lead | 0.002 (=$0.002 per lead) |
| `COST_P99_THRESHOLD` | 0.004 | P99 cost threshold per lead | 0.004 (=$0.004 per lead) |

## Environment-Specific Configuration

### Development Environment
```bash
# Lower costs for development
COST_LLM_GPT4O_INPUT_RATE=0.005
COST_LLM_GPT4O_OUTPUT_RATE=0.015
COST_API_REQUEST_RATE=0.0001
COST_INFRASTRUCTURE_PER_LEAD_RATE=0.0005
COST_DAILY_BUDGET_USD=50.0
```

### Staging Environment
```bash
# Moderate costs for staging
COST_LLM_GPT4O_INPUT_RATE=0.005
COST_LLM_GPT4O_OUTPUT_RATE=0.015
COST_API_REQUEST_RATE=0.0001
COST_INFRASTRUCTURE_PER_LEAD_RATE=0.0005
COST_DAILY_BUDGET_USD=200.0
```

### Production Environment
```bash
# Production costs with higher budget
COST_LLM_GPT4O_INPUT_RATE=0.005
COST_LLM_GPT4O_OUTPUT_RATE=0.015
COST_API_REQUEST_RATE=0.0001
COST_INFRASTRUCTURE_PER_LEAD_RATE=0.0005
COST_DAILY_BUDGET_USD=1000.0
```

## Cost Calculation

The system calculates costs as follows:

1. **LLM Cost**: `(input_tokens * COST_LLM_GPT4O_INPUT_RATE / 1000) + (output_tokens * COST_LLM_GPT4O_OUTPUT_RATE / 1000)`
2. **API Cost**: `COST_API_REQUEST_RATE` per request
3. **Infrastructure Cost**: `lead_count * COST_INFRASTRUCTURE_PER_LEAD_RATE`
4. **Total Cost**: `LLM Cost + API Cost + Infrastructure Cost`

## Budget Enforcement

The system enforces budgets through:

1. **Daily Budget**: Tracks total daily spending and alerts when approaching/exceeding `COST_DAILY_BUDGET_USD`
2. **Average Cost Threshold**: Alerts when average cost per lead exceeds `COST_AVG_THRESHOLD`
3. **P99 Cost Threshold**: Alerts when P99 cost per lead exceeds `COST_P99_THRESHOLD`

## Monitoring

Cost telemetry provides:

- Real-time cost tracking via Prometheus metrics
- Historical cost data stored in Redis
- Budget violation alerts in application logs
- Cost statistics API endpoints

## Updating Costs

To update cost rates or budgets:

1. Set the appropriate environment variables
2. Restart the application
3. Monitor the new rates in application logs
4. Verify budget enforcement is working correctly

No code changes are required - all cost configuration is externalized. 