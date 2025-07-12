# Leadpoet Intent Model API Documentation

## Overview

The Leadpoet Intent Model API is a B2B lead scoring and ranking service that returns the **K** leads most likely to purchase any B2B product specified in a free-text prompt. The API achieves &lt; **400 ms** P95 latency at ≤ **$0.002** COGS per lead under 250 QPS load.
## API Base URL

- **Production**: `https://api.leadpoet.com`
- **Staging**: `https://api-staging.leadpoet.com`
- **Development**: `http://localhost:8000`

## Authentication

The API uses API key authentication. Include your API key in the request headers:

```http
Authorization: Bearer YOUR_API_KEY
```

## API Endpoints

### 1. Query Leads

**POST** `/query`

Submit a natural language query to find relevant B2B leads.

#### Request Body

```json
{
  "query": "Find companies looking for CRM software",
  "limit": 25,
  "filters": {
    "industries": ["technology", "saas"],
    "company_size": ["10-50", "51-200"],
    "location": "United States",
    "exclude": ["competitor1.com", "competitor2.com"]
  }
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Natural language description of target companies |
| `limit` | integer | No | Number of leads to return (default: 25, max: 2000) |
| `filters` | object | No | Additional filtering criteria |

#### Filter Options

```json
{
  "industries": ["technology", "healthcare", "finance"],
  "company_size": ["1-10", "11-50", "51-200", "201-1000", "1000+"],
  "location": "United States",
  "exclude": ["domain1.com", "domain2.com"],
  "technologies": ["salesforce", "hubspot", "zendesk"],
  "job_titles": ["CTO", "VP Engineering", "Head of Sales"]
}
```

#### Response

```json
{
  "leads": [
    {
      "lead_id": "lead_12345",
      "company_name": "TechCorp Inc",
      "domain": "techcorp.com",
      "industry": "Technology",
      "company_size": "51-200",
      "location": "San Francisco, CA",
      "score": 0.92,
      "confidence": 0.85,
      "contact_info": {
        "name": "John Smith",
        "title": "CTO",
        "email": "john.smith@techcorp.com",
        "phone": "+1-555-0123"
      },
      "intent_signals": [
        "Recently posted job for CRM administrator",
        "Company website mentions 'customer relationship management'",
        "LinkedIn shows recent CRM-related activity"
      ]
    }
  ],
  "metadata": {
    "total_leads": 150,
    "query_processed": "Find companies looking for CRM software",
    "processing_time_ms": 245,
    "cost_usd": 0.0018,
    "llm_calls": 3,
    "cache_hit_rate": 0.75
  }
}
```

### 2. Health Check

**GET** `/healthz`

Check API health and status.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-12-19T10:30:00Z",
  "version": "1.1.0",
  "uptime_seconds": 86400,
  "database": "connected",
  "redis": "connected",
  "llm_service": "available"
}
```

### 3. Metrics

**GET** `/metrics`

Prometheus-compatible metrics endpoint.

#### Response

```
# HELP leadpoet_api_requests_total Total number of API requests
# TYPE leadpoet_api_requests_total counter
leadpoet_api_requests_total{endpoint="/query",method="POST"} 1250

# HELP leadpoet_api_request_duration_seconds Request duration in seconds
# TYPE leadpoet_api_request_duration_seconds histogram
leadpoet_api_request_duration_seconds_bucket{endpoint="/query",le="0.1"} 100
leadpoet_api_request_duration_seconds_bucket{endpoint="/query",le="0.4"} 950
leadpoet_api_request_duration_seconds_bucket{endpoint="/query",le="+Inf"} 1250

# HELP leadpoet_cost_per_lead_usd Cost per lead in USD
# TYPE leadpoet_cost_per_lead_usd gauge
leadpoet_cost_per_lead_usd 0.0018
```

### 4. Query Quality Reputation

**GET** `/query-quality/reputation`

Get query quality reputation for the current API key.

#### Response

```json
{
  "api_key": "sk_...",
  "reputation_score": 0.85,
  "total_queries": 1250,
  "successful_queries": 1180,
  "failed_queries": 70,
  "average_response_time_ms": 245,
  "cost_efficiency": 0.92,
  "throttle_status": "active",
  "throttle_reason": "High cost per lead",
  "recommendations": [
    "Consider using more specific queries",
    "Add industry filters to reduce noise",
    "Limit results to improve cost efficiency"
  ]
}
```

### 5. Query Quality Alerts

**GET** `/query-quality/alerts`

Get active alerts for query quality issues.

#### Response

```json
{
  "alerts": [
    {
      "alert_id": "alert_123",
      "type": "high_cost",
      "severity": "warning",
      "message": "Cost per lead exceeded threshold",
      "threshold": 0.002,
      "current_value": 0.0025,
      "created_at": "2024-12-19T10:00:00Z",
      "status": "active"
    }
  ],
  "summary": {
    "total_alerts": 1,
    "critical_alerts": 0,
    "warning_alerts": 1,
    "info_alerts": 0
  }
}
```

## Error Responses

### Standard Error Format

```json
{
  "error": {
    "code": "INVALID_QUERY",
    "message": "Query must be at least 10 characters long",
    "details": {
      "field": "query",
      "min_length": 10,
      "provided_length": 5
    }
  },
  "timestamp": "2024-12-19T10:30:00Z",
  "request_id": "req_abc123"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_QUERY` | 400 | Query validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `COST_LIMIT_EXCEEDED` | 429 | Cost per lead limit exceeded |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | API key lacks required permissions |
| `QUERY_THROTTLED` | 429 | Query quality reputation too low |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |
| `INTERNAL_ERROR` | 500 | Internal server error |

## SDKs and Libraries

### Python

```bash
pip install leadpoet-client
```

```python
from leadpoet import LeadpoetClient

client = LeadpoetClient(api_key="your_api_key")

# Query leads
response = client.query_leads(
    query="Find companies looking for CRM software",
    limit=25,
    filters={
        "industries": ["technology"],
        "company_size": ["51-200"]
    }
)

print(f"Found {len(response.leads)} leads")
for lead in response.leads:
    print(f"{lead.company_name} - {lead.score}")
```

### JavaScript/Node.js

```bash
npm install @leadpoet/client
```

```javascript
const { LeadpoetClient } = require('@leadpoet/client');

const client = new LeadpoetClient('your_api_key');

// Query leads
const response = await client.queryLeads({
  query: 'Find companies looking for CRM software',
  limit: 25,
  filters: {
    industries: ['technology'],
    companySize: ['51-200']
  }
});

console.log(`Found ${response.leads.length} leads`);
response.leads.forEach(lead => {
  console.log(`${lead.companyName} - ${lead.score}`);
});
```

### cURL Examples

#### Basic Query

```bash
curl -X POST "https://api.leadpoet.com/query" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find companies looking for CRM software",
    "limit": 25
  }'
```

#### Query with Filters

```bash
curl -X POST "https://api.leadpoet.com/query" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find technology companies in San Francisco looking for marketing automation",
    "limit": 50,
    "filters": {
      "industries": ["technology"],
      "location": "San Francisco",
      "company_size": ["51-200", "201-1000"],
      "technologies": ["hubspot", "marketo"]
    }
  }'
```

## Performance Guidelines

### Best Practices

1. **Specific Queries**: Use specific, detailed queries for better results
2. **Appropriate Limits**: Start with smaller limits (25-50) and increase as needed
3. **Use Filters**: Leverage filters to reduce noise and improve relevance
4. **Monitor Costs**: Track cost per lead to stay within budget
5. **Cache Results**: Implement client-side caching for repeated queries

### Query Examples

#### Good Queries
- "Find SaaS companies with 50-200 employees looking for customer support software"
- "Identify healthcare companies in Boston seeking electronic health records systems"
- "Find manufacturing companies using legacy ERP systems that need modernization"

#### Poor Queries
- "Find companies" (too vague)
- "CRM" (too short)
- "All businesses" (too broad)

## Rate Limiting and Throttling

### Rate Limits
- **Standard**: 100 requests/minute
- **Burst**: 1000 requests/minute
- **Daily**: 10,000 requests/day

### Cost Limits
- **Per Lead**: $0.002 maximum (automatically enforced)
- **Per Query**: $4.00 maximum (2000 leads)
- **Daily**: $100 maximum

### Throttling
The API may throttle requests based on:
- Query quality reputation score
- Cost per lead efficiency
- Service load and capacity

### Rate Limit Headers
When approaching rate limits, the API includes the following headers:
- `X-RateLimit-Limit`: Maximum requests allowed in the window
- `X-RateLimit-Remaining`: Number of requests remaining in the current window
- `X-RateLimit-Reset`: Time when the rate limit window resets (Unix timestamp)

### Handling Rate Limits
When you receive a `429 Too Many Requests` response:
1. **Wait for the reset time** indicated in the `X-RateLimit-Reset` header
2. **Implement exponential backoff** for retry attempts
3. **Consider reducing request frequency** or implementing caching
4. **Monitor your usage** to stay within limits

## Support and Resources

### Documentation
- **API Reference**: [https://docs.leadpoet.com/api](https://docs.leadpoet.com/api)
- **SDK Documentation**: [https://docs.leadpoet.com/sdks](https://docs.leadpoet.com/sdks)
- **Best Practices**: [https://docs.leadpoet.com/best-practices](https://docs.leadpoet.com/best-practices)

### Support
- **Email**: api-support@leadpoet.com
- **Slack**: [leadpoet-api-support](https://leadpoet.slack.com/archives/api-support)
- **Status Page**: [https://status.leadpoet.com](https://status.leadpoet.com)

### Community
- **GitHub**: [https://github.com/leadpoet/api-examples](https://github.com/leadpoet/api-examples)
- **Discord**: [https://discord.gg/leadpoet](https://discord.gg/leadpoet)

---

*Last Updated: December 2024*
*API Version: 1.1.0* 