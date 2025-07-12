# Leadpoet Intent Model API - Training Guide

## Overview

This training guide provides comprehensive information for engineering and miner-relations teams to understand, use, and support the Leadpoet Intent Model API.

## Target Audience

- **Engineering Team**: Developers, DevOps, SREs
- **Miner Relations Team**: Customer success, sales engineers, support
- **Product Team**: Product managers, technical writers

## Training Objectives

By the end of this training, participants will be able to:

1. **Understand** the API architecture and components
2. **Use** the API effectively for lead discovery
3. **Troubleshoot** common issues and errors
4. **Optimize** queries for better results and cost efficiency
5. **Support** customers with API integration and usage

## Module 1: API Architecture Overview

### System Components

The Leadpoet Intent Model API consists of:

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL with TimescaleDB
- **Cache**: Redis
- **ML Models**: LightGBM, OpenAI GPT-4o
- **Infrastructure**: Kubernetes, Helm
- **Monitoring**: Prometheus, Grafana, Datadog

### Data Flow

1. **Query Submission**: Client sends natural language query
2. **Query Parsing**: Extract intent and filters
3. **Lead Retrieval**: BM25 search + vector similarity
4. **Scoring**: Multi-factor scoring with time decay
5. **LLM Enhancement**: GPT-4o for intent scoring (if needed)
6. **Fallback**: LightGBM model if LLM fails
7. **Caching**: Store results for future queries
8. **Response**: Return ranked leads with scores

## Module 2: API Usage and Best Practices

### Authentication

```bash
# API Key Authentication
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.leadpoet.com/query
```

### Basic Query Example

```python
import requests

# Basic query
response = requests.post(
    "https://api.leadpoet.com/query",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "query": "Find companies looking for CRM software",
        "limit": 25
    }
)

leads = response.json()["leads"]
for lead in leads:
    print(f"{lead['company_name']} - Score: {lead['score']}")
```

### Query Best Practices

#### ✅ Good Queries
- **Specific and Detailed**: "Find SaaS companies with 50-200 employees looking for customer support software"
- **Industry Focused**: "Identify healthcare companies in Boston seeking electronic health records systems"
- **Technology Specific**: "Find manufacturing companies using legacy ERP systems that need modernization"

#### ❌ Poor Queries
- **Too Vague**: "Find companies" (no specific criteria)
- **Too Short**: "CRM" (insufficient context)
- **Too Broad**: "All businesses" (no targeting)

## Module 3: Performance and Cost Optimization

### Understanding Costs

- **Cost per Lead**: Target < $0.002
- **LLM Call Ratio**: Target < 30%
- **Cache Hit Rate**: Target > 70%

### Cost Optimization Strategies

#### 1. Query Specificity
```python
# High cost - too broad
{"query": "Find companies", "limit": 1000}

# Lower cost - specific
{"query": "Find SaaS companies with 50-200 employees looking for customer support software", "limit": 50}
```

#### 2. Use Filters Effectively
```python
# Reduce search space with filters
"filters": {
    "industries": ["technology"],
    "company_size": ["51-200"],
    "location": "United States"
}
```

## Module 4: Error Handling and Troubleshooting

### Common Error Codes

| Code | HTTP Status | Description | Solution |
|------|-------------|-------------|----------|
| `INVALID_QUERY` | 400 | Query validation failed | Check query length and format |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded | Implement backoff strategy |
| `COST_LIMIT_EXCEEDED` | 429 | Cost per lead limit exceeded | Optimize query or reduce limit |
| `UNAUTHORIZED` | 401 | Invalid API key | Check API key validity |

### Error Handling Example

```python
import requests
import time

def query_leads_with_retry(query_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.leadpoet.com/query",
                headers={"Authorization": "Bearer YOUR_API_KEY"},
                json=query_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = 2 ** attempt
                print(f"Rate limited, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                error_data = response.json()
                print(f"Error: {error_data['error']['message']}")
                return None
                
        except Exception as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    
    return None
```

## Module 5: Customer Support Scenarios

### Common Customer Issues

#### 1. "My queries are returning poor results"

**Investigation Steps:**
1. Check query specificity
2. Review filter usage
3. Analyze query quality reputation
4. Suggest query improvements

**Example Response:**
```
Your query "Find companies" is quite broad. Try:
- "Find SaaS companies with 50-200 employees looking for customer support software"
- Add industry filters: "industries": ["technology"]
- Specify company size: "company_size": ["51-200"]
```

#### 2. "The API is too slow"

**Investigation Steps:**
1. Check response times in metrics
2. Review query complexity
3. Verify cache hit rates
4. Check for rate limiting

**Example Response:**
```
Let's optimize your query for better performance:
- Reduce limit from 1000 to 100
- Add specific filters to narrow search
- Implement client-side caching
- Check if you're hitting rate limits
```

## Module 6: Integration Examples

### Python SDK Example

```python
from leadpoet import LeadpoetClient

# Initialize client
client = LeadpoetClient(api_key="your_api_key")

# Basic query
response = client.query_leads(
    query="Find companies looking for CRM software",
    limit=25
)

# Advanced query with filters
response = client.query_leads(
    query="Find technology companies looking for marketing automation",
    limit=50,
    filters={
        "industries": ["technology"],
        "company_size": ["51-200"],
        "location": "United States"
    }
)

# Process results
for lead in response.leads:
    print(f"{lead.company_name} - {lead.score}")
    print(f"Contact: {lead.contact_info.name} ({lead.contact_info.title})")
    print(f"Email: {lead.contact_info.email}")
    print("---")
```

### JavaScript/Node.js Example

```javascript
const { LeadpoetClient } = require('@leadpoet/client');

// Initialize client
const client = new LeadpoetClient('your_api_key');

// Query leads
async function findLeads() {
    try {
        const response = await client.queryLeads({
            query: 'Find companies looking for CRM software',
            limit: 25,
            filters: {
                industries: ['technology'],
                companySize: ['51-200']
            }
        });

        response.leads.forEach(lead => {
            console.log(`${lead.companyName} - ${lead.score}`);
            console.log(`Contact: ${lead.contactInfo.name} (${lead.contactInfo.title})`);
            console.log(`Email: ${lead.contactInfo.email}`);
            console.log('---');
        });
    } catch (error) {
        console.error('Error:', error.message);
    }
}

findLeads();
```

## Module 7: Advanced Features

### Query Quality Reputation

```python
# Monitor query quality
response = requests.get(
    "https://api.leadpoet.com/query-quality/reputation",
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

reputation = response.json()
print(f"Reputation Score: {reputation['reputation_score']}")
print(f"Total Queries: {reputation['total_queries']}")
print(f"Success Rate: {reputation['successful_queries'] / reputation['total_queries']:.2%}")
print(f"Cost Efficiency: {reputation['cost_efficiency']:.2%}")

# Follow recommendations
for rec in reputation['recommendations']:
    print(f"Recommendation: {rec}")
```

### Bulk Operations

```python
# Process multiple queries efficiently
queries = [
    "Find companies looking for CRM software",
    "Find companies looking for marketing automation",
    "Find companies looking for project management tools"
]

results = []
for query in queries:
    response = requests.post(
        "https://api.leadpoet.com/query",
        headers={"Authorization": "Bearer YOUR_API_KEY"},
        json={"query": query, "limit": 25}
    )
    results.append(response.json())

# Combine and deduplicate results
all_leads = []
seen_companies = set()

for result in results:
    for lead in result['leads']:
        if lead['domain'] not in seen_companies:
            all_leads.append(lead)
            seen_companies.add(lead['domain'])

# Sort by score
all_leads.sort(key=lambda x: x['score'], reverse=True)
```

## Assessment and Certification

### Knowledge Check

1. **What is the target cost per lead?**
   - A) $0.001
   - B) $0.002
   - C) $0.005
   - D) $0.010

2. **What is the maximum response time for P95?**
   - A) 200ms
   - B) 400ms
   - C) 600ms
   - D) 800ms

3. **Which filter is most effective for reducing costs?**
   - A) Location
   - B) Industry
   - C) Company size
   - D) All of the above

### Practical Exercise

**Scenario**: A customer wants to find 100 technology companies in the United States looking for customer support software. They have a budget of $0.20 for this search.

**Tasks**:
1. Design an optimal query
2. Calculate estimated cost
3. Suggest optimization strategies
4. Handle potential errors

**Solution**:
```python
# Optimal query
query_data = {
    "query": "Find technology companies in United States looking for customer support software",
    "limit": 100,
    "filters": {
        "industries": ["technology"],
        "location": "United States",
        "company_size": ["51-200", "201-1000"]
    }
}

# Cost estimation
estimated_cost = 100 * 0.002  # $0.20
# This fits within budget

# Optimization
# - Use specific filters to reduce LLM calls
# - Target appropriate company sizes
# - Consider caching for repeated searches
```

## Resources and Support

### Documentation
- **API Reference**: [https://docs.leadpoet.com/api](https://docs.leadpoet.com/api)
- **SDK Documentation**: [https://docs.leadpoet.com/sdks](https://docs.leadpoet.com/sdks)
- **Best Practices**: [https://docs.leadpoet.com/best-practices](https://docs.leadpoet.com/best-practices)

### Support Channels
- **Email**: api-support@leadpoet.com
- **Slack**: [leadpoet-api-support](https://leadpoet.slack.com/archives/api-support)
- **Status Page**: [https://status.leadpoet.com](https://status.leadpoet.com)

### Training Materials
- **Video Tutorials**: [https://leadpoet.com/training](https://leadpoet.com/training)
- **Interactive Demos**: [https://demo.leadpoet.com](https://demo.leadpoet.com)
- **Case Studies**: [https://leadpoet.com/case-studies](https://leadpoet.com/case-studies)

## Feedback Survey

After completing the training, please provide feedback:

1. **Overall satisfaction** (1-5 scale)
2. **Content clarity** (1-5 scale)
3. **Practical applicability** (1-5 scale)
4. **Areas for improvement**
5. **Additional topics needed**

**Target**: Average satisfaction ≥ 4/5

---

*Last Updated: June 2025*
*Training Version: 1.1* 