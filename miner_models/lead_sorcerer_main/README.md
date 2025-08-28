# Lead Sorcerer

A comprehensive lead generation and enrichment system designed to automate the process of identifying, scoring, and enriching business leads.

## Overview

Lead Sorcerer is composed of three main tools:

- **Domain**: Generates and scores potential leads based on predefined criteria
- **Crawl**: Extracts detailed information about companies and contacts
- **Enrich**: Fills in missing data and verifies contact information

The orchestrator coordinates these tools, ensuring data flows seamlessly from one stage to the next while maintaining compliance with data handling and privacy standards.

## Architecture & Constraints

- **Single-file isolation**: Each tool contains all its logic, caching, prompts, and retries
- **No cross-imports**: Tools are completely isolated from each other
- **Schema-driven**: All tools validate against `schemas/unified_lead_record.json`
- **Deterministic IDs**: Uses UUID5 with NAMESPACE_DNS/NAMESPACE_URL for consistency
- **Modular enrichment**: Enrichment providers are pluggable and configurable via tiers
- **Provider routing**: Intelligent fallback between enrichment providers based on availability and cost

## Installation

1. Install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:

```bash
git clone <repository-url>
cd lead-sorcerer
```

3. Install dependencies:

```bash
poetry install
```

4. Copy environment template:

```bash
cp env.template .env
# Edit .env with your API keys
```

## Configuration

### ICP Configuration (`icp_config.json`)

The ICP configuration defines your ideal customer profile and search parameters:

```json
{
  "name": "SaaS Companies",
  "icp_text": "B2B SaaS companies with 50-500 employees",
  "queries": ["${industry} companies in ${region}"],
  "threshold": 0.7,
  "mode": "fast"
}
```

### Cost Configuration (`config/costs.yaml`)

Define unit pricing for all providers:

```yaml
proxycurl:
  unit: 'lookup'
  usd_per_unit: 0.0100
```

## Usage

### Individual Tools

Each tool can be run independently:

```bash
# Domain tool
echo '{"icp_config": {...}}' | poetry run domain

# Crawl tool
echo '{"lead_records": [...], "icp_config": {...}}' | poetry run crawl

# Enrich tool
echo '{"lead_records": [...], "icp_config": {...}, "caps": {...}}' | poetry run enrich
```

### Orchestrator

Run the complete pipeline:

```bash
poetry run orchestrator --config icp_config.json
```

## Data Flow

1. **Domain** → Generates leads from ICP queries, scores them, and filters by threshold
2. **Crawl** → Extracts company and contact information from websites
3. **Enrich** → Fills missing data and verifies contact information

## Enrichment Provider Architecture

The enrichment system uses a modular, tier-based architecture:

### Provider Tiers

- **TIER_0**: Primary providers (CompanyEnrich, Anymail Finder)
- **TIER_1**: Secondary providers (fallback options)
- **TIER_2**: Legacy providers (Coresignal, Snovio)

### Provider Router

The `EnrichmentProviderRouter` intelligently routes requests:

- Automatically selects the best available provider
- Implements fallback logic when primary providers fail
- Tracks provider health and availability
- Manages cost optimization across providers

### Available Providers

- **CompanyEnrich**: Company data and contact enrichment
- **Anymail Finder**: Email discovery and verification
- **Coresignal**: Company information (legacy)
- **Snovio**: Contact discovery (legacy)
- **Apollo**: Lead generation and enrichment (new!)

## Apollo Integration

Lead Sorcerer now supports Apollo as an alternative lead generation path that can bypass the traditional Domain + Crawl tools while maintaining all existing functionality.

### Key Features

- **Alternative Pipeline**: Apollo Search + Enrich instead of Domain → Crawl → Enrich
- **High-Quality Leads**: Advanced filtering based on ICP criteria
- **Search Strategies**: Company-first and person-first approaches
- **Data Quality**: Built-in quality validation and scoring
- **Cost Tracking**: Credit-based pricing with detailed cost monitoring
- **Hybrid Mode**: Use Apollo enrichment within traditional pipeline

### Setup

1. **Environment Variables**:

```bash
# Add to your .env file
APOLLO_API_KEY=your_api_key_here
APOLLO_REFRESH_TOKEN=your_refresh_token_here  # Optional
```

2. **ICP Configuration**:

```json
{
  "lead_generation_mode": "apollo",
  "apollo": {
    "enabled": true,
    "search": {
      "strategy": "company_first",
      "company_filters": {
        "industry": ["Technology", "SaaS"],
        "size": ["11-50", "51-200"],
        "location": ["United States"]
      }
    }
  }
}
```

3. **Run Apollo Pipeline**:

```bash
poetry run orchestrator --config icp_config.json
```

### Search Strategies

- **Company-First**: Find companies matching ICP, then discover people
- **Person-First**: Find people matching ICP, then enrich company data

### Cost Structure

- **Search**: 1 credit ($0.015) per request
- **Enrichment**: 1 credit ($0.015) per request
- **Credits**: Can be used for both search and enrichment

### Documentation

- [Apollo API Documentation](docs/apolloapidocs.md)
- [Configuration Guide](docs/icp_configuration_guide.md)
- [Sample Configuration](config/apollo_example_icp.json)

## LLM-Based Contact Selection

Lead Sorcerer now includes intelligent contact selection using Large Language Models (LLMs) to automatically choose the most relevant contacts based on your ICP business context.

### Key Features

- **Context-Aware Selection**: Understands business context beyond simple rules
- **ICP-Specific Configuration**: Tailored settings for different industry types
- **Intelligent Fallback**: Automatically falls back to rule-based selection on LLM failures
- **Cost Optimization**: Built-in caching and cost tracking
- **Comprehensive Monitoring**: Detailed metrics and quality analysis

### Supported ICP Types

| ICP Type       | Focus                        | Example Use Cases                       |
| -------------- | ---------------------------- | --------------------------------------- |
| **Investment** | Capital allocation decisions | Family offices, VCs, investment funds   |
| **Healthcare** | Medical practice leadership  | Practice owners, medical directors      |
| **Technology** | Technical decision-making    | CTOs, engineering leaders               |
| **Enterprise** | Executive leadership         | C-level executives, business unit heads |
| **Default**    | General business decisions   | Standard business contacts              |

### Configuration

Enable LLM selection by adding configuration to your ICP file:

```json
{
  "icp_type": "investment",
  "icp_text": "Family offices and VCs investing in AI startups",
  "llm_config": {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": 600,
    "timeout": 30.0
  }
}
```

### Environment Setup

Add your OpenRouter API key:

```bash
# Add to your .env file
OPENROUTER_API_KEY=your-api-key-here
```

### Performance Benefits

- **Higher Accuracy**: Context-aware selection improves contact relevance
- **Scalable Intelligence**: Easily adapt to new ICP requirements
- **Reduced Manual Review**: Better selections reduce manual validation
- **Cost Effective**: Caching minimizes API usage costs

For detailed configuration and usage information, see:

- [LLM Contact Selection Guide](docs/llm_contact_selection_guide.md)
- [Enrichment Architecture](docs/enrichment_architecture.md)
- [Maintenance Procedures](docs/llm_maintenance_procedures.md)

## Schema Validation & Contract

Each tool validates inputs/outputs against the canonical schema (`schemas/unified_lead_record.json`). On schema validation failure, tools return `SCHEMA_VALIDATION` errors while still returning partial results.

## Status History

Every tool appends a status history entry when changing record status:

```json
{
  "status": "scored",
  "ts": "2024-01-01T00:00:00Z",
  "notes": "LLM scoring completed"
}
```

## Caching & Revisit Policy

- **Domain**: SERP cache with configurable TTL (default: 24 hours)
- **Crawl**: Artifact cache with configurable TTL (default: 14 days)
- **Revisit Policy**: Uses max-rule: `max(existing_next_revisit_at, now + revisit_after_days)`

## Versioning Policy

Tools bump versions when:

- **MAJOR**: Envelope shape or required I/O fields change
- **MINOR**: Backward-incompatible schema or selection/scoring changes
- **PATCH**: Prompts, heuristics, or bug fixes without contract changes

## Error Handling & PII Masking

Tools never crash the pipeline. Instead, they:

- Return partial results
- Append structured errors to `errors[]`
- Mask PII in logs and errors
- Use exponential backoff with 45s wall-clock cap

### Error Codes

- `SCHEMA_VALIDATION`: Input/output schema mismatch
- `HTTP_429`: Rate limited (retryable)
- `PROVIDER_ERROR`: Provider API errors
- `BUDGET_EXCEEDED`: Cost cap reached
- `UNKNOWN`: Unhandled exceptions

## State Machine (Authoritative)

Allowed transitions:

- `new` → `scored` → `crawled` → `enrich_ready` → `enriched`
- `scored` → `crawl_failed`
- `crawled` → `enrich_partial`

## Field Ownership

Each tool can only update specific fields:

- **Domain**: `icp.pre_*`, `provenance.scored_at`, `cost.domain_usd`
- **Crawl**: `company.*`, `contacts[]`, `icp.crawl_*`, `cost.crawl_usd`
- **Enrich**: `best_contact_id`, `contacts[*].email*`, `cost.enrich_usd`

## Metrics Semantics

Every tool returns metrics:

```json
{
  "count_in": 100,
  "count_out": 95,
  "duration_ms": 15000,
  "cache_hit_rate": 0.25,
  "pass_rate": 0.95,
  "cost_usd": {
    "domain": 0.5,
    "crawl": 1.2,
    "enrich": 2.85,
    "total": 4.55
  }
}
```

## Exports

When enabled, exports are created in:

- `data/exports/{icp_name}/{YYYYMMDD_HHMM}/leads.jsonl`
- `data/exports/{icp_name}/{YYYYMMDD_HHMM}/leads.csv`

CSV exports include only the best contact with flattened dot-notation.

## Testing

Run the test suite:

```bash
poetry run pytest
```

Run with coverage:

```bash
poetry run pytest --cov=src --cov-report=html
```

## Development

### Pre-commit Hooks

Install pre-commit hooks:

```bash
poetry run pre-commit install
```

### Code Quality

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

## Data Retention

- **Artifacts**: Keep latest 3 versions per domain
- **Cleanup**: Delete files older than 365 days
- **GC**: Nightly cleanup via GitHub Actions

## Environment Variables

Required environment variables:

- `GSE_API_KEY`: Google Programmable Search API key
- `GSE_CX`: Google Search Engine ID
- `OPENROUTER_KEY`: OpenRouter API key
- `FIRECRAWL_KEY`: Firecrawl API key
- `CORESIGNAL_API_TOKEN`: Coresignal API token for company and contact enrichment
- `MAILGUN_SMTP_LOGIN`: Mailgun SMTP login
- `MAILGUN_SMTP_PW`: Mailgun SMTP password
- `SNOV_IO_ID`: Snov.io API ID
- `SNOV_IO_SECRET`: Snov.io API secret

### New Enrichment Providers

- `COMPANY_ENRICH_API_KEY`: CompanyEnrich API key for company data enrichment
- `ANYMAIL_FINDER_API_KEY`: Anymail Finder API key for email discovery

## License

[Leadpoet License]
