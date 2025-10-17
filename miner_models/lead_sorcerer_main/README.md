# Lead Sorcerer

A comprehensive lead generation system designed to automate the process of identifying, scoring, and extracting business leads.

## Overview

Lead Sorcerer is composed of two main tools:

- **Domain**: Generates and scores potential leads based on predefined criteria
- **Crawl**: Extracts detailed information about companies and contacts

The orchestrator coordinates these tools, ensuring data flows seamlessly from one stage to the next while maintaining compliance with data handling and privacy standards.

## Architecture & Constraints

- **Single-file isolation**: Each tool contains all its logic, caching, prompts, and retries
- **No cross-imports**: Tools are completely isolated from each other
- **Schema-driven**: All tools validate against `schemas/unified_lead_record.json`
- **Deterministic IDs**: Uses UUID5 with NAMESPACE_DNS/NAMESPACE_URL for consistency

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
```

### Orchestrator

Run the complete pipeline:

```bash
poetry run orchestrator --config icp_config.json
```

## Data Flow

1. **Domain** → Generates leads from ICP queries, scores them, and filters by threshold
2. **Crawl** → Extracts company and contact information from websites

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

- `new` → `scored` → `crawled`
- `scored` → `crawl_failed`

## Field Ownership

Each tool can only update specific fields:

- **Domain**: `icp.pre_*`, `provenance.scored_at`, `cost.domain_usd`
- **Crawl**: `company.*`, `contacts[]`, `icp.crawl_*`, `cost.crawl_usd`

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
    "total": 1.7
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
- `OPENROUTER_KEY`: OpenRouter API key for LLM-based classification
- `FIRECRAWL_KEY`: Firecrawl API key for web scraping
