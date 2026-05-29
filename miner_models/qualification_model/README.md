# qualification_model

Universal evidence-first qualification miner. Single code path runs against
any ICP — no per-ICP hardcoding.

## Contract

```python
from miner_models.qualification_model import qualify

leads = qualify(icp_dict)   # list of up to 5 CompanyOutput dicts
```

`qualify(icp)` returns 0–5 CompanyOutput-shaped dicts matching the schema in
`gateway.qualification.models.CompanyOutput`. Empty list = honest abstention
(no candidates were verifiable for this ICP).

## Pipeline

```
1. DISCOVERY      Exa news (date-bound primary sources)
                  + Sonar 3-pass exclusion-driven (forces diversity)

2. STRICT EXTRACT LLM filter — drops listicles, intent-mismatched headlines,
                  ambiguous-subject titles

3. ANCHOR LOOKUP  Sonar resolves website / LinkedIn / industry-aligned description

4. URL SEARCH     Two parallel Exa keyword queries
                  - site:<company-domain> {headline}
                  - {company} {headline} site:businesswire OR prnewswire OR globenewswire

5. VERIFY         Production `verify_three_stage`:
                    Stage 1 — Sonar independent corroboration
                    Stage 2 — ScrapingDog scrape (4-tier cascade + Wayback)
                    Stage 3 — Sonar-pro judge on scraped content
                  Throttled to 8 concurrent calls per ICP

6. SCORE & RANK   Production `score_company` for internal ranking,
                  dedupe by company, keep top-5
```

## Files

```
__init__.py        Re-exports qualify, MAX_LEADS_PER_ICP
qualify.py         Sync validator entry point (wraps the async pipeline)
_model.py          Async implementation
requirements.txt   httpx, pydantic
```

## Environment

Auto-loaded from `{repo_root}/.env` if present. Required:

| Key                   | Used by                                                                   |
| --------------------- | ------------------------------------------------------------------------- |
| `OPENROUTER_API_KEY`  | Sonar discovery, article extractor, anchor lookup, verifier Stage 1/3     |
| `EXA_API_KEY`         | News search (discovery), keyword search (URL discovery)                   |
| `SCRAPINGDOG_API_KEY` | Verifier Stage 2 (`/scrape` + specialized `/x/post`, `/profile/post`)     |

## Sandbox note

This implementation calls external APIs (OpenRouter / Exa / ScrapingDog)
directly. The validator's TEE sandbox blocks outbound network except via
`QUALIFICATION_PROXY_URL`. For deployment to the sandbox, replace the direct
client calls (`call_sonar`, `exa_news_search`, `exa_keyword_search`,
`_scrape_sd_hardened`, etc.) with proxy-routed equivalents.

## Benchmark

On the 20260525 corpus (20 ICPs, sequential, Sonar+Exa merged, top-5
submission, sum/5 normalization):

```
Total per-ICP scores sum  : 408.10  (today's run, 20260529 corpus)
Avg per-ICP score          : 20.40
Wall time                  : ~56 min
```

Floor: `MINIMUM_CHAMPION_SCORE = 20.0` (per `gateway/qualification/config.py:87`).
