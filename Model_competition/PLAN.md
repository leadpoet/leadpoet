# Lead Qualification Engine — End-to-End Plan

> **Goal:** Build a top-tier lead qualification engine that, given an `ICPPrompt`, returns one genuinely matching company with intent signals whose every claim is verifiable. Abstain when not confident.
>
> **Constraints:** Not for the on-chain competition. No sandbox, no 500 KB cap, no per-ICP cost budget. Accuracy is the only target.
>
> **Stack:** OpenRouter (all LLMs including Perplexity Sonar) + Exa + ScrapingDog. Optional providers (Crunchbase, TheirStack, Desearch, Data Universe, NewsAPI) are not required.

---

## 0. Success Criteria

Measured on a 20-ICP golden eval set:

| Metric | Target |
|---|---|
| Precision (of non-abstain answers) | ≥ 95% |
| False positives (returned a `must_not_return`) | 0 |
| Abstention rate | 10–30% (preferred to wrong answer) |
| Calibration error (confidence vs actual) | ≤ ±5% |
| Cost P50 / P95 per ICP | ≤ $1.00 / ≤ $1.50 |
| Latency P50 / P95 | ≤ 180s / ≤ 240s |

---

## 1. Architectural Lessons Already Folded In

From inspecting `gateway/fulfillment/scoring.py` + `validator_models/fulfillment_company_verification.py`:

1. **LinkedIn is the anchor of truth** for structural facts (name, employee count, HQ, website, industry). Triple fallback: ScrapingDog → GSE slug-existence check → Apify (deferred).
2. **Strict structural validators**: lift `_validate_name_match`, `_validate_size_match`, `_normalize_domain`, `_llm_hq_match`, `validate_lead_geography`.
3. **3-stage industry classification**: Sonar enrich → Gemini validate → taxonomy-gated sub-industry pick (uses `validator_models/industry_taxonomy.py` + `taxonomy_embeddings.json.gz`).
4. **Dual-date time decay**: `signal.date` AND `content_found_date` — catches "selective date dropping" gaming.
5. **Negative-attribute proxy**: for "does NOT have X" claims, search for the positive; if found, reject.
6. **Peak-weighted intent aggregation**: best signal dominates, additional quality signals give diminishing breadth bonus.
7. **Required-spec gating**: ICP signals tagged `required=True` must each have ≥1 verified signal, else abstain.
8. **Defense-in-depth tiering**: cheap deterministic gates first, expensive LLM gates last.

---

## 2. Pipeline — 9 Layers

```
qualify(icp: ICPPrompt) → QualificationResult
│
├─ L1   ICP Parser ............. structure the intent (1 Sonnet call)
├─ L2   Wide Discovery ......... 200–500 raw candidates (Sonar + Exa + ScrapingDog, parallel)
├─ L3   Entity Resolver ........ LinkedIn-anchored facts (SD → GSE fallback) → 80–120 unique
├─ L2.5 Cheap Triage ........... gpt-4o-mini batch ranks → top 20
├─ L4   Hard Filters ........... country, industry, stage, size → 5–12 survivors
├─ L5   Evidence Gathering ..... 5–10 evidence URLs per candidate (parallel)
├─ L6   Strict Grounding (10 gates) verify every (company, url, claim) tuple
├─ L7   Final Ranker ........... Opus picks best, peak-weighted, abstain if < 80
└─ L8   Output Builder ......... CompanyOutput + audit trail
```

Each layer has typed input/output (see `Model_competition/SCHEMAS.md` after Sprint 1).

---

## 3. Tool Roles

| Capability | Provider |
|---|---|
| All LLM reasoning (Claude Opus 4.7, Sonnet 4.6, GPT-4o-mini) | OpenRouter |
| Real-time web search + synthesis (Perplexity Sonar/Sonar-pro) | OpenRouter |
| Semantic search + clean content extraction | Exa (`/search`, `/findSimilar`, `/contents`) |
| LinkedIn company / jobs / Google search / generic scrape | ScrapingDog |

---

## 4. Layer Details

### L1 — ICP Parser (Claude Sonnet 4.6, T=0)
- One JSON-mode LLM call
- Outputs: intent_class, is_time_bound, time_window_days, hard_filters, 3 semantic queries, 3 Sonar angles, 2 keyword queries, LinkedIn jobs query (if hiring), news keywords
- Retry once on schema failure

### L2 — Wide Discovery (parallel)
~12 tasks fired via `asyncio.gather`:
- 3× Sonar synthesized lists (different angles)
- 3× Exa neural search (semantic queries)
- 1× Exa /findSimilar if buyer named a peer
- 1× Exa keyword fallback
- 2× ScrapingDog Google (operator queries)
- 1× ScrapingDog LinkedIn jobs (if hiring intent)
- 1× intent-class-specific Sonar (funding, expansion, etc.)

Output: 200–500 `CandidateCompany` records (with dupes).

### L3 — LinkedIn-Anchored Resolver
Per unique domain:
1. Derive LinkedIn slug from discovery sources or canonical company name
2. **ScrapingDog LinkedIn company endpoint** (primary source of truth)
3. **GSE fallback**: if SD fetch fails, `site:linkedin.com/company/{slug}` to confirm slug exists. If not → drop candidate.
4. (Apify fallback deferred until proven necessary)
5. Extract: `canonical_name, primary_domain, country, industry_tags, employee_count_band, linkedin_url`
6. Cross-check homepage mentions company name (Exa /contents on `https://{domain}`)
7. Sonar enrichment for `funding_stage` only (not primary facts)

If no LinkedIn anchor possible → drop candidate. Hard gate.

### L2.5 — Cheap Triage (gpt-4o-mini)
- Batch candidates in groups of 20
- LLM scores each 0–10 on ICP fit using only the info provided
- Keep top 20 where score ≥ 5
- ~$0.02 per ICP regardless of candidate count

### L4 — Hard Filters (no LLM)
- Country compatibility (only reject if both known and incompatible)
- Industry tag overlap (at least one tag must match)
- Stage hard-incompatibility (Public vs Series B)
- Size egregious mismatch (10 employees vs 1000-5000 band)

If < 3 survivors → abstain at L4.

### L5 — Evidence Gathering (parallel per candidate)
- Always: company homepage (Exa /contents), LinkedIn company page (ScrapingDog)
- Intent-specific:
  - hiring → SD LinkedIn jobs, Exa "careers" search (60-day window)
  - funding → Exa "funding round" search (180-day window)
  - product_launch → Exa "launches announces" (90 days)
  - leadership_change → Exa "CEO CTO appointed" (120 days)
  - expansion → Exa "expands opens office" (120 days)
  - tech_adoption → Exa "uses adopts {keyword}"
- Catch-all: Sonar evidence search for the specific intent

Output: `dict[company → list[EvidenceURL]]`, 5–10 URLs per candidate.

### L6 — Strict Grounding (10 gates, fail-closed)

For each `(company, evidence_url, icp_signal)` tuple:

| Gate | Cost | What it checks |
|---|---|---|
| 1 URL reachable | ~$0.002 | 2xx, ≥100 chars, no parked-domain markers |
| 2 Company linkage | free | Host match OR ≥2 name mentions in first 5KB |
| 3 Negation scan | free | "0 open positions", "no longer", "404", "expired"… |
| 4 LinkedIn anchor | free | Candidate has confirmed LinkedIn (from L3) |
| 5 Strict grounding LLM (Sonnet 4.6) | ~$0.005 | Direct evidence + verbatim proof_quote + confidence ≥ 80 |
| 6 Anti-hallucination | free | proof_quote literal substring (or ≥92% fuzzy) in content |
| 7 Independent cross-check (Sonar-pro) | ~$0.005 | YES/UNKNOWN passes, NO rejects |
| 8 Dual-date sanity | ~$0.001 | claimed vs content-found ≤ 30 days apart; both not future; within window |
| 9 Negative-claim proxy | ~$0.005 | for "does NOT have X" claims, positive search must fail |
| 10 Generic-content blocklist | free | description not majority marketing fluff |

Pass all 10 → emit `VerifiedSignal`. Any fail → reject silently, log reason.

### L7 — Final Ranker (Claude Opus 4.7)
- Only candidates with ≥1 verified signal AND all required-spec ICP signals satisfied
- Per candidate: Opus scores 0–100 with reasoning + best_signals_to_use indices
- Pick highest. Abstain if best < 80 OR top-two within 5 points and both ≥ 80
- Aggregate confidence: peak-weighted (not pure mean)

### L8 — Output Builder
- Assemble `CompanyOutput` with top 3 signals per ranker's `best_signals_to_use`
- Compute `overall_confidence` from ranker_score, avg_grounding, cross_check_rate, sources_diversity
- Build `QualificationResult` with `alternatives_considered`, `reasoning_trace`, `cost_breakdown`, `latency_ms`
- If abstaining: `company=None`, `abstention_reason="<which gate or rule fired>"`

---

## 5. Abstention Rules

Return `QualificationResult(company=None, abstention_reason=...)` if any of:

- L4 leaves < 3 candidates
- L6 produces 0 verified signals across all candidates
- L7 best score < 80
- L7 top-two scores within 5 points and both ≥ 80 (ambiguous)
- Required-spec ICP signal has no verified evidence on the top candidate
- Per-ICP cost guard hit ($2 hard ceiling)

---

## 6. Cost & Latency Budget (per ICP)

| Layer | Cost | Latency |
|---|---|---|
| L1 | $0.005 | 3s |
| L2 | $0.15 | 25s |
| L3 (LinkedIn-anchored) | $0.10 | 25s |
| L2.5 | $0.02 | 10s |
| L4 | $0 | 1s |
| L5 | $0.10 | 25s |
| L6 (10 gates × ~30 tuples) | $0.50 | 50s |
| L7 | $0.10 | 25s |
| L8 | $0 | <1s |
| **Total (cold)** | **~$0.99** | **~165s** |
| **Total (50% cache warm)** | **~$0.50** | **~90s** |

Hard per-ICP ceiling: $2.00 → abstain if hit.

---

## 7. Project Structure

All files live under `Model_competition/`.

```
Model_competition/
├── PLAN.md                           (this file)
├── SCHEMAS.md                        (data shapes — written in Sprint 1)
├── README.md
├── pyproject.toml
├── .env.example
│
├── src/qual_engine/
│   ├── __init__.py
│   ├── entry.py                      # public API: qualify(icp)
│   ├── models.py                     # all Pydantic schemas
│   ├── config.py                     # env, thresholds, model IDs
│   │
│   ├── layers/
│   │   ├── l1_icp_parser.py
│   │   ├── l2_discovery.py
│   │   ├── l2_5_triage.py
│   │   ├── l3_resolver.py            # LinkedIn-anchored
│   │   ├── l4_filter.py
│   │   ├── l5_evidence.py
│   │   ├── l6_grounding.py           # 10 gates
│   │   ├── l7_ranker.py
│   │   └── l8_output.py
│   │
│   ├── providers/
│   │   ├── openrouter.py             # llm_json, llm_chat, sonar
│   │   ├── exa.py                    # search, find_similar, contents
│   │   └── scrapingdog.py            # google, linkedin_company, linkedin_jobs
│   │
│   ├── validators/                   # lifted from fulfillment patterns
│   │   ├── name_match.py             # _validate_name_match equivalent
│   │   ├── size_match.py             # _validate_size_match equivalent
│   │   ├── domain.py                 # _normalize_domain
│   │   ├── geography.py              # rule US / LLM non-US
│   │   └── industry_taxonomy.py      # taxonomy-gated picker
│   │
│   ├── infra/
│   │   ├── cache.py                  # SQLite, key by stable hash
│   │   ├── retry.py                  # tenacity wrappers
│   │   ├── trace.py                  # structured trace events
│   │   ├── cost_tracker.py
│   │   └── rate_limit.py             # asyncio semaphores
│   │
│   └── utils/
│       ├── dates.py
│       ├── fuzzy.py
│       └── text.py
│
├── tests/
│   ├── unit/
│   │   ├── test_grounding.py         # gates 1–10 in isolation
│   │   ├── test_resolver.py
│   │   ├── test_filter.py
│   │   ├── test_triage.py
│   │   └── test_validators.py        # name/size/domain matchers
│   └── integration/
│       └── test_full_pipeline.py
│
├── eval/
│   ├── golden_icps.yaml              # 20 hand-curated truth ICPs
│   ├── harness.py                    # run pipeline against golden
│   ├── metrics.py                    # precision, abstention, calibration
│   └── reports/                      # historical JSON results
│
├── scripts/
│   ├── run_single.py                 # qualify one ICP, print result + trace
│   ├── run_eval.py                   # full golden eval run
│   └── inspect_trace.py              # post-hoc debug for a single trace_id
│
└── cache/                            # SQLite cache (gitignored)
    └── .gitkeep
```

---

## 8. Configuration (`src/qual_engine/config.py`)

```python
class Config(BaseSettings):
    # API keys
    OPENROUTER_API_KEY: str
    EXA_API_KEY: str
    SCRAPINGDOG_API_KEY: str
    
    # Thresholds (calibrated against golden eval)
    GROUNDING_MIN_CONFIDENCE: int = 80
    RANKER_MIN_SCORE: int = 80
    RANKER_AMBIGUITY_MARGIN: int = 5
    MIN_CANDIDATES_TO_RANK: int = 3
    MAX_CANDIDATES_TO_GROUND: int = 15
    
    # Cost guards
    PER_ICP_SOFT_CEILING_USD: float = 1.50
    PER_ICP_HARD_CEILING_USD: float = 2.00
    
    # Model IDs
    PARSER_MODEL: str = "anthropic/claude-sonnet-4.6"
    GROUNDING_MODEL: str = "anthropic/claude-sonnet-4.6"
    TRIAGE_MODEL: str = "openai/gpt-4o-mini"
    RANKER_MODEL: str = "anthropic/claude-opus-4.7"
    SONAR_MODEL: str = "perplexity/sonar-pro"
    
    # Concurrency
    EXA_CONCURRENCY: int = 10
    SCRAPINGDOG_CONCURRENCY: int = 20
    OPENROUTER_CONCURRENCY: int = 10
```

---

## 9. Sprint Plan (10 sprints, each ≈ one PR)

| # | Sprint | Deliverable | Done when |
|---|---|---|---|
| 1 | **Bootstrap** | Project skeleton, `OpenRouter` wrapper, SQLite cache, cost tracker, env loading | `llm_json("anthropic/claude-sonnet-4.6", prompt)` works + is cached + records cost |
| 2 | **Providers** | Exa + ScrapingDog clients with retry, rate limit, cache | Each returns typed results; handles 429 with backoff |
| 3 | **Grounding (L6) in isolation** | 10-gate verifier + 20-pair hand-curated test set | 19/20 correct accept/reject on `(url, claim, expected)` pairs |
| 4 | **Golden eval set** | 20 ICPs YAML + harness scaffold + metrics module | Harness runs (initially against L6 only) and emits Markdown report |
| 5 | **L1 ICP Parser** | LLM-driven structuring | 10 sample ICPs parse correctly to intent_class + queries |
| 6 | **L2 Discovery + L3 Resolver** | Wide-net + LinkedIn-anchored resolution | 5 test ICPs each yield ≥30 LinkedIn-anchored candidates |
| 7 | **L2.5 Triage + L4 Filter + L5 Evidence** | Funnel + per-candidate evidence | Cuts to ≤10 candidates with ≥5 evidence URLs each |
| 8 | **L7 Ranker + L8 Output** | End-to-end pipeline | Runs full golden ICP, produces valid `QualificationResult` |
| 9 | **Eval iteration** | Tune thresholds against golden set | Precision ≥ 95%, abstention 10–30%, 0 false positives |
| 10 | **Observability + ops** | Tracing, cost dashboard, runbook | Can debug a single bad output in <5 min from trace |

---

## 10. The First Three PRs (concrete)

### PR-1 — Bootstrap (~600 LOC)
- `pyproject.toml`, `.env.example`, `README.md`
- `src/qual_engine/__init__.py`, `config.py`, `models.py` (skeleton schemas)
- `src/qual_engine/providers/openrouter.py` (`llm_json`, `llm_chat`, `sonar_search` with retry + cache)
- `src/qual_engine/infra/cache.py` (SQLite, TTL-aware)
- `src/qual_engine/infra/cost_tracker.py`
- `src/qual_engine/infra/retry.py` (tenacity wrappers)
- `tests/unit/test_openrouter_wrapper.py`
- `tests/unit/test_cache.py`

**Done**: `pytest tests/` passes; manual smoke test of `llm_json` shows a cache hit on second call.

### PR-2 — Providers (~800 LOC)
- `src/qual_engine/providers/exa.py` (`/search` neural+keyword, `/findSimilar`, `/contents`)
- `src/qual_engine/providers/scrapingdog.py` (Google, LinkedIn company, LinkedIn jobs)
- `src/qual_engine/infra/rate_limit.py` (per-provider semaphores)
- `tests/unit/test_exa.py`, `tests/unit/test_scrapingdog.py`
- Manual integration check against real endpoints with `scripts/test_providers.py`

**Done**: Each provider returns typed Pydantic objects, retries on 429/5xx, caches results.

### PR-3 — Grounding (L6) in isolation (~700 LOC)
- `src/qual_engine/layers/l6_grounding.py` — all 10 gates
- `src/qual_engine/validators/*` (name match, size match, domain normalize, etc. — lift patterns from fulfillment)
- `tests/unit/test_grounding.py` with **20 hand-curated `(url, claim, expected)` pairs**
  - 10 should accept: real job postings, real funding announcements, etc.
  - 10 should reject: 404s, wrong companies, negated content, hallucinatable claims
- `tests/fixtures/grounding_cases.yaml`

**Done**: 19/20 pairs return the expected accept/reject decision. The 1 allowed miss must be documented with reason.

After PR-3 we have provable grounding accuracy in isolation. From here, sprints 4–9 each build one layer on top with regression gates.

---

## 11. Eval Framework

Golden set covers all 9 intent classes + at least one "should abstain" case:

```yaml
# Model_competition/eval/golden_icps.yaml (excerpt)
- icp_id: hiring_fintech_001
  prompt: "Series B fintech in NYC hiring senior backend engineers"
  industry: Financial Services
  sub_industry: Fintech
  geography: United States, New York
  country: US
  employee_count: 50-200
  company_stage: Series B
  product_service: Developer infrastructure
  intent_signals: ["Hiring senior backend engineers"]
  
  expected_one_of:
    - {name: Mercury, domain: mercury.com}
    - {name: Modern Treasury, domain: moderntreasury.com}
    - {name: Petal, domain: petalcard.com}
  
  must_not_return:
    - {name: Stripe, reason: "too large, 4000+ employees"}
    - {name: SoFi,   reason: "public, not Series B"}
  
  must_have_signal_about: ["hiring", "backend", "engineer"]

# 19 more covering: funding / product_launch / tech_adoption / expansion /
# leadership_change / partnership / compliance_event / other / abstain_case
```

Metrics (`Model_competition/eval/metrics.py`):
- **Precision** = (correct returns) / (non-abstain returns)
- **False positives** = count of `must_not_return` returned (must be 0)
- **Abstention rate** = abstain / total
- **Calibration** = bucket by confidence; actual precision per bucket
- **Signal quality** = mean `grounding_confidence` on returned signals
- **Cost & latency** P50 / P95

Run `python -m scripts.run_eval` → produces `eval/reports/<timestamp>.md`. Block PR merge on regression in precision or false positives.

---

## 12. Operational Considerations

- **Tracing**: every layer emits `{trace_id, layer, action, inputs_hash, outputs_summary, latency_ms, cost_usd, decision}`. Persisted to SQLite. `scripts/inspect_trace.py <trace_id>` reconstructs the full decision tree.
- **Cost guards**: warn at $1.50, hard-abstain at $2.00 per ICP. Daily aggregate dashboard.
- **Error handling**: catch all provider exceptions; never bubble to caller. Each layer has a fallback. Circuit breaker: 10 consecutive 5xx → mark provider down 5 min.
- **Concurrency**: per-provider semaphores; `asyncio.gather(*, return_exceptions=True)` so one failure doesn't sink the batch. Timeouts: Exa 30s, ScrapingDog 60s, LLM 90s.
- **Caching**: SQLite with TTL by call type (search 6h, contents 24h, LLM @ T=0 30d, Sonar cross-check 1h).

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LinkedIn rate-limits via ScrapingDog | Cache 7d for company pages, 6h for job searches; per-provider semaphore caps concurrency |
| LLM hallucination of proof quotes | Gate 6 literal substring check (anti-hallucination) |
| Sonar drift / freshness gaps | Treat as one of two signals; require URL grounding as primary, Sonar as corroboration |
| Wide-net cost explosion | L2.5 triage caps grounding to top 20; hard $2 ceiling abstains |
| Industry-tag ambiguity ("fintech" vs "financial services") | Taxonomy-gated pick + multi-tag overlap (≥1 match) |
| Sparse public footprint companies | Abstention is preferred over guessing |
| Time-bound intent + no fresh evidence | L5 returns no evidence → L6 produces no signals → L7 abstains |

---

## 14. What's Explicitly Out of Scope

- No on-chain submission flow, signing, payment verification
- No TEE sandbox / hardcoding detection / 500 KB cap
- No competition scoring formula optimization
- No miner-side packaging
- Apify fallback is deferred until SD+GSE coverage proves insufficient
- Optional providers (Crunchbase, TheirStack, etc.) deferred — only add if eval shows coverage gaps

---

**Ready to start with PR-1 (Bootstrap) on approval.**
