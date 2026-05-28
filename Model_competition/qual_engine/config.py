"""Configuration — loaded from environment, with sensible defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


# Load .env from repo root if present
_REPO_ROOT = Path(__file__).parent.parent.parent
_load_env_file(_REPO_ROOT / ".env")
_load_env_file(Path(__file__).parent.parent / ".env")


class Config:
    # --- API keys ---
    OPENROUTER_API_KEY: Optional[str] = (
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    )
    EXA_API_KEY: Optional[str] = os.environ.get("EXA_API_KEY")
    SCRAPINGDOG_API_KEY: Optional[str] = os.environ.get("SCRAPINGDOG_API_KEY")

    # --- Model IDs ---
    # Cost-tuned mix. Empirical findings:
    #   - Sonnet 4.6 was the ONLY model that hit 5/5 on the L6 grounding test.
    #     Gemini Flash and Pro each failed 1/5 (different cases). Don't downgrade.
    #   - Flash is fine for L1 (JSON structuring) and L7 (per-candidate scoring).
    #   - sonar (regular) is fine for binary cross-checks; sonar-pro for synthesis.
    PARSER_MODEL = "google/gemini-2.5-flash"      # OK — pure JSON parsing
    GROUNDING_MODEL = "anthropic/claude-sonnet-4.6"  # MUST stay — 5/5 grounding accuracy
    TRIAGE_MODEL = "openai/gpt-4o-mini"            # already cheap
    RANKER_MODEL = "google/gemini-2.5-flash"      # OK with multi-answer mode (less critical)
    SONAR_MODEL = "perplexity/sonar-pro"           # L2 discovery + L3 facts — needs synthesis quality
    SONAR_VERIFY_MODEL = "perplexity/sonar"        # L6 cross-check + neg-claim — binary yes/no
    QUERY_GEN_MODEL = "openai/gpt-4o-mini"

    # --- Thresholds ---
    GROUNDING_MIN_CONFIDENCE = 60  # Sonnet routinely returns 60-75 for "moderately supported" evidence on non-funding intents; 75 was over-strict and killed most non-funding signals
    RANKER_MIN_SCORE = 70          # multi-answer inclusion threshold
    RANKER_AMBIGUITY_MARGIN = 0    # disabled — multi-answer doesn't tie-break
    MIN_CANDIDATES_TO_RANK = 1     # was 3 — multi-answer doesn't need a minimum pool
    MAX_CANDIDATES_TO_GROUND = 25  # was 12 — verify more candidates → more answers
    NAME_MENTIONS_REQUIRED = 1  # Gate 2: minimum company-name mentions in evidence

    # --- Cost guards ---
    PER_ICP_SOFT_CEILING_USD = 1.50
    PER_ICP_HARD_CEILING_USD = 2.00

    # --- Concurrency ---
    EXA_CONCURRENCY = 10
    SCRAPINGDOG_CONCURRENCY = 20
    OPENROUTER_CONCURRENCY = 10

    # --- Timeouts (seconds) ---
    EXA_TIMEOUT = 45
    SCRAPINGDOG_TIMEOUT = 60
    OPENROUTER_TIMEOUT = 90

    # --- Cache TTLs (seconds) ---
    EXA_SEARCH_TTL = 6 * 3600       # 6h
    EXA_CONTENTS_TTL = 24 * 3600    # 24h
    SD_LINKEDIN_COMPANY_TTL = 7 * 24 * 3600  # 7d
    SD_LINKEDIN_JOBS_TTL = 6 * 3600  # 6h
    SD_GOOGLE_TTL = 6 * 3600        # 6h
    SONAR_CROSS_CHECK_TTL = 3600    # 1h
    SONAR_FACT_TTL = 24 * 3600      # 24h
    LLM_TEMP0_TTL = 30 * 24 * 3600  # 30d

    # --- Discovery defaults ---
    EXA_DEFAULT_DAYS_BACK = 180
    DISCOVERY_PER_QUERY_RESULTS = 18  # was 12 — wider net per query

    # --- Paths ---
    CACHE_DB_PATH = str(Path(__file__).parent.parent / "cache" / "cache.sqlite")
    TRACE_DB_PATH = str(Path(__file__).parent.parent / "cache" / "trace.sqlite")

    # --- Misc ---
    USER_AGENT = "Model_competition/qual_engine 0.1"

    @classmethod
    def validate(cls) -> list[str]:
        """Return list of missing required env vars."""
        missing = []
        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        if not cls.EXA_API_KEY:
            missing.append("EXA_API_KEY")
        if not cls.SCRAPINGDOG_API_KEY:
            missing.append("SCRAPINGDOG_API_KEY")
        return missing


CONFIG = Config()
