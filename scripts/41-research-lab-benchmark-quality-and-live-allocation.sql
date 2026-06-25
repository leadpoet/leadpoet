-- Research Lab benchmark quality gates and live allocation retry support.
--
-- Deployment policy:
--   * Apply after scripts 36, 37, 38, 39, and 40.
--   * Allows corrected same-day private baseline/public benchmark attempts to
--     be appended when an earlier attempt failed quality checks.
--   * Keeps benchmark and allocation rows append-only and service-role only.
--   * No raw OpenRouter keys, service-role keys, private repo material, hidden
--     ICP plaintext, judge prompts, private image refs, candidate patch
--     manifests, proxy credentials, credential URLs, or raw provider secrets
--     may be stored here.

BEGIN;

ALTER TABLE public.research_lab_private_model_benchmark_bundles
    ADD COLUMN IF NOT EXISTS benchmark_attempt INTEGER NOT NULL DEFAULT 0
        CHECK (benchmark_attempt >= 0);
ALTER TABLE public.research_lab_private_model_benchmark_bundles
    ADD COLUMN IF NOT EXISTS benchmark_quality TEXT NOT NULL DEFAULT 'legacy_unverified'
        CHECK (benchmark_quality IN ('passed', 'failed', 'legacy_unverified'));

ALTER TABLE public.research_lab_public_benchmark_reports
    ADD COLUMN IF NOT EXISTS benchmark_attempt INTEGER NOT NULL DEFAULT 0
        CHECK (benchmark_attempt >= 0);
ALTER TABLE public.research_lab_public_benchmark_reports
    ADD COLUMN IF NOT EXISTS benchmark_quality TEXT NOT NULL DEFAULT 'legacy_unverified'
        CHECK (benchmark_quality IN ('passed', 'failed', 'legacy_unverified'));

ALTER TABLE public.research_lab_private_model_benchmark_bundles
    DROP CONSTRAINT IF EXISTS research_lab_private_model_benchmark_unique_day;
ALTER TABLE public.research_lab_private_model_benchmark_bundles
    DROP CONSTRAINT IF EXISTS research_lab_private_model_benchmark_unique_attempt;
ALTER TABLE public.research_lab_private_model_benchmark_bundles
    ADD CONSTRAINT research_lab_private_model_benchmark_unique_attempt
    UNIQUE (
        benchmark_date,
        private_model_manifest_hash,
        rolling_window_hash,
        benchmark_attempt
    );

ALTER TABLE public.research_lab_public_benchmark_reports
    DROP CONSTRAINT IF EXISTS research_lab_public_benchmark_reports_unique_day;
ALTER TABLE public.research_lab_public_benchmark_reports
    DROP CONSTRAINT IF EXISTS research_lab_public_benchmark_reports_unique_attempt;
ALTER TABLE public.research_lab_public_benchmark_reports
    ADD CONSTRAINT research_lab_public_benchmark_reports_unique_attempt
    UNIQUE (
        benchmark_date,
        private_model_manifest_hash,
        rolling_window_hash,
        benchmark_attempt
    );

CREATE INDEX IF NOT EXISTS idx_research_lab_private_benchmark_quality
    ON public.research_lab_private_model_benchmark_bundles(
        benchmark_date DESC,
        benchmark_quality,
        benchmark_attempt DESC,
        created_at DESC
    );

CREATE INDEX IF NOT EXISTS idx_research_lab_public_benchmark_quality
    ON public.research_lab_public_benchmark_reports(
        benchmark_date DESC,
        benchmark_quality,
        benchmark_attempt DESC,
        created_at DESC
    );

COMMENT ON COLUMN public.research_lab_private_model_benchmark_bundles.benchmark_attempt IS
    'Append-only same-day retry number for private baseline rebenchmarks.';
COMMENT ON COLUMN public.research_lab_private_model_benchmark_bundles.benchmark_quality IS
    'Quality status for benchmark output; new valid rows use passed.';
COMMENT ON COLUMN public.research_lab_public_benchmark_reports.benchmark_attempt IS
    'Append-only same-day retry number matching the source private benchmark.';
COMMENT ON COLUMN public.research_lab_public_benchmark_reports.benchmark_quality IS
    'Quality status for public report output; new valid rows use passed.';

COMMIT;
