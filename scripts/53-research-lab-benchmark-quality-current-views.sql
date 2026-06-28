-- Research Lab: expose benchmark quality fields to benchmark current views.
--
-- Deployment policy:
--   * Apply after scripts 41 and 52.
--   * Safe to apply repeatedly.
--   * Fixes stale current-view shapes where benchmark_quality and
--     benchmark_attempt were added after the views were first created.

BEGIN;

DROP VIEW IF EXISTS public.research_lab_private_model_benchmark_current;

CREATE VIEW public.research_lab_private_model_benchmark_current
WITH (security_invoker = true) AS
SELECT
    b.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.benchmark_status AS current_benchmark_status,
    e.created_at AS current_status_at
FROM public.research_lab_private_model_benchmark_bundles b
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_private_model_benchmark_events e
    WHERE e.benchmark_bundle_id = b.benchmark_bundle_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

DROP VIEW IF EXISTS public.research_lab_public_benchmark_report_current;

CREATE VIEW public.research_lab_public_benchmark_report_current
WITH (security_invoker = true) AS
SELECT
    r.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.report_status AS current_report_status,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_lab_public_benchmark_reports r
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_public_benchmark_report_events e
    WHERE e.report_id = r.report_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

REVOKE ALL ON TABLE public.research_lab_private_model_benchmark_current FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_public_benchmark_report_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_private_model_benchmark_current TO service_role;
GRANT SELECT ON TABLE public.research_lab_public_benchmark_report_current TO service_role;

COMMENT ON VIEW public.research_lab_private_model_benchmark_current IS
    'Latest private model benchmark status joined to immutable benchmark bundle rows, including benchmark quality fields.';
COMMENT ON VIEW public.research_lab_public_benchmark_report_current IS
    'Latest public benchmark report status joined to immutable report rows, including benchmark quality fields.';

COMMIT;

-- Verification:
-- SELECT table_name, column_name
-- FROM information_schema.columns
-- WHERE table_schema = 'public'
--   AND table_name IN (
--       'research_lab_private_model_benchmark_current',
--       'research_lab_public_benchmark_report_current'
--   )
--   AND column_name IN ('benchmark_quality', 'benchmark_attempt')
-- ORDER BY table_name, column_name;
