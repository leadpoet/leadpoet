-- Research Lab: DB-side duplicate reimbursement guard.
--
-- Deployment policy:
--   * Apply after scripts 28 and 29.
--   * Enforces one reimbursement award per paid Research Lab run.
--   * Does not enable reimbursements or mutate existing rows.

BEGIN;

CREATE UNIQUE INDEX IF NOT EXISTS ux_research_reimbursement_awards_run_id
    ON public.research_reimbursement_awards(run_id);

COMMIT;

-- Smoke check after applying:
--
--   SELECT indexname
--   FROM pg_indexes
--   WHERE schemaname = 'public'
--     AND tablename = 'research_reimbursement_awards'
--     AND indexname = 'ux_research_reimbursement_awards_run_id';
