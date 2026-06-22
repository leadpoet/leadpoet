-- Research Lab Phase 1: receipt run_id foreign-key correction.
--
-- Deployment policy:
--   * Apply after scripts 27, 28, 29, 30, and 31.
--   * This fixes an early Research Lab persistence draft where
--     research_loop_receipts.run_id pointed at execution_traces.
--   * Hosted Research Lab run IDs are event-sourced in
--     research_loop_run_queue_events and are not execution_trace IDs.
--   * No data is mutated. Only the incorrect FK constraint is removed.
--   * No anon/authenticated grants are created.

BEGIN;

ALTER TABLE public.research_loop_receipts
    DROP CONSTRAINT IF EXISTS research_loop_receipts_run_id_fkey;

COMMENT ON COLUMN public.research_loop_receipts.run_id IS
    'Hosted Research Lab run UUID from research_loop_run_queue_events. Not an execution_traces.run_id foreign key.';

COMMIT;

-- Smoke check after applying this migration:
--
--   SELECT conname
--   FROM pg_constraint
--   WHERE conrelid = 'public.research_loop_receipts'::regclass
--     AND conname = 'research_loop_receipts_run_id_fkey';
--
-- Expected: zero rows.
