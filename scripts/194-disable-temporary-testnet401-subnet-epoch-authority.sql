-- Close the temporary fresh-testnet401 epoch-authority and result write paths.
--
-- Migration 191 allowed one measured testnet401 candidate and cutover to coexist
-- with the unrelated Finney singleton. Migration 192 allowed exact testnet401
-- execution results and signed weight-submission events to use that keyed
-- cutover. The proof rows and their receipt graphs are append-only evidence and
-- remain stored. This migration restores the unconditional Finney fence on all
-- four relations, restores legacy cutover validation, and removes only the
-- temporary functions.

BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Restore the exact Finney singleton fence on all four relations. Existing rows
-- are not updated or revalidated; all future inserts and updates again require
-- the active singleton authority.
DROP TRIGGER IF EXISTS enforce_research_lab_stateful_epoch_fence_v1
    ON public.research_lab_stateful_subnet_epoch_candidates_v1;
CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
    BEFORE INSERT OR UPDATE
    ON public.research_lab_stateful_subnet_epoch_candidates_v1
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();

DROP TRIGGER IF EXISTS enforce_research_lab_stateful_epoch_fence_v1
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
    BEFORE INSERT OR UPDATE
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();

DROP TRIGGER IF EXISTS enforce_research_lab_stateful_epoch_fence_v1
    ON public.research_lab_attested_execution_results_v2;
DROP TRIGGER IF EXISTS enforce_temporary_testnet401_execution_result_epoch_scope_v1
    ON public.research_lab_attested_execution_results_v2;
CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
    BEFORE INSERT OR UPDATE
    ON public.research_lab_attested_execution_results_v2
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();

DROP TRIGGER IF EXISTS enforce_research_lab_stateful_epoch_fence_v1
    ON public.transparency_log;
DROP TRIGGER IF EXISTS enforce_temporary_testnet401_weight_submission_epoch_scope_v1
    ON public.transparency_log;
CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
    BEFORE INSERT OR UPDATE
    ON public.transparency_log
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();

-- Restore the migration-105 cutover validator for every future cutover insert.
-- Its v1/v2 Finney semantics are unchanged. A v3 insert also fails closed.
DROP TRIGGER IF EXISTS validate_research_lab_stateful_epoch_cutover_v1
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
DROP TRIGGER IF EXISTS validate_research_lab_stateful_epoch_cutover_legacy_v3
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
DROP TRIGGER IF EXISTS validate_research_lab_fresh_network_epoch_cutover_v1
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
CREATE TRIGGER validate_research_lab_stateful_epoch_cutover_v1
    BEFORE INSERT
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1
    FOR EACH ROW
    EXECUTE FUNCTION public.validate_research_lab_stateful_epoch_cutover_v2();

-- The CHECK constraints added by migration 191 intentionally remain because
-- removing their v3 allowance would invalidate the retained proof row.
DROP FUNCTION IF EXISTS
    public.research_lab_fresh_network_epoch_cutover_public_state_v1(TEXT, INTEGER);
DROP FUNCTION IF EXISTS
    public.validate_research_lab_fresh_network_epoch_cutover_v1();
DROP FUNCTION IF EXISTS
    public.enforce_temporary_testnet401_execution_result_epoch_scope_v1();
DROP FUNCTION IF EXISTS
    public.enforce_temporary_testnet401_weight_submission_epoch_scope_v1();

NOTIFY pgrst, 'reload schema';

COMMIT;
