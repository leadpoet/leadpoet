-- Retire caller-free Research Lab helpers while preserving their historical tables.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DROP FUNCTION IF EXISTS public.put_research_lab_provider_evidence_cache_v2(jsonb) RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_attested_execution_result_secret_free_v2(text, jsonb) RESTRICT;

-- Remove callers before their shared deterministic UUID helper.
DROP FUNCTION IF EXISTS public.research_lab_execution_trace_id(uuid) RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_trajectory_id(uuid) RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_deterministic_uuid(text) RESTRICT;

-- Remove the hash caller before its canonical JSON helper.
DROP FUNCTION IF EXISTS public.research_lab_routing_jsonb_hash_v2(jsonb) RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_routing_canonical_jsonb_v2(jsonb) RESTRICT;

DROP FUNCTION IF EXISTS public.research_lab_unpaid_ticket_expires_at(timestamp with time zone) RESTRICT;

NOTIFY pgrst, 'reload schema';
COMMIT;
