-- Additive minimum-participation evidence. Apply after migration 215.
-- Historical accepted rows intentionally remain NULL: updated_at is mutable.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_runs
  ADD COLUMN IF NOT EXISTS participation_accepted_at TIMESTAMPTZ;

CREATE OR REPLACE FUNCTION public.lab_arena_runs_participation_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog
AS $participation$
BEGIN
  IF TG_OP = 'INSERT' THEN
    IF NEW.participation_accepted_at IS NOT NULL THEN
      RAISE EXCEPTION 'participation timestamp is database-owned' USING ERRCODE = '42501';
    END IF;
    RETURN NEW;
  END IF;
  IF NEW.participation_accepted_at IS DISTINCT FROM OLD.participation_accepted_at THEN
    RAISE EXCEPTION 'participation timestamp is immutable' USING ERRCODE = '42501';
  END IF;
  IF OLD.participation_accepted_at IS NOT NULL AND (
      NEW.runner_hotkey IS DISTINCT FROM OLD.runner_hotkey
      OR NEW.round_id IS DISTINCT FROM OLD.round_id) THEN
    RAISE EXCEPTION 'participation identity is immutable' USING ERRCODE = '42501';
  END IF;
  -- Only the normal completion RPC can make this transition: the service role
  -- has SELECT, not direct table writes. Its existing lease, generation, output
  -- and accounting checks remain authoritative and run in this transaction.
  -- Cache-inserted rows and pending followers never held an original lease.
  IF OLD.status = 'leased' AND NEW.status = 'accepted'
      AND NEW.terminal_cause = 'accepted'
      AND OLD.runner_hotkey IS NOT NULL
      AND NEW.runner_hotkey IS NOT DISTINCT FROM OLD.runner_hotkey
      AND NEW.round_id IS NOT DISTINCT FROM OLD.round_id
      AND OLD.lease_token_hash IS NOT NULL
      AND OLD.claim_request_id IS NOT NULL
      AND (NEW.result_doc ->> 'schema_version') IS DISTINCT FROM
          'leadpoet.lab_arena.cached_run_result.v1' THEN
    NEW.participation_accepted_at := pg_catalog.clock_timestamp();
  END IF;
  RETURN NEW;
END;
$participation$;
ALTER FUNCTION public.lab_arena_runs_participation_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_runs_participation_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DROP TRIGGER IF EXISTS lab_arena_runs_participation ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_runs_participation
  BEFORE INSERT OR UPDATE ON public.lab_arena_runs
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_runs_participation_v1();

CREATE INDEX IF NOT EXISTS lab_arena_runs_recent_participation_idx
  ON public.lab_arena_runs (runner_hotkey, participation_accepted_at DESC)
  WHERE status = 'accepted' AND participation_accepted_at IS NOT NULL;

CREATE OR REPLACE FUNCTION public.lab_arena_has_recent_participation_v1(
  p_network TEXT, p_netuid INTEGER, p_runner_hotkey TEXT
)
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = pg_catalog
AS $recent_participation$
  SELECT pg_catalog.jsonb_build_object('eligible', EXISTS (
    SELECT 1
    FROM public.lab_arena_runs AS run
    JOIN public.lab_arena_rounds AS round ON round.round_id = run.round_id
    WHERE run.runner_hotkey = p_runner_hotkey
      AND run.status = 'accepted'
      AND run.participation_accepted_at IS NOT NULL
      AND run.participation_accepted_at > pg_catalog.statement_timestamp() - INTERVAL '24 hours'
      AND run.participation_accepted_at <= pg_catalog.statement_timestamp()
      AND round.arena_network_name = p_network
      AND round.arena_netuid = p_netuid
  ));
$recent_participation$;
ALTER FUNCTION public.lab_arena_has_recent_participation_v1(TEXT, INTEGER, TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_has_recent_participation_v1(TEXT, INTEGER, TEXT)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_has_recent_participation_v1(TEXT, INTEGER, TEXT)
  TO lab_arena_service;

COMMENT ON COLUMN public.lab_arena_runs.participation_accepted_at IS
  'Database time of original leased-run acceptance; immutable, never backfilled or refreshed by replay/cache reuse.';
COMMENT ON FUNCTION public.lab_arena_has_recent_participation_v1(TEXT, INTEGER, TEXT) IS
  'Gateway-only 24-hour accepted-work lookup, scoped to network, subnet and validator hotkey. No idle exemption.';

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
