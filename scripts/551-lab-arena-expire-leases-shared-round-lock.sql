-- Take a shared, not exclusive, hold on the round row in lab_arena_expire_leases.
--
-- The lease sweep reads the round (status and stage_generation) and never writes
-- it; its writes are all to lab_arena_runs. The exclusive hold it takes therefore
-- buys nothing it needs, and it conflicts with the FOR SHARE hold that
-- lab_arena_claim_assignment takes on the same row. Once a claim is slow enough
-- that one is almost always in flight, the sweep waits, sits at the head of the
-- lock queue, and every claim arriving behind it waits too and dies on its 5s
-- lock_timeout (55P03). Observed 2026-10-24 from 14:25Z: the sweep timed out on
-- every attempt and 13-20% of run claims were refused.
--
-- FOR SHARE still excludes lab_arena_close_stage and the other progression RPCs,
-- which take FOR UPDATE — the migration-490 lock order is unchanged. Two sweeps
-- may now overlap; under READ COMMITTED each re-checks a run row after taking
-- FOR UPDATE on it, so a run already moved to 'failed' no longer matches the
-- status='leased' predicate and is not processed twice.
--
-- Only the lock acquisition changes. The guard below asserts the swapped
-- definition is byte-identical to the live one apart from that single clause,
-- and that no function metadata (owner, acl, security, volatility, config)
-- moves with it.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $expire_leases_shared_round_lock551$
DECLARE
  d TEXT;
  updated TEXT;
  metadata JSONB;
  target OID := 'public.lab_arena_expire_leases(text)'::regprocedure;
  old CONSTANT TEXT := $old$SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;$old$;
  replacement CONSTANT TEXT := $new$SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR SHARE;$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid), to_jsonb(p)-'prosrc' INTO STRICT d, metadata
    FROM pg_proc p JOIN pg_language l ON l.oid = p.prolang
   WHERE p.oid = target
     AND pg_get_userbyid(p.proowner) = 'lab_arena_owner'
     AND p.proacl::TEXT = '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
     AND p.prosecdef AND p.provolatile = 'v'
     AND p.proconfig = ARRAY['search_path=pg_catalog, public']::TEXT[]
     AND l.lanname = 'plpgsql';

  -- Already applied.
  IF (length(d) - length(replace(d, replacement, ''))) = length(replacement)
     AND strpos(d, old) = 0 THEN
    RETURN;
  END IF;

  -- The round lock must appear exactly once, in the form this script expects.
  IF (length(d) - length(replace(d, old, ''))) <> length(old) THEN
    RAISE EXCEPTION 'expire leases round lock source differs' USING ERRCODE = '55000';
  END IF;

  updated := replace(d, old, replacement);
  IF replace(updated, replacement, old) IS DISTINCT FROM d THEN
    RAISE EXCEPTION 'expire leases round lock swap is not confined to the lock clause' USING ERRCODE = '55000';
  END IF;

  EXECUTE updated;

  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid = target) IS DISTINCT FROM metadata
     OR pg_get_functiondef(target) IS DISTINCT FROM updated THEN
    RAISE EXCEPTION 'expire leases round lock readback differs' USING ERRCODE = '55000';
  END IF;
END;
$expire_leases_shared_round_lock551$;
COMMIT;
