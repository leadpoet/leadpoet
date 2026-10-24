-- Take a shared, not exclusive, hold on the round row in lab_arena_expire_leases.
--
-- The lease sweep reads the round (status and stage_generation) and never writes
-- it; its writes are all to lab_arena_runs. Provider and completion RPCs hold
-- FOR SHARE on the round before locking their run. An exclusive sweep waits
-- behind them while holding the claim-control advisory lock, which blocks
-- subsequent claims. A healthy sweep needs no run locks and can instead finish
-- under FOR SHARE. Recovery still locks each affected run before checking its
-- current provider accounting, so it excludes provider changes on that run.
--
-- FOR SHARE still excludes lab_arena_close_stage and the other progression RPCs,
-- which take FOR UPDATE. The control -> round -> run lock order is unchanged.
-- Sweeps retain their existing exclusive claim-control advisory lock, which
-- serializes sweeps and claims and prevents duplicate retry creation.
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

  -- Accept only the measured current definition or its exact postimage.
  IF encode(extensions.digest(d,'sha256'),'hex') =
      'd1ffdf531e598254d3fd4241e44ca2a1d5721580e559d1c613ad8ccfd3b3f0ad' THEN
    RETURN;
  END IF;
  IF encode(extensions.digest(d,'sha256'),'hex') <>
      '61e1dd5c5c472ce8060d356794bd9f621a8fc0f579ce091941826e19bf261468' THEN
    RAISE EXCEPTION 'expire leases round lock preimage differs' USING ERRCODE = '55000';
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
     OR pg_get_functiondef(target) IS DISTINCT FROM updated
     OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex') <>
       'd1ffdf531e598254d3fd4241e44ca2a1d5721580e559d1c613ad8ccfd3b3f0ad' THEN
    RAISE EXCEPTION 'expire leases round lock readback differs' USING ERRCODE = '55000';
  END IF;
END;
$expire_leases_shared_round_lock551$;
COMMIT;
