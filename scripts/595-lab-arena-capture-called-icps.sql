-- Narrow capture ICPs to existing provider-call activity, matching public cost traces.
-- No rows, return columns, model/token totals, function ACLs or role grants change.
-- Apply through the existing exact owner-migration procedure used for 593/594.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
DO $capture_traces595$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_capture_summary_v1(text,text)';
  v_preimage CONSTANT TEXT := 'ee81b35c076f8ff630b21adbcbb43f257ae5a2b68ee4508ebf0b788f00ca0ec9';
  v_postimage CONSTANT TEXT := '10c9ee3a9543bf65fb95c198efdbf14bf6ece7b340f9fceccc4e1637de5cb2c4';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_old TEXT := $old595$  SELECT count(*)::INTEGER,
    coalesce(array_agg(DISTINCT r.icp_position::INTEGER ORDER BY r.icp_position::INTEGER), '{}'),
    coalesce(array_agg(r.run_id), '{}')
  INTO execute_runs, icp_positions, run_ids
  FROM public.lab_arena_runs AS r
  WHERE r.round_id = p_round_id AND r.submission_id = p_submission_id AND r.kind = 'execute';$old595$;
  v_new TEXT := $new595$  SELECT count(*)::INTEGER, coalesce(array_agg(r.run_id), '{}')
  INTO execute_runs, run_ids
  FROM public.lab_arena_runs AS r
  WHERE r.round_id = p_round_id AND r.submission_id = p_submission_id AND r.kind = 'execute';

  -- Match the public cost summary's call_count > 0 trace definition across
  -- providers and terminal states. Merely creating an execute run is not a call.
  SELECT coalesce(array_agg(DISTINCT r.icp_position::INTEGER ORDER BY r.icp_position::INTEGER), '{}')
  INTO icp_positions
  FROM public.lab_arena_runs AS r
  WHERE r.run_id = ANY (run_ids)
    AND EXISTS (
      SELECT 1 FROM public.lab_arena_ledger AS l
      WHERE l.run_id = r.run_id AND l.round_id = p_round_id
        AND l.submission_id = p_submission_id AND l.call_identity IS NOT NULL
    );$new595$;
BEGIN
  v_oid := pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid), pg_catalog.to_jsonb(p)-'prosrc'
  INTO v_definition, v_metadata FROM pg_catalog.pg_proc p WHERE p.oid=v_oid;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'capture traces595 function missing' USING ERRCODE='55000';
  END IF;
  IF (SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT,
      p.prosecdef, p.provolatile, p.proconfig, l.lanname)
    FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid=p.prolang WHERE p.oid=v_oid)
    IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,leadpoet_capture_reader=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'capture traces595 security differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'capture traces595 preimage differs' USING ERRCODE='55000';
  END IF;
  IF length(v_definition)-length(pg_catalog.replace(v_definition,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'capture traces595 anchor differs' USING ERRCODE='55000';
  END IF;
  v_updated := pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'capture traces595 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata
     OR pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'capture traces595 catalog readback differs' USING ERRCODE='55000';
  END IF;
END;
$capture_traces595$;
NOTIFY pgrst, 'reload schema';
COMMIT;
