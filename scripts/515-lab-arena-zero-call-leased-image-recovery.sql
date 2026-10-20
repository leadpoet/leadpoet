-- Recover authenticated zero-call image setup failures through ordinary expiry.
-- Keep guard holds, frozen claims, ledger fences and retry creation unchanged.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $zero_call_leased_image515$
DECLARE
  target OID:=to_regprocedure('public.lab_arena_expire_leases(text)');
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  old_gate CONSTANT TEXT := $old$host_error.content ->> 'error_class' = 'RunnerError'$old$;
  new_gate CONSTANT TEXT := $new$host_error.content ->> 'error_class' IN ('RunnerError','LeasedImageError')$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname)
  INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target;
  IF definition IS NULL OR identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Leased image recovery security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='7c70db8f77d1552c315a1ef6ab4d0ad325f0a93850bcb3d302424858680ad002' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')<>'7835213f7d4991be60194c60589da3d18f62f05f8640136e7df2dfce1d8e256f'
    OR (length(definition)-length(replace(definition,old_gate,'')))<>2*length(old_gate) THEN
    RAISE EXCEPTION 'Leased image recovery preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=replace(definition,old_gate,new_gate);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>'7c70db8f77d1552c315a1ef6ab4d0ad325f0a93850bcb3d302424858680ad002' THEN
    RAISE EXCEPTION 'Leased image recovery postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'7c70db8f77d1552c315a1ef6ab4d0ad325f0a93850bcb3d302424858680ad002' THEN
    RAISE EXCEPTION 'Leased image recovery readback differs' USING ERRCODE='55000';
  END IF;
END;
$zero_call_leased_image515$;
-- A zero-call expiry implies a runner handoff only within the same frozen
-- rolling window already used by the three-expiry cooldown. Exact authenticated
-- runtime host faults retain the prior-runner ban after that window.
DO $bounded_score_handoff515$
DECLARE
  target OID:=to_regprocedure('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)');
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  old_gate CONSTANT TEXT := $old$          AND prior_score.lease_expires_at<=pg_catalog.clock_timestamp()$old$;
  new_gate CONSTANT TEXT := $new$          AND prior_score.lease_expires_at<=pg_catalog.clock_timestamp()
          AND (prior_score.lease_expires_at>
            pg_catalog.clock_timestamp()-pg_catalog.make_interval(
              secs => (v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER)
            OR EXISTS (
              SELECT 1 FROM public.lab_arena_trajectory_events AS host_error
              WHERE host_error.run_id=prior_score.run_id
                AND (host_error.runner_hotkey,host_error.round_id,host_error.submission_id,
                  host_error.miner_hotkey,host_error.assignment_id,host_error.stage,
                  host_error.icp_position,host_error.attempt,host_error.run_kind)
                  IS NOT DISTINCT FROM
                  (prior_score.runner_hotkey,prior_score.round_id,prior_score.submission_id,
                   prior_score.miner_hotkey,prior_score.assignment_id,prior_score.stage,
                   prior_score.icp_position,prior_score.attempt,prior_score.kind)
                AND host_error.event_kind='runtime.error'
                AND host_error.content->>'status'='abandoned'
                AND host_error.content->>'failure_stage'='runtime'
                AND host_error.content->>'error_class'='RuntimeHostError'
                AND host_error.created_at BETWEEN
                  (prior_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
                    -pg_catalog.make_interval(secs=>(v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER)
                  AND COALESCE((prior_score.terminal_doc->>'original_lease_expires_at')::TIMESTAMPTZ,
                    prior_score.lease_expires_at)
                AND host_error.occurred_at BETWEEN pg_catalog.date_trunc('second',
                  (prior_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
                    -pg_catalog.make_interval(secs=>(v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER))
                  AND COALESCE((prior_score.terminal_doc->>'original_lease_expires_at')::TIMESTAMPTZ,
                    prior_score.lease_expires_at)
            ))$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname)
  INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target;
  IF definition IS NULL OR identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Bounded score handoff security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='584a2b835f0d74ede019f4076b7000c9a3e8e681342e5e6d356c7fb90c8c7ad1' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')<>'ccf306a3d6fddbe92f22f90cc2c0c401847eb4d80eb3a160cf06fa6ba6aa9987'
    OR (length(definition)-length(replace(definition,old_gate,'')))<>length(old_gate) THEN
    RAISE EXCEPTION 'Bounded score handoff preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=replace(definition,old_gate,new_gate);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>'584a2b835f0d74ede019f4076b7000c9a3e8e681342e5e6d356c7fb90c8c7ad1' THEN
    RAISE EXCEPTION 'Bounded score handoff postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'584a2b835f0d74ede019f4076b7000c9a3e8e681342e5e6d356c7fb90c8c7ad1' THEN
    RAISE EXCEPTION 'Bounded score handoff readback differs' USING ERRCODE='55000';
  END IF;
END;
$bounded_score_handoff515$;
COMMIT;
