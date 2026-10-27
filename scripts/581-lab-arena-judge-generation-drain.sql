-- Keep captured judges valid across a round stage transition only when the
-- existing immutable-input continuation proof still accepts them. Preserve
-- owner, capture, claim, accounting, expiry, retry and accepted-result guards.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $judge_generation_drain581$
DECLARE
  target RECORD; patch RECORD; definition TEXT; updated TEXT;
  identity JSONB; metadata JSONB; function_oid OID;
BEGIN
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,
    p.prosecdef,p.provolatile,p.proconfig) INTO definition,identity
  FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=to_regprocedure(
    'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)');
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM
    '3a0ff0eb9209d66f766daaee4eebab92a81f5f1d05e5d189130d7ec1dc69b0af'
    OR identity IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'judge generation drain authority differs' USING ERRCODE='55000';
  END IF;
  FOR target IN SELECT * FROM (VALUES
    ('public.lab_arena__restart_drain_state_v1()','6de3a3673357bbe823992075fd00443247e1b19c50e7cc3d01307ea381eab984','c1d1a09cd55369d53d8d845a581d8f5470e7cc06ada550861b78ff9874011b4c','{lab_arena_owner=X/lab_arena_owner}','s'),
    ('public.lab_arena_expire_leases(text)','f8b2d491b97404f6dbea55cedeac2fb277897b9bdd7d67447b4fd43e56340a83','b14f3018d626ae76fdd4151d9106161a89cfc264ded814c7abb7d49c48da42c4','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}','v'),
    ('public.lab_arena_restart_quiescence_v1(text,text,bigint)','155bdb39497e08055e51ff57c2c87e4b1efac8888f2202fcbc75e2989fa8a501','c75f61711ea080fc2e50a0c68207635882129e6757cceb1211dc79a45f893d9a','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner,service_role=X/lab_arena_owner}','v')
  ) AS targets(signature,preimage,postimage,acl,volatility) LOOP
    function_oid:=to_regprocedure(target.signature);
    SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig),to_jsonb(p)-'prosrc'
    INTO definition,identity,metadata FROM pg_proc p
    JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=function_oid;
    IF definition IS NULL OR identity IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner',target.acl,TRUE,target.volatility,ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'judge generation drain security differs' USING ERRCODE='55000';
    END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')<>target.preimage THEN
      RAISE EXCEPTION 'judge generation drain preimage differs' USING ERRCODE='55000';
    END IF;
    updated:=definition;
    FOR patch IN SELECT * FROM (VALUES
    ('public.lab_arena__restart_drain_state_v1()',$old$       AND runs.stage_generation = recovery_round.stage_generation$old$,$new$       AND (runs.stage_generation = recovery_round.stage_generation
         OR (runs.stage_generation < recovery_round.stage_generation
           AND public.lab_arena__judge_continuation_v1(recovery_round,runs)))$new$),
    ('public.lab_arena_expire_leases(text)',$old$candidate.stage_generation<>v_round.stage_generation$old$,$new$(candidate.stage_generation<>v_round.stage_generation AND NOT (candidate.stage_generation<v_round.stage_generation AND public.lab_arena__judge_continuation_v1(v_round,candidate)))$new$),
    ('public.lab_arena_expire_leases(text)',$old$      AND candidate.stage_generation = v_round.stage_generation
      AND (v_round.status$old$,$new$      AND (candidate.stage_generation = v_round.stage_generation
        OR (candidate.stage_generation < v_round.stage_generation
          AND public.lab_arena__judge_continuation_v1(v_round,candidate)))
      AND (v_round.status$new$),
    ('public.lab_arena_expire_leases(text)',$old$       AND v_run.stage_generation = v_round.stage_generation
       AND (v_round.status$old$,$new$       AND (v_run.stage_generation = v_round.stage_generation
         OR (v_run.stage_generation < v_round.stage_generation
           AND public.lab_arena__judge_continuation_v1(v_round,v_run)))
       AND (v_round.status$new$),
    ('public.lab_arena_restart_quiescence_v1(text,text,bigint)',$old$overdue.stage_generation<>round.stage_generation$old$,$new$(overdue.stage_generation<>round.stage_generation AND NOT (overdue.stage_generation<round.stage_generation AND public.lab_arena__judge_continuation_v1(round,overdue)))$new$)
    ) AS patches(signature,old_text,new_text) WHERE signature=target.signature LOOP
      IF length(updated)-length(replace(updated,patch.old_text,''))<>length(patch.old_text) THEN
        RAISE EXCEPTION 'judge generation drain anchor differs' USING ERRCODE='55000';
      END IF;
      updated:=replace(updated,patch.old_text,patch.new_text);
    END LOOP;
    IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
      RAISE EXCEPTION 'judge generation drain postimage differs' USING ERRCODE='55000';
    END IF;
    EXECUTE updated;
    IF encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage
      OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata THEN
      RAISE EXCEPTION 'judge generation drain readback differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$judge_generation_drain581$;
NOTIFY pgrst,'reload schema';
COMMIT;
