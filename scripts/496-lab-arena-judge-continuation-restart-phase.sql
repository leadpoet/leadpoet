-- Restart drain must recognize the same unfinished judges as ordinary claims.
-- Reuse the existing full-input continuation proof. Preserve capture, owner,
-- generation, stopped-host, accounting, expiry and immutable-result fences.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $judge_continuation_drain496$
DECLARE
  v_target RECORD;
  v_patch RECORD;
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_metadata JSONB;
  v_oid OID;
BEGIN
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,
    p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_proc p
  JOIN pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=to_regprocedure(
    'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)');
  IF encode(extensions.digest(v_definition,'sha256'),'hex') IS DISTINCT FROM
       '89cfed833d6bdeb3c15b0677b155679c33cf524d0055cd8ccab56986138e3c0d'
    OR v_identity IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
      ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena judge continuation drain authority differs' USING ERRCODE='55000';
  END IF;
  FOR v_target IN SELECT * FROM (VALUES
    ('public.lab_arena_expire_leases(text)',
      '35cbc97d9d8390f2d208e36617e7684cfcad4e1d1a8296ca4fbb461cc625f594',
      '7835213f7d4991be60194c60589da3d18f62f05f8640136e7df2dfce1d8e256f',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}','v'),
    ('public.lab_arena__restart_drain_state_v1()',
      'f00bdd46ba168268a5541fa3f47baa5edbb9aa0db0e41b03e6420259b0e2c469',
      '3ed11092880c10bad608420b33c076773557a308288d39ca0a45df9a95990e5a',
      '{lab_arena_owner=X/lab_arena_owner}','s'),
    ('public.lab_arena_restart_quiescence_v1(text,text,bigint)',
      '052c937b72b622d3e627ffcb02a7364c98d479bbbc58c151129d0085658dce7b',
      'f11ba1674937d27e4a5378c2a4fe074f01611c17d361326393851d4414dfb349',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner,service_role=X/lab_arena_owner}','v')
  ) AS targets(signature,preimage,postimage,acl,volatility) LOOP
    v_oid:=to_regprocedure(v_target.signature);
    SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig),to_jsonb(p)-'prosrc'
    INTO v_definition,v_identity,v_metadata FROM pg_proc p
    JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=v_oid;
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner',v_target.acl,TRUE,v_target.volatility,
      ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'Arena judge continuation drain security shape differs: %',
        v_target.signature USING ERRCODE='55000';
    END IF;
    IF encode(extensions.digest(v_definition,'sha256'),'hex')=v_target.postimage THEN
      CONTINUE;
    END IF;
    IF encode(extensions.digest(v_definition,'sha256'),'hex')<>v_target.preimage THEN
      RAISE EXCEPTION 'Arena judge continuation drain preimage differs: %',
        v_target.signature USING ERRCODE='55000';
    END IF;
    v_updated:=v_definition;
    FOR v_patch IN SELECT * FROM (VALUES
      ('public.lab_arena_expire_leases(text)',
        'public.lab_arena__baseline_drain_phase_v1(v_round,candidate)',
        '(public.lab_arena__baseline_drain_phase_v1(v_round,candidate) OR public.lab_arena__judge_continuation_v1(v_round,candidate))',3),
      ('public.lab_arena_expire_leases(text)',
        'public.lab_arena__baseline_drain_phase_v1(v_round,v_run)',
        '(public.lab_arena__baseline_drain_phase_v1(v_round,v_run) OR public.lab_arena__judge_continuation_v1(v_round,v_run))',2),
      ('public.lab_arena__restart_drain_state_v1()',
        'public.lab_arena__baseline_drain_phase_v1(recovery_round,runs)',
        '(public.lab_arena__baseline_drain_phase_v1(recovery_round,runs) OR public.lab_arena__judge_continuation_v1(recovery_round,runs))',1),
      ('public.lab_arena_restart_quiescence_v1(text,text,bigint)',
        'public.lab_arena__baseline_recovery_phase_v1(round,overdue)',
        '(public.lab_arena__baseline_recovery_phase_v1(round,overdue) OR public.lab_arena__judge_continuation_v1(round,overdue))',1)
    ) AS patches(signature,old_text,new_text,expected_count)
    WHERE signature=v_target.signature LOOP
      IF (length(v_updated)-length(replace(v_updated,v_patch.old_text,'')))
           <>v_patch.expected_count*length(v_patch.old_text) THEN
        RAISE EXCEPTION 'Arena judge continuation drain anchor differs' USING ERRCODE='55000';
      END IF;
      v_updated:=replace(v_updated,v_patch.old_text,v_patch.new_text);
    END LOOP;
    IF encode(extensions.digest(v_updated,'sha256'),'hex')<>v_target.postimage THEN
      RAISE EXCEPTION 'Arena judge continuation drain postimage differs' USING ERRCODE='55000';
    END IF;
    EXECUTE v_updated;
    IF encode(extensions.digest(pg_get_functiondef(v_oid),'sha256'),'hex')<>v_target.postimage
      OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=v_oid)
        IS DISTINCT FROM v_metadata THEN
      RAISE EXCEPTION 'Arena judge continuation drain readback differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$judge_continuation_drain496$;
NOTIFY pgrst,'reload schema';
COMMIT;
