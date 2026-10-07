-- Preserve pre-cutoff live baseline-first executions for one frozen host TTL.
-- Stop new execute claims/retries at the original cutoff. Pending work remains
-- infrastructure-incomplete; it does not become a zero or a completed result.
-- No frozen configuration, runtime, source, ICP, score, or accounting changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $stage_cutoff_drain_420$
DECLARE
  v_patch RECORD;
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
  v_old_count INTEGER := 0;
  v_new_count INTEGER := 0;
BEGIN
  -- Validate the whole coupled seam before changing any function.
  FOR v_patch IN SELECT * FROM (VALUES
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)', '836d8a6594e8c61dca9db084a5311f2a5bce86c1f94b137768aa07c7f0aba467', '990baa411ac35d075ded6a56f14c5bd23c72416f85706eab4a802e3f08f51b3e'),
    ('public.lab_arena_close_stage(text,smallint)', '3f2ba09c8f3a545496033289aff1029396919f068fbccf31acb5f24b3f8237fd', '75f7b992f8a71091fceef9f6cb72c209eca5a54381e3fc50cc8bb0462aeaad10'),
    ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)', '0f3242b7d24e066c0acf1ab56d7891102a5d3956f9a13733c8d3025f55e45513', '6e71deaba402ae37b0676ab0c0347d23679c621ac21efd7a3d93835cfcf72043'),
    ('public.lab_arena_expire_leases(text)', '823ce89127543ab7d59dd642903b4b5d9a5dd0bfb79fde412fb3a0963b8e81da', 'ef61f7bf17884267c7c126529148099e48591eb973e88ecb841eea4498439688')
  ) AS patches(signature,preimage,postimage)
  LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),
      pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
        p.prosecdef,p.provolatile,p.proconfig)
      INTO v_definition,v_identity FROM pg_catalog.pg_proc p
      JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
      WHERE p.oid=pg_catalog.to_regprocedure(v_patch.signature);
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM
      pg_catalog.jsonb_build_array('lab_arena_owner',
        '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
        TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'Arena stage-cutoff drain security shape differs';
    END IF;
    v_hash:=pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex');
    IF v_hash=v_patch.preimage THEN v_old_count:=v_old_count+1;
    ELSIF v_hash=v_patch.postimage THEN v_new_count:=v_new_count+1;
    ELSE RAISE EXCEPTION 'Arena stage-cutoff drain preimage differs';
    END IF;
  END LOOP;
  IF v_new_count=4 THEN RETURN; END IF;
  IF v_old_count<>4 THEN RAISE EXCEPTION 'Arena stage-cutoff drain partial state differs'; END IF;
  FOR v_patch IN SELECT * FROM (VALUES
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)', '836d8a6594e8c61dca9db084a5311f2a5bce86c1f94b137768aa07c7f0aba467', '990baa411ac35d075ded6a56f14c5bd23c72416f85706eab4a802e3f08f51b3e',
     $old0$    AND runs.status = 'pending'
    AND runs.stage_generation = v_round.stage_generation$old0$,
     $new0$    AND runs.status = 'pending'
    -- lab_arena_stage_cutoff_drain_v1: this is admission, not an extension
    -- of any model's frozen runtime or retry allowance.
    AND (
      runs.kind <> 'execute'
      OR v_round.configuration_doc ->> 'execution_sequence_policy'
           IS DISTINCT FROM 'baseline_scored_first_v1'
      OR pg_catalog.clock_timestamp() <
           (v_round.configuration_doc #>> ARRAY[
             'schedule', 'stage_' || v_stage::TEXT || '_close'
           ])::TIMESTAMPTZ
    )
    AND runs.stage_generation = v_round.stage_generation$new0$),
    ('public.lab_arena_close_stage(text,smallint)', '3f2ba09c8f3a545496033289aff1029396919f068fbccf31acb5f24b3f8237fd', '75f7b992f8a71091fceef9f6cb72c209eca5a54381e3fc50cc8bb0462aeaad10',
     $old1$  v_generation := v_round.stage_generation + 1;$old1$,
     $new1$  -- lab_arena_stage_cutoff_drain_v1: stop admission at the frozen cutoff,
  -- but let existing live execution leases drain for one frozen host TTL.
  -- The absolute bound never follows settlement-renewed lease_expires_at.
  -- No row, generation, result, or accounting entry changes while draining.
  IF v_round.configuration_doc ->> 'execution_sequence_policy'
       = 'baseline_scored_first_v1'
     AND (v_round.configuration_doc ->> 'checkpoint_deadline_policy',
          v_round.configuration_doc ->> 'icp_wall_clock_seconds',
          v_round.configuration_doc ->> 'lease_ttl_seconds') IN (
       ('atomic_checkpoint_45m_v1', '2700', '3600'),
       ('atomic_checkpoint_60m_v1', '3600', '4500'),
       ('atomic_checkpoint_90m_v1', '5400', '6300')
     )
     AND pg_catalog.clock_timestamp() >=
          (v_round.configuration_doc #>> ARRAY[
            'schedule', 'stage_' || p_stage::TEXT || '_close'
          ])::TIMESTAMPTZ
     AND pg_catalog.clock_timestamp() <
          (v_round.configuration_doc #>> ARRAY[
            'schedule', 'stage_' || p_stage::TEXT || '_close'
          ])::TIMESTAMPTZ + pg_catalog.make_interval(
            secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER)
     AND EXISTS (
       SELECT 1 FROM public.lab_arena_runs AS active
       WHERE active.round_id=p_round_id AND active.stage=p_stage
         AND active.kind='execute' AND active.status='leased'
         AND active.stage_generation=v_round.stage_generation
         AND active.lease_expires_at > pg_catalog.clock_timestamp()
     ) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'draining', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation
    );
  END IF;
  v_generation := v_round.stage_generation + 1;$new1$),
    ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)', '0f3242b7d24e066c0acf1ab56d7891102a5d3956f9a13733c8d3025f55e45513', '6e71deaba402ae37b0676ab0c0347d23679c621ac21efd7a3d93835cfcf72043',
     $old2$    IF v_run.stage_generation = v_round.stage_generation THEN$old2$,
     $new2$    IF v_run.stage_generation = v_round.stage_generation
       AND (
         v_run.kind <> 'execute'
         OR v_round.configuration_doc ->> 'execution_sequence_policy'
              IS DISTINCT FROM 'baseline_scored_first_v1'
         OR pg_catalog.clock_timestamp() <
              (v_round.configuration_doc #>> ARRAY[
                'schedule', 'stage_' || v_run.stage::TEXT || '_close'
              ])::TIMESTAMPTZ
       ) THEN$new2$),
    ('public.lab_arena_expire_leases(text)', '823ce89127543ab7d59dd642903b4b5d9a5dd0bfb79fde412fb3a0963b8e81da', 'ef61f7bf17884267c7c126529148099e48591eb973e88ecb841eea4498439688',
     $old3$AND v_run.stage_generation = v_round.stage_generation THEN$old3$,
     $new3$AND v_run.stage_generation = v_round.stage_generation
       AND (
         v_run.kind <> 'execute'
         OR v_round.configuration_doc ->> 'execution_sequence_policy'
              IS DISTINCT FROM 'baseline_scored_first_v1'
         OR pg_catalog.clock_timestamp() <
              (v_round.configuration_doc #>> ARRAY[
                'schedule', 'stage_' || v_run.stage::TEXT || '_close'
              ])::TIMESTAMPTZ
       ) THEN$new3$)
  ) AS patches(signature,preimage,postimage,old_fragment,new_fragment)
  LOOP
    v_definition:=pg_catalog.pg_get_functiondef(v_patch.signature::REGPROCEDURE);
    v_definition:=pg_catalog.replace(v_definition,v_patch.old_fragment,v_patch.new_fragment);
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_patch.postimage THEN
      RAISE EXCEPTION 'Arena stage-cutoff drain postimage differs';
    END IF;
    EXECUTE v_definition;
    IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_patch.signature::REGPROCEDURE),'sha256'),'hex')<>v_patch.postimage THEN
      RAISE EXCEPTION 'Arena stage-cutoff drain readback differs';
    END IF;
  END LOOP;
END;
$stage_cutoff_drain_420$;
COMMIT;
