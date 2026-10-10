-- Canonical quiescence may recover its authenticated stopped score hosts.
-- Ordinary operator holds, provider liabilities and captured lease identity remain.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $stopped_score_drain454$
DECLARE
  v_expiry TEXT;
  v_quiescence TEXT;
  v_drain TEXT;
  v_before TEXT[];
  v_after TEXT[];
  v_identity JSONB;
  v_signatures TEXT[] := ARRAY['public.lab_arena_expire_leases(text)',
    'public.lab_arena_restart_quiescence_v1(text,text,bigint)',
    'public.lab_arena__restart_drain_state_v1()'];
  v_preimages TEXT[] := ARRAY['1539836b593cee82a8e9ac237199208464da5aa5bd960e9179275c9f51387f97',
    '94c8c23deeccd6b89bb575b0d2782709b0c030810e00bef5691b254b5bb84d35',
    'b9cfb9b59ef428d8e43b052258cac72f1fff3d141a736080c8012d644211497b'];
  v_postimages TEXT[] := ARRAY['dadb3e139bb972ca8d3ef4894e0c2e3ba087f55c57973adaa21ed1a8d46d6c0a','9e7334a5d2d4d05e673351627061a84558fc865585e479bd2b35231b405e4abc','8d4147a0ade1685f14fbdda3af24106384bb6173f375cac417dc099ab627b943'];
  v_i INTEGER;
  v_paid_start INTEGER;
  v_proof_start INTEGER;
  v_proof_end INTEGER;
  v_proof TEXT;
  v_anchor TEXT;
BEGIN
  FOR v_i IN 1..3 LOOP
    SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::text,
      p.prosecdef,p.provolatile,p.proconfig)
    INTO v_anchor,v_identity FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
    WHERE p.oid=v_signatures[v_i]::regprocedure;
    IF v_anchor IS NULL OR v_identity->>0 IS DISTINCT FROM 'lab_arena_owner'
      OR v_identity->>1 IS DISTINCT FROM (CASE WHEN v_i=3
        THEN '{lab_arena_owner=X/lab_arena_owner}'
        WHEN v_i=2 THEN '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner,service_role=X/lab_arena_owner}'
        ELSE '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' END)
      OR v_identity->2 IS DISTINCT FROM 'true'::jsonb
      OR v_identity->>3 IS DISTINCT FROM (CASE WHEN v_i=3 THEN 's' ELSE 'v' END)
      OR v_identity->4 IS DISTINCT FROM '["search_path=pg_catalog, public"]'::jsonb THEN
      RAISE EXCEPTION 'Arena stopped score drain security shape differs' USING ERRCODE='55000';
    END IF;
    v_before := array_append(v_before,v_anchor);
  END LOOP;
  IF ARRAY(SELECT encode(extensions.digest(x,'sha256'),'hex') FROM unnest(v_before) x)=v_postimages THEN RETURN; END IF;
  IF ARRAY(SELECT encode(extensions.digest(x,'sha256'),'hex') FROM unnest(v_before) x)<>v_preimages THEN
    RAISE EXCEPTION 'Arena stopped score drain preimage differs' USING ERRCODE='55000';
  END IF;
  v_expiry:=v_before[1]; v_quiescence:=v_before[2]; v_drain:=v_before[3];
  -- Reuse the entire production441/450 proof for durable drain classification.
  -- The only receipt substitutions are terminal status and original expiry.
  v_paid_start:=strpos(v_expiry,'  -- lab_arena_settled_execute_host_recovery_v1:');
  v_proof_start:=strpos(substr(v_expiry,v_paid_start),'    IF v_run.status = ''leased''')+v_paid_start-1;
  v_proof_end:=strpos(substr(v_expiry,v_proof_start),E' THEN\n      UPDATE public.lab_arena_runs')+v_proof_start-1;
  IF v_paid_start=0 OR v_proof_start<v_paid_start OR v_proof_end<v_proof_start THEN
    RAISE EXCEPTION 'Arena stopped score drain proof anchor differs' USING ERRCODE='55000';
  END IF;
  v_proof:=substr(v_expiry,v_proof_start+7,v_proof_end-v_proof_start-7);
  v_proof:=replace(v_proof,'v_run.status = ''leased''','v_run.status = ''failed''');
  v_proof:=replace(v_proof,'v_run.lease_expires_at >=',
    '(v_run.terminal_doc ->> ''original_lease_expires_at'')::timestamptz >=');
  v_proof:=replace(replace(v_proof,'v_run.','runs.'),'v_round.','recovery_round.');
  v_expiry:=replace(v_expiry,'  v_early_recovery_allowed BOOLEAN;',
$declarations$  v_early_recovery_allowed BOOLEAN;
  v_recovery_context JSONB;
  v_recovery_control public.lab_arena_restart_claim_control;
  v_captured_recovery_allowed BOOLEAN := FALSE;$declarations$);
  v_anchor:=E'  SELECT * INTO v_round';
  v_expiry:=replace(v_expiry,v_anchor,
$authorization$  -- lab_arena_owner_authorized_stopped_score_drain_v1
  -- A setting is not authority: verify both secret commitments and generation.
  v_recovery_context := nullif(current_setting('lab_arena.restart_recovery_owner',TRUE),'')::jsonb;
  SELECT * INTO v_recovery_control FROM public.lab_arena_restart_claim_control WHERE singleton;
  v_captured_recovery_allowed := COALESCE(
    v_recovery_control.restart_phase='draining'
    AND v_recovery_control.guard_expires_at>clock_timestamp()
    AND v_recovery_control.guard_commitment='sha256:'||encode(extensions.digest(
      convert_to(v_recovery_context->>'guard_id','UTF8'),'sha256'),'hex')
    AND v_recovery_control.owner_commitment='sha256:'||encode(extensions.digest(
      convert_to(v_recovery_context->>'owner_id','UTF8'),'sha256'),'hex')
    AND v_recovery_context->'guard_generation'=to_jsonb(v_recovery_control.guard_generation),FALSE);
  SELECT * INTO v_round$authorization$);
  v_expiry:=replace(v_expiry,
    '      AND v_early_recovery_allowed AND candidate.kind IN (''score'', ''execute'')',
$captured$      AND (v_early_recovery_allowed OR (v_captured_recovery_allowed
        AND candidate.kind='score' AND EXISTS (
          SELECT 1 FROM jsonb_array_elements(v_recovery_control.captured_leases) item
          WHERE item->>'run_id'=candidate.run_id
            AND item->'lease_generation'=to_jsonb(candidate.lease_generation))))
      AND candidate.kind IN ('score', 'execute')$captured$);
  v_expiry:=replace(v_expiry,'  -- lab_arena_settled_execute_host_recovery_v1:',
$write_set$  -- The existing whole-round expiry must never touch foreign captured work.
  IF v_captured_recovery_allowed THEN
    PERFORM 1 FROM public.lab_arena_runs WHERE round_id=p_round_id AND status='leased'
      ORDER BY run_id FOR UPDATE;
    IF EXISTS (SELECT 1 FROM public.lab_arena_runs candidate
      WHERE candidate.round_id=p_round_id AND candidate.status='leased'
        AND (candidate.kind<>'score' OR candidate.stage_generation<>v_round.stage_generation
          OR v_round.status<>'stage'||candidate.stage::text||'_scoring'
          OR candidate.result_doc IS NOT NULL OR candidate.output_ref IS NOT NULL
          OR NOT EXISTS (SELECT 1 FROM jsonb_array_elements(v_recovery_control.captured_leases) item
            WHERE item->>'run_id'=candidate.run_id
              AND item->'lease_generation'=to_jsonb(candidate.lease_generation)))) THEN
      RAISE EXCEPTION 'Arena stopped score drain captured write set differs' USING ERRCODE='55000';
    END IF;
    -- Preserve404: a whole-round call cannot naturally expire in-flight work.
    IF EXISTS (SELECT 1 FROM public.lab_arena_runs candidate
      WHERE candidate.round_id=p_round_id AND candidate.status='leased'
        AND candidate.lease_expires_at<=clock_timestamp()
        AND (EXISTS (SELECT 1 FROM (
          SELECT DISTINCT ON (cost.call_identity) cost.entry_kind
          FROM public.lab_arena_ledger cost WHERE cost.run_id=candidate.run_id
            AND cost.call_identity IS NOT NULL
          ORDER BY cost.call_identity,cost.entry_id DESC
        ) heads WHERE heads.entry_kind IN ('reservation','dispatch'))
        OR EXISTS (SELECT 1 FROM public.lab_arena_ledger cost
          WHERE cost.run_id=candidate.run_id AND cost.call_identity IS NULL))) THEN
      RETURN jsonb_build_object('status','ok','expired',0,'retried',0);
    END IF;
  END IF;
  -- lab_arena_settled_execute_host_recovery_v1:$write_set$);
  v_quiescence:=replace(v_quiescence,'  v_round_id TEXT;',
    E'  v_round_id TEXT;\n  v_prior_recovery_context TEXT;');
  v_quiescence:=replace(v_quiescence,'  -- lab_arena_guard_owned_natural_expiry_v1',
$quiescence$  -- Authenticated owner context exists only around the existing expiry call.
  IF v_control.restart_phase='draining' AND v_control.guard_expires_at>clock_timestamp() THEN
    v_prior_recovery_context:=current_setting('lab_arena.restart_recovery_owner',TRUE);
    BEGIN
      PERFORM set_config('lab_arena.restart_recovery_owner',jsonb_build_object(
        'guard_id',p_guard_id,'owner_id',p_owner_id,'guard_generation',p_guard_generation)::text,TRUE);
      FOR v_round_id IN SELECT DISTINCT runs.round_id
        FROM jsonb_array_elements(v_control.captured_leases) item
        JOIN public.lab_arena_runs runs ON runs.run_id=item->>'run_id'
        WHERE runs.lease_generation=(item->>'lease_generation')::bigint
          AND runs.status='leased' AND runs.kind='score'
      LOOP
        PERFORM public.lab_arena_expire_leases(v_round_id);
      END LOOP;
      PERFORM set_config('lab_arena.restart_recovery_owner',coalesce(v_prior_recovery_context,''),TRUE);
    EXCEPTION WHEN OTHERS THEN
      PERFORM set_config('lab_arena.restart_recovery_owner',coalesce(v_prior_recovery_context,''),TRUE);
      RAISE;
    END;
  END IF;
  -- lab_arena_guard_owned_natural_expiry_v1$quiescence$);
  v_anchor:=$receipt$        ELSE 'lost'$receipt$;
  v_drain:=replace(v_drain,v_anchor,
$receipt$        -- Authenticated early recovery retains the original frozen claim.
        WHEN runs.status='failed' AND runs.terminal_cause='lease_expired'
          AND runs.kind='score' AND runs.result_doc IS NULL AND runs.output_ref IS NULL
          AND runs.claim_response->'lease_generation'=to_jsonb(captured.lease_generation)
          AND runs.terminal_doc=jsonb_build_object(
            'original_lease_expires_at',runs.terminal_doc->>'original_lease_expires_at',
            'recovery_reason',runs.terminal_doc->>'recovery_reason',
            'expired_at',runs.terminal_doc->>'expired_at')
          AND runs.terminal_doc->>'recovery_reason' IN (
            'authenticated_settled_score_runtime_host_error',
            'authenticated_completed_unpriced_score_runtime_host_error')
          AND (runs.terminal_doc->>'expired_at')::timestamptz>=runs.lease_expires_at
          AND (runs.terminal_doc->>'expired_at')::timestamptz<=clock_timestamp()
          AND (runs.terminal_doc->>'original_lease_expires_at')::timestamptz>runs.lease_expires_at
          AND EXISTS (SELECT 1 FROM public.lab_arena_rounds recovery_round
            WHERE recovery_round.round_id=runs.round_id AND ($receipt$ || v_proof || $receipt$)) THEN 'expired'
        ELSE 'lost'$receipt$);
  v_after:=ARRAY[v_expiry,v_quiescence,v_drain];
  FOR v_i IN 1..3 LOOP
    EXECUTE v_after[v_i];
    IF encode(extensions.digest(pg_get_functiondef(v_signatures[v_i]::regprocedure),'sha256'),'hex')<>v_postimages[v_i] THEN
      RAISE EXCEPTION 'Arena stopped score drain postimage differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$stopped_score_drain454$;
NOTIFY pgrst,'reload schema';
COMMIT;
