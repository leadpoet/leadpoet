-- Bound new judge admission by the frozen deadline; drain before safe failure.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $judge_final_deadline$
DECLARE d TEXT; metadata JSONB; target OID;
BEGIN
  target:='public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
      AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='v'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
      AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')<>'ed57dcfcf6ded84f9c8ae44ee5dc390e89afc8ed9bc142e5a5ce4d3b21b7dadf' THEN
    IF encode(extensions.digest(d,'sha256'),'hex')<>'17c0b8d87e6662d01c56d9c3c31d01f833b229e2e5f825d848efa2aca00d4106'
      OR (length(d)-length(replace(d,$old_claim$    AND runs.round_id = p_round_id
    AND (runs.kind<>'score' OR runs.scoring_generation_floor=v_round.scoring_generation_floor)$old_claim$,'')))<>length($old_claim$    AND runs.round_id = p_round_id
    AND (runs.kind<>'score' OR runs.scoring_generation_floor=v_round.scoring_generation_floor)$old_claim$) THEN
      RAISE EXCEPTION 'judge deadline claim source differs' USING ERRCODE='55000';
    END IF;
    d:=replace(d,$old_claim$    AND runs.round_id = p_round_id
    AND (runs.kind<>'score' OR runs.scoring_generation_floor=v_round.scoring_generation_floor)$old_claim$,$new_claim$    AND runs.round_id = p_round_id
    AND (runs.kind<>'score' OR runs.scoring_generation_floor=v_round.scoring_generation_floor)
    AND (runs.kind<>'score' OR pg_catalog.clock_timestamp() <
      v_publication_deadline::TIMESTAMPTZ)$new_claim$);
    IF (length(d)-length(replace(d,$old_claim_decl$  v_claims_paused BOOLEAN;$old_claim_decl$,'')))<>length($old_claim_decl$  v_claims_paused BOOLEAN;$old_claim_decl$) THEN RAISE EXCEPTION 'judge deadline scalar anchor differs' USING ERRCODE='55000'; END IF;
    d:=replace(d,$old_claim_decl$  v_claims_paused BOOLEAN;$old_claim_decl$,$new_claim_decl$  v_claims_paused BOOLEAN;
  v_publication_deadline TEXT;$new_claim_decl$);
    IF (length(d)-length(replace(d,$old_claim_read$  WITH round_quarantined_runners AS MATERIALIZED ($old_claim_read$,'')))<>length($old_claim_read$  WITH round_quarantined_runners AS MATERIALIZED ($old_claim_read$) THEN RAISE EXCEPTION 'judge deadline scalar anchor differs' USING ERRCODE='55000'; END IF;
    d:=replace(d,$old_claim_read$  WITH round_quarantined_runners AS MATERIALIZED ($old_claim_read$,$new_claim_read$  -- Extract once; rejected wide queues must not detoast the frozen config per row.
  v_publication_deadline := v_round.configuration_doc->'schedule'->>'publication_deadline';
  WITH round_quarantined_runners AS MATERIALIZED ($new_claim_read$);
    EXECUTE d;
  END IF;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'ed57dcfcf6ded84f9c8ae44ee5dc390e89afc8ed9bc142e5a5ce4d3b21b7dadf' THEN
    RAISE EXCEPTION 'judge deadline claim readback differs' USING ERRCODE='55000';
  END IF;
  target:='public.lab_arena_cancel_round(text,text)'::regprocedure;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
      AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='v'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
      AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')<>'befdc48d4bf66284c0bdb45861caad294ad736060290eaa06c1b115191092f29' THEN
    IF encode(extensions.digest(d,'sha256'),'hex')<>'038b7462e8ae7702e6d21b098142e8fd626b5cadcb9adf1c4f1d858be1246364'
      OR (length(d)-length(replace(d,$old_cancel$  v_generation := v_round.stage_generation + 1;$old_cancel$,'')))<>length($old_cancel$  v_generation := v_round.stage_generation + 1;$old_cancel$) THEN
      RAISE EXCEPTION 'judge deadline cancel source differs' USING ERRCODE='55000';
    END IF;
    d:=replace(d,$old_cancel$  v_generation := v_round.stage_generation + 1;$old_cancel$,$new_cancel$  -- Deadline failure drains existing work; it never terminates a live call.
  -- The existing control -> round lock order serializes claims and admission.
  IF p_reason='scoring_incomplete' AND (
    EXISTS (SELECT 1 FROM public.lab_arena_runs AS active
      WHERE active.round_id=p_round_id AND active.status IN ('leased','submitted'))
    OR EXISTS (SELECT 1 FROM public.lab_arena_ledger AS opened
      WHERE opened.round_id=p_round_id
        AND opened.call_identity IS NOT NULL
        AND opened.entry_kind IN ('reservation','dispatch')
        AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger AS later
          WHERE later.call_identity=opened.call_identity AND later.entry_id>opened.entry_id))
  ) THEN
    RETURN pg_catalog.jsonb_build_object('status','retry','round_status',v_round.status,
      'reason','evaluation_draining');
  END IF;
  v_generation := v_round.stage_generation + 1;$new_cancel$);
    EXECUTE d;
  END IF;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'befdc48d4bf66284c0bdb45861caad294ad736060290eaa06c1b115191092f29' THEN
    RAISE EXCEPTION 'judge deadline cancel readback differs' USING ERRCODE='55000';
  END IF;
END;
$judge_final_deadline$;
COMMIT;
