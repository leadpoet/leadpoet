-- Reject new score leases at the frozen scoring cutoff despite driver lag.
-- Preserve existing leases, exact claim replays, execute timing and all accounting.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $score_claim_cutoff458$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)';
  v_preimage CONSTANT TEXT := 'cd889cc33098309a32effa193f6eb389b26d6ac1beaa0f448b6a01b7edf2b16e';
  v_postimage CONSTANT TEXT := '4e5335081b6fb3f8007f9e3a74c61a84bee47c07f7260616c36cd39ede272621';
  v_definition TEXT;
  v_identity JSONB;
  v_old TEXT := $old$    -- lab_arena_stage_cutoff_drain_v1: this is admission, not an extension$old$;
  v_new TEXT := $new$    -- Frozen score windows gate admission even while the driver is delayed.
    AND (
      runs.kind <> 'score'
      OR pg_catalog.clock_timestamp() <
           (v_round.configuration_doc #>> ARRAY[
             'schedule', CASE v_stage WHEN 1 THEN 'stage_1_scoring_close'
               ELSE 'final_scoring_close' END
           ])::TIMESTAMPTZ
    )
    -- lab_arena_stage_cutoff_drain_v1: this is admission, not an extension$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena score-claim cutoff security shape differs';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena score-claim cutoff preimage differs';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena score-claim cutoff postimage differs';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena score-claim cutoff readback differs';
  END IF;
END;
$score_claim_cutoff458$;
COMMIT;
