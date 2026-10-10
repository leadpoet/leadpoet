-- Archived judgments cannot witness current-round publication authority.
-- Add only same-round binding to the two accepted-judge vetoes. Preserve
-- all current accepted authority, incomplete proofs, costs and deadlines.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
DO $publication_round_witness_460$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena__publication_execution_incomplete_v1(text,text)';
  v_preimage CONSTANT TEXT := '1d0b203f8e0b8d2ece0db3c288806e85dc310a69b9786fce27e0ca9c3d308ecf';
  v_postimage CONSTANT TEXT := '5fb513506ad7ecd78de3009bce3d004cbc8e9883af163fafb1934b7cebe904d7';
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
  v_old_first TEXT := $first_old$AND judge.kind = 'score'
        AND judge.status = 'accepted'$first_old$;
  v_new_first TEXT := $first_new$AND judge.kind = 'score'
        AND judge.round_id = p_round_id
        AND judge.status = 'accepted'$first_new$;
  v_old_last TEXT := $last_old$AND judge.kind = 'score'
             AND judge.scored_run_id = v_execution.run_id
             AND judge.status = 'accepted'$last_old$;
  v_new_last TEXT := $last_new$AND judge.kind = 'score'
             AND judge.round_id = p_round_id
             AND judge.scored_run_id = v_execution.run_id
             AND judge.status = 'accepted'$last_new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena publication round witness security shape differs';
  END IF;
  v_hash:=pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex');
  IF v_hash=v_postimage THEN RETURN; END IF;
  IF v_hash<>v_preimage THEN RAISE EXCEPTION 'Arena publication round witness preimage differs'; END IF;
  IF pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_old_first,''))<>pg_catalog.length(v_old_first)
     OR pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_old_last,''))<>pg_catalog.length(v_old_last) THEN
    RAISE EXCEPTION 'Arena publication round witness fragment differs';
  END IF;
  v_definition:=pg_catalog.replace(pg_catalog.replace(v_definition,v_old_first,v_new_first),v_old_last,v_new_last);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena publication round witness postimage differs';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena publication round witness readback differs';
  END IF;
END;
$publication_round_witness_460$;
COMMIT;
