-- Recover fully settled, explicitly abandoned execute leases through ordinary expiry.
-- Preserve migration 412, score recovery, ledger, retry bounds and claim quarantine.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $settled_execute_host_recovery$
DECLARE
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
  v_index INTEGER;
  v_preimage CONSTANT TEXT := '9735de28e3a59f9143546b3a12d5345fc1627ea8d9b39a80f8863177286ff654';
  v_postimage CONSTANT TEXT := '9aa15b0f1aa66e61a668c708384600954b28bef7ef530781b0a25da357d40c71';
  v_old CONSTANT TEXT[] := ARRAY[
$old0$  -- lab_arena_settled_score_host_recovery_v1: RuntimeHostError is raised
  -- only after runsc deletion and namespace cleanup succeed. Score socket
  -- requests cannot replay paid calls. The round/run locks also fence any
  -- queued reservation or dispatch against the old lease after recovery.$old0$,
$old1$      AND v_early_recovery_allowed AND candidate.kind = 'score'$old1$,
$old2$      AND v_round.status = 'stage' || candidate.stage::TEXT || '_scoring'$old2$,
$old3$    IF v_run.status = 'leased' AND v_run.kind = 'score'
       AND v_run.stage_generation = v_round.stage_generation
       AND v_round.status = 'stage' || v_run.stage::TEXT || '_scoring'$old3$,
$old4$              'recovery_reason', 'authenticated_settled_score_runtime_host_error'$old4$
  ];
  v_new CONSTANT TEXT[] := ARRAY[
$new0$  -- lab_arena_settled_execute_host_recovery_v1: extend the settled score
  -- path to execute leases. RuntimeHostError follows successful sandbox
  -- deletion and namespace cleanup for both kinds. The round/run locks fence
  -- queued reservation, dispatch and execute Responses retries after recovery.$new0$,
$new1$      AND v_early_recovery_allowed AND candidate.kind IN ('score', 'execute')$new1$,
$new2$      AND v_round.status = 'stage' || candidate.stage::TEXT
          || CASE candidate.kind WHEN 'score' THEN '_scoring' ELSE '' END$new2$,
$new3$    IF v_run.status = 'leased' AND v_run.kind IN ('score', 'execute')
       AND v_run.stage_generation = v_round.stage_generation
       AND v_round.status = ('stage' || v_run.stage::TEXT
           || CASE v_run.kind WHEN 'score' THEN '_scoring' ELSE '' END)$new3$,
$new4$              'recovery_reason', CASE v_run.kind
                WHEN 'score' THEN 'authenticated_settled_score_runtime_host_error'
                ELSE 'authenticated_settled_execute_runtime_host_error' END$new4$
  ];
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner
  WHERE p.oid='public.lab_arena_expire_leases(text)'::REGPROCEDURE;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena settled execute host recovery security shape differs'
      USING ERRCODE='55000';
  END IF;
  v_hash:=pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex');
  IF v_hash=v_postimage THEN RETURN; END IF;
  IF v_hash<>v_preimage THEN
    RAISE EXCEPTION 'Arena settled execute host recovery preimage differs'
      USING ERRCODE='55000';
  END IF;
  FOR v_index IN 1..pg_catalog.cardinality(v_old) LOOP
    IF (pg_catalog.length(v_definition)-pg_catalog.length(
         pg_catalog.replace(v_definition,v_old[v_index],'')))
         / pg_catalog.length(v_old[v_index]) <> 1 THEN
      RAISE EXCEPTION 'Arena settled execute host recovery anchor differs'
        USING ERRCODE='55000';
    END IF;
    v_definition:=pg_catalog.replace(v_definition,v_old[v_index],v_new[v_index]);
  END LOOP;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena settled execute host recovery postimage differs'
      USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
    'public.lab_arena_expire_leases(text)'::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena settled execute host recovery readback differs'
      USING ERRCODE='55000';
  END IF;
END;
$settled_execute_host_recovery$;
COMMIT;
