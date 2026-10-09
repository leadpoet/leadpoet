-- Recover exact host-funded judge charges after an accepted score run.
-- The accepted score, publication snapshot, miner admission and unknown cost
-- stay unchanged. Extend only the existing host-only closed-billing predicate.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $accepted_host_score_billing$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)';
  v_before_hash CONSTANT TEXT := '38733983232f1317485a568349f8da76';
  v_after_hash CONSTANT TEXT := 'b096b9440dd66037b9f286632c5b27a9';
  v_old CONSTANT TEXT := $old$      AND runs.status = 'failed'
      AND runs.terminal_cause IN (
        'lease_expired', 'worker_lost', 'result_rejected', 'provider_error',
        'stage_closed', 'judge_error', 'judge_timeout'
      )$old$;
  v_new CONSTANT TEXT := $new$      AND (
        (runs.status = 'failed'
         AND runs.terminal_cause IN (
           'lease_expired', 'worker_lost', 'result_rejected', 'provider_error',
           'stage_closed', 'judge_error', 'judge_timeout'
         ))
        OR (runs.status = 'accepted' AND runs.terminal_cause = 'accepted')
      )$new$;
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_expected_identity CONSTANT JSONB := pg_catalog.jsonb_build_array(
    'lab_arena_owner', '{lab_arena_owner=X/lab_arena_owner}',
    TRUE, 's', ARRAY['search_path=pg_catalog, public']
  );
  v_granted_create BOOLEAN := FALSE;
BEGIN
  IF NOT pg_catalog.has_schema_privilege('lab_arena_owner', 'public', 'CREATE') THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
    v_granted_create := TRUE;
  END IF;
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(
           owner.rolname, p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig
         )
    INTO v_definition, v_identity
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM v_expected_identity THEN
    RAISE EXCEPTION 'accepted host score billing security shape changed';
  END IF;
  IF pg_catalog.md5(v_definition) = v_after_hash THEN
    -- Exact repeat after the first successful application.
    NULL;
  ELSIF pg_catalog.md5(v_definition) = v_before_hash THEN
    IF (pg_catalog.length(v_definition) - pg_catalog.length(
          pg_catalog.replace(v_definition, v_old, '')))
          / pg_catalog.length(v_old) <> 1 THEN
      RAISE EXCEPTION 'accepted host score billing preimage shape changed';
    END IF;
    v_updated := pg_catalog.replace(v_definition, v_old, v_new);
    IF pg_catalog.md5(v_updated) <> v_after_hash THEN
      RAISE EXCEPTION 'accepted host score billing postimage differs';
    END IF;
    EXECUTE v_updated;
  ELSE
    RAISE EXCEPTION 'accepted host score billing preimage differs';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(
           owner.rolname, p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig
         )
    INTO v_definition, v_identity
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = pg_catalog.to_regprocedure(v_signature);
  IF pg_catalog.md5(v_definition) <> v_after_hash
     OR v_identity IS DISTINCT FROM v_expected_identity THEN
    RAISE EXCEPTION 'accepted host score billing readback differs';
  END IF;
  IF v_granted_create THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
END;
$accepted_host_score_billing$;

NOTIFY pgrst, 'reload schema';
COMMIT;
