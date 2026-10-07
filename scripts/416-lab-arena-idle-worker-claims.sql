-- Use free physical slots while older model tails remain active.
-- Keep frozen inputs, physical concurrency, ownership and all retry guards.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $idle_worker_claims$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)';
  v_preimage CONSTANT TEXT :=
    '59a5e9dc26f9e3d5157beb251f55c08db46e020dd30103c8a38db18ee18d1992';
  v_postimage CONSTANT TEXT :=
    '836d8a6594e8c61dca9db084a5311f2a5bce86c1f94b137768aa07c7f0aba467';
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(owner.rolname, p.proacl::TEXT,
           p.prosecdef, p.provolatile, p.proconfig)
  INTO v_definition, v_identity
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
       pg_catalog.jsonb_build_array('lab_arena_owner',
         '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
         TRUE, 'v', ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena idle worker claim security shape differs'
      USING ERRCODE = '55000';
  END IF;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = v_postimage THEN
    RETURN;
  END IF;
  IF v_hash <> v_preimage THEN
    RAISE EXCEPTION 'Arena idle worker claim preimage differs'
      USING ERRCODE = '55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    $old$  v_model_limit := GREATEST(1, v_limit / v_benchmark_count);$old$,
    $new$  -- lab_arena_idle_worker_claims_v1: each live model holds at least one
  -- physical slot; admitting another model must not reserve its whole ICP bank.
  v_model_limit := v_limit;$new$);
  v_definition := pg_catalog.replace(v_definition,
    $old$    CASE WHEN v_round.status = 'stage1'
                  AND v_round.configuration_doc ->>
                        'parallel_twenty_icp_execution' = 'true'
                  AND runs.kind = 'execute'
      THEN (
        SELECT participant.ordinality$old$,
    $new$    CASE WHEN runs.kind = 'execute' AND (
        (v_round.status = 'stage1'
         AND v_round.configuration_doc ->> 'parallel_twenty_icp_execution' = 'true')
        OR (v_round.status = 'stage2'
            AND v_round.configuration_doc ->> 'execution_sequence_policy'
                = 'baseline_scored_first_v1')
      )
      THEN (
        SELECT participant.ordinality$new$);
  IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex')
       <> v_postimage THEN
    RAISE EXCEPTION 'Arena idle worker claim postimage differs'
      USING ERRCODE = '55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(
       pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE), 'sha256'), 'hex')
       <> v_postimage THEN
    RAISE EXCEPTION 'Arena idle worker claim readback differs'
      USING ERRCODE = '55000';
  END IF;
END;
$idle_worker_claims$;
COMMIT;
