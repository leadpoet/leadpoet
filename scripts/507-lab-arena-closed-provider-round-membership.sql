-- Compute closed-round membership once. A generic SQL-function plan otherwise
-- estimates one matching round and can reread the spilled uncertainty CTE for
-- every closed round. Keep all candidate, scope, proof and cursor predicates.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $closed_provider_round_array$
DECLARE
  signature CONSTANT TEXT :=
    'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)';
  definition TEXT; updated TEXT; security_before JSONB; security_after JSONB;
BEGIN
  SELECT pg_get_functiondef(p.oid), jsonb_build_array(o.rolname,p.proacl::TEXT,
    p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO definition,security_before
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
  JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF definition IS NULL OR security_before IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'s',ARRAY['search_path=pg_catalog, public'],'sql') THEN
    RAISE EXCEPTION 'Closed provider round-array security preimage differs';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='e8076fea79b802cbe4d91da25cd53e560059e609cbee419dad337b9240a879f1' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM 'a65adc61115dbb1fa7258bffcadc96218cf889534f3de189b82134e2c5effaf0' THEN
    RAISE EXCEPTION 'Closed provider round-array exact preimage differs';
  END IF;
  updated := replace(definition,$join$      JOIN public.lab_arena_rounds AS rounds
        ON rounds.round_id = uncertainty.round_id
$join$,'');
  updated := replace(updated,$before$      WHERE rounds.status IN ('cancelled', 'published')
        AND rounds.configuration_doc ->> 'mode' = p_mode
        AND rounds.configuration_doc ->> 'network_name' = p_network_name
        AND rounds.configuration_doc ->> 'netuid' = p_netuid::TEXT
        AND (COALESCE(p_round_id, '') = '' OR uncertainty.round_id = p_round_id)
        AND (
          rounds.status = 'cancelled'
          OR rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
               IN ('successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')
        )
$before$,$after$      WHERE uncertainty.round_id = ANY (ARRAY(
        SELECT rounds.round_id
        FROM public.lab_arena_rounds AS rounds
        WHERE rounds.status IN ('cancelled', 'published')
          AND rounds.configuration_doc ->> 'mode' = p_mode
          AND rounds.configuration_doc ->> 'network_name' = p_network_name
          AND rounds.configuration_doc ->> 'netuid' = p_netuid::TEXT
          AND (COALESCE(p_round_id, '') = '' OR rounds.round_id = p_round_id)
          AND (
            rounds.status = 'cancelled'
            OR rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
                 IN ('successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')
          )
      ))
$after$);
  updated := replace(updated,'lab_arena_closed_provider_candidate_query_436',
    'lab_arena_closed_provider_round_array_507');
  IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM 'e8076fea79b802cbe4d91da25cd53e560059e609cbee419dad337b9240a879f1' THEN
    RAISE EXCEPTION 'Closed provider round-array target differs';
  END IF;
  -- CREATE OR REPLACE retains the deployed owner, ACL and function settings.
  EXECUTE updated;
  SELECT jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO security_after
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
  JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF encode(extensions.digest(pg_get_functiondef(signature::regprocedure),'sha256'),'hex')
       IS DISTINCT FROM 'e8076fea79b802cbe4d91da25cd53e560059e609cbee419dad337b9240a879f1' OR security_after IS DISTINCT FROM security_before THEN
    RAISE EXCEPTION 'Closed provider round-array readback differs';
  END IF;
END;
$closed_provider_round_array$;

NOTIFY pgrst, 'reload schema';
COMMIT;
