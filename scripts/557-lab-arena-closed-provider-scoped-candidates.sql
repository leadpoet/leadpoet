-- Scope the existing closed-cost candidate CTEs before materialization.
-- Keep original outer scope, latest heads, cursor order and every full proof.
-- No index, payer, billing, settlement, history or publication change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $closed_provider_scope557$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)';
  definition TEXT; updated TEXT; security_before JSONB; security_after JSONB;
  prerequisite RECORD;
BEGIN
  -- These exact full proofs make non-null call identity a necessary condition.
  -- Validate dependencies even on replay; changed eligibility needs review.
  FOR prerequisite IN SELECT * FROM (VALUES
    ('public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)','e347886710c2461366607941044af1123d092622cfb65ecae1c5b70a1a22f914'),
    ('public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)','67fcd9dc83d29491aa4608e069b6680ac652e85b93b1feac64cc01a72b1f93f8'),
    ('public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)','0b876ab01b0c9e88ca9dc4a1cfcd3eb9e8cc13dc453a2bd418593e5dd3a96f82'),
    ('public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)','d5885be424024b95f7aac98b175872857baa6b0856e407ee9108b2382581ff0d'),
    ('public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)','8a973077fb121ef9f01866b301aba0ec5dd05da8143f87644ad8e28afa530f49')
  ) AS required(signature,sha256) LOOP
    IF encode(extensions.digest(pg_get_functiondef(to_regprocedure(prerequisite.signature)),
        'sha256'),'hex') IS DISTINCT FROM prerequisite.sha256 THEN
      RAISE EXCEPTION 'Closed provider scope helper preimage differs: %',prerequisite.signature;
    END IF;
  END LOOP;
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(o.rolname,p.proacl::TEXT,
    p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO definition,security_before
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
    JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF definition IS NULL OR security_before IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'s',ARRAY['search_path=pg_catalog, public'],'sql') THEN
    RAISE EXCEPTION 'Closed provider scope security preimage differs';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='629daf05a1f6aa4415450d3e39b69186ffa1ad121182189ca295b71939abd289' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM
      'e8076fea79b802cbe4d91da25cd53e560059e609cbee419dad337b9240a879f1' THEN
    RAISE EXCEPTION 'Closed provider scope exact preimage differs';
  END IF;
  updated:=replace(definition,$old_uncertainties557$    SELECT entry_id, round_id, run_id, call_identity
    FROM public.lab_arena_ledger
    WHERE entry_kind = 'uncertain' AND provider = 'deepline'
$old_uncertainties557$,$new_uncertainties557$    SELECT uncertainty.entry_id, uncertainty.round_id, uncertainty.run_id,
      uncertainty.call_identity
    FROM public.lab_arena_ledger AS uncertainty
    JOIN public.lab_arena_rounds AS scoped_round
      ON scoped_round.round_id = uncertainty.round_id
    WHERE uncertainty.entry_kind = 'uncertain' AND uncertainty.provider = 'deepline'
      -- Every unchanged full helper requires an exact non-null call binding.
      AND uncertainty.call_identity IS NOT NULL
      AND scoped_round.status IN ('cancelled', 'published')
      AND scoped_round.configuration_doc ->> 'mode' = p_mode
      AND scoped_round.configuration_doc ->> 'network_name' = p_network_name
      AND scoped_round.configuration_doc ->> 'netuid' = p_netuid::TEXT
      AND (COALESCE(p_round_id, '') = '' OR scoped_round.round_id = p_round_id)
      AND (
        scoped_round.status = 'cancelled'
        OR scoped_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
             IN ('successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')
      )
      -- Preserve the same global latest-head fence before materialization.
      AND NOT EXISTS (
        SELECT 1 FROM public.lab_arena_ledger AS later
        WHERE later.call_identity = uncertainty.call_identity
          AND later.entry_id > uncertainty.entry_id
      )
$new_uncertainties557$);
  updated:=replace(updated,$old_runs557$    WHERE eligible_run.kind = 'score'
$old_runs557$,$new_runs557$    WHERE eligible_run.kind = 'score'
       -- The existing outer inner join already requires this membership.
       AND eligible_run.run_id IN (SELECT run_id FROM uncertainties)
$new_runs557$);
  updated:=replace(updated,'lab_arena_closed_provider_round_array_507',
    'lab_arena_closed_provider_scoped_candidates_557');
  IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM
      '629daf05a1f6aa4415450d3e39b69186ffa1ad121182189ca295b71939abd289' THEN
    RAISE EXCEPTION 'Closed provider scope target differs';
  END IF;
  EXECUTE updated;
  SELECT jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO security_after
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
    JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF encode(extensions.digest(pg_get_functiondef(signature::regprocedure),'sha256'),'hex')
      IS DISTINCT FROM '629daf05a1f6aa4415450d3e39b69186ffa1ad121182189ca295b71939abd289' OR security_after IS DISTINCT FROM security_before THEN
    RAISE EXCEPTION 'Closed provider scope readback differs';
  END IF;
END;
$closed_provider_scope557$;
NOTIFY pgrst,'reload schema';
COMMIT;
