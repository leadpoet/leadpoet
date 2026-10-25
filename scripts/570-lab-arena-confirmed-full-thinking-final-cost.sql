-- Frozen full-thinking coverage rounds use confirmed competition cost at final
-- eligibility. Open calls still block publication; terminal uncertainty stays
-- in the ledger and makes its ICP unavailable for strict training export.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $confirmed_full_thinking_cost569$
DECLARE
  v_signature REGPROCEDURE := 'public.lab_arena_icp_cost_eligibility(text,text,integer,integer)'::REGPROCEDURE;
  v_source TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_after_identity JSONB;
  v_old_spend TEXT := $old_spend$  v_spend := COALESCE(v_state ->> 'competition_successful_microusd',v_state ->> 'successful_microusd')::BIGINT;$old_spend$;
  v_new_spend TEXT := $new_spend$  v_spend := CASE WHEN v_round.configuration_doc ->> 'reasoning_capture_policy'
       = 'full_generated_thinking_v1'
     AND pg_catalog.jsonb_typeof(v_round.configuration_doc -> 'training_minimum_icp_coverage_percent') = 'number'
     AND v_round.configuration_doc ->> 'training_minimum_icp_coverage_percent'
       ~ '^(100|[1-9][0-9]?)$'
    THEN COALESCE(v_state ->> 'competition_settled_microusd',v_state ->> 'settled_microusd')::BIGINT
    ELSE COALESCE(v_state ->> 'competition_successful_microusd',v_state ->> 'successful_microusd')::BIGINT END;
  -- lab_arena_confirmed_full_thinking_spend_569$new_spend$;
  v_old TEXT := $old$  ELSIF (v_state ->> 'success_unresolved_calls')::BIGINT > 0 THEN
    v_reason := 'provider_cost_uncertain';$old$;
  v_new TEXT := $new$  ELSIF (v_state ->> 'success_unresolved_calls')::BIGINT > 0
     AND NOT COALESCE((v_round.configuration_doc ->> 'reasoning_capture_policy'
       = 'full_generated_thinking_v1'
       AND pg_catalog.jsonb_typeof(v_round.configuration_doc -> 'training_minimum_icp_coverage_percent') = 'number'
       AND v_round.configuration_doc ->> 'training_minimum_icp_coverage_percent'
         ~ '^(100|[1-9][0-9]?)$'),FALSE) THEN
    v_reason := 'provider_cost_uncertain';
  -- lab_arena_confirmed_full_thinking_final_cost_569$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
           p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO v_source,v_identity FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_language l ON l.oid=p.prolang WHERE p.oid=v_signature;
  IF v_source IS NULL OR v_identity IS DISTINCT FROM
      '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Arena confirmed final cost security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.strpos(v_source,'lab_arena_confirmed_full_thinking_final_cost_569') > 0 THEN
    IF pg_catalog.strpos(v_source,v_new) = 0 OR pg_catalog.strpos(v_source,v_old) > 0
       OR pg_catalog.strpos(v_source,v_new_spend) = 0
       OR pg_catalog.strpos(v_source,v_old_spend) > 0 THEN
      RAISE EXCEPTION 'Arena confirmed final cost patch differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_source,'sha256'),'hex')
       IS DISTINCT FROM '97b5b528c0f81f902f7ce1ed6e0390dbce2822d0224a60b28b6cab9397f353a6'
     OR pg_catalog.length(v_source)-pg_catalog.length(pg_catalog.replace(v_source,v_old,''))
        <> pg_catalog.length(v_old)
     OR pg_catalog.length(v_source)-pg_catalog.length(pg_catalog.replace(v_source,v_old_spend,''))
        <> pg_catalog.length(v_old_spend) THEN
    RAISE EXCEPTION 'Arena confirmed final cost exact preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated := pg_catalog.replace(pg_catalog.replace(v_source,v_old_spend,v_new_spend),v_old,v_new);
  EXECUTE v_updated;
  SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
           p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO v_after_identity FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_language l ON l.oid=p.prolang WHERE p.oid=v_signature;
  IF v_after_identity IS DISTINCT FROM v_identity
     OR pg_catalog.pg_get_functiondef(v_signature) IS DISTINCT FROM v_updated THEN
    RAISE EXCEPTION 'Arena confirmed final cost readback differs' USING ERRCODE='55000';
  END IF;
END;
$confirmed_full_thinking_cost569$;
DO $confirmed_full_thinking_baseline569$
DECLARE
  v_signature REGPROCEDURE := 'public.lab_arena__baseline_final_ready_v1(text)'::REGPROCEDURE;
  v_source TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_after_identity JSONB;
  v_old TEXT := $old$  IF v_round.configuration_doc->>'sourcing_cost_eligibility_policy'
       IN ('successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1') THEN
    IF COALESCE((v_cost->>'success_unresolved_calls')::BIGINT,-1) <> 0 THEN RETURN FALSE; END IF;
  ELSE
    IF COALESCE((v_cost->>'uncertain_calls')::BIGINT,-1) <> 0 THEN RETURN FALSE; END IF;
  END IF;$old$;
  v_new TEXT := $new$  IF v_round.configuration_doc ->> 'reasoning_capture_policy'
       = 'full_generated_thinking_v1'
     AND pg_catalog.jsonb_typeof(v_round.configuration_doc -> 'training_minimum_icp_coverage_percent') = 'number'
     AND v_round.configuration_doc ->> 'training_minimum_icp_coverage_percent'
       ~ '^(100|[1-9][0-9]?)$' THEN
    -- lab_arena_confirmed_full_thinking_baseline_569: terminal uncertainty
    -- does not hold a fully scored baseline. The inflight guard above remains.
    NULL;
  ELSIF v_round.configuration_doc->>'sourcing_cost_eligibility_policy'
       IN ('successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1') THEN
    IF COALESCE((v_cost->>'success_unresolved_calls')::BIGINT,-1) <> 0 THEN RETURN FALSE; END IF;
  ELSE
    IF COALESCE((v_cost->>'uncertain_calls')::BIGINT,-1) <> 0 THEN RETURN FALSE; END IF;
  END IF;$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
           p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO v_source,v_identity FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_language l ON l.oid=p.prolang WHERE p.oid=v_signature;
  IF v_source IS NULL OR v_identity IS DISTINCT FROM
      '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Arena confirmed baseline security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.strpos(v_source,'lab_arena_confirmed_full_thinking_baseline_569') > 0 THEN
    IF pg_catalog.strpos(v_source,v_new) = 0 OR pg_catalog.strpos(v_source,v_old) > 0 THEN
      RAISE EXCEPTION 'Arena confirmed baseline patch differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_source,'sha256'),'hex')
       IS DISTINCT FROM '637c20c4b2a69741e72921797a78c7c94967e9cfcd8f8aae30d75e5dd0add52f'
     OR pg_catalog.length(v_source)-pg_catalog.length(pg_catalog.replace(v_source,v_old,''))
        <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION 'Arena confirmed baseline exact preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated := pg_catalog.replace(v_source,v_old,v_new);
  EXECUTE v_updated;
  SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
           p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO v_after_identity FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_language l ON l.oid=p.prolang WHERE p.oid=v_signature;
  IF v_after_identity IS DISTINCT FROM v_identity
     OR pg_catalog.pg_get_functiondef(v_signature) IS DISTINCT FROM v_updated THEN
    RAISE EXCEPTION 'Arena confirmed baseline readback differs' USING ERRCODE='55000';
  END IF;
END;
$confirmed_full_thinking_baseline569$;
NOTIFY pgrst,'reload schema';
COMMIT;
