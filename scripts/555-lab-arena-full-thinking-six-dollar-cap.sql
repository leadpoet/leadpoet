-- Only frozen full-generated-thinking rounds admit and score up to $6 per ICP.
-- Historical $4 flat rounds and $0.80 qualified-pair rounds keep their rules.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $full_thinking_cap555$
DECLARE
  item RECORD;
  source TEXT;
  updated TEXT;
  identity_before JSONB;
  identity_after JSONB;
  target OID;
  old_icp TEXT := $old_icp$IS DISTINCT FROM 4000000$old_icp$;
  new_icp TEXT := $new_icp$IS DISTINCT FROM (CASE WHEN v_round.configuration_doc ->> 'reasoning_capture_policy'
           = 'full_generated_thinking_v1' THEN 6000000 ELSE 4000000 END)$new_icp$;
  old_company TEXT := $old_company$= 'successful_calls_flat_per_icp_v1' THEN 4000000 ELSE 800000 END$old_company$;
  new_company TEXT := $new_company$= 'successful_calls_flat_per_icp_v1' THEN
             CASE WHEN v_round.configuration_doc ->> 'reasoning_capture_policy'
               = 'full_generated_thinking_v1' THEN 6000000 ELSE 4000000 END
           ELSE 800000 END$new_company$;
  old_admission TEXT := $old_admission$    IF COALESCE(v_icp_cost ->> 'competition_settled_microusd',v_icp_cost ->> 'settled_microusd')::BIGINT >=
         (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT THEN
      v_reason := 'money_cap';
    END IF;$old_admission$;
  new_admission TEXT := $new_admission$    IF v_run.kind='execute' AND v_round.configuration_doc ->> 'reasoning_capture_policy'
         = 'full_generated_thinking_v1' THEN
      -- The existing per-ICP advisory lock also serializes these holds.
      -- A settled head replaces its reservation, so unused room returns.
      -- Admission pays the actual host bill, including a failed retry.
      -- The competition eligibility function keeps its separate exemptions.
      v_locked_spend := (v_icp_cost ->> 'settled_microusd')::BIGINT
                        + (v_icp_cost ->> 'reserved_or_uncertain_microusd')::BIGINT;
      IF (v_dynamic OR p_amount_microusd > 0) AND
         (v_locked_spend >= (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
          OR (NOT v_dynamic AND p_amount_microusd >
             (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT - v_locked_spend)) THEN
        IF (v_icp_cost ->> 'inflight_calls')::BIGINT > 0
           OR (v_icp_cost ->> 'uncertain_calls')::BIGINT > 0 THEN
          v_expires := pg_catalog.clock_timestamp()
            + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
          UPDATE public.lab_arena_runs SET lease_expires_at=v_expires WHERE run_id=p_run_id;
          RETURN pg_catalog.jsonb_build_object(
            'status','budget_busy', 'idempotent',FALSE,
            'reason', CASE WHEN (v_icp_cost ->> 'uncertain_calls')::BIGINT > 0
                            THEN 'provider_cost_uncertain' ELSE 'provider_calls_inflight' END,
            'call_identity',p_call_identity,'lease_expires_at',v_expires);
        END IF;
        v_reason := 'money_cap';
      ELSIF v_dynamic THEN
        p_amount_microusd := (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
                              - v_locked_spend;
      END IF;
    ELSIF COALESCE(v_icp_cost ->> 'competition_settled_microusd',v_icp_cost ->> 'settled_microusd')::BIGINT >=
         (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT THEN
      v_reason := 'money_cap';
    END IF;$new_admission$;
  old_hold TEXT := $old_hold$     AND v_run.kind IN ('execute', 'score') THEN
    p_amount_microusd := 0;$old_hold$;
  new_hold TEXT := $new_hold$     AND v_run.kind IN ('execute', 'score')
     AND NOT (v_run.kind='execute' AND v_round.configuration_doc ->> 'reasoning_capture_policy'
              = 'full_generated_thinking_v1') THEN
    p_amount_microusd := 0;$new_hold$;
BEGIN
  FOR item IN SELECT * FROM (VALUES
    ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',
     '8cd8236c6f185956035c327357f5c7d7ea67bee9f83bbab8f7976c97e1b5b4f8',
     'ef9909b1d2c39a76e6630d228cd43a3560d9732e159dbdacd9e5de52c6393643'),
    ('public.lab_arena_icp_cost_eligibility(text,text,integer,integer)',
     'f190d9727c331c967966f8552a43e12f6cbafad8e8281be7d68749d5b8704571',
     '97b5b528c0f81f902f7ce1ed6e0390dbce2822d0224a60b28b6cab9397f353a6')
  ) AS t(signature,preimage,postimage) LOOP
    target := to_regprocedure(item.signature);
    SELECT pg_get_functiondef(p.oid),
           jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
             p.prosecdef,p.provolatile,p.proconfig,l.lanname)
      INTO source,identity_before
      FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
    IF source IS NULL OR identity_before IS DISTINCT FROM
       jsonb_build_array('lab_arena_owner',
         '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
         true, CASE WHEN item.signature LIKE '%reserve_call%' THEN 'v' ELSE 's' END,
         ARRAY['search_path=pg_catalog, public']::TEXT[], 'plpgsql') THEN
      RAISE EXCEPTION 'full thinking cost security identity differs: %',item.signature
        USING ERRCODE='55000';
    END IF;
    IF encode(extensions.digest(source,'sha256'),'hex')=item.postimage THEN CONTINUE; END IF;
    IF encode(extensions.digest(source,'sha256'),'hex')<>item.preimage
       OR (length(source)-length(replace(source,old_icp,'')))<>length(old_icp)
       OR (length(source)-length(replace(source,old_company,'')))<>length(old_company)
       OR (item.signature LIKE '%reserve_call%' AND (
          (length(source)-length(replace(source,old_admission,'')))<>length(old_admission)
          OR (length(source)-length(replace(source,old_hold,'')))<>length(old_hold))) THEN
      RAISE EXCEPTION 'full thinking cost preimage differs: %',item.signature
        USING ERRCODE='55000';
    END IF;
    updated := replace(replace(source,old_icp,new_icp),old_company,new_company);
    IF item.signature LIKE '%reserve_call%' THEN
      updated := replace(replace(updated,old_admission,new_admission),old_hold,new_hold);
    END IF;
    IF encode(extensions.digest(updated,'sha256'),'hex')<>item.postimage THEN
      RAISE EXCEPTION 'full thinking cost target differs: %',item.signature
        USING ERRCODE='55000';
    END IF;
    EXECUTE updated;
    SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
             p.prosecdef,p.provolatile,p.proconfig,l.lanname)
      INTO identity_after
      FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
    IF identity_after IS DISTINCT FROM identity_before
       OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>item.postimage THEN
      RAISE EXCEPTION 'full thinking cost readback differs: %',item.signature
        USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$full_thinking_cap555$;
NOTIFY pgrst,'reload schema';
COMMIT;
