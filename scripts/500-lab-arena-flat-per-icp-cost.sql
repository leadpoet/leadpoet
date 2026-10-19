-- Future rounds can opt in to a fixed $4 successful sourcing allowance per ICP.
-- Frozen legacy policies, raw costs, provider admissions and retry proofs remain.
-- No round, run, ledger, score or publication rows are changed by this migration.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $flat_per_icp_cost$
DECLARE
  item RECORD; patch RECORD;
  definition TEXT; updated TEXT; security_before JSONB; security_after JSONB;
BEGIN
  FOR item IN SELECT * FROM (VALUES
    ('public.lab_arena__baseline_final_ready_v1(text)','98fe2309206c5a3a0ea429b0d26f57c45cf97a657c42fbca13606da22450ff29','acb8d403ce29dc306db08d43dc65f2aaeefe578177dc9580cc171eb251bae294','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena__baseline_recovery_run_v1(lab_arena_runs)','2f4352b22ac198b10f70dd6b90e9320db890b395ca53e3a6a56375cce04dcab6','fb5c263a796c4ef3dc49174c18e68b08d639ed7222c813dea401cd139ec8eebd','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena__execute_host_retry_ready_v1(lab_arena_runs)','713c87e768ccb10db8b9270c19d50afbfd63d2d54601b852e4d7a3a46cf6d460','d2c350acd090cc96d34dd749b4215d2583853a68b405a6f567f6ec07ee40ced0','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena__per_icp_publication_valid(text,jsonb)','b55034a2541a72c115bb38b9eda86a4b193ee7b6275effb8490aee90cdbecaba','1ed2b288d7bc614088c05b245cb8ee43a3c0e8eea3281de595a8694871e266ad','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena__publication_execution_incomplete_v1(text,text)','ef0dcc626e8aa5ef9c204db85af82ca98add76b58cf0fefff2e2e5c4b0b62508','ac255ffd6bbcd2c28d8c0c9421b9c13cdaa87e477d55a34c5f48032a1b8a4458','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','b9da18be2a587210faae63a5f340e956dc437fd3e2d0b706e9621757aad75913','aa00d7f981f0c255d85e84d650b6986ed0d400feccebacb1fc9aa702192c3b52','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)','dcc78b9c7615def5db85a9544b8ae8269a10ed2f6fa153293acb68611ee69c09','db63d4251758f6590cddd7c659b37bbec3bce9585491392627175e3549bfe26b','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_commit_round_v3(text,jsonb,text,text,date,text,text,jsonb)','a245d37af1ec82e5b065cfe74b1222d6aef7966b3cd56c06bfc75777d359e8b3','3434a1c212c79fb698146a1598be2cb86a182b8ac8edefe2a38a28f0c303f5fa','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_icp_cost_eligibility(text,text,integer,integer)','e2926293aba28e4e8a0b3ef2ed6605790fa2dc0f5ea865a77d33b40d7ed6be5b','f190d9727c331c967966f8552a43e12f6cbafad8e8281be7d68749d5b8704571','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_integrity_publication_guard_v1()','9b1c54afa98c4918c53b678352c5588afee05f1d89270eb70b036e2f70f1b74b','0897b15b52a28f5ccaf08523c38c92c19fa24dbb056796f867a79e71524c9bb3','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}",false,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)','a05339ccfc8611816f273307b2f665a439e0c76ed837cf02b4304c596997ed68','ed9585bada5629700c89f12a204a26df9e84460e6ec27f38da0817f335a2231e','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB),
    ('public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)','19a81f024bc51efa9c3fcdfa2a2f2a8b7fa8cd914eb02d5efc7cdbee4275d9bf','6267d70170c18f3303b6923f2bf5b68deb63eb79e1ee51ec244c3ba1030b43ed','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB),
    ('public.lab_arena_list_deepline_cost_reconciliations_v3(text,text,bigint,integer)','8b88dd0d77f8890fd3ea133fd0a89ce1c59a1a3b38c5bd978488538c6483aa57','696a7cdf26748b711813f135e4f807b7e626ccb1c91225fc8ef738368c50011f','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB),
    ('public.lab_arena_list_openrouter_cost_reconciliations_v1(text,text,bigint,integer)','380311d654e1e7b5a8b4f20d7a41eed277f9d8a2a5e2c8a2ba24850d195a58c7','e2d75725cb2fa7c5e901f27a0625b86c9a9d5f0727fcbfafb5ebcb235d46ab78','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB),
    ('public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)','982950a12738d17989c05c3e6bc61513a59b0b4188c90f3d99b7836e3a93e5d0','58f74de8066a3c9b0300a66c32820f92a098d584a964574d886c830dbc7f2547','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB),
    ('public.lab_arena_next_closed_provider_reconciliation_v1(text,text,integer,text,bigint)','220b9508eb2fbc31b16a6689864e1b1822e5a2444f7d778566f5aef766eac824','89a779452595cbab4537fec883dfaa0191eb327244a311421d297c73fecf1476','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB),
    ('public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)','610a6e24a10d5a3f046bd360c2025f1f0234a34204d9a77af53b355d38591a90','dad21e757318741a77ffea122b384485ef53dda7cfa826d41146baa6fd2d49ee','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)','5c67e1d48e19c22cc0de51de32173e9e16fe53cc194cec9550e40406bbba9113','1abfea580cdcce6b195072709691dcf44a0eb76b2ad6480d884048cc271f9adc','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_reconcile_openrouter_cost_v1(text,text,text,bigint,text,text,bigint,text)','98609f8097c1395b3086e1cccd838a685f9a2edf86fabd0b3dbffd8f6ecbe1b0','c3932833cad2a51cb7b19cde1b27d3631b71d4aac4e8859a4a78f709d58afaf3','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_record_run_scores(text,smallint,jsonb)','23dd26cb833078f06e58f2d1716475f35af65740764f787c8d3e98d6652d0895','949f568393c5a81dc2b804fadd8b0ccbfb0c4d3ae89b3f3e2f5afd35b5804c3f','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)','6c4aa26ea4fe38c070e4096cc91b979463d99583ae33c8faf8ad0c52fd6e304a','ebf02eae4ca75ea99d420fb585bb3daa22ccdb2a97b7a28d39a5368c13134128','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_rounds_write_once_v1()','f4213c6f65d83ce631be50df1ca0a2a3ee6ad6bbf7f47c327496b8b716dd0b18','89a4b69e12ad828377eddf16e63cfcb70beb1e4af4c103451025616821fe7a80','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",false,"v",["search_path=pg_catalog"],"plpgsql"]'::JSONB),
    ('public.lab_arena_run_quota_snapshot_v1(text,text)','1d81024e92bda081979bb4efdb114d74840d30bb40ce516219c335f7ef7d0512','71c927c6036b626ede1573896a4a930e486157016830d9fe8c3adc52fa000c3c','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_submission_costs(text)','37f25734ab5bde112d4c7909beb59d0fbdc083130b493accb41a100a63020822','842f97ecd9b339be8501fab18485431fd61dd4a81ba877ca48055b83cfc9e369','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB),
    ('public.lab_arena_transition_round(text,text,text,jsonb)','24875eaab94a0977170e0b2f5baa07a43998e36a2f17a7f1b79b23e8acbf2e23','1ab1ebf04fbc1920ca9667681f385c7e48e6e55b82ac19aa364249c8c9decb07','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public","statement_timeout=600s"],"plpgsql"]'::JSONB)
  ) expected(signature,preimage,postimage,security_identity) LOOP
    SELECT pg_get_functiondef(p.oid), jsonb_build_array(o.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig,l.lanname)
      INTO definition,security_before
    FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
    JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(item.signature);
    IF definition IS NULL OR security_before IS DISTINCT FROM item.security_identity THEN
      RAISE EXCEPTION 'Flat per-ICP cost security preimage differs: %',item.signature;
    END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')=item.postimage THEN CONTINUE; END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM item.preimage THEN
      RAISE EXCEPTION 'Flat per-ICP cost exact preimage differs: %',item.signature;
    END IF;
    -- Null policies remain unrecognized. Admit the new marker only through
    -- the same named legacy policy paths; no unknown policy gains access.
    updated := regexp_replace(definition,$pattern$((?:[A-Za-z_][A-Za-z_0-9]*\.)?[A-Za-z_][A-Za-z_0-9]*\s*->>\s*'sourcing_cost_eligibility_policy')\s+IS DISTINCT FROM 'successful_calls_per_icp_v1'$pattern$,$replacement$COALESCE(\1, '') NOT IN ('successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')$replacement$,'g');
    updated := regexp_replace(updated,$pattern$=\s*'successful_calls_per_icp_v1'$pattern$,$replacement$IN ('successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')$replacement$,'g');
    updated := regexp_replace(updated,$pattern$'successful_calls_v1',\s*'successful_calls_per_icp_v1'$pattern$,$replacement$'successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1'$replacement$,'g');
    FOR patch IN SELECT * FROM (VALUES
      ('public.lab_arena__per_icp_publication_valid(text,jsonb)',$old$COALESCE(v_cost ->> 'sourcing_cost_eligibility_policy', '') NOT IN ('successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')$old$,$new$v_cost ->> 'sourcing_cost_eligibility_policy' IS DISTINCT FROM
       (SELECT configuration_doc ->> 'sourcing_cost_eligibility_policy'
        FROM public.lab_arena_rounds WHERE round_id = p_round_id)$new$),
      ('public.lab_arena_icp_cost_eligibility(text,text,integer,integer)',$old$IS DISTINCT FROM 800000$old$,$new$IS DISTINCT FROM (CASE WHEN v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
           = 'successful_calls_flat_per_icp_v1' THEN 4000000 ELSE 800000 END)$new$),
      ('public.lab_arena_icp_cost_eligibility(text,text,integer,integer)',$old$  v_company_cap BIGINT;$old$,$new$  v_company_cap BIGINT;
  v_eligibility_cap BIGINT;$new$),
      ('public.lab_arena_icp_cost_eligibility(text,text,integer,integer)',$old$  v_company_cap := (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT;$old$,$new$  v_company_cap := (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT;
  -- lab_arena_flat_per_icp_cost_v1: qualified yield never scales this cap.
  v_eligibility_cap := CASE WHEN v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
    = 'successful_calls_flat_per_icp_v1' THEN v_icp_cap
    ELSE v_company_cap * p_qualified_company_count END;$new$),
      ('public.lab_arena_icp_cost_eligibility(text,text,integer,integer)',$old$ELSIF v_spend > v_company_cap * p_qualified_company_count THEN
    v_reason := 'cost_per_company_exceeded';$old$,$new$ELSIF v_spend > v_eligibility_cap THEN
    v_reason := CASE WHEN v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
      = 'successful_calls_flat_per_icp_v1' THEN 'execution_cap_exceeded' ELSE 'cost_per_company_exceeded' END;$new$),
      ('public.lab_arena_icp_cost_eligibility(text,text,integer,integer)',$old$'eligibility_cap_microusd', v_company_cap * p_qualified_company_count,$old$,$new$'eligibility_cap_microusd', v_eligibility_cap,$new$),
      ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',$old$IS DISTINCT FROM 800000$old$,$new$IS DISTINCT FROM (CASE WHEN v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
           = 'successful_calls_flat_per_icp_v1' THEN 4000000 ELSE 800000 END)$new$)
    ) changes(signature,before_text,after_text) WHERE signature=item.signature LOOP
      IF (length(updated)-length(replace(updated,patch.before_text,'')))<>length(patch.before_text) THEN
        RAISE EXCEPTION 'Flat per-ICP cost patch shape differs: %',item.signature;
      END IF;
      updated := replace(updated,patch.before_text,patch.after_text);
    END LOOP;
    IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM item.postimage THEN
      RAISE EXCEPTION 'Flat per-ICP cost target differs: %',item.signature;
    END IF;
    EXECUTE updated;
    SELECT jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
      INTO security_after FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
      JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(item.signature);
    IF security_after IS DISTINCT FROM security_before
       OR encode(extensions.digest(pg_get_functiondef(item.signature::REGPROCEDURE),'sha256'),'hex')
          IS DISTINCT FROM item.postimage THEN
      RAISE EXCEPTION 'Flat per-ICP cost readback differs: %',item.signature;
    END IF;
  END LOOP;
END;
$flat_per_icp_cost$;

NOTIFY pgrst, 'reload schema';
COMMIT;
