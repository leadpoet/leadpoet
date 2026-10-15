-- Match the claim-control -> round/run order in every reachable progression RPC.
-- Only lock acquisition changes; final hold guards and all function metadata stay.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $progression_lock_order490$
DECLARE
  v_target RECORD;
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_metadata JSONB;
  v_expected_acl TEXT;
BEGIN
  FOR v_target IN SELECT * FROM (VALUES
    ('public.lab_arena_activate_preexecuted_stage2_v1(text)','bdb16e09687a4c016429d23ebbc565c218e7c0898814467a3cfaf6082d1e18a8','eecc7d959e81cb881b25a1365c0441251b374fa5ed981bfa214520e8a105733d',
     $marker$SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public']::TEXT[]),
    ('public.lab_arena_cancel_round(text,text)','5e5b109bedf347afddefc7b2c9a646602b41e8592222cbf4b771da4c07fb24c2','038b7462e8ae7702e6d21b098142e8fd626b5cadcb9adf1c4f1d858be1246364',
     $marker$SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public']::TEXT[]),
    ('public.lab_arena_close_parallel_execution_v1(text)','ffadfbf44dc0583879495f90eeda37d653f01357f69e924de30f5d7fd1ac6c9e','a4f0ea976da82d09ae19ef7867090ec819cda465a4f5307d30df54dbef8a7609',
     $marker$SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public']::TEXT[]),
    ('public.lab_arena_close_stage(text,smallint)','676aa701fd9a81031a4f5f36e4029c9960331dd43065824a714587d50c7444ea','dc50c521f73ff996c28437ce3321d62cecbf4a5adfb69126f94a35708ecde481',
     $marker$SELECT * INTO v_round
  FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public']::TEXT[]),
    ('public.lab_arena_open_parallel_execution_v1(text,jsonb)','8126955c423b440353fd52f9027f86e92188d6cd8487ff47256c9be18d2fde9a','0d5b381c9e389780dddfae84133992c72d34f5849ffa91360c7bd55d40a46010',
     $marker$SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public']::TEXT[]),
    ('public.lab_arena_open_scoring(text,smallint,jsonb)','5dcc89131da07629df5bedafd53357941de32c0e41b274dcd729cd0a6428cd79','d6b84f982a6ce25c8c0563f469326f1f7d2f13f21198fd60ecc63d4c19cc17d3',
     $marker$SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public']::TEXT[]),
    ('public.lab_arena_open_scoring_sep15_baseline_only_v1(text,smallint,jsonb)','c71e43caf400c28fb7ece5172686cef09fd5542a95945b2a8a79beb6a43ba750','f8dea3dbd0a59d4e6f788dc85b1f4faa664a8b2658c1f368529c0332223f4410',
     $marker$SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public']::TEXT[]),
    ('public.lab_arena_open_scoring_sep16_baseline_only_v1(text,smallint,jsonb)','9316a0792b0b601bac3c8e0b58b8289f4ac4a107d6531aa63b396d7faf6aa1e7','1a81dd717af21bd60a087109539c475faa1b955ad7b5d0ce630ec9fc9750cd01',
     $marker$SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public']::TEXT[]),
    ('public.lab_arena_open_scoring_v2(text,smallint,jsonb)','e99dcc4eb2f4a532336553fe19002aeb8fd86bc10e42871aa37ea745c4f174a1','e6d08167d9cad6df8444320972f6f6c26b97bbddccedb48473bd270d06d90b34',
     $marker$SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public']::TEXT[]),
    ('public.lab_arena_open_scoring_v3(text,smallint,jsonb)','1599d0f6eee691f7057306955a8e52e0c2de8a4db4c3ede9529601a49685794b','0e931641a2106db809708264dac25a13bede2be7a32e21db44010a6311b8bed2',
     $marker$SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public','statement_timeout=60s']::TEXT[]),
    ('public.lab_arena_open_stage(text,smallint,jsonb,integer[])','09c0a684a35dcc245d4e7dba18cb5bac61d70a73ad12a14d0ee3e6d6654909a0','58b4f914cc2eab90f99d76da4a336aed83f7030f5172147a1b22cfdd9207cf75',
     $marker$SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public','statement_timeout=60s']::TEXT[]),
    ('public.lab_arena_record_run_scores(text,smallint,jsonb)','a0d4f0ea4bff2b642ecfa7b62b6ecabfb330ae7c404e3c6d083c5a9da9c952e0','23dd26cb833078f06e58f2d1716475f35af65740764f787c8d3e98d6652d0895',
     $marker$SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR SHARE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public']::TEXT[]),
    ('public.lab_arena_transition_round(text,text,text,jsonb)','f1826c1ca0789fb00ebef9f3ec6a4bea9de0e02ae4caedd3291d56d901576b90','24875eaab94a0977170e0b2f5baa07a43998e36a2f17a7f1b79b23e8acbf2e23',
     $marker$SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;$marker$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',ARRAY['search_path=pg_catalog, public','statement_timeout=600s']::TEXT[])
  ) AS targets(signature,preimage,postimage,marker,acl,config)
  LOOP
    v_oid:=pg_catalog.to_regprocedure(v_target.signature);
    SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
      owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig),
      pg_catalog.to_jsonb(p)-'prosrc'
    INTO v_definition,v_identity,v_metadata
    FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
    WHERE p.oid=v_oid;
    v_expected_acl:=v_target.acl;
    -- The separate491 privilege fix can precede this migration. Accept only
    -- that reviewed owner-only shape for the unused Sep15 backup, and retain it.
    IF v_target.signature='public.lab_arena_open_scoring_sep15_baseline_only_v1(text,smallint,jsonb)'
       AND v_identity->>1='{lab_arena_owner=X/lab_arena_owner}' THEN
      v_expected_acl:='{lab_arena_owner=X/lab_arena_owner}';
    END IF;
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner',v_expected_acl,TRUE,'v',v_target.config) THEN
      RAISE EXCEPTION 'Arena progression490 security shape differs: %',v_target.signature
        USING ERRCODE='55000';
    END IF;
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_target.postimage THEN
      CONTINUE;
    END IF;
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_target.preimage
       OR pg_catalog.length(v_definition)-pg_catalog.length(
         pg_catalog.replace(v_definition,v_target.marker,''))<>pg_catalog.length(v_target.marker) THEN
      RAISE EXCEPTION 'Arena progression490 preimage differs: %',v_target.signature
        USING ERRCODE='55000';
    END IF;
    v_updated:=pg_catalog.replace(v_definition,v_target.marker,$lock490$-- Keep claims and progression in one order: control, then round/run.
  -- Final transition and score triggers still enforce operator holds.
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0));
  $lock490$||v_target.marker);
    IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_target.postimage THEN
      RAISE EXCEPTION 'Arena progression490 postimage differs: %',v_target.signature
        USING ERRCODE='55000';
    END IF;
    EXECUTE v_updated;
    IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')
       <>v_target.postimage OR (SELECT pg_catalog.to_jsonb(p)-'prosrc'
         FROM pg_catalog.pg_proc p WHERE p.oid=v_oid) IS DISTINCT FROM v_metadata THEN
      RAISE EXCEPTION 'Arena progression490 readback differs: %',v_target.signature
        USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$progression_lock_order490$;
NOTIFY pgrst, 'reload schema';
COMMIT;
