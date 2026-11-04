-- One explicit operator continuation for eleven unfinished G3 assignments.
-- Wait for natural expiry. Preserve old rows, costs, sources, quotas and clocks.
-- The apply owner must freshly verify all eleven canonical S3 output keys absent.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $patch611$
DECLARE target RECORD; d TEXT; updated TEXT; metadata JSONB; function_oid OID;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)','4f78a1c47e9afa29f4162083eec62728aaaa3271bfd2da91e11741b624a5b32c','2b95e5566b8fed2ddee1fae3525a6bbc0ab5d598001eee650b45653f2dee22a0','v','plpgsql',$old611$'arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:3'))$old611$,$new611$'arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:3','arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:2:3','arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:3:3','arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:4:3','arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:5:3','arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:6:3','arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:7:3','arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:8:3','arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:9:3','arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4:4','arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5:4','arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:4'))$new611$),
 ('public.lab_arena_scoring_policy_history_v1(text)','7b14f6720f04d04018f73eac023b56419aa054445cbb44c9dd159c7d70f2ce9d','a2bbd7410aacc8b1e4f814f2c475b8c8c8b6f2b689e7f39fb79b9eeace92874d','s','sql',$old611$'nov03_baseline8_checkpoint_continuation610_v1') THEN$old611$,$new611$'nov03_baseline8_checkpoint_continuation610_v1','nov03_g3_natural_expiry_executions611_v1') THEN$new611$)
 ) AS patches(signature,preimage,postimage,volatility,language,old_text,new_text) LOOP
  function_oid:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=function_oid;
  IF d IS NULL OR (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
   p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid)
   IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,target.volatility,ARRAY['search_path=pg_catalog, public']::TEXT[],target.language) THEN
   RAISE EXCEPTION 'Nov03 G3 natural expiry security differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage
   OR length(d)-length(replace(d,target.old_text,''))<>length(target.old_text) THEN
   RAISE EXCEPTION 'Nov03 G3 natural expiry preimage differs: %',target.signature; END IF;
  updated:=replace(d,target.old_text,target.new_text);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov03 G3 natural expiry postimage differs: %',target.signature; END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov03 G3 natural expiry readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch611$;

DO $append611$
DECLARE rid CONSTANT TEXT:='arena-2026-11-03';
 request CONSTANT TEXT:='nov03-g3-natural-expiry611-v1';
 manifest CONSTANT JSONB:=$manifest611${"configuration_sha256":"9c396e3565a7f9d84d636cc55537ac95968d97770f5de798b1f45b56fbfc8d57","original_failed_hashes":{"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:2:1":"22da94e685bb468b41f700c7918d03179d941ccb578baa78ffc59d9470c24c1e","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:3:1":"9476f764992b7f8c849bcb0fbf28139de0833822f88bae8cecb6567e21cebbe6","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:4:1":"0e64f163a1414ef8a8d2d6bdd17658daaa2a533fc14333e451eb26c5cfe835c0","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:5:1":"d54d49493f3aabdb0cfa6c76a8ce902e0b094b44232df30ab52df7ed290ba654","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:6:1":"c4bdf55ae91a3cac261e385bff609486cef582e28b77154b899317675064b88c","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:7:1":"a88c536e0bbbca9af30df8a0220498784a23aea505bf078a4b51eaee9f73d814","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:8:1":"a3a67aebb132109a8098f4e28cf4ceb5b757348cea4499e47932a6bd45b150e4","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:9:1":"850aed80ca4f0887033b67163a56a13af2104a7907b9eae977dbb5e4df2f73c6","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4:1":"d72c986ce4ed06a3d0eaf7a246d397fd637c6f85ad3c9cdca3545c6dcb370df4","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4:2":"2b65a8e071ea44b5733ea9ab87ba91c65bfa42ac2465a51ead24473b013563b8","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5:1":"54745206a409c23b2f1c359e487bc283d299538e2fa7ca10b271b076780257a9","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5:2":"3dae7cc8b28c7dce5dcd9bf8f7950f255ccf14cf5110c9fc80beecbd6d13738c","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:1":"bb6c5d6f2eb8561a617522a0672c8f3751be7d3023000006f99b62a4e674035f","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:2":"7726f47dcc0e1cc7e55bf3933e20623ee92d4e20db654c2cd631a910e7dfb2c0"},"source_identities":{"sub-a4fe920d6204d92824d54d9cd0149bc6":{"miner_hotkey":"5CFFbNiTpYiv7VwhN3c2n1NRUtBGBmRkpZMxasKkfo35k1SL","round_id":"arena-2026-11-03","source_archive_sha256":"sha256:2888040e46763219fd18d9b8c3a0d5b4f5d2bed19872c99f356b2fe46b6a0216","source_ref":"arena/arena-2026-11-03/sources/sub-a4fe920d6204d92824d54d9cd0149bc6.tar.gz","source_size_bytes":470314,"submission_id":"sub-a4fe920d6204d92824d54d9cd0149bc6"},"sub-f013f522a7c764e18df792e094e7b820":{"miner_hotkey":"5DZrBs1WQzpopHuMjtP74vbYDCKQhDUVtLWLAuYAyGnM3AFh","round_id":"arena-2026-11-03","source_archive_sha256":"sha256:0e7a88a9f901be0ab14a4d6566c12b2f4c6507a6b2e253ef875af42ad0185c16","source_ref":"arena/arena-2026-11-03/sources/sub-f013f522a7c764e18df792e094e7b820.tar.gz","source_size_bytes":460081,"submission_id":"sub-f013f522a7c764e18df792e094e7b820"}},"targets":[{"assignment_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:2","icp_position":2,"lease_expires_at":"2026-11-03T12:19:55.451228+00:00","new_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:2:3","original_terminal_doc":null,"parent_attempt":2,"parent_full_sha256":"efd5e34d226277699b09f416f10b7b803bbec007915ff51bda2ed5dcc4056a53","parent_immutable_sha256":"aec7c43ea08a7488310563ec460fc37ec81ff370f9004640459130587f2f9a73","parent_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:2:2","submission_id":"sub-a4fe920d6204d92824d54d9cd0149bc6"},{"assignment_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:3","icp_position":3,"lease_expires_at":"2026-11-03T12:19:57.594301+00:00","new_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:3:3","original_terminal_doc":null,"parent_attempt":2,"parent_full_sha256":"33704192dc15a64945cec66af06b894c70c3201a321bbd74858157841a018843","parent_immutable_sha256":"1d0e422c3e17c2ae886c0e9189bd7cfeb2c0acf0af914e1e2b97605fe50631d1","parent_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:3:2","submission_id":"sub-a4fe920d6204d92824d54d9cd0149bc6"},{"assignment_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:4","icp_position":4,"lease_expires_at":"2026-11-03T12:19:35.990819+00:00","new_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:4:3","original_terminal_doc":null,"parent_attempt":2,"parent_full_sha256":"684739a5a5b1b0fbef211a83250139fa35b97e8faa78b9192ad33da7d1ab7c4d","parent_immutable_sha256":"f4b8c53d92ce757d18b66332cda729169f0f7d08451a6b95b4a0cef0206a6d49","parent_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:4:2","submission_id":"sub-a4fe920d6204d92824d54d9cd0149bc6"},{"assignment_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:5","icp_position":5,"lease_expires_at":"2026-11-03T12:19:46.449306+00:00","new_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:5:3","original_terminal_doc":null,"parent_attempt":2,"parent_full_sha256":"9655e33f60109509b0481ca472f3730f90e6dccfa623064fc68584274c43cd48","parent_immutable_sha256":"83f38feafbb8cbda6c5edbf6a3ad82e875c82dbe40e23e25f92fcda4c448aca0","parent_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:5:2","submission_id":"sub-a4fe920d6204d92824d54d9cd0149bc6"},{"assignment_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:6","icp_position":6,"lease_expires_at":"2026-11-03T12:19:43.552753+00:00","new_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:6:3","original_terminal_doc":null,"parent_attempt":2,"parent_full_sha256":"1447ced6c8f162e5f978507779e25011dfe64ccf04fbcc145a35fc20f39f95ff","parent_immutable_sha256":"73be298dd9833b436ac466cba3f253ba8e04fcc7beaebe9b6301e0ac3f4c4313","parent_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:6:2","submission_id":"sub-a4fe920d6204d92824d54d9cd0149bc6"},{"assignment_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:7","icp_position":7,"lease_expires_at":"2026-11-03T12:19:52.200303+00:00","new_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:7:3","original_terminal_doc":null,"parent_attempt":2,"parent_full_sha256":"444ca564e8041e36208420ec82a707b8dacddb5029e8628199ee136e113771eb","parent_immutable_sha256":"da515e0bd45b4e6bd1aad6d350d7b5e9e8047f1e5ab03a17c294cda65fbbc2a5","parent_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:7:2","submission_id":"sub-a4fe920d6204d92824d54d9cd0149bc6"},{"assignment_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:8","icp_position":8,"lease_expires_at":"2026-11-03T12:19:48.838567+00:00","new_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:8:3","original_terminal_doc":null,"parent_attempt":2,"parent_full_sha256":"733b6d4cb7cb9dd83845854573dc2c8dc6e5061e087a3dc31815b3dfe9df1ba9","parent_immutable_sha256":"78840a304d6692a2ca0628eee29ffd2ed1f9133cd2e39ff6196c6d4247db668b","parent_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:8:2","submission_id":"sub-a4fe920d6204d92824d54d9cd0149bc6"},{"assignment_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:9","icp_position":9,"lease_expires_at":"2026-11-03T12:19:50.295588+00:00","new_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:9:3","original_terminal_doc":null,"parent_attempt":2,"parent_full_sha256":"0d950655132d1d4d690da4fcfe0fd41ff3eee9d299b7ddf8a26421989cecf10f","parent_immutable_sha256":"d813384a87bc71a347dedb7f804557c34de3dfc789f96a7d0f8783ca81c0ffd5","parent_run_id":"arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:9:2","submission_id":"sub-a4fe920d6204d92824d54d9cd0149bc6"},{"assignment_id":"arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4","icp_position":4,"lease_expires_at":"2026-11-03T12:19:48.894231+00:00","new_run_id":"arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4:4","original_terminal_doc":null,"parent_attempt":3,"parent_full_sha256":"715f373332c569f509cb2d5b0e68dbf893fe23a045cba4cc6bd17f8503c19242","parent_immutable_sha256":"3106f32f9b9bf1677ef81b89ff57b88b72eb0fef3403d186e6582b73c9e7c990","parent_run_id":"arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4:3","submission_id":"sub-f013f522a7c764e18df792e094e7b820"},{"assignment_id":"arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5","icp_position":5,"lease_expires_at":"2026-11-03T12:19:48.080377+00:00","new_run_id":"arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5:4","original_terminal_doc":null,"parent_attempt":3,"parent_full_sha256":"0b406b08e2af022cff27c666b1e77db23d3f5adf993a1918368635faae42bb0a","parent_immutable_sha256":"fd0bf8c542b0c279e6f57263bf728614ed3ea8df403160029596ace0c538dedd","parent_run_id":"arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5:3","submission_id":"sub-f013f522a7c764e18df792e094e7b820"},{"assignment_id":"arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6","icp_position":6,"lease_expires_at":"2026-11-03T12:19:52.431189+00:00","new_run_id":"arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:4","original_terminal_doc":null,"parent_attempt":3,"parent_full_sha256":"cdf8ec5f149f99c4654b8478ee1dc8487dd1239bfcf533fcb55fb1a9a4fe60bb","parent_immutable_sha256":"662b6e66360cf709f5faf5bbc2e5ff6b5ecf08a020c890203d71385af71f2e44","parent_run_id":"arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:3","submission_id":"sub-f013f522a7c764e18df792e094e7b820"}]}$manifest611$::JSONB;
 r public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
 s public.lab_arena_submissions; parent public.lab_arena_runs; successor public.lab_arena_runs;
 h public.lab_arena_scoring_supersessions; target JSONB; source JSONB; n INTEGER;
 revision BIGINT; request_hash TEXT; old_hashes JSONB; failed_hashes JSONB;
 accepted_before JSONB; accepted_immutable JSONB; control_before JSONB; round_before JSONB;
 costs JSONB:='{}'; quotas JSONB:='{}'; cost JSONB; provider RECORD; calls BIGINT; available BOOLEAN;
BEGIN
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 control_before:=to_jsonb(c);round_before:=to_jsonb(r);
 request_hash:='sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex');
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=rid AND request_id=request;
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc->>'operation' IS DISTINCT FROM 'nov03_g3_natural_expiry_executions611_v1'
   OR h.recovery_doc->'manifest' IS DISTINCT FROM manifest
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc THEN
   RAISE EXCEPTION 'Nov03 G3 natural expiry replay differs'; END IF;
  FOR target IN SELECT value FROM jsonb_array_elements(manifest->'targets') LOOP
   SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=target->>'new_run_id';
   IF successor.assignment_id IS DISTINCT FROM target->>'assignment_id'
    OR successor.round_id IS DISTINCT FROM rid OR successor.kind IS DISTINCT FROM 'execute'
    OR successor.submission_id IS DISTINCT FROM target->>'submission_id'
    OR successor.icp_position IS DISTINCT FROM (target->>'icp_position')::INTEGER
    OR successor.attempt IS DISTINCT FROM (target->>'parent_attempt')::INTEGER+1
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=successor.assignment_id AND attempt>successor.attempt) THEN
    RAISE EXCEPTION 'Nov03 G3 natural expiry replay row differs'; END IF;
  END LOOP;
  RETURN; -- Never requeue a terminal continuation or append a further attempt.
 END IF;
 IF current_setting('session_replication_role')<>'origin' OR c.operator_paused
  OR c.guard_commitment IS DISTINCT FROM '' OR c.owner_commitment IS DISTINCT FROM ''
  OR c.guard_expires_at IS NOT NULL OR c.candidate_commit IS DISTINCT FROM ''
  OR c.restart_scope IS DISTINCT FROM '' OR c.restart_phase IS DISTINCT FROM ''
  OR c.captured_leases IS DISTINCT FROM '[]'::JSONB
  OR r.status IS DISTINCT FROM 'stage2' OR r.scoring_generation_floor IS DISTINCT FROM 0
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
  OR encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'configuration_sha256'
  OR r.configuration_doc->'max_attempts_per_assignment' IS DISTINCT FROM '2'::JSONB
  OR r.configuration_doc->'execution_icp_cap_microusd' IS DISTINCT FROM '6000000'::JSONB
  OR r.configuration_doc->'cost_per_company_microusd' IS DISTINCT FROM '6000000'::JSONB
  OR r.configuration_doc#>>'{schedule,stage_2_close}' IS DISTINCT FROM '2026-11-03T14:00:02Z'
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR jsonb_array_length(manifest->'targets')<>11
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions verifier WHERE verifier.round_id=rid
   AND verifier.request_id='nov03-forward-only-verifier604-v1'
   AND verifier.recovery_doc->>'operation'='nov03_forward_only_verifier_v1'
   AND verifier.new_round_doc->'configuration_doc'=r.configuration_doc)
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions prior WHERE prior.round_id=rid
   AND prior.request_id='nov03-three-unfinished-executions609-v1'
   AND prior.recovery_doc->>'operation'='nov03_three_unfinished_executions609_v1'
   AND prior.new_round_doc->'configuration_doc'=r.configuration_doc) THEN
  RAISE EXCEPTION 'Nov03 G3 natural expiry round, guard or clock differs'; END IF;
 PERFORM 1 FROM public.lab_arena_runs a WHERE a.assignment_id IN(
  SELECT t->>'assignment_id' FROM jsonb_array_elements(manifest->'targets')t) ORDER BY run_id FOR UPDATE;
 SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO failed_hashes
 FROM public.lab_arena_runs a WHERE (manifest->'original_failed_hashes') ? a.run_id;
 IF failed_hashes IS DISTINCT FROM manifest->'original_failed_hashes'
  OR (SELECT count(*) FROM jsonb_object_keys(manifest->'original_failed_hashes'))<>14 THEN
  RAISE EXCEPTION 'Nov03 G3 natural expiry fourteen original failures differ'; END IF;
 SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO old_hashes
 FROM public.lab_arena_runs a WHERE a.assignment_id IN(
  SELECT t->>'assignment_id' FROM jsonb_array_elements(manifest->'targets')t);
 SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO accepted_before
 FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.status='accepted';
 SELECT jsonb_object_agg(a.run_id,encode(extensions.digest((to_jsonb(a)-ARRAY['per_icp_score','qualification_doc','updated_at'])::TEXT,'sha256'),'hex')) INTO accepted_immutable
 FROM public.lab_arena_runs a WHERE accepted_before ? a.run_id;
 FOR source IN SELECT value FROM jsonb_each(manifest->'source_identities') LOOP
  SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=source->>'submission_id' FOR SHARE;
  IF s.round_id IS DISTINCT FROM rid OR s.is_king OR s.status IS DISTINCT FROM 'frozen'
   OR s.code_review_status IS DISTINCT FROM 'passed'
   OR source IS DISTINCT FROM jsonb_build_object('round_id',s.round_id,'submission_id',s.submission_id,
    'miner_hotkey',s.miner_hotkey,'source_ref',s.source_ref,'source_size_bytes',s.source_size_bytes,'source_archive_sha256',s.source_archive_sha256)
   OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants)p WHERE p->>'submission_id'=s.submission_id AND p->>'miner_hotkey'=s.miner_hotkey
    AND p->>'source_ref'=s.source_ref AND p->'source_size_bytes'=to_jsonb(s.source_size_bytes)
    AND p->'is_king'='false'::JSONB) THEN
   RAISE EXCEPTION 'Nov03 G3 natural expiry source identity differs'; END IF;
 END LOOP;
 FOR target IN SELECT value FROM jsonb_array_elements(manifest->'targets') LOOP
  n:=(target->>'parent_attempt')::INTEGER;
  SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=target->>'parent_run_id';
  IF parent.round_id IS DISTINCT FROM rid OR parent.kind IS DISTINCT FROM 'execute' OR parent.stage IS DISTINCT FROM 2
   OR parent.submission_id IS DISTINCT FROM target->>'submission_id'
   OR parent.miner_hotkey IS DISTINCT FROM manifest#>>ARRAY['source_identities',parent.submission_id,'miner_hotkey']
   OR parent.assignment_id IS DISTINCT FROM target->>'assignment_id'
   OR parent.icp_position IS DISTINCT FROM (target->>'icp_position')::INTEGER OR parent.attempt IS DISTINCT FROM n
   OR parent.status IS DISTINCT FROM 'failed' OR parent.terminal_cause IS DISTINCT FROM 'lease_expired'
   OR parent.result_doc IS NOT NULL OR parent.output_ref IS NOT NULL
   OR parent.per_icp_score IS NOT NULL OR parent.qualification_doc IS NOT NULL
   OR parent.champion_restart_required OR parent.stage_generation IS DISTINCT FROM r.stage_generation
   OR parent.lease_expires_at IS DISTINCT FROM (target->>'lease_expires_at')::TIMESTAMPTZ
   OR parent.lease_expires_at>=clock_timestamp()
   OR parent.terminal_doc-'expired_at' IS DISTINCT FROM coalesce(nullif(target->'original_terminal_doc','null'::JSONB),'{}'::JSONB)
   OR jsonb_typeof(parent.terminal_doc->'expired_at') IS DISTINCT FROM 'string'
   OR (parent.terminal_doc->>'expired_at')::TIMESTAMPTZ<parent.lease_expires_at
   OR (parent.terminal_doc->>'expired_at')::TIMESTAMPTZ>clock_timestamp()
   OR encode(extensions.digest((to_jsonb(parent)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])::TEXT,'sha256'),'hex') IS DISTINCT FROM target->>'parent_immutable_sha256'
   OR public.lab_arena_execution_attempt_count(parent) IS DISTINCT FROM n
   OR (SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=parent.assignment_id)<>n
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=parent.assignment_id
    AND (status IN('pending','leased','submitted','accepted') OR output_ref IS NOT NULL OR result_doc IS NOT NULL)) THEN
   RAISE EXCEPTION 'Nov03 G3 natural expiry parent proof differs: %',parent.run_id; END IF;
  cost:=public.lab_arena__successful_icp_cost_state(rid,parent.submission_id,parent.icp_position);
  IF (cost->>'inflight_calls')::BIGINT<>0 OR (cost->>'success_unresolved_calls')::BIGINT<>0
   OR coalesce(cost->>'competition_settled_microusd',cost->>'settled_microusd')::BIGINT>=6000000 THEN
   RAISE EXCEPTION 'Nov03 G3 natural expiry cost headroom differs'; END IF;
  costs:=costs||jsonb_build_object(parent.assignment_id,cost);available:=FALSE;
  FOR provider IN SELECT key,value FROM jsonb_each_text(r.configuration_doc->'call_quotas') LOOP
   calls:=public.lab_arena__execute_assignment_calls_v1(parent.assignment_id,provider.key);
   quotas:=quotas||jsonb_build_object(parent.assignment_id||':'||provider.key,calls);
   IF provider.value::BIGINT=0 AND provider.key='deepline'
     AND r.configuration_doc#>>'{deepline_catalog,schema_version}'='leadpoet.lab_arena.deepline_catalog.v1' THEN available:=TRUE;
   ELSIF provider.value::BIGINT>0 AND calls<provider.value::BIGINT*2
    AND public.lab_arena__submission_stage_consumed(parent.submission_id,2::SMALLINT,provider.key,'execute')<
     provider.value::BIGINT*2*(r.configuration_doc->>'stage_2_icp_count')::BIGINT THEN available:=TRUE;
   END IF;
  END LOOP;
  IF NOT available THEN RAISE EXCEPTION 'Nov03 G3 natural expiry provider quota exhausted'; END IF;
 END LOOP;
 SELECT greatest(1,coalesce(max(generation_floor),0)+1) INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,request,request_hash,'nov03-g3-natural-expiry611',txid_current(),pg_backend_pid(),
  to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',jsonb_build_object(
   'operation','nov03_g3_natural_expiry_executions611_v1','scoring_generation_floor',0,
   'manifest',manifest,'original_run_hashes',old_hashes,'original_accepted_run_hashes',accepted_before,'original_accepted_run_immutable_hashes',accepted_immutable,'original_icp_costs',costs,'original_assignment_provider_counts',quotas,
   'reason','One explicit operator continuation per eleven unfinished assignments after natural lease expiry following a distinct external validator session replacement. The apply owner freshly checks canonical S3 output absence. Preserve every prior cost and uncertain liability, original sources, $6 per-ICP cap, original two-attempt provider and stage quotas and all deadlines. No host credit, count reset, automatic fourth or fifth attempt.'));
 FOR target IN SELECT value FROM jsonb_array_elements(manifest->'targets') LOOP
  SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=target->>'parent_run_id';
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
   stage,icp_position,attempt,status,lease_generation,stage_generation,kind,previous_runner_hotkey)
  VALUES(target->>'new_run_id',parent.assignment_id,rid,parent.submission_id,parent.miner_hotkey,
   2,parent.icp_position,parent.attempt+1,'pending',parent.lease_generation,r.stage_generation,'execute',parent.runner_hotkey);
 END LOOP;
 IF (SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) FROM public.lab_arena_runs a
   WHERE old_hashes ? a.run_id) IS DISTINCT FROM old_hashes
  OR (SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) FROM public.lab_arena_runs a
   WHERE accepted_before ? a.run_id) IS DISTINCT FROM accepted_before
  OR (SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE round_id=rid) IS DISTINCT FROM round_before
  OR (SELECT to_jsonb(a) FROM public.lab_arena_restart_claim_control a WHERE singleton) IS DISTINCT FROM control_before THEN
  RAISE EXCEPTION 'Nov03 G3 natural expiry preserved state differs'; END IF;
END;
$append611$;
NOTIFY pgrst,'reload schema';
COMMIT;
