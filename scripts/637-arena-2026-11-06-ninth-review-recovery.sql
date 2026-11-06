-- User-approved ninth November6 review. One unchanged-artifact Sol attempt4.
-- Reuse the existing receipt-version semantics; each exact request has its own
-- thirty-minute exception. No prior receipt, round schedule, hold, or cost changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
LOCK TABLE public.lab_arena_rounds,public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
CREATE TEMP TABLE arena637_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $patch637_helper$
DECLARE target OID:=to_regprocedure('public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)'); d TEXT; updated TEXT; metadata JSONB; ident JSONB; old TEXT; new TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN RAISE EXCEPTION 'Nov06 ninth637 helper security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='515f66f04664bd8cd9ec8327b75db89e41164018d3c07c99869696ea9b8a19f1' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'80a5e95464b11019a55d0faf526a3295070003c1a13686cfefb9f8d35deeb491' THEN RAISE EXCEPTION 'Nov06 ninth637 helper preimage differs'; END IF;
 old:=$old637_helper$ WHERE h.round_id=p_round.round_id AND h.request_id='nov06-late-review635-v1'
  AND h.recovery_doc->>'operation'='nov06_late_review635_v1'$old637_helper$;
 new:=$new637_helper$ WHERE h.round_id=p_round.round_id
  AND ((h.request_id='nov06-late-review635-v1' AND h.recovery_doc->>'operation'='nov06_late_review635_v1')
   OR (h.request_id='nov06-ninth-review637-v1' AND h.recovery_doc->>'operation'='nov06_late_review635_v1'
    AND p_submission.submission_id='sub-43a929eb0e2e487f68014f48e7d80e6b'))
  AND h.recovery_doc->'originals' ? p_submission.submission_id$new637_helper$;
 IF length(d)-length(replace(d,old,''))<>length(old) THEN RAISE EXCEPTION 'Nov06 ninth637 helper anchor differs'; END IF;
 updated:=replace(d,old,new);
 old:=$old637_state$  AND p_round.round_id='arena-2026-11-06' AND p_round.status='committed'
  AND (to_jsonb(p_round)-'updated_at')=(h.old_round_doc-'updated_at')$old637_state$;
 new:=$new637_state$  AND p_round.round_id='arena-2026-11-06'
  AND ((h.request_id='nov06-late-review635-v1' AND p_round.status='committed'
    AND (to_jsonb(p_round)-'updated_at')=(h.old_round_doc-'updated_at'))
   OR (h.request_id='nov06-ninth-review637-v1'
    AND p_round.status IN('stage1','stage1_closed','stage1_scoring','stage1_judged')
    AND p_round.configuration_doc->>'execution_sequence_policy'='baseline_scored_first_v1'
    AND p_round.stage2_scoring_plan_doc IS NULL AND p_round.finalists IS NULL
    AND (p_round.stage1_scoring_plan_doc IS NULL OR COALESCE((jsonb_typeof(p_round.stage1_scoring_plan_doc)='object'
     AND jsonb_typeof(p_round.stage1_scoring_plan_doc->'work_items')='array' AND jsonb_typeof(p_round.stage1_scoring_plan_doc->'zero_rows')='array'
     AND NOT EXISTS(SELECT 1 FROM jsonb_array_elements((CASE WHEN jsonb_typeof(p_round.stage1_scoring_plan_doc->'work_items')='array' THEN p_round.stage1_scoring_plan_doc->'work_items' ELSE '[]'::JSONB END)||(CASE WHEN jsonb_typeof(p_round.stage1_scoring_plan_doc->'zero_rows')='array' THEN p_round.stage1_scoring_plan_doc->'zero_rows' ELSE '[]'::JSONB END)) plan_item(value)
      WHERE plan_item.value->>'submission_id' IS DISTINCT FROM 'baseline-2026-11-06')),FALSE))
    AND p_round.status_generation>=(h.old_round_doc->>'status_generation')::BIGINT
    AND p_round.stage_generation>=(h.old_round_doc->>'stage_generation')::BIGINT
    AND (to_jsonb(p_round)-ARRAY['status','status_generation','stage_generation','stage1_scoring_plan_doc','updated_at']::TEXT[])=(h.old_round_doc-ARRAY['status','status_generation','stage_generation','stage1_scoring_plan_doc','updated_at']::TEXT[])))$new637_state$;
 IF length(updated)-length(replace(updated,old,''))<>length(old) THEN RAISE EXCEPTION 'Nov06 ninth637 helper state anchor differs'; END IF;
 updated:=replace(updated,old,new);
 old:=$old637_control$   AND c.operator_paused AND c.pause_reason=h.recovery_doc->>'pause_reason')$old637_control$;
 new:=$new637_control$   AND ((h.request_id='nov06-late-review635-v1' AND c.operator_paused AND c.pause_reason=h.recovery_doc->>'pause_reason')
    OR (h.request_id='nov06-ninth-review637-v1' AND NOT c.operator_paused)))$new637_control$;
 IF length(updated)-length(replace(updated,old,''))<>length(old) THEN RAISE EXCEPTION 'Nov06 ninth637 helper control anchor differs'; END IF;
 updated:=replace(updated,old,new);
 old:=$old637_runs$  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=p_round.round_id)$old637_runs$;
 new:=$new637_runs$  AND ((h.request_id='nov06-late-review635-v1'
    AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=p_round.round_id))
   OR (h.request_id='nov06-ninth-review637-v1' AND NOT EXISTS(
    SELECT 1 FROM public.lab_arena_runs WHERE round_id=p_round.round_id
     AND (submission_id IS DISTINCT FROM 'baseline-2026-11-06' OR stage IS DISTINCT FROM 1))))$new637_runs$;
 IF length(updated)-length(replace(updated,old,''))<>length(old) THEN RAISE EXCEPTION 'Nov06 ninth637 helper runs anchor differs'; END IF;
 updated:=replace(updated,old,new);
 old:=$old637_deadline$ SELECT h.created_at+INTERVAL '30 minutes'$old637_deadline$;
 new:=$new637_deadline$ SELECT CASE WHEN h.request_id='nov06-ninth-review637-v1' THEN
  LEAST(h.created_at+INTERVAL '30 minutes',(h.old_round_doc#>>'{configuration_doc,schedule,stage_1_close}')::TIMESTAMPTZ)
  ELSE h.created_at+INTERVAL '30 minutes' END$new637_deadline$;
 IF length(updated)-length(replace(updated,old,''))<>length(old) THEN RAISE EXCEPTION 'Nov06 ninth637 helper deadline anchor differs'; END IF;
 updated:=replace(updated,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'515f66f04664bd8cd9ec8327b75db89e41164018d3c07c99869696ea9b8a19f1' THEN RAISE EXCEPTION 'Nov06 ninth637 helper postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'515f66f04664bd8cd9ec8327b75db89e41164018d3c07c99869696ea9b8a19f1'
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Nov06 ninth637 helper readback differs'; END IF;
END;
$patch637_helper$;
DO $patch637_frozen$
DECLARE target OID:=to_regprocedure('public.lab_arena_submissions_frozen_v1()'); d TEXT; updated TEXT; metadata JSONB; ident JSONB; old TEXT; new TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',FALSE,'v',ARRAY['search_path=pg_catalog']::TEXT[]) THEN RAISE EXCEPTION 'Nov06 ninth637 frozen security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='a225ce8971679ea2d389789c7d3b940a871b202fff46a3859562efb8d49650cb' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'21d54ab2e05c1497b38e3320ddf1836bd2a7980231f6dcabe62f970e34350108' THEN RAISE EXCEPTION 'Nov06 ninth637 frozen preimage differs'; END IF;
 old:=$old637_frozen$      WHERE h.round_id=OLD.round_id AND h.request_id='nov06-late-review635-v1'
       AND h.recovery_doc->>'operation'='nov06_late_review635_v1'$old637_frozen$;
 new:=$new637_frozen$      WHERE h.round_id=OLD.round_id
       AND ((h.request_id='nov06-late-review635-v1' AND h.recovery_doc->>'operation'='nov06_late_review635_v1')
        OR (h.request_id='nov06-ninth-review637-v1' AND h.recovery_doc->>'operation'='nov06_late_review635_v1'
         AND OLD.submission_id='sub-43a929eb0e2e487f68014f48e7d80e6b'))$new637_frozen$;
 IF length(d)-length(replace(d,old,''))<>length(old) THEN RAISE EXCEPTION 'Nov06 ninth637 frozen anchor differs'; END IF;
 updated:=replace(d,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'a225ce8971679ea2d389789c7d3b940a871b202fff46a3859562efb8d49650cb' THEN RAISE EXCEPTION 'Nov06 ninth637 frozen postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'a225ce8971679ea2d389789c7d3b940a871b202fff46a3859562efb8d49650cb'
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Nov06 ninth637 frozen readback differs'; END IF;
END;
$patch637_frozen$;
DO $recover637$
DECLARE r public.lab_arena_rounds; s public.lab_arena_submissions; h public.lab_arena_scoring_supersessions;
 c public.lab_arena_restart_claim_control; runs_before JSONB; control_before JSONB; item JSONB; sid TEXT; originals JSONB:='{}';
 expected JSONB:=$targets635$[{"ledger_ids":[8558233,8558234,8558235,8558239,8558240,8558241,8558245,8558246,8558247],"manifest_sha256":"sha256:30d0d5baddcde4edc7f11ad17a5724108dd9138bf3115a4cbf9eca0d9bee06fd","row_sha256":"05bc566c4697099d791402b3c617b11c317ba45d0a6681c8ba95de98eebfcdb5","source_sha256":"sha256:5ae54960007f2cc87130634b62f5dff3cbe40d5978dd59640fce4574b9855301","submission_id":"sub-43a929eb0e2e487f68014f48e7d80e6b"}]$targets635$::JSONB;
 previous JSONB:=$previous637${"nov06-admit-late-reviewed636-v1":"sha256:8a594d5f09a755019080bbd386e52fda2f36df42c0db39d0676b6b854f983b23","nov06-late-review635-v1":"sha256:d7ae1461e01727775e6824b6cbd4a20a4dbc61c96476d7dda454885c339a43f6"}$previous637$::JSONB;
 preserved JSONB:=$preserved635${"baseline-2026-11-06":"fafe9db43d264d9ad98c4a7edbeaf4210ed5d5b2f1b7e864fda336b87d3e8c86","sub-01f3260a133e59901869ca1a22ca34a3":"30e384126c1efe0633beff76b181e6bc6014aaba29a61cd53b26230c0ed4aceb","sub-0598bdeaa2643fc14cc2de959d7e99ca":"5b2436901a506660ac74f0eb317269bf5986b848a44b01680754765681680864","sub-12fb3ace2cd16a4edb6cc14b20138812":"283d71da89a0edb2e2e1d9411d72219c199d8b3a7923e2bf998afc016ea7f34e","sub-229d2e0d0c4c19b32dca23b9e1aa0739":"d373b61b413a624e6b6dbb95e8e64d82823145c7407c177c2bfc80656c392fb2","sub-2524e20e6ce4b4f3c09b34021e58b789":"10fc8890b62fb994984e56523f87930b8412be605d8490224911c91c4f6a4c0a","sub-2eafc097dca3fb8f862b86f0ea311ea4":"28f40e2025d98592cd00f0bc8b21a976dc8dca56f71417767162717513b80fbb","sub-3b3b4f8f2bce3583b6067d6699e96126":"98fb23c71b630f168335d5b747accfa467b6a6a83eae7d42991d55ea45a77491","sub-432ae8f60ab65d3d37a3a0beb329dd9c":"fe7958b9cf76745e130e69bac977862629d2bbdd759d82f77aae1aed241ab329","sub-4f9dcb8b1b1b85392283b8c8afcf68cb":"82b6d384b3454dddced7b4a561e520c3c354ca1125161103ff7ba78343c2524f","sub-5ca8f69d36dfb626b8241ea247381a78":"0e68cd2b9f530c04d54ed45fc4e74ebc9baf2087bfdb035aac2edb33c80679fe","sub-653798a4f3525c6de9a0da97d52ad5f8":"6449099f1b774f6344a30a967422bd907bb1bc30adbb6c9efe7bdcc73d2066a3","sub-696973ddfe30ef19396925b21166345e":"0920324a5d5b7c3f15fbec128e118a20e1eec7ec8d225f2acea1d0fa30ad65ba","sub-6a76e67caa37b34504e565584845f795":"607a6de51ac4aa39bd784318f3f17fdbec900f8345593b616aac635e1da051a8","sub-6bcdee3048e3b579150478166005b968":"2e5b8403eda957e86f761e5058d5fbcf2c54ca779472cdd167783cddede21e81","sub-86adc0ddbe50ef55afbbf8bfffa173e9":"0f38ea42083a1c9a082c4f7fc4f1655a8bb6e58907904ae276c4128a09a758b3","sub-8dcabdc1a405016fd93a3d4959e0ed6f":"6cf46ee6e3ae2bdb70c4af7b4c94d3f31e270551736c0cd510c7a027ba6753ad","sub-94f9e4dc04fbeb96625d423f445d7dd7":"bae143582dca50437a53e9c787164a642d0ae7daa964a3b7898cbc643f72e6bb","sub-971f053097b4b940a0a6a3ae5dc108d9":"60eb7b6de3542a49cade7c56fabc5908eb8b1fe0fedf502d2865439fe9ca721b","sub-9e7290663ae9819dc6ef0d55d33fbe0e":"f39fa1139f7d514684f5d8dc3500f3803f9a16095e5ba1ebd3c8674ee4147f08","sub-9fefd027e8e133f63a4a3e0be429c8ba":"b73ba98a16032838939c879898f9e6caab7c17ff359f0af45a367c862191ec63","sub-a0fff9b8e9550b935a4b7e5989219ad9":"b221c2efe862c60a6f6dc38294fe3a4e67dc5aa095a39a387d4c73dd04fe7d18","sub-a5947fae18bfc9c28ab593821825374e":"4174d18edd50130ee3b9af9e88703e816767a74e7434948c893c71f745113f8b","sub-ab2a4a5834748704157465ca6a435bf8":"b53a64d14f32f6ac004f9a2d9155dd4a9af9c2011f5ec9440c30882331b69416","sub-ae3bd1cf6d3dd41aac6bf1063d3fb77d":"5978b7b2aba5012677a399b42c6e608c6285a0f9c10b9d4ce2a6d796b1cb3a7b","sub-c8deb2248a68a1946cbbbe343dc07f89":"50f8fc6c08600b2aab3f000ef045fbae3a98e2fa72e1be431d03bf3b6c25afae","sub-ec5ccbef0972655f788ea41850102746":"52584208ba2955bc57f2af69170e29b0fc9281d2daf41550f85cc15be2871d84","sub-ed89cc69891b0222f523a730a9335063":"0f24a52c7a5477288b6a2aaedca4779cbe148bf136493eaabe7d52ec5ccfad50","sub-f5345040943729f422be5c33da8d53fe":"35be204188ba9ceb4fe77bbe773377645fef9a55305a03ef1f9907bdb305b9fa"}$preserved635$::JSONB;
 ledger JSONB; ids JSONB; generation BIGINT; created TIMESTAMPTZ:=clock_timestamp(); deadline TIMESTAMPTZ;
BEGIN
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-06' FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id AND request_id='nov06-ninth-review637-v1';
 IF FOUND THEN
  IF h.recovery_doc->>'operation' IS DISTINCT FROM 'nov06_late_review635_v1' THEN RAISE EXCEPTION 'Nov06 ninth637 replay differs'; END IF;
  RETURN; -- Never regrant or move the original receipt window.
 END IF;
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 deadline:=LEAST(created+INTERVAL '30 minutes',(r.configuration_doc#>>'{schedule,stage_1_close}')::TIMESTAMPTZ);
 control_before:=to_jsonb(c);
 SELECT coalesce(jsonb_object_agg(run_id,to_jsonb(a)),'{}') INTO runs_before FROM public.lab_arena_runs a WHERE round_id=r.round_id;
 IF current_setting('session_replication_role')<>'origin' OR c.operator_paused
  OR c.guard_commitment<>'' OR c.owner_commitment<>'' OR c.guard_expires_at IS NOT NULL
  OR c.candidate_commit<>'' OR c.restart_scope<>'' OR c.restart_phase<>'' OR c.captured_leases<>'[]'::JSONB
  OR r.status NOT IN('stage1','stage1_closed','stage1_scoring','stage1_judged')
  OR r.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
  OR r.stage2_scoring_plan_doc IS NOT NULL OR r.finalists IS NOT NULL
  OR NOT (r.stage1_scoring_plan_doc IS NULL OR COALESCE((jsonb_typeof(r.stage1_scoring_plan_doc)='object'
     AND jsonb_typeof(r.stage1_scoring_plan_doc->'work_items')='array' AND jsonb_typeof(r.stage1_scoring_plan_doc->'zero_rows')='array'
     AND NOT EXISTS(SELECT 1 FROM jsonb_array_elements((CASE WHEN jsonb_typeof(r.stage1_scoring_plan_doc->'work_items')='array' THEN r.stage1_scoring_plan_doc->'work_items' ELSE '[]'::JSONB END)||(CASE WHEN jsonb_typeof(r.stage1_scoring_plan_doc->'zero_rows')='array' THEN r.stage1_scoring_plan_doc->'zero_rows' ELSE '[]'::JSONB END)) plan_item(value)
      WHERE plan_item.value->>'submission_id' IS DISTINCT FROM 'baseline-2026-11-06')),FALSE))
  OR r.status_generation<2 OR r.stage_generation<1
  OR (to_jsonb(r)-ARRAY['status','status_generation','stage_generation','stage1_scoring_plan_doc','updated_at']::TEXT[]) IS DISTINCT FROM
   ((SELECT new_round_doc FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id
      AND request_id='nov06-admit-late-reviewed636-v1')-ARRAY['status','status_generation','stage_generation','stage1_scoring_plan_doc','updated_at']::TEXT[])
  OR EXISTS(SELECT 1 FROM jsonb_each_text(previous) p LEFT JOIN public.lab_arena_scoring_supersessions prior
   ON prior.round_id=r.round_id AND prior.request_id=p.key WHERE prior.request_hash IS DISTINCT FROM p.value)
  OR jsonb_array_length(r.participants)<>18 OR r.benchmark_ref IS DISTINCT FROM 'arena/arena-2026-11-06/benchmark.json'
  OR deadline IS NULL OR created>=deadline
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=r.round_id
   AND (submission_id IS DISTINCT FROM 'baseline-2026-11-06' OR stage IS DISTINCT FROM 1))
  OR EXISTS(SELECT 1 FROM public.lab_arena_submissions WHERE round_id=r.round_id AND code_review_status='reviewing')
  OR EXISTS(SELECT 1 FROM jsonb_each_text(preserved) keep LEFT JOIN public.lab_arena_submissions kept ON kept.submission_id=keep.key
   WHERE encode(extensions.digest(to_jsonb(kept)::TEXT,'sha256'),'hex') IS DISTINCT FROM keep.value) THEN
  RAISE EXCEPTION 'Nov06 ninth637 pre-miner round, control, clock or preserved originals differ'; END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(expected) LOOP
  sid:=item->>'submission_id';
  IF sid IS DISTINCT FROM 'sub-43a929eb0e2e487f68014f48e7d80e6b' THEN RAISE EXCEPTION 'Nov06 ninth637 exact identity differs'; END IF;
  SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=sid FOR UPDATE;
  SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id),jsonb_agg(l.entry_id ORDER BY entry_id) INTO ledger,ids
   FROM public.lab_arena_ledger l WHERE l.submission_id=sid AND l.operation_id='openrouter.code_review';
  IF encode(extensions.digest(to_jsonb(s)::TEXT,'sha256'),'hex') IS DISTINCT FROM item->>'row_sha256'
   OR s.round_id<>r.round_id OR s.status<>'rejected' OR s.rejection_rule IS DISTINCT FROM 'code_review_incomplete'
   OR s.is_king OR s.frozen_at IS NOT NULL OR s.code_review_status<>'error' OR s.code_review_attempts<>3
   OR s.source_archive_sha256 IS DISTINCT FROM item->>'source_sha256'
   OR s.code_review_doc->>'error_code' IS DISTINCT FROM 'review_response_invalid'
   OR s.code_review_doc->>'error_reason' NOT IN('research_evidence','research_evidence_mismatch')
   OR ids IS DISTINCT FROM item->'ledger_ids' OR jsonb_array_length(ledger)<>9
   OR (SELECT count(*) FILTER (WHERE value->>'entry_kind'='reservation')<>3
     OR count(*) FILTER (WHERE value->>'entry_kind'='reservation' AND value#>>'{entry_doc,review_model}' IS DISTINCT FROM 'anthropic/claude-sonnet-5.5')<>0
     FROM jsonb_array_elements(ledger))
   OR EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants) p WHERE p->>'submission_id'=sid) THEN
   RAISE EXCEPTION 'Nov06 ninth637 exact target or original ledger differs: %',sid; END IF;
  originals:=originals||jsonb_build_object(sid,jsonb_build_object('submission',to_jsonb(s),
   'row_sha256',item->>'row_sha256','manifest_sha256',item->>'manifest_sha256','ledger',ledger,
   'last_entry_id',(ledger->8->>'entry_id')::BIGINT));
 END LOOP;
 IF (SELECT count(*) FROM jsonb_object_keys(originals))<>1 THEN RAISE EXCEPTION 'Nov06 ninth637 target count differs'; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO generation FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc,created_at)
 VALUES(r.round_id,generation,'nov06-ninth-review637-v1','sha256:'||encode(extensions.digest(originals::TEXT,'sha256'),'hex'),
  'operator:nov06-ninth-review637',txid_current(),pg_backend_pid(),to_jsonb(r),to_jsonb(r),'{}',jsonb_build_object(
   'operation','nov06_late_review635_v1','originals',originals,'preserved',preserved,'pause_reason',c.pause_reason,
   'baseline_runs_at_grant',runs_before,'control_at_grant',control_before,
   'max_additional_reservations',1,'review_model','openai/gpt-6.1-sol','expires_at',deadline),created);
 FOR sid IN SELECT jsonb_object_keys(originals) LOOP
  UPDATE public.lab_arena_submissions SET status='accepted' WHERE submission_id=sid;
  UPDATE public.lab_arena_submissions SET code_review_doc=code_review_doc||jsonb_build_object('retryable',TRUE,
   'nov06_late_review',jsonb_build_object('request_id','nov06-ninth-review637-v1','expires_at',deadline))
   WHERE submission_id=sid;
 END LOOP;
 IF (SELECT coalesce(jsonb_object_agg(run_id,to_jsonb(a)),'{}') FROM public.lab_arena_runs a WHERE round_id=r.round_id) IS DISTINCT FROM runs_before
  OR (SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton) IS DISTINCT FROM control_before THEN
  RAISE EXCEPTION 'Nov06 ninth637 changed existing baseline work or control'; END IF;
END;
$recover637$;
DO $restore637$
BEGIN
 IF NOT (SELECT had_create FROM arena637_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM arena637_schema_acl) THEN
  RAISE EXCEPTION 'Nov06 ninth637 schema ACL differs'; END IF;
END;
$restore637$;
COMMIT;
