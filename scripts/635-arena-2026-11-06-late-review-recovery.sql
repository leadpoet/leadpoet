-- User-approved exact eight late November 6 admission reviews. No approval,
-- counter reset, execution, round schedule/rules change, or operator unpause.
-- The exact receipt grants only a thirty-minute late review deadline exception.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
LOCK TABLE public.lab_arena_rounds,public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
CREATE TEMP TABLE arena635_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $helper_guard635$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)');
 IF FOUND AND (p.prosrc IS DISTINCT FROM $expected_helper635$
 SELECT h.created_at+INTERVAL '30 minutes'
 FROM public.lab_arena_scoring_supersessions h
 WHERE h.round_id=p_round.round_id AND h.request_id='nov06-late-review635-v1'
  AND h.recovery_doc->>'operation'='nov06_late_review635_v1'
  AND p_round.round_id='arena-2026-11-06' AND p_round.status='committed'
  AND (to_jsonb(p_round)-'updated_at')=(h.old_round_doc-'updated_at')
  AND p_submission.round_id=p_round.round_id AND p_submission.status='accepted'
  AND NOT p_submission.is_king AND p_submission.frozen_at IS NULL
  AND p_submission.code_review_attempts IN(3,4)
  AND (to_jsonb(p_submission)-ARRAY['status','updated_at','code_review_status','code_review_doc',
   'code_review_claim','code_review_started_at','code_review_expires_at','code_review_attempts']::TEXT[])=
   ((h.recovery_doc->'originals'->p_submission.submission_id->'submission')-
    ARRAY['status','updated_at','code_review_status','code_review_doc','code_review_claim',
     'code_review_started_at','code_review_expires_at','code_review_attempts']::TEXT[])
  AND EXISTS(SELECT 1 FROM public.lab_arena_restart_claim_control c WHERE c.singleton
   AND c.operator_paused AND c.pause_reason=h.recovery_doc->>'pause_reason')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=p_round.round_id)
  AND NOT EXISTS(SELECT 1 FROM jsonb_each_text(h.recovery_doc->'preserved') keep
   LEFT JOIN public.lab_arena_submissions s ON s.submission_id=keep.key
   WHERE encode(extensions.digest(to_jsonb(s)::TEXT,'sha256'),'hex') IS DISTINCT FROM keep.value)
  AND (SELECT jsonb_agg(to_jsonb(l) ORDER BY l.entry_id) FROM public.lab_arena_ledger l
   WHERE l.submission_id=p_submission.submission_id AND l.operation_id='openrouter.code_review'
    AND l.entry_id<=(h.recovery_doc->'originals'->p_submission.submission_id->>'last_entry_id')::BIGINT)
   =h.recovery_doc->'originals'->p_submission.submission_id->'ledger'
  AND (SELECT count(*)=p_submission.code_review_attempts FROM public.lab_arena_ledger l
   WHERE l.submission_id=p_submission.submission_id AND l.operation_id='openrouter.code_review'
    AND l.entry_kind='reservation')
  AND (p_submission.code_review_attempts=3 OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l
   WHERE l.submission_id=p_submission.submission_id AND l.operation_id='openrouter.code_review'
    AND l.entry_kind='reservation' AND l.entry_doc->'attempt'='4'::JSONB
    AND l.entry_doc->>'review_model'='openai/gpt-6.1-sol'));
$expected_helper635$
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'sql') THEN
  RAISE EXCEPTION 'Nov06 late635 helper differs'; END IF;
END;
$helper_guard635$;
CREATE OR REPLACE FUNCTION public.lab_arena__nov06_late_review635_v1(
 p_round public.lab_arena_rounds,p_submission public.lab_arena_submissions)
RETURNS TIMESTAMPTZ LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $helper635$
 SELECT h.created_at+INTERVAL '30 minutes'
 FROM public.lab_arena_scoring_supersessions h
 WHERE h.round_id=p_round.round_id AND h.request_id='nov06-late-review635-v1'
  AND h.recovery_doc->>'operation'='nov06_late_review635_v1'
  AND p_round.round_id='arena-2026-11-06' AND p_round.status='committed'
  AND (to_jsonb(p_round)-'updated_at')=(h.old_round_doc-'updated_at')
  AND p_submission.round_id=p_round.round_id AND p_submission.status='accepted'
  AND NOT p_submission.is_king AND p_submission.frozen_at IS NULL
  AND p_submission.code_review_attempts IN(3,4)
  AND (to_jsonb(p_submission)-ARRAY['status','updated_at','code_review_status','code_review_doc',
   'code_review_claim','code_review_started_at','code_review_expires_at','code_review_attempts']::TEXT[])=
   ((h.recovery_doc->'originals'->p_submission.submission_id->'submission')-
    ARRAY['status','updated_at','code_review_status','code_review_doc','code_review_claim',
     'code_review_started_at','code_review_expires_at','code_review_attempts']::TEXT[])
  AND EXISTS(SELECT 1 FROM public.lab_arena_restart_claim_control c WHERE c.singleton
   AND c.operator_paused AND c.pause_reason=h.recovery_doc->>'pause_reason')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=p_round.round_id)
  AND NOT EXISTS(SELECT 1 FROM jsonb_each_text(h.recovery_doc->'preserved') keep
   LEFT JOIN public.lab_arena_submissions s ON s.submission_id=keep.key
   WHERE encode(extensions.digest(to_jsonb(s)::TEXT,'sha256'),'hex') IS DISTINCT FROM keep.value)
  AND (SELECT jsonb_agg(to_jsonb(l) ORDER BY l.entry_id) FROM public.lab_arena_ledger l
   WHERE l.submission_id=p_submission.submission_id AND l.operation_id='openrouter.code_review'
    AND l.entry_id<=(h.recovery_doc->'originals'->p_submission.submission_id->>'last_entry_id')::BIGINT)
   =h.recovery_doc->'originals'->p_submission.submission_id->'ledger'
  AND (SELECT count(*)=p_submission.code_review_attempts FROM public.lab_arena_ledger l
   WHERE l.submission_id=p_submission.submission_id AND l.operation_id='openrouter.code_review'
    AND l.entry_kind='reservation')
  AND (p_submission.code_review_attempts=3 OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l
   WHERE l.submission_id=p_submission.submission_id AND l.operation_id='openrouter.code_review'
    AND l.entry_kind='reservation' AND l.entry_doc->'attempt'='4'::JSONB
    AND l.entry_doc->>'review_model'='openai/gpt-6.1-sol'));
$helper635$;
ALTER FUNCTION public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $patch635_begin$
DECLARE target OID:=to_regprocedure('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'); d TEXT; updated TEXT; metadata JSONB; ident JSONB; old TEXT; new TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Nov06 late635 begin security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='44e1e4125cd373913cb1f02419ba17b7ffb1f45e8b55e7190aa256e85f5a6b1c' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'2bd23a14f1ba30bfdb4f4425c2adfef3226631301f41534143de5b49876065ee' THEN RAISE EXCEPTION 'Nov06 late635 begin preimage differs'; END IF;
 updated:=d;
  old:=$old635_begin0$  v_nov03_recovery BOOLEAN;$old635_begin0$;
  new:=$new635_begin0$  v_nov03_recovery BOOLEAN;
  v_nov06_deadline TIMESTAMPTZ;$new635_begin0$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*1 THEN RAISE EXCEPTION 'Nov06 late635 anchor begin0 differs'; END IF;
  updated:=replace(updated,old,new);
  old:=$old635_begin1$  v_now := pg_catalog.clock_timestamp();
  v_freeze_at :=$old635_begin1$;
  new:=$new635_begin1$  v_nov06_deadline := public.lab_arena__nov06_late_review635_v1(v_round,v_submission);
  v_now := pg_catalog.clock_timestamp();
  v_freeze_at :=$new635_begin1$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*1 THEN RAISE EXCEPTION 'Nov06 late635 anchor begin1 differs'; END IF;
  updated:=replace(updated,old,new);
  old:=$old635_begin2$  v_is_baseline :=$old635_begin2$;
  new:=$new635_begin2$  IF v_nov06_deadline IS NOT NULL THEN v_deadline:=v_nov06_deadline; END IF;
  v_is_baseline :=$new635_begin2$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*1 THEN RAISE EXCEPTION 'Nov06 late635 anchor begin2 differs'; END IF;
  updated:=replace(updated,old,new);
  old:=$old635_begin3$v_round.status <> 'open' AND NOT v_nov03_recovery$old635_begin3$;
  new:=$new635_begin3$v_round.status <> 'open' AND NOT v_nov03_recovery AND v_nov06_deadline IS NULL$new635_begin3$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*3 THEN RAISE EXCEPTION 'Nov06 late635 anchor begin3 differs'; END IF;
  updated:=replace(updated,old,new);
  old:=$old635_begin4$(v_invalid_start IS NOT NULL AND v_invalid_count>=3)$old635_begin4$;
  new:=$new635_begin4$(v_invalid_start IS NOT NULL AND v_invalid_count>=3
      AND NOT (v_nov06_deadline IS NOT NULL AND v_submission.code_review_attempts=3 AND v_invalid_count=3))$new635_begin4$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*1 THEN RAISE EXCEPTION 'Nov06 late635 anchor begin4 differs'; END IF;
  updated:=replace(updated,old,new);
  old:=$old635_begin5$  ELSIF NOT v_refusal_recovery AND v_invalid_start IS NOT NULL$old635_begin5$;
  new:=$new635_begin5$  ELSIF v_nov06_deadline IS NOT NULL AND v_submission.code_review_attempts=3 THEN
    IF p_review_model IS DISTINCT FROM 'openai/gpt-6.1-sol' THEN
      RETURN jsonb_build_object('status','route_required','review_model','openai/gpt-6.1-sol');
    END IF;
  ELSIF NOT v_refusal_recovery AND v_invalid_start IS NOT NULL$new635_begin5$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*1 THEN RAISE EXCEPTION 'Nov06 late635 anchor begin5 differs'; END IF;
  updated:=replace(updated,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'44e1e4125cd373913cb1f02419ba17b7ffb1f45e8b55e7190aa256e85f5a6b1c' THEN RAISE EXCEPTION 'Nov06 late635 begin postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'44e1e4125cd373913cb1f02419ba17b7ffb1f45e8b55e7190aa256e85f5a6b1c'
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Nov06 late635 begin readback differs'; END IF;
END;
$patch635_begin$;
DO $patch635_finish$
DECLARE target OID:=to_regprocedure('public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)'); d TEXT; updated TEXT; metadata JSONB; ident JSONB; old TEXT; new TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Nov06 late635 finish security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='167edcc637fc4d3c3ac5b624850788895a736cb88907e97172e43c077b609f6d' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'a84213a8c7affd3e6dd15c97b4cb20fa3b155c2e67d52e72de8de50eb536960b' THEN RAISE EXCEPTION 'Nov06 late635 finish preimage differs'; END IF;
 updated:=d;
  old:=$old635_finish0$  v_nov03_recovery BOOLEAN;$old635_finish0$;
  new:=$new635_finish0$  v_nov03_recovery BOOLEAN;
  v_nov06_deadline TIMESTAMPTZ;$new635_finish0$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*1 THEN RAISE EXCEPTION 'Nov06 late635 anchor finish0 differs'; END IF;
  updated:=replace(updated,old,new);
  old:=$old635_finish1$  v_nov03_recovery := public.lab_arena__nov03_review605_v1(v_round,v_submission);$old635_finish1$;
  new:=$new635_finish1$  v_nov03_recovery := public.lab_arena__nov03_review605_v1(v_round,v_submission);
  IF v_round.round_id='arena-2026-11-06' THEN
    PERFORM 1 FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
  END IF;
  v_nov06_deadline := public.lab_arena__nov06_late_review635_v1(v_round,v_submission);$new635_finish1$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*1 THEN RAISE EXCEPTION 'Nov06 late635 anchor finish1 differs'; END IF;
  updated:=replace(updated,old,new);
  old:=$old635_finish2$  v_is_baseline :=$old635_finish2$;
  new:=$new635_finish2$  IF v_nov06_deadline IS NOT NULL THEN v_deadline:=v_nov06_deadline; END IF;
  v_is_baseline :=$new635_finish2$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*1 THEN RAISE EXCEPTION 'Nov06 late635 anchor finish2 differs'; END IF;
  updated:=replace(updated,old,new);
  old:=$old635_finish3$(v_round.status <> 'open' AND NOT v_nov03_recovery)$old635_finish3$;
  new:=$new635_finish3$(v_round.status <> 'open' AND NOT v_nov03_recovery AND v_nov06_deadline IS NULL)$new635_finish3$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*2 THEN RAISE EXCEPTION 'Nov06 late635 anchor finish3 differs'; END IF;
  updated:=replace(updated,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'167edcc637fc4d3c3ac5b624850788895a736cb88907e97172e43c077b609f6d' THEN RAISE EXCEPTION 'Nov06 late635 finish postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'167edcc637fc4d3c3ac5b624850788895a736cb88907e97172e43c077b609f6d'
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Nov06 late635 finish readback differs'; END IF;
END;
$patch635_finish$;
DO $patch635_frozen$
DECLARE target OID:=to_regprocedure('public.lab_arena_submissions_frozen_v1()'); d TEXT; updated TEXT; metadata JSONB; ident JSONB; old TEXT; new TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',FALSE,'v',ARRAY['search_path=pg_catalog']::TEXT[]) THEN
  RAISE EXCEPTION 'Nov06 late635 frozen security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='21d54ab2e05c1497b38e3320ddf1836bd2a7980231f6dcabe62f970e34350108' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'c7f91e17eaca4d726472d3e80c2af452ba4854ced1e3a200d64c9a1304e20ad4' THEN RAISE EXCEPTION 'Nov06 late635 frozen preimage differs'; END IF;
 updated:=d;
  old:=$old635_frozen0$  IF OLD.status = 'rejected' AND NEW.status <> 'rejected' AND NOT (
$old635_frozen0$;
  new:=$new635_frozen0$  IF OLD.status = 'rejected' AND NEW.status <> 'rejected' AND NOT (
    NEW.status='accepted' AND OLD.round_id='arena-2026-11-06'
    AND (to_jsonb(NEW)-ARRAY['status','updated_at'])=(to_jsonb(OLD)-ARRAY['status','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.request_id='nov06-late-review635-v1'
       AND h.recovery_doc->>'operation'='nov06_late_review635_v1'
       AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->'originals'->OLD.submission_id->>'row_sha256'=
        encode(extensions.digest(to_jsonb(OLD)::TEXT,'sha256'),'hex'))
  ) AND NOT (
$new635_frozen0$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*1 THEN RAISE EXCEPTION 'Nov06 late635 anchor frozen0 differs'; END IF;
  updated:=replace(updated,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'21d54ab2e05c1497b38e3320ddf1836bd2a7980231f6dcabe62f970e34350108' THEN RAISE EXCEPTION 'Nov06 late635 frozen postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'21d54ab2e05c1497b38e3320ddf1836bd2a7980231f6dcabe62f970e34350108'
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Nov06 late635 frozen readback differs'; END IF;
END;
$patch635_frozen$;
DO $patch635_history$
DECLARE target OID:=to_regprocedure('public.lab_arena_scoring_policy_history_v1(text)'); d TEXT; updated TEXT; metadata JSONB; ident JSONB; old TEXT; new TEXT; expected_post TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Nov06 late635 history security differs'; END IF;
 expected_post:=CASE WHEN encode(extensions.digest(d,'sha256'),'hex')='27da80592b12e32477348f106756d4e5bf9ca8dabe9e604291530519a3a93d43' THEN 'e56c877e3c2ba5440b2fc7b55caa65b2e94935c5117dddd12ed1320c9e3f4a45' ELSE 'c263bf0dcdc639093f4321351bfc212cd983a6270414f81b5c7cab05df401aae' END;
 IF encode(extensions.digest(d,'sha256'),'hex') IN('c263bf0dcdc639093f4321351bfc212cd983a6270414f81b5c7cab05df401aae','e56c877e3c2ba5440b2fc7b55caa65b2e94935c5117dddd12ed1320c9e3f4a45') THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex') NOT IN('1bc738f6e73c29ac756106ff233af894915a492547334f6ecb66815654f6031a','27da80592b12e32477348f106756d4e5bf9ca8dabe9e604291530519a3a93d43') THEN RAISE EXCEPTION 'Nov06 late635 history preimage differs'; END IF;
 updated:=d;
  old:=$old635_history0$'nov03_admit_reviewed_originals606_v1'$old635_history0$;
  new:=$new635_history0$'nov03_admit_reviewed_originals606_v1','nov06_late_review635_v1','nov06_admit_late_reviewed636_v1'$new635_history0$;
  IF length(updated)-length(replace(updated,old,''))<>length(old)*1 THEN RAISE EXCEPTION 'Nov06 late635 anchor history0 differs'; END IF;
  updated:=replace(updated,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>expected_post THEN RAISE EXCEPTION 'Nov06 late635 history postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>expected_post
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Nov06 late635 history readback differs'; END IF;
END;
$patch635_history$;
DO $recover635$
DECLARE r public.lab_arena_rounds; s public.lab_arena_submissions; h public.lab_arena_scoring_supersessions;
 c public.lab_arena_restart_claim_control; item JSONB; sid TEXT; originals JSONB:='{}';
 expected JSONB:=$targets635$[{"ledger_ids":[8558126,8558127,8558133,8558177,8558178,8558186,8558215,8558216,8558218],"manifest_sha256":"sha256:a7ccdae8b553c5f7f5ef92cc4c733aa355e33f9419fa6f191c18fb134c0b331d","row_sha256":"ac201446847eba9c1dc1fa97d55c7d43452e00542394d6755a289639d42e021a","source_sha256":"sha256:13822703258e768cc0116a91379968d3d8068265b527714d7ccdeb5273d3963f","submission_id":"sub-0598bdeaa2643fc14cc2de959d7e99ca"},{"ledger_ids":[8558076,8558077,8558082,8558140,8558141,8558148,8558188,8558189,8558192],"manifest_sha256":"sha256:540447ee9f42168f054b185d0486d1cd7a7542710972a2da6c9b1db6f643365b","row_sha256":"1d4de081de31fd6f7aef1a6632216a2b908be5df26b21c4683d9508eff5eb871","source_sha256":"sha256:8e61615547e3e7b3ef98714a39d6b7ad18c2d25e876eae0659f04e69172505b4","submission_id":"sub-2eafc097dca3fb8f862b86f0ea311ea4"},{"ledger_ids":[8558128,8558129,8558136,8558181,8558182,8558185,8558219,8558220,8558225],"manifest_sha256":"sha256:c3d6d0eb2f9bf7553bfe84987916218be9c0e874b8880c1e2dc4006da21e2cfe","row_sha256":"b805c60ada8aadfe36f5cf49b00670710f4fbb9afbaa6a101b6507b41a196b27","source_sha256":"sha256:b491a4e81dd8a22fb8bbac164155e6d427c551c49e677a27b5867c7aedf273d8","submission_id":"sub-3b3b4f8f2bce3583b6067d6699e96126"},{"ledger_ids":[8558134,8558135,8558138,8558183,8558184,8558187,8558221,8558222,8558226],"manifest_sha256":"sha256:4fdece173bf825f671bb0ffe51d6a9ff5ee363d961185ce597a9a759e15fc8d8","row_sha256":"eec7a09aa17a62b36a67a10cdb2dcb62ddf229d08671d84d2f965d0d1d9fc330","source_sha256":"sha256:d544e2a54bb3f8cd792fb3e46f85e3fdc09efe307fae26c65a36bf9ff48a2496","submission_id":"sub-4f9dcb8b1b1b85392283b8c8afcf68cb"},{"ledger_ids":[8558112,8558113,8558119,8558161,8558162,8558168,8558204,8558205,8558208],"manifest_sha256":"sha256:94230e1756d370208181bf06a9a0b78bde0bd9a05a55d2c4259499d623d4119e","row_sha256":"82915578462d9d0914670cfa8ab8d4c3700ef0e94a3af513ca5837e1b010d4c8","source_sha256":"sha256:06f015519295fbfd473a6c8a2bf2e3cf92dd4babf262b9423769f1390e71c44f","submission_id":"sub-6bcdee3048e3b579150478166005b968"},{"ledger_ids":[8558090,8558091,8558097,8558150,8558151,8558163,8558198,8558199,8558203],"manifest_sha256":"sha256:255b8de909d0f49349b60c8da003767426f31d8c97676b8c2fe1953514bc7f04","row_sha256":"e72ea02386225f358e59b6f1f1c30475f1c3aa9b841af4083045ae52081e63ec","source_sha256":"sha256:f8666bdea4c43b7175bbf6c3ea329146ffbf2a43d4b1c8cf7ba48b2e209f3041","submission_id":"sub-8dcabdc1a405016fd93a3d4959e0ed6f"},{"ledger_ids":[8558114,8558115,8558125,8558164,8558165,8558175,8558209,8558210,8558224],"manifest_sha256":"sha256:f1d0c0825b67862941340cfb624dafecc8cd9b7dc132eadee73c90132e9b85aa","row_sha256":"b247a3c0d6a4262bc9e565a8471217e780488fa4926e10883002c571512ba4a6","source_sha256":"sha256:964b7917d81dbec93f733e2a9958617766dd82a89c46457d04e6ae88e9f55034","submission_id":"sub-94f9e4dc04fbeb96625d423f445d7dd7"},{"ledger_ids":[8558120,8558121,8558130,8558171,8558172,8558176,8558211,8558212,8558217],"manifest_sha256":"sha256:8e50b1de14026ea2e859a9f9afed1072b8c566aeec34a87a487f8b55b21b935c","row_sha256":"bd84fbae7501b530599de9e4d89a948d7a36f2f46f4be8ab0594d132c76a3d16","source_sha256":"sha256:96b6ca1426dbbf9161e483aeb79c82e254d3fc4ba2a1f28e067b2fa3389c493d","submission_id":"sub-ab2a4a5834748704157465ca6a435bf8"}]$targets635$::JSONB;
 preserved JSONB:=$preserved635${"baseline-2026-11-06":"fafe9db43d264d9ad98c4a7edbeaf4210ed5d5b2f1b7e864fda336b87d3e8c86","sub-01f3260a133e59901869ca1a22ca34a3":"30e384126c1efe0633beff76b181e6bc6014aaba29a61cd53b26230c0ed4aceb","sub-12fb3ace2cd16a4edb6cc14b20138812":"283d71da89a0edb2e2e1d9411d72219c199d8b3a7923e2bf998afc016ea7f34e","sub-2524e20e6ce4b4f3c09b34021e58b789":"10fc8890b62fb994984e56523f87930b8412be605d8490224911c91c4f6a4c0a","sub-5ca8f69d36dfb626b8241ea247381a78":"0e68cd2b9f530c04d54ed45fc4e74ebc9baf2087bfdb035aac2edb33c80679fe","sub-653798a4f3525c6de9a0da97d52ad5f8":"6449099f1b774f6344a30a967422bd907bb1bc30adbb6c9efe7bdcc73d2066a3","sub-696973ddfe30ef19396925b21166345e":"0920324a5d5b7c3f15fbec128e118a20e1eec7ec8d225f2acea1d0fa30ad65ba","sub-6a76e67caa37b34504e565584845f795":"607a6de51ac4aa39bd784318f3f17fdbec900f8345593b616aac635e1da051a8","sub-86adc0ddbe50ef55afbbf8bfffa173e9":"0f38ea42083a1c9a082c4f7fc4f1655a8bb6e58907904ae276c4128a09a758b3","sub-971f053097b4b940a0a6a3ae5dc108d9":"60eb7b6de3542a49cade7c56fabc5908eb8b1fe0fedf502d2865439fe9ca721b","sub-9e7290663ae9819dc6ef0d55d33fbe0e":"f39fa1139f7d514684f5d8dc3500f3803f9a16095e5ba1ebd3c8674ee4147f08","sub-9fefd027e8e133f63a4a3e0be429c8ba":"b73ba98a16032838939c879898f9e6caab7c17ff359f0af45a367c862191ec63","sub-a0fff9b8e9550b935a4b7e5989219ad9":"b221c2efe862c60a6f6dc38294fe3a4e67dc5aa095a39a387d4c73dd04fe7d18","sub-a5947fae18bfc9c28ab593821825374e":"4174d18edd50130ee3b9af9e88703e816767a74e7434948c893c71f745113f8b","sub-c8deb2248a68a1946cbbbe343dc07f89":"50f8fc6c08600b2aab3f000ef045fbae3a98e2fa72e1be431d03bf3b6c25afae","sub-ec5ccbef0972655f788ea41850102746":"52584208ba2955bc57f2af69170e29b0fc9281d2daf41550f85cc15be2871d84","sub-ed89cc69891b0222f523a730a9335063":"0f24a52c7a5477288b6a2aaedca4779cbe148bf136493eaabe7d52ec5ccfad50","sub-f5345040943729f422be5c33da8d53fe":"35be204188ba9ceb4fe77bbe773377645fef9a55305a03ef1f9907bdb305b9fa"}$preserved635$::JSONB;
 ledger JSONB; ids JSONB; generation BIGINT; created TIMESTAMPTZ:=clock_timestamp();
BEGIN
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-06' FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id AND request_id='nov06-late-review635-v1';
 IF FOUND THEN
  IF h.recovery_doc->>'operation' IS DISTINCT FROM 'nov06_late_review635_v1' THEN RAISE EXCEPTION 'Nov06 late635 replay differs'; END IF;
  RETURN; -- Never regrant or move the original receipt window.
 END IF;
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 IF current_setting('session_replication_role')<>'origin' OR NOT c.operator_paused
  OR c.pause_reason IS DISTINCT FROM 'await_reviewed_recovery633_before_publication'
  OR c.guard_commitment<>'' OR r.status<>'committed'
  OR encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex')<>'c3386e4c79a54d2c5b3102e152ed45aed09d8771c21318dc3dad4c0a7f373cc6'
  OR jsonb_array_length(r.participants)<>18 OR r.benchmark_ref IS DISTINCT FROM 'arena/arena-2026-11-06/benchmark.json'
  OR created+INTERVAL '30 minutes'>=(r.configuration_doc#>>'{schedule,stage_1_close}')::TIMESTAMPTZ
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=r.round_id)
  OR EXISTS(SELECT 1 FROM public.lab_arena_submissions WHERE round_id=r.round_id AND code_review_status='reviewing')
  OR EXISTS(SELECT 1 FROM jsonb_each_text(preserved) keep LEFT JOIN public.lab_arena_submissions kept ON kept.submission_id=keep.key
   WHERE encode(extensions.digest(to_jsonb(kept)::TEXT,'sha256'),'hex') IS DISTINCT FROM keep.value) THEN
  RAISE EXCEPTION 'Nov06 late635 committed round, hold, clock or preserved originals differ'; END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(expected) LOOP
  sid:=item->>'submission_id';
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
   RAISE EXCEPTION 'Nov06 late635 exact target or original ledger differs: %',sid; END IF;
  originals:=originals||jsonb_build_object(sid,jsonb_build_object('submission',to_jsonb(s),
   'row_sha256',item->>'row_sha256','manifest_sha256',item->>'manifest_sha256','ledger',ledger,
   'last_entry_id',(ledger->8->>'entry_id')::BIGINT));
 END LOOP;
 IF (SELECT count(*) FROM jsonb_object_keys(originals))<>8 THEN RAISE EXCEPTION 'Nov06 late635 target count differs'; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO generation FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc,created_at)
 VALUES(r.round_id,generation,'nov06-late-review635-v1','sha256:'||encode(extensions.digest(originals::TEXT,'sha256'),'hex'),
  'operator:nov06-late-review635',txid_current(),pg_backend_pid(),to_jsonb(r),to_jsonb(r),'{}',jsonb_build_object(
   'operation','nov06_late_review635_v1','originals',originals,'preserved',preserved,'pause_reason',c.pause_reason,
   'max_additional_reservations',1,'review_model','openai/gpt-6.1-sol','expires_at',created+INTERVAL '30 minutes'),created);
 FOR sid IN SELECT jsonb_object_keys(originals) LOOP
  UPDATE public.lab_arena_submissions SET status='accepted' WHERE submission_id=sid;
  UPDATE public.lab_arena_submissions SET code_review_doc=code_review_doc||jsonb_build_object('retryable',TRUE,
   'nov06_late_review',jsonb_build_object('request_id','nov06-late-review635-v1','expires_at',created+INTERVAL '30 minutes'))
   WHERE submission_id=sid;
 END LOOP;
END;
$recover635$;
DO $restore635$
BEGIN
 IF NOT (SELECT had_create FROM arena635_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM arena635_schema_acl) THEN
  RAISE EXCEPTION 'Nov06 late635 schema ACL differs'; END IF;
END;
$restore635$;
COMMIT;
