-- One source/receipt-bound alternate fourth review for eight November 6 errors.
-- This never approves a submission, resets history, extends a cutoff, or
-- changes the ordinary three-reservation or explicit-refusal allowances.
-- Apply only through the canonical drained-release flow before admission closes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
LOCK TABLE public.lab_arena_rounds,public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
CREATE TEMP TABLE arena635_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $guard635$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov06_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)');
 IF FOUND AND (p.prosrc IS DISTINCT FROM $expected635_helper$
 SELECT COALESCE(p_round.round_id='arena-2026-11-06' AND p_round.status='open'
  AND p_round.published_at IS NULL AND p_round.publication_doc IS NULL
  AND p_round.reward_activated_at IS NULL
  AND p_submission.round_id=p_round.round_id AND p_submission.status='accepted'
  AND p_submission.frozen_at IS NULL AND NOT p_submission.is_king
  AND p_submission.code_review_status='error' AND p_submission.code_review_attempts=3
  AND p_submission.code_review_expires_at IS NULL
  AND clock_timestamp()<LEAST((p_round.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ,
   (p_round.configuration_doc#>>'{schedule,benchmark_deadline}')::TIMESTAMPTZ)
  AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
   WHERE h.round_id=p_round.round_id AND h.request_id='nov06-invalid-review635-v1'
    AND h.recovery_doc->>'operation'='nov06_invalid_review635_v1'
    AND h.old_round_doc->'configuration_doc'=p_round.configuration_doc
    AND h.recovery_doc->'originals' ? p_submission.submission_id
    AND (to_jsonb(p_submission)-ARRAY['updated_at','code_review_doc']::TEXT[])=
     ((h.recovery_doc->'originals'->p_submission.submission_id->'submission')-ARRAY['updated_at','code_review_doc']::TEXT[])
    AND p_submission.code_review_doc=(h.recovery_doc->'originals'->p_submission.submission_id->'submission'->'code_review_doc')
     ||jsonb_build_object('retryable',TRUE,'nov06_review_recovery','nov06-invalid-review635-v1')
    AND h.recovery_doc->'originals'->p_submission.submission_id->>'terminal_sha256'=
     encode(extensions.digest(to_jsonb(public.lab_arena__ledger_head(
      h.recovery_doc->'originals'->p_submission.submission_id->>'call_identity'))::TEXT,'sha256'),'hex'))
  AND (SELECT count(*)=3 AND count(*) FILTER (WHERE l.entry_doc->>'review_model' IS DISTINCT FROM 'anthropic/claude-sonnet-5.5')=0
   FROM public.lab_arena_ledger l WHERE l.submission_id=p_submission.submission_id
    AND l.operation_id='openrouter.code_review' AND l.entry_kind='reservation'),FALSE);
$expected635_helper$
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'sql') THEN
  RAISE EXCEPTION 'Nov06 review635 helper differs'; END IF;
END;
$guard635$;
CREATE OR REPLACE FUNCTION public.lab_arena__nov06_review635_v1(
 p_round public.lab_arena_rounds,p_submission public.lab_arena_submissions)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $helper635$
 SELECT COALESCE(p_round.round_id='arena-2026-11-06' AND p_round.status='open'
  AND p_round.published_at IS NULL AND p_round.publication_doc IS NULL
  AND p_round.reward_activated_at IS NULL
  AND p_submission.round_id=p_round.round_id AND p_submission.status='accepted'
  AND p_submission.frozen_at IS NULL AND NOT p_submission.is_king
  AND p_submission.code_review_status='error' AND p_submission.code_review_attempts=3
  AND p_submission.code_review_expires_at IS NULL
  AND clock_timestamp()<LEAST((p_round.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ,
   (p_round.configuration_doc#>>'{schedule,benchmark_deadline}')::TIMESTAMPTZ)
  AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
   WHERE h.round_id=p_round.round_id AND h.request_id='nov06-invalid-review635-v1'
    AND h.recovery_doc->>'operation'='nov06_invalid_review635_v1'
    AND h.old_round_doc->'configuration_doc'=p_round.configuration_doc
    AND h.recovery_doc->'originals' ? p_submission.submission_id
    AND (to_jsonb(p_submission)-ARRAY['updated_at','code_review_doc']::TEXT[])=
     ((h.recovery_doc->'originals'->p_submission.submission_id->'submission')-ARRAY['updated_at','code_review_doc']::TEXT[])
    AND p_submission.code_review_doc=(h.recovery_doc->'originals'->p_submission.submission_id->'submission'->'code_review_doc')
     ||jsonb_build_object('retryable',TRUE,'nov06_review_recovery','nov06-invalid-review635-v1')
    AND h.recovery_doc->'originals'->p_submission.submission_id->>'terminal_sha256'=
     encode(extensions.digest(to_jsonb(public.lab_arena__ledger_head(
      h.recovery_doc->'originals'->p_submission.submission_id->>'call_identity'))::TEXT,'sha256'),'hex'))
  AND (SELECT count(*)=3 AND count(*) FILTER (WHERE l.entry_doc->>'review_model' IS DISTINCT FROM 'anthropic/claude-sonnet-5.5')=0
   FROM public.lab_arena_ledger l WHERE l.submission_id=p_submission.submission_id
    AND l.operation_id='openrouter.code_review' AND l.entry_kind='reservation'),FALSE);
$helper635$;
ALTER FUNCTION public.lab_arena__nov06_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov06_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $patch635$
DECLARE
 target OID:=to_regprocedure('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)');
 d TEXT; updated TEXT; m JSONB; ident JSONB; old TEXT; new TEXT;
 preimage CONSTANT TEXT:='2bd23a14f1ba30bfdb4f4425c2adfef3226631301f41534143de5b49876065ee';
 postimage CONSTANT TEXT:='e6bc89509657584b31d5d57a5babd51905e0d9ecb67625c63e1ef13efe93f8fa';
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,m,ident FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
  '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',
  ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN RAISE EXCEPTION 'Nov06 review635 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')=postimage THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>preimage THEN RAISE EXCEPTION 'Nov06 review635 preimage differs'; END IF;
 updated:=d;
 old:=$old635a$  v_invalid_doc JSONB := '{}'::JSONB;$old635a$;
 new:=$new635a$  v_invalid_doc JSONB := '{}'::JSONB;
  v_nov06_recovery BOOLEAN := FALSE;$new635a$;
 IF length(updated)-length(replace(updated,old,''))<>length(old) THEN RAISE EXCEPTION 'Nov06 review635 anchor a differs'; END IF;
 updated:=replace(updated,old,new);
 old:=$old635b$  -- Always recover/settle an expired lease above before enforcing the cap.
  -- Refusal does not reset an existing invalid-review episode.
  IF (v_invalid_start IS NOT NULL AND v_invalid_count>=3)$old635b$;
 new:=$new635b$  -- Always recover/settle an expired lease above before enforcing the cap.
  -- Refusal does not reset an existing invalid-review episode.
  v_nov06_recovery := public.lab_arena__nov06_review635_v1(v_round,v_submission);
  IF (v_invalid_start IS NOT NULL AND v_invalid_count>=3
      AND NOT (v_nov06_recovery AND v_invalid_count=3))$new635b$;
 IF length(updated)-length(replace(updated,old,''))<>length(old) THEN RAISE EXCEPTION 'Nov06 review635 anchor b differs'; END IF;
 updated:=replace(updated,old,new);
 old:=$old635c$  ELSIF NOT v_refusal_recovery AND v_invalid_start IS NOT NULL$old635c$;
 new:=$new635c$  ELSIF v_nov06_recovery THEN
    IF p_review_model IS DISTINCT FROM 'openai/gpt-6.1-sol' THEN
      RETURN pg_catalog.jsonb_build_object('status','route_required','review_model','openai/gpt-6.1-sol');
    END IF;
  ELSIF NOT v_refusal_recovery AND v_invalid_start IS NOT NULL$new635c$;
 IF length(updated)-length(replace(updated,old,''))<>length(old) THEN RAISE EXCEPTION 'Nov06 review635 anchor c differs'; END IF;
 updated:=replace(updated,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>postimage THEN RAISE EXCEPTION 'Nov06 review635 postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>postimage
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m THEN
  RAISE EXCEPTION 'Nov06 review635 readback differs'; END IF;
END;
$patch635$;

DO $recover635$
DECLARE r public.lab_arena_rounds; s public.lab_arena_submissions; h public.lab_arena_scoring_supersessions;
 head public.lab_arena_ledger; reservation public.lab_arena_ledger;
 expected JSONB:=$expected635$[{"code_review_attempts":3,"code_review_doc":{"cost_microusd":1517778,"cost_status":"settled","error_code":"review_response_invalid","error_reason":"research_evidence","file_count":49,"invalid_episode_start":1,"model":"anthropic/claude-sonnet-5.5","retryable":false,"review_cost_microusd":4553344,"review_invalid_attempts":3,"review_output_token_recommendation":32768,"schema_version":"leadpoet.lab_arena.code_review.result.v1","source_bytes":1855272,"verdict":"error"},"code_review_status":"error","generation_id":"gen-1793920095-EfMFopCRXC86DY6jzolX","source_archive_sha256":"sha256:13822703258e768cc0116a91379968d3d8068265b527714d7ccdeb5273d3963f","source_normalized_sha256":"sha256:e9eae3c198d882f69b3eb104a8ee2dc7ef410bbc5b5536e4169fa868a49fb170","source_ref":"arena/arena-2026-11-06/sources/sub-0598bdeaa2643fc14cc2de959d7e99ca.tar.gz","source_size_bytes":466881,"submission_id":"sub-0598bdeaa2643fc14cc2de959d7e99ca"},{"code_review_attempts":3,"code_review_doc":{"cost_microusd":1521622,"cost_status":"settled","error_code":"review_response_invalid","error_reason":"research_evidence","file_count":50,"invalid_episode_start":1,"model":"anthropic/claude-sonnet-5.5","retryable":false,"review_cost_microusd":4564886,"review_invalid_attempts":3,"review_output_token_recommendation":32768,"schema_version":"leadpoet.lab_arena.code_review.result.v1","source_bytes":1859837,"verdict":"error"},"code_review_status":"error","generation_id":"gen-1793919971-k4P0oyL7NiS4zeFsQkKt","source_archive_sha256":"sha256:8e61615547e3e7b3ef98714a39d6b7ad18c2d25e876eae0659f04e69172505b4","source_normalized_sha256":"sha256:c8cb82f671afc6d5ed2fbd8eea45750c249e09e4a09a69c8d1d0de6f998b0b57","source_ref":"arena/arena-2026-11-06/sources/sub-2eafc097dca3fb8f862b86f0ea311ea4.tar.gz","source_size_bytes":468560,"submission_id":"sub-2eafc097dca3fb8f862b86f0ea311ea4"},{"code_review_attempts":3,"code_review_doc":{"cost_microusd":1517426,"cost_status":"settled","error_code":"review_response_invalid","error_reason":"research_evidence","file_count":49,"invalid_episode_start":1,"model":"anthropic/claude-sonnet-5.5","retryable":false,"review_cost_microusd":4552298,"review_invalid_attempts":3,"review_output_token_recommendation":32768,"schema_version":"leadpoet.lab_arena.code_review.result.v1","source_bytes":1854675,"verdict":"error"},"code_review_status":"error","generation_id":"gen-1793920119-JuahbT9tPzPOLJpIuifN","source_archive_sha256":"sha256:b491a4e81dd8a22fb8bbac164155e6d427c551c49e677a27b5867c7aedf273d8","source_normalized_sha256":"sha256:88cdf8dbc6d0f3f051feac8a23dada0c7361e30641b4fd35ff85d6d7eba617d4","source_ref":"arena/arena-2026-11-06/sources/sub-3b3b4f8f2bce3583b6067d6699e96126.tar.gz","source_size_bytes":466649,"submission_id":"sub-3b3b4f8f2bce3583b6067d6699e96126"},{"code_review_attempts":3,"code_review_doc":{"cost_microusd":1536866,"cost_status":"settled","error_code":"review_response_invalid","error_reason":"research_evidence_mismatch","file_count":49,"invalid_episode_start":1,"model":"anthropic/claude-sonnet-5.5","retryable":false,"review_cost_microusd":4603388,"review_invalid_attempts":3,"review_output_token_recommendation":32768,"schema_version":"leadpoet.lab_arena.code_review.result.v1","source_bytes":1854962,"verdict":"error"},"code_review_status":"error","generation_id":"gen-1793920123-yWQUseLx9t9vnRjcaMYI","source_archive_sha256":"sha256:d544e2a54bb3f8cd792fb3e46f85e3fdc09efe307fae26c65a36bf9ff48a2496","source_normalized_sha256":"sha256:3dd09d046a5c806606ef6d098a81beebb36ba9a025b1ebd79f827fe791646061","source_ref":"arena/arena-2026-11-06/sources/sub-4f9dcb8b1b1b85392283b8c8afcf68cb.tar.gz","source_size_bytes":466852,"submission_id":"sub-4f9dcb8b1b1b85392283b8c8afcf68cb"},{"code_review_attempts":3,"code_review_doc":{"cost_microusd":1544982,"cost_status":"settled","error_code":"review_response_invalid","error_reason":"research_evidence_mismatch","file_count":50,"invalid_episode_start":1,"model":"anthropic/claude-sonnet-5.5","retryable":false,"review_cost_microusd":4608476,"review_invalid_attempts":3,"review_output_token_recommendation":32768,"schema_version":"leadpoet.lab_arena.code_review.result.v1","source_bytes":1865320,"verdict":"error"},"code_review_status":"error","generation_id":"gen-1793920040-VEXu8lD7XyeJt1GKoFQf","source_archive_sha256":"sha256:06f015519295fbfd473a6c8a2bf2e3cf92dd4babf262b9423769f1390e71c44f","source_normalized_sha256":"sha256:1b767c2e490b838f1d70c3d294a3ff1fc888a132bb06ed31a64b98154602384d","source_ref":"arena/arena-2026-11-06/sources/sub-6bcdee3048e3b579150478166005b968.tar.gz","source_size_bytes":470283,"submission_id":"sub-6bcdee3048e3b579150478166005b968"},{"code_review_attempts":3,"code_review_doc":{"cost_microusd":1524802,"cost_status":"settled","error_code":"review_response_invalid","error_reason":"research_evidence","file_count":50,"invalid_episode_start":1,"model":"anthropic/claude-sonnet-5.5","retryable":false,"review_cost_microusd":4588146,"review_invalid_attempts":3,"review_output_token_recommendation":32768,"schema_version":"leadpoet.lab_arena.code_review.result.v1","source_bytes":1863842,"verdict":"error"},"code_review_status":"error","generation_id":"gen-1793920017-IrTaMYCnnqUABrRS3DWg","source_archive_sha256":"sha256:f8666bdea4c43b7175bbf6c3ea329146ffbf2a43d4b1c8cf7ba48b2e209f3041","source_normalized_sha256":"sha256:6946d386d9482605e7408c66d2cc0b8053cf1df215d8ae6e80a07771808ece3d","source_ref":"arena/arena-2026-11-06/sources/sub-8dcabdc1a405016fd93a3d4959e0ed6f.tar.gz","source_size_bytes":469832,"submission_id":"sub-8dcabdc1a405016fd93a3d4959e0ed6f"},{"code_review_attempts":3,"code_review_doc":{"cost_microusd":1532548,"cost_status":"settled","error_code":"review_response_invalid","error_reason":"research_evidence_mismatch","file_count":49,"invalid_episode_start":1,"model":"anthropic/claude-sonnet-5.5","retryable":false,"review_cost_microusd":4599084,"review_invalid_attempts":3,"review_output_token_recommendation":32768,"schema_version":"leadpoet.lab_arena.code_review.result.v1","source_bytes":1850346,"verdict":"error"},"code_review_status":"error","generation_id":"gen-1793920092-Z19NShcUsHyY61g4RzPN","source_archive_sha256":"sha256:964b7917d81dbec93f733e2a9958617766dd82a89c46457d04e6ae88e9f55034","source_normalized_sha256":"sha256:bcaa44fa4e767820cf252b631dcce12c93e85872aff57bc665cb8e876524f317","source_ref":"arena/arena-2026-11-06/sources/sub-94f9e4dc04fbeb96625d423f445d7dd7.tar.gz","source_size_bytes":465144,"submission_id":"sub-94f9e4dc04fbeb96625d423f445d7dd7"},{"code_review_attempts":3,"code_review_doc":{"cost_microusd":1520026,"cost_status":"settled","error_code":"review_response_invalid","error_reason":"research_evidence","file_count":49,"invalid_episode_start":1,"model":"anthropic/claude-sonnet-5.5","retryable":false,"review_cost_microusd":4560258,"review_invalid_attempts":3,"review_output_token_recommendation":32768,"schema_version":"leadpoet.lab_arena.code_review.result.v1","source_bytes":1857706,"verdict":"error"},"code_review_status":"error","generation_id":"gen-1793920094-DiV7zudIuQbMZO4WgrK9","source_archive_sha256":"sha256:96b6ca1426dbbf9161e483aeb79c82e254d3fc4ba2a1f28e067b2fa3389c493d","source_normalized_sha256":"sha256:c0824956d10c8b67ecf7d40d5ce93362f469d9afeac86bd68e565a4f522aa6ef","source_ref":"arena/arena-2026-11-06/sources/sub-ab2a4a5834748704157465ca6a435bf8.tar.gz","source_size_bytes":467817,"submission_id":"sub-ab2a4a5834748704157465ca6a435bf8"}]$expected635$::JSONB;
 item JSONB; sid TEXT; identity TEXT; originals JSONB:='{}'; generation BIGINT;
BEGIN
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-06' FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=r.round_id AND request_id='nov06-invalid-review635-v1';
 IF FOUND THEN
  IF h.recovery_doc->>'operation' IS DISTINCT FROM 'nov06_invalid_review635_v1'
   OR h.old_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc THEN
   RAISE EXCEPTION 'Nov06 review635 replay differs'; END IF;
  RETURN; -- A consumed fourth claim can never be granted again.
 END IF;
 IF current_setting('session_replication_role')<>'origin' OR r.status<>'open'
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR (r.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ<>'2026-11-06T00:00:00Z'::TIMESTAMPTZ
  OR clock_timestamp()>=LEAST((r.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ,
   (r.configuration_doc#>>'{schedule,benchmark_deadline}')::TIMESTAMPTZ) THEN
  RAISE EXCEPTION 'Nov06 review635 round or cutoff differs'; END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(expected) LOOP
  sid:=item->>'submission_id';
  SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=sid FOR UPDATE;
  identity:='sha256:'||encode(extensions.digest(convert_to('lab_arena.code_review:'||sid||':3','UTF8'),'sha256'),'hex');
  head:=public.lab_arena__ledger_head(identity);
  SELECT * INTO STRICT reservation FROM public.lab_arena_ledger WHERE call_identity=identity AND entry_kind='reservation';
  IF s.round_id<>r.round_id OR s.status<>'accepted' OR s.is_king OR s.frozen_at IS NOT NULL
   OR s.code_review_status<>'error' OR s.code_review_attempts<>3 OR s.code_review_expires_at IS NOT NULL
   OR s.source_archive_sha256 IS DISTINCT FROM item->>'source_archive_sha256'
   OR s.source_normalized_sha256 IS DISTINCT FROM item->>'source_normalized_sha256'
   OR s.source_ref IS DISTINCT FROM item->>'source_ref'
   OR s.source_size_bytes IS DISTINCT FROM (item->>'source_size_bytes')::BIGINT
   OR s.code_review_doc IS DISTINCT FROM item->'code_review_doc'
   OR head.entry_kind IS DISTINCT FROM 'settlement' OR head.round_id IS DISTINCT FROM r.round_id OR head.submission_id IS DISTINCT FROM sid
   OR head.operation_id IS DISTINCT FROM 'openrouter.code_review' OR head.entry_doc->>'review_status' IS DISTINCT FROM 'error'
   OR head.entry_doc->>'error_code' IS DISTINCT FROM 'review_response_invalid'
   OR head.entry_doc->>'error_reason' IS DISTINCT FROM item#>>'{code_review_doc,error_reason}'
   OR head.entry_doc#>>'{provider_diagnostic,generation_id}' IS DISTINCT FROM item->>'generation_id'
   OR head.amount_microusd IS DISTINCT FROM (item#>>'{code_review_doc,cost_microusd}')::BIGINT
   OR reservation.round_id IS DISTINCT FROM r.round_id OR reservation.submission_id IS DISTINCT FROM sid
   OR reservation.operation_id IS DISTINCT FROM 'openrouter.code_review' OR reservation.entry_doc->'attempt' IS DISTINCT FROM '3'::JSONB
   OR reservation.entry_doc->>'review_model' IS DISTINCT FROM 'anthropic/claude-sonnet-5.5'
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE submission_id=sid)
   OR (SELECT count(*)<>3 OR count(*) FILTER (WHERE l.entry_doc->>'review_model' IS DISTINCT FROM 'anthropic/claude-sonnet-5.5')<>0
    FROM public.lab_arena_ledger l WHERE l.submission_id=sid AND l.operation_id='openrouter.code_review' AND l.entry_kind='reservation') THEN
   RAISE EXCEPTION 'Nov06 review635 exact source or third receipt differs: %',sid; END IF;
  originals:=originals||jsonb_build_object(sid,jsonb_build_object('submission',to_jsonb(s),
   'call_identity',identity,'reservation',to_jsonb(reservation),'terminal',to_jsonb(head),
   'terminal_sha256',encode(extensions.digest(to_jsonb(head)::TEXT,'sha256'),'hex')));
 END LOOP;
 SELECT coalesce(max(generation_floor),0)+1 INTO generation FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,generation,'nov06-invalid-review635-v1',
  'sha256:'||encode(extensions.digest(originals::TEXT,'sha256'),'hex'),'operator:nov06-review635',
  txid_current(),pg_backend_pid(),to_jsonb(r),to_jsonb(r),'{}',jsonb_build_object(
   'operation','nov06_invalid_review635_v1','originals',originals,'max_additional_reservations',1,
   'review_model','openai/gpt-6.1-sol'));
 FOR sid IN SELECT jsonb_object_keys(originals) LOOP
  UPDATE public.lab_arena_submissions SET code_review_doc=code_review_doc||jsonb_build_object(
   'retryable',TRUE,'nov06_review_recovery','nov06-invalid-review635-v1'),updated_at=clock_timestamp()
  WHERE submission_id=sid;
 END LOOP;
END;
$recover635$;
DO $restore635$
DECLARE prior RECORD;
BEGIN
 SELECT * INTO STRICT prior FROM arena635_schema_acl;
 IF NOT prior.had_create THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM prior.nspacl THEN
  RAISE EXCEPTION 'Nov06 review635 schema ACL differs'; END IF;
END;
$restore635$;
COMMIT;
