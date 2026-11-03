-- Bind company-quality continuation judgments to their frozen ordered refs.
-- Existing legacy cache policy, accepted evidence, scores and costs do not change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
CREATE TEMP TABLE arena589_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $helper_preflight589$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__company_quality_refs_v1(public.lab_arena_rounds,jsonb)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'16fa8ed0b29a34ffc8ae50fa8ba42c907cbb61f9faa57875b658eb2cc4a6bc8b'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Arena quality judgment helper differs' USING ERRCODE='55000'; END IF;
END;
$helper_preflight589$;
CREATE OR REPLACE FUNCTION public.lab_arena__company_quality_refs_v1(p_round public.lab_arena_rounds,p_refs JSONB)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $quality589$
DECLARE c JSONB:=p_round.configuration_doc; ref JSONB; scope JSONB; ordinal INTEGER:=0;
BEGIN
 IF jsonb_typeof(c) IS DISTINCT FROM 'object'
  OR jsonb_typeof(c->'scorer_policy') IS DISTINCT FROM 'object'
  OR c->'integrity_policy' IS DISTINCT FROM '"arena_integrity_v1"'::JSONB
  OR c->'company_quality_policy' IS DISTINCT FROM '"company_quality_v1"'::JSONB
  OR c#>'{scorer_policy,company_quality_policy}' IS DISTINCT FROM '"company_quality_v1"'::JSONB
  OR c ? 'company_judgment_policy' OR (c->'scorer_policy') ? 'company_judgment_policy'
  OR p_round.round_id IS NULL OR p_round.arena_network_name IS NULL
  OR p_round.arena_netuid IS NULL OR p_round.evaluation_date IS NULL
  OR jsonb_typeof(c->'scorer_image_digest') IS DISTINCT FROM 'string'
  OR jsonb_typeof(c->'scorer_image_reference') IS DISTINCT FROM 'string'
  OR jsonb_typeof(p_refs) IS DISTINCT FROM 'array' THEN RETURN FALSE; END IF;
 IF jsonb_array_length(p_refs)>5 THEN RETURN FALSE; END IF;
 FOR ref IN SELECT value FROM jsonb_array_elements(p_refs) LOOP
  scope:=ref->'scope_doc';
  IF jsonb_typeof(ref) IS DISTINCT FROM 'object'
   OR ref->'schema_version' IS DISTINCT FROM '"leadpoet.lab_arena.company_judgment_ref.v1"'::JSONB
   OR ref->'company_index' IS DISTINCT FROM to_jsonb(ordinal)
   OR jsonb_typeof(ref->'cache_key') IS DISTINCT FROM 'string'
   OR ref->>'cache_key' !~ '^sha256:[0-9a-f]{64}$'
   OR jsonb_typeof(ref->'company_input_hash') IS DISTINCT FROM 'string'
   OR ref->>'company_input_hash' !~ '^sha256:[0-9a-f]{64}$'
   OR jsonb_typeof(scope) IS DISTINCT FROM 'object'
   OR scope->'cache_key' IS DISTINCT FROM ref->'cache_key'
   OR scope->'company_input_hash' IS DISTINCT FROM ref->'company_input_hash'
   OR scope->'round_id' IS DISTINCT FROM to_jsonb(p_round.round_id)
   OR scope->'network_name' IS DISTINCT FROM to_jsonb(p_round.arena_network_name)
   OR scope->'netuid' IS DISTINCT FROM to_jsonb(p_round.arena_netuid)
   OR scope->'evaluation_date' IS DISTINCT FROM to_jsonb(p_round.evaluation_date)
   OR scope->'integrity_policy' IS DISTINCT FROM '"arena_integrity_v1"'::JSONB
   OR scope->'company_quality_policy' IS DISTINCT FROM '"company_quality_v1"'::JSONB
   OR scope ? 'company_judgment_policy'
   OR scope->'scorer_image_digest' IS DISTINCT FROM c->'scorer_image_digest'
   OR scope->'scorer_image_reference' IS DISTINCT FROM c->'scorer_image_reference'
   THEN RETURN FALSE; END IF;
  ordinal:=ordinal+1;
 END LOOP;
 RETURN TRUE;
END;
$quality589$;
ALTER FUNCTION public.lab_arena__company_quality_refs_v1(public.lab_arena_rounds,JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__company_quality_refs_v1(public.lab_arena_rounds,JSONB)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $helper_verify589$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid='public.lab_arena__company_quality_refs_v1(public.lab_arena_rounds,jsonb)'::regprocedure
   AND encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')='16fa8ed0b29a34ffc8ae50fa8ba42c907cbb61f9faa57875b658eb2cc4a6bc8b'
   AND pg_get_userbyid(p.proowner)='lab_arena_owner' AND p.prosecdef AND p.provolatile='s'
   AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
   AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[] AND l.lanname='plpgsql') THEN
  RAISE EXCEPTION 'Arena quality judgment helper readback differs' USING ERRCODE='55000'; END IF;
END;
$helper_verify589$;
DO $patch_judgments_589$
DECLARE target OID:='public.lab_arena__all_judgments_ready_v1(text)'::regprocedure; d TEXT; metadata JSONB; expected TEXT;
 old_text TEXT:=$old589$
                AND (CASE WHEN public.lab_arena__company_judgment_policy_v1(validation.round_row) THEN$old589$;
 new_text TEXT:=$new589$
                AND (CASE WHEN (validation.round_row).configuration_doc ? 'company_quality_policy'
                  OR ((validation.round_row).configuration_doc->'scorer_policy') ? 'company_quality_policy' THEN
                  public.lab_arena__company_quality_refs_v1(validation.round_row,judge.company_judgment_refs)
                  AND judge.judgment_cache_key IS NULL AND judge.judgment_input_hash IS NULL
                  AND judge.judgment_scope_doc IS NULL
                 WHEN public.lab_arena__company_judgment_policy_v1(validation.round_row) THEN$new589$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
     FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target)
   IS DISTINCT FROM $security589$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]$security589$::JSONB THEN
  RAISE EXCEPTION 'Arena quality judgment security differs: judgments' USING ERRCODE='42501'; END IF;
 expected:=encode(extensions.digest(d,'sha256'),'hex');
 IF expected='68a8126d0729337cd2e12adc8bb4be0b7cfd1eadc8ca9935032bf869fd7bd33e' THEN RETURN; END IF;
 IF expected<>'39e4ca21bb87af27fd0114b5f2d562a445ca93c2ef6ae3a515e1445dc0d6d1dc' THEN
  RAISE EXCEPTION 'Arena quality judgment preimage differs: judgments' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,old_text,'')))<>length(old_text) THEN
  RAISE EXCEPTION 'Arena quality judgment anchor differs: judgments' USING ERRCODE='55000'; END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'68a8126d0729337cd2e12adc8bb4be0b7cfd1eadc8ca9935032bf869fd7bd33e' THEN
  RAISE EXCEPTION 'Arena quality judgment postimage differs: judgments' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'68a8126d0729337cd2e12adc8bb4be0b7cfd1eadc8ca9935032bf869fd7bd33e' THEN
  RAISE EXCEPTION 'Arena quality judgment readback differs: judgments' USING ERRCODE='55000'; END IF;
END;
$patch_judgments_589$;

DO $patch_scores_589$
DECLARE target OID:='public.lab_arena_record_run_scores(text,smallint,jsonb)'::regprocedure; d TEXT; metadata JSONB; expected TEXT;
 old_text TEXT:=$old589$
          AND (CASE WHEN public.lab_arena__company_judgment_policy_v1(v_round) THEN$old589$;
 new_text TEXT:=$new589$
          AND (CASE WHEN v_round.configuration_doc ? 'company_quality_policy'
            OR (v_round.configuration_doc->'scorer_policy') ? 'company_quality_policy' THEN
            public.lab_arena__company_quality_refs_v1(v_round,judged.company_judgment_refs)
            AND judged.judgment_cache_key IS NULL AND judged.judgment_input_hash IS NULL
            AND judged.judgment_scope_doc IS NULL
           WHEN public.lab_arena__company_judgment_policy_v1(v_round) THEN$new589$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
     FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target)
   IS DISTINCT FROM $security589$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]$security589$::JSONB THEN
  RAISE EXCEPTION 'Arena quality judgment security differs: scores' USING ERRCODE='42501'; END IF;
 expected:=encode(extensions.digest(d,'sha256'),'hex');
 IF expected='759119aafc1f0041da8faa38eb4066abc0d10206e7192956f9741ef989e94834' THEN RETURN; END IF;
 IF expected<>'9d8e6f4576dbc7ba83853405e00117ea901152213610617b347f60d1d6cad684' THEN
  RAISE EXCEPTION 'Arena quality judgment preimage differs: scores' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,old_text,'')))<>length(old_text) THEN
  RAISE EXCEPTION 'Arena quality judgment anchor differs: scores' USING ERRCODE='55000'; END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'759119aafc1f0041da8faa38eb4066abc0d10206e7192956f9741ef989e94834' THEN
  RAISE EXCEPTION 'Arena quality judgment postimage differs: scores' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'759119aafc1f0041da8faa38eb4066abc0d10206e7192956f9741ef989e94834' THEN
  RAISE EXCEPTION 'Arena quality judgment readback differs: scores' USING ERRCODE='55000'; END IF;
END;
$patch_scores_589$;

DO $patch_incomplete_589$
DECLARE target OID:='public.lab_arena__participant_incomplete_reason_v1(text,text)'::regprocedure; d TEXT; metadata JSONB; expected TEXT;
 old_text TEXT:=$old589$
     OR (CASE WHEN public.lab_arena__company_judgment_policy_v1(r) THEN$old589$;
 new_text TEXT:=$new589$
     OR (CASE WHEN r.configuration_doc ? 'company_quality_policy'
       OR (r.configuration_doc->'scorer_policy') ? 'company_quality_policy' THEN
       public.lab_arena__company_quality_refs_v1(r,judge.company_judgment_refs)
       AND judge.judgment_cache_key IS NULL AND judge.judgment_input_hash IS NULL
       AND judge.judgment_scope_doc IS NULL
      WHEN public.lab_arena__company_judgment_policy_v1(r) THEN$new589$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
     FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target)
   IS DISTINCT FROM $security589$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]$security589$::JSONB THEN
  RAISE EXCEPTION 'Arena quality judgment security differs: incomplete' USING ERRCODE='42501'; END IF;
 expected:=encode(extensions.digest(d,'sha256'),'hex');
 IF expected='f083615eea5f2b3e440ce7bed39ef7111b1ef5fe242bb685a79febaab0ca757a' THEN RETURN; END IF;
 IF expected<>'f0947c0d204d0883d926781cfe03ba22ad2b940ea16cf7618de90a9533cd44c4' THEN
  RAISE EXCEPTION 'Arena quality judgment preimage differs: incomplete' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,old_text,'')))<>length(old_text) THEN
  RAISE EXCEPTION 'Arena quality judgment anchor differs: incomplete' USING ERRCODE='55000'; END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'f083615eea5f2b3e440ce7bed39ef7111b1ef5fe242bb685a79febaab0ca757a' THEN
  RAISE EXCEPTION 'Arena quality judgment postimage differs: incomplete' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'f083615eea5f2b3e440ce7bed39ef7111b1ef5fe242bb685a79febaab0ca757a' THEN
  RAISE EXCEPTION 'Arena quality judgment readback differs: incomplete' USING ERRCODE='55000'; END IF;
END;
$patch_incomplete_589$;

DO $patch_continuation_589$
DECLARE target OID:='public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)'::regprocedure;
 d TEXT; metadata JSONB; expected TEXT;
 old_text TEXT:=$old589$    OR (CASE WHEN public.lab_arena__company_judgment_policy_v1(p_round) THEN$old589$;
 new_text TEXT:=$new589$    OR (CASE WHEN p_round.configuration_doc ? 'company_quality_policy'
      OR (p_round.configuration_doc->'scorer_policy') ? 'company_quality_policy' THEN
      NOT public.lab_arena__company_quality_refs_v1(p_round,p_run.company_judgment_refs)
      OR p_run.judgment_cache_key IS NOT NULL OR p_run.judgment_input_hash IS NOT NULL
      OR p_run.judgment_scope_doc IS NOT NULL
     WHEN public.lab_arena__company_judgment_policy_v1(p_round) THEN$new589$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
     FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target)
   IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
  RAISE EXCEPTION 'Arena quality judgment security differs: continuation' USING ERRCODE='42501'; END IF;
 expected:=encode(extensions.digest(d,'sha256'),'hex');
 IF expected='dd24da9152f1787e8a556b2ec972feeff6d5c73938c2f0b071cb67db39a78246' THEN RETURN; END IF;
 IF expected<>'3a0ff0eb9209d66f766daaee4eebab92a81f5f1d05e5d189130d7ec1dc69b0af' THEN
  RAISE EXCEPTION 'Arena quality judgment preimage differs: continuation' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,old_text,'')))<>length(old_text) THEN
  RAISE EXCEPTION 'Arena quality judgment anchor differs: continuation' USING ERRCODE='55000'; END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'dd24da9152f1787e8a556b2ec972feeff6d5c73938c2f0b071cb67db39a78246' THEN
  RAISE EXCEPTION 'Arena quality judgment postimage differs: continuation' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'dd24da9152f1787e8a556b2ec972feeff6d5c73938c2f0b071cb67db39a78246' THEN
  RAISE EXCEPTION 'Arena quality judgment readback differs: continuation' USING ERRCODE='55000'; END IF;
END;
$patch_continuation_589$;

DO $restore589$
BEGIN
 IF NOT (SELECT had_create FROM arena589_schema_acl) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
    (SELECT nspacl FROM arena589_schema_acl) THEN
  RAISE EXCEPTION 'Arena quality judgment schema ACL differs' USING ERRCODE='42501'; END IF;
END;
$restore589$;
NOTIFY pgrst,'reload schema';
COMMIT;
