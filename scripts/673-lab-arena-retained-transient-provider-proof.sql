-- Extend the existing finite owner grant's retained provider proof only.
-- Keep frozen runtime/payer, parent identity, budgets, checkpoints and one child.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena673_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE')AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $pre673$
DECLARE p RECORD;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure('public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)');
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'e8aab74f0d9a3ddd37d01cd3e4dc8df375030ab3fa6f4e532bdd829e3326f235'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'s' OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u') THEN
  RAISE EXCEPTION 'retained transient673 existing proof helper differs'; END IF;
END;
$pre673$;
CREATE OR REPLACE FUNCTION public.lab_arena__retained_provider_transient_v1(p_entry public.lab_arena_ledger)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $proof673$
DECLARE call_doc JSONB; receipt JSONB; document JSONB; data JSONB; source JSONB;
 raw BYTEA; request_hash TEXT; eligible BOOLEAN:=FALSE;
BEGIN
 IF p_entry.entry_kind NOT IN ('uncertain','settlement') THEN RETURN FALSE; END IF;
 call_doc:=CASE p_entry.entry_kind WHEN 'uncertain' THEN p_entry.entry_doc->'call' ELSE p_entry.terminal_response END;
 receipt:=call_doc->'upstream_response_receipt';request_hash:=receipt->>'request_hash';
 IF (jsonb_typeof(call_doc)='object' AND jsonb_typeof(receipt)='object'
  AND receipt->>'call_identity'=p_entry.call_identity
  AND receipt->>'operation_id'=p_entry.operation_id
  AND request_hash ~ '^sha256:[0-9a-f]{64}$') IS NOT TRUE THEN RETURN FALSE; END IF;
 raw:=decode(receipt->>'body_b64','base64');
 IF (receipt->>'body_sha256'='sha256:'||encode(extensions.digest(raw,'sha256'),'hex')) IS NOT TRUE THEN RETURN FALSE; END IF;
 IF NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger r
   WHERE r.run_id=p_entry.run_id AND r.round_id=p_entry.round_id AND r.call_identity=p_entry.call_identity
    AND r.entry_kind='reservation' AND r.entry_id<p_entry.entry_id
    AND r.provider=p_entry.provider AND r.operation_id=p_entry.operation_id
    AND r.funding_source=p_entry.funding_source AND r.entry_doc->>'request_hash'=request_hash)
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger d
   WHERE d.run_id=p_entry.run_id AND d.round_id=p_entry.round_id AND d.call_identity=p_entry.call_identity
    AND d.entry_kind='dispatch' AND d.entry_id<p_entry.entry_id
    AND d.provider=p_entry.provider AND d.operation_id=p_entry.operation_id
    AND d.funding_source=p_entry.funding_source) THEN RETURN FALSE; END IF;
 BEGIN document:=convert_from(raw,'UTF8')::JSONB;
 EXCEPTION WHEN invalid_text_representation OR character_not_in_repertoire THEN document:=NULL;
 END;
 -- Explicit credential, funding and managed capability denials cannot qualify.
 IF jsonb_typeof(document)='object' AND (
  document#>>'{error,code}' IN ('401','402','403','invalid_api_key','insufficient_credits')
  OR document->>'code' IN ('PROVIDER_AUTHORIZATION_FAILED','INVALID_API_KEY','INSUFFICIENT_CREDITS')
  OR document->>'upstream_status' IN ('401','402','403')
  OR document#>>'{error,metadata,provider_code}' IN ('401','402','403')
  OR document#>>'{error,type}' IN ('authentication_error','permission_error','invalid_request_error')
  OR document->>'failure_origin' IN ('caller','workspace')
  OR document->>'error_category' IN ('authentication','authorization','provider_auth','provider_account','validation')) THEN RETURN FALSE; END IF;
 -- Preserve671's shared-pool proof and add actual, failed upstream5xx replies.
 IF call_doc->'call_succeeded'='false'::JSONB THEN
  eligible:=(receipt->'call_succeeded'='false'::JSONB AND receipt->'status' IN ('500'::JSONB,'502'::JSONB,'503'::JSONB,'504'::JSONB))
   OR (p_entry.entry_kind='uncertain' AND p_entry.provider='openrouter'
    AND p_entry.operation_id='openrouter.chat' AND receipt->'status'='429'::JSONB
    AND document#>'{error,code}'='429'::JSONB
    AND document#>>'{error,metadata,limit_source}'='upstream_provider_shared_pool'
    AND document#>'{error,metadata,is_byok}'='false'::JSONB);
  IF eligible IS TRUE THEN RETURN TRUE; END IF;
 END IF;
 -- Contents API success is distinct from a failed exact-source crawl. Use the
 -- normalized protocol, regardless of which approved provider backs it.
 IF (p_entry.entry_kind='settlement' AND p_entry.operation_id='exa.contents'
  AND receipt->'status'='200'::JSONB AND call_doc->'status'='200'::JSONB
  AND call_doc->'call_succeeded'='true'::JSONB AND receipt->'call_succeeded'='true'::JSONB
  AND jsonb_typeof(document)='object') IS NOT TRUE THEN RETURN FALSE; END IF;
 IF document ? 'result' THEN
  IF document->>'status' IS DISTINCT FROM 'completed' THEN RETURN FALSE; END IF;
  data:=document#>'{result,data}';
 ELSE data:=document;
 END IF;
 IF (jsonb_typeof(data)='object' AND data->'results'='[]'::JSONB
  AND (data->'error' IS NULL OR data->'error'='null'::JSONB)
  AND (data->'errors' IS NULL OR data->'errors'='null'::JSONB)
  AND jsonb_typeof(data->'statuses')='array') IS NOT TRUE THEN RETURN FALSE; END IF;
 IF jsonb_array_length(data->'statuses')<>1 THEN RETURN FALSE; END IF;
 source:=data#>'{statuses,0}';
 RETURN coalesce(source->>'status'='error' AND jsonb_typeof(source->'id')='string'
  AND source->>'id' ~ '^https://[^[:space:]]+$'
  AND source#>'{error,httpStatusCode}' IN ('500'::JSONB,'502'::JSONB,'503'::JSONB,'504'::JSONB)
  AND (source#>>'{error,tag}'='CRAWL_HTTP_'||(source#>>'{error,httpStatusCode}')
    OR (source#>'{error,httpStatusCode}'='504'::JSONB
     AND source#>>'{error,tag}'='CRAWL_LIVECRAWL_TIMEOUT')),FALSE);
EXCEPTION WHEN invalid_parameter_value OR invalid_text_representation OR character_not_in_repertoire THEN
 -- Invalid retained bytes are ineligible, never proof of a provider fault.
 RETURN FALSE;
END;
$proof673$;
ALTER FUNCTION public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $patch673$
DECLARE p RECORD; d TEXT; metadata JSONB; digest TEXT;
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid='public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)'::regprocedure;
 metadata:=to_jsonb(p)-'prosrc';d:=p.prosrc;digest:=encode(extensions.digest(d,'sha256'),'hex');
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v' OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN RAISE EXCEPTION 'retained transient673 grant security differs'; END IF;
 IF digest='d223ae52b42ec0db4d6af3cf2a026c5a3d829456c9f20965b0820866067c00bb' THEN RETURN; END IF;
 IF digest<>'6bfec20fb4155365534ccbe7369c5fb41657115a500896d685d080f70fd84429' THEN RAISE EXCEPTION 'retained transient673 grant preimage differs'; END IF;
 IF (length(d)-length(replace(d,$old673$   OR NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger l
    WHERE l.entry_id=p_provider_entry_id AND l.run_id=parent.run_id AND l.round_id=parent.round_id
     AND l.provider='openrouter' AND l.operation_id='openrouter.chat' AND l.entry_kind='uncertain'
     AND l.entry_doc#>'{call,call_succeeded}'='false'::JSONB
     AND l.entry_doc#>'{call,upstream_response_receipt,status}'='429'::JSONB
     AND l.entry_doc#>>'{call,upstream_response_receipt,call_identity}'=l.call_identity
     AND l.entry_doc#>>'{call,upstream_response_receipt,operation_id}'=l.operation_id
     AND l.entry_doc#>>'{call,upstream_response_receipt,body_sha256}'='sha256:'||encode(extensions.digest(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'sha256'),'hex')
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>'{error,code}'='429'::JSONB
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>>'{error,metadata,limit_source}'='upstream_provider_shared_pool'
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>'{error,metadata,is_byok}'='false'::JSONB
     AND EXISTS(SELECT 1 FROM public.lab_arena_ledger dispatch WHERE dispatch.run_id=l.run_id AND dispatch.call_identity=l.call_identity AND dispatch.entry_kind='dispatch' AND dispatch.entry_id<l.entry_id AND dispatch.provider=l.provider AND dispatch.operation_id=l.operation_id AND dispatch.funding_source=l.funding_source)
     AND EXISTS(SELECT 1 FROM public.lab_arena_ledger reservation WHERE reservation.run_id=l.run_id AND reservation.call_identity=l.call_identity AND reservation.entry_kind='reservation' AND reservation.entry_id<l.entry_id AND reservation.provider=l.provider AND reservation.operation_id=l.operation_id AND reservation.funding_source=l.funding_source AND reservation.entry_doc->>'request_hash'=l.entry_doc#>>'{call,upstream_response_receipt,request_hash}'))$old673$,'')))/length($old673$   OR NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger l
    WHERE l.entry_id=p_provider_entry_id AND l.run_id=parent.run_id AND l.round_id=parent.round_id
     AND l.provider='openrouter' AND l.operation_id='openrouter.chat' AND l.entry_kind='uncertain'
     AND l.entry_doc#>'{call,call_succeeded}'='false'::JSONB
     AND l.entry_doc#>'{call,upstream_response_receipt,status}'='429'::JSONB
     AND l.entry_doc#>>'{call,upstream_response_receipt,call_identity}'=l.call_identity
     AND l.entry_doc#>>'{call,upstream_response_receipt,operation_id}'=l.operation_id
     AND l.entry_doc#>>'{call,upstream_response_receipt,body_sha256}'='sha256:'||encode(extensions.digest(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'sha256'),'hex')
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>'{error,code}'='429'::JSONB
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>>'{error,metadata,limit_source}'='upstream_provider_shared_pool'
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>'{error,metadata,is_byok}'='false'::JSONB
     AND EXISTS(SELECT 1 FROM public.lab_arena_ledger dispatch WHERE dispatch.run_id=l.run_id AND dispatch.call_identity=l.call_identity AND dispatch.entry_kind='dispatch' AND dispatch.entry_id<l.entry_id AND dispatch.provider=l.provider AND dispatch.operation_id=l.operation_id AND dispatch.funding_source=l.funding_source)
     AND EXISTS(SELECT 1 FROM public.lab_arena_ledger reservation WHERE reservation.run_id=l.run_id AND reservation.call_identity=l.call_identity AND reservation.entry_kind='reservation' AND reservation.entry_id<l.entry_id AND reservation.provider=l.provider AND reservation.operation_id=l.operation_id AND reservation.funding_source=l.funding_source AND reservation.entry_doc->>'request_hash'=l.entry_doc#>>'{call,upstream_response_receipt,request_hash}'))$old673$)<>1 THEN RAISE EXCEPTION 'retained transient673 grant anchor differs'; END IF;
 d:=replace(replace(d,$old673$   OR NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger l
    WHERE l.entry_id=p_provider_entry_id AND l.run_id=parent.run_id AND l.round_id=parent.round_id
     AND l.provider='openrouter' AND l.operation_id='openrouter.chat' AND l.entry_kind='uncertain'
     AND l.entry_doc#>'{call,call_succeeded}'='false'::JSONB
     AND l.entry_doc#>'{call,upstream_response_receipt,status}'='429'::JSONB
     AND l.entry_doc#>>'{call,upstream_response_receipt,call_identity}'=l.call_identity
     AND l.entry_doc#>>'{call,upstream_response_receipt,operation_id}'=l.operation_id
     AND l.entry_doc#>>'{call,upstream_response_receipt,body_sha256}'='sha256:'||encode(extensions.digest(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'sha256'),'hex')
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>'{error,code}'='429'::JSONB
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>>'{error,metadata,limit_source}'='upstream_provider_shared_pool'
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>'{error,metadata,is_byok}'='false'::JSONB
     AND EXISTS(SELECT 1 FROM public.lab_arena_ledger dispatch WHERE dispatch.run_id=l.run_id AND dispatch.call_identity=l.call_identity AND dispatch.entry_kind='dispatch' AND dispatch.entry_id<l.entry_id AND dispatch.provider=l.provider AND dispatch.operation_id=l.operation_id AND dispatch.funding_source=l.funding_source)
     AND EXISTS(SELECT 1 FROM public.lab_arena_ledger reservation WHERE reservation.run_id=l.run_id AND reservation.call_identity=l.call_identity AND reservation.entry_kind='reservation' AND reservation.entry_id<l.entry_id AND reservation.provider=l.provider AND reservation.operation_id=l.operation_id AND reservation.funding_source=l.funding_source AND reservation.entry_doc->>'request_hash'=l.entry_doc#>>'{call,upstream_response_receipt,request_hash}'))$old673$,$new673$   OR NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger l
    WHERE l.entry_id=p_provider_entry_id AND l.run_id=parent.run_id AND l.round_id=parent.round_id
     AND public.lab_arena__retained_provider_transient_v1(l))$new673$),'provider fault recovery retained shared-pool proof differs','provider fault recovery retained transient proof differs');
 IF encode(extensions.digest(d,'sha256'),'hex')<>'d223ae52b42ec0db4d6af3cf2a026c5a3d829456c9f20965b0820866067c00bb' THEN RAISE EXCEPTION 'retained transient673 grant postimage differs'; END IF;
 EXECUTE 'CREATE OR REPLACE FUNCTION public.lab_arena__grant_failed_judge_recovery_v1(p_parent_run_id TEXT,p_parent_sha256 TEXT,p_runtime_image_digest TEXT,p_request_id TEXT,p_actor_ref TEXT,p_correction_ref TEXT,p_dry_run BOOLEAN,p_recovery_kind TEXT,p_provider_entry_id BIGINT) RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS '||quote_literal(d);
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM metadata
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex')FROM pg_proc q WHERE q.oid=p.oid)<>'d223ae52b42ec0db4d6af3cf2a026c5a3d829456c9f20965b0820866067c00bb' THEN RAISE EXCEPTION 'retained transient673 grant readback differs'; END IF;
END;
$patch673$;
DO $restore673$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena673_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena673_acl)THEN RAISE EXCEPTION 'retained transient673 schema ACL differs'; END IF;
END;
$restore673$;
NOTIFY pgrst,'reload schema';COMMIT;
