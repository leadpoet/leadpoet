-- Keep validated unscored miner judge gaps NOT EVALUATED despite unknown billing.
-- No score, cost, receipt, privacy proof or saved round/result is changed.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $unscored664$
DECLARE target OID:='public.lab_arena__participant_incomplete_reason_v1(text,text)'::regprocedure;
 definition TEXT; body TEXT; metadata JSONB; strict_before JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc,to_jsonb(p)-'prosrc'
 INTO STRICT definition,body,metadata FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
 WHERE p.oid=target AND pg_get_userbyid(p.proowner)='lab_arena_owner'
  AND p.prosecdef AND p.provolatile='s'
  AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
  AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[] AND l.lanname='plpgsql';
 SELECT to_jsonb(p) INTO STRICT strict_before FROM pg_proc p
  WHERE p.oid='public.lab_arena__execution_privacy_route_unavailable_v1(text)'::regprocedure;
 IF encode(extensions.digest(body,'sha256'),'hex')='3eafb43ffb22a98e64640c0cce6bd0298a7c8f0f3ed13d724871c00a6ef581e5' THEN RETURN; END IF;
 IF encode(extensions.digest(body,'sha256'),'hex')<>'4066cdc359bc23fedcf1efc7a002410231bb9f79080bad3e7f7b2c45cdd1124f'
  OR length(definition)-length(replace(definition,$decl664$ position INTEGER; total INTEGER; reason TEXT; cause TEXT; route BOOLEAN;$decl664$,''))<>length($decl664$ position INTEGER; total INTEGER; reason TEXT; cause TEXT; route BOOLEAN;$decl664$)
  OR length(definition)-length(replace(definition,$old664$    route:=public.lab_arena__execution_privacy_route_unavailable_v1(source.run_id);$old664$,''))<>length($old664$    route:=public.lab_arena__execution_privacy_route_unavailable_v1(source.run_id);$old664$) THEN
  RAISE EXCEPTION 'Unscored uncertain omission664 preimage differs' USING ERRCODE='55000';
 END IF;
 definition:=replace(definition,$decl664$ position INTEGER; total INTEGER; reason TEXT; cause TEXT; route BOOLEAN;$decl664$,$newdecl664$ position INTEGER; total INTEGER; reason TEXT; cause TEXT; route BOOLEAN;
 issuance public.lab_arena_ledger; terminal public.lab_arena_ledger; route_unknown BOOLEAN;$newdecl664$);
 definition:=replace(definition,$old664$    route:=public.lab_arena__execution_privacy_route_unavailable_v1(source.run_id);$old664$,$new664$    route_unknown:=FALSE;
    -- An exhausted miner judgment is NOT EVALUATED, not an inferred score.
    -- A genuine receipt-less uncertain call cannot prove a privacy-route cause.
    -- Validate its complete local lifecycle before omitting this diagnostic;
    -- settled/malformed receipts and all measured/baseline paths stay strict.
    IF source.stage=2 AND source.per_icp_score IS NULL AND source.qualification_doc IS NULL
      AND judge.status IS DISTINCT FROM 'accepted'
      AND EXISTS(SELECT 1 FROM public.lab_arena_submissions ordinary
        WHERE ordinary.submission_id=source.submission_id AND ordinary.round_id=r.round_id
          AND NOT ordinary.is_king) THEN
     SELECT * INTO issuance FROM public.lab_arena_ledger l WHERE l.run_id=source.run_id
       AND l.provider='openrouter' AND l.entry_kind='reservation'
       ORDER BY (l.entry_doc->'action_sequence') DESC NULLS FIRST,
         (l.entry_doc->'provider_attempt') DESC NULLS LAST,l.entry_id DESC LIMIT 1;
     IF FOUND THEN
      SELECT * INTO terminal FROM public.lab_arena_ledger l
        WHERE l.call_identity=issuance.call_identity ORDER BY l.entry_id DESC LIMIT 1;
      IF terminal.entry_kind='uncertain' AND terminal.terminal_response IS NULL THEN
       IF issuance.operation_id NOT IN('openrouter.chat','openrouter.responses')
        OR jsonb_typeof(issuance.entry_doc->'action_sequence') IS DISTINCT FROM 'number'
        OR coalesce(issuance.entry_doc->>'action_sequence','') !~ '^[0-9]+$'
        OR jsonb_typeof(issuance.entry_doc->'request_hash') IS DISTINCT FROM 'string'
        OR coalesce(issuance.entry_doc->>'request_hash','') !~ '^sha256:[0-9a-f]{64}$'
        OR coalesce(issuance.call_identity,'') !~ '^sha256:[0-9a-f]{64}$'
        OR (issuance.run_id,issuance.round_id,issuance.submission_id,issuance.miner_hotkey,issuance.stage)
          IS DISTINCT FROM (source.run_id,source.round_id,source.submission_id,source.miner_hotkey,source.stage)
        OR (SELECT count(*) FROM public.lab_arena_ledger l WHERE l.call_identity=issuance.call_identity)<>3
        OR EXISTS(SELECT kind FROM unnest(ARRAY['reservation','dispatch','uncertain'])kind
          WHERE (SELECT count(*) FROM public.lab_arena_ledger l
            WHERE l.call_identity=issuance.call_identity AND l.entry_kind=kind)<>1)
        OR NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger l
          WHERE l.call_identity=issuance.call_identity AND l.entry_kind='dispatch'
            AND issuance.entry_id<l.entry_id AND l.entry_id<terminal.entry_id)
        OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.call_identity=issuance.call_identity
          AND (l.run_id,l.round_id,l.submission_id,l.miner_hotkey,l.stage,l.provider,l.operation_id,l.funding_source)
            IS DISTINCT FROM (issuance.run_id,issuance.round_id,issuance.submission_id,issuance.miner_hotkey,
              issuance.stage,issuance.provider,issuance.operation_id,issuance.funding_source)) THEN
        RAISE EXCEPTION 'Arena unscored uncertain receipt binding invalid' USING ERRCODE='22023';
       END IF;
       route_unknown:=TRUE;
      END IF;
     END IF;
    END IF;
    IF NOT route_unknown THEN
     route:=public.lab_arena__execution_privacy_route_unavailable_v1(source.run_id);
    END IF;$new664$);
 EXECUTE definition;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR (SELECT encode(extensions.digest(p.prosrc,'sha256'),'hex') FROM pg_proc p WHERE p.oid=target)<>'3eafb43ffb22a98e64640c0cce6bd0298a7c8f0f3ed13d724871c00a6ef581e5'
  OR (SELECT to_jsonb(p) FROM pg_proc p WHERE p.oid='public.lab_arena__execution_privacy_route_unavailable_v1(text)'::regprocedure)
    IS DISTINCT FROM strict_before THEN
  RAISE EXCEPTION 'Unscored uncertain omission664 readback differs' USING ERRCODE='55000';
 END IF;
END;
$unscored664$;
NOTIFY pgrst,'reload schema';
COMMIT;
