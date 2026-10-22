-- Preserve completed Arena participants when another miner remains unmeasured.
-- No historical rows, accepted work, scores, provider receipts or costs change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE arena540_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $preflight_lab_arena__execution_privacy_route_unavailable_v1$
DECLARE p RECORD;
BEGIN
 SELECT a.*,l.lanname INTO p FROM pg_proc a JOIN pg_language l ON l.oid=a.prolang
 WHERE a.oid=to_regprocedure('public.lab_arena__execution_privacy_route_unavailable_v1(text)');
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'4ac0ef13106cfe30274bc06247d830d384bc8693314ccbe0a619c49b24886bd5'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Arena incomplete helper source/security differs'; END IF;
END;
$preflight_lab_arena__execution_privacy_route_unavailable_v1$;
CREATE OR REPLACE FUNCTION public.lab_arena__execution_privacy_route_unavailable_v1(p_run_id TEXT) RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $body540$
DECLARE issuance public.lab_arena_ledger; terminal public.lab_arena_ledger; body JSON; is_accepted BOOLEAN;

BEGIN
 SELECT status='accepted' INTO is_accepted FROM public.lab_arena_runs WHERE run_id=p_run_id;
 SELECT * INTO issuance FROM public.lab_arena_ledger l WHERE l.run_id=p_run_id
  AND l.provider='openrouter' AND l.entry_kind='reservation'
  ORDER BY (l.entry_doc->'action_sequence') DESC NULLS FIRST,
   (l.entry_doc->'provider_attempt') DESC NULLS LAST,l.entry_id DESC LIMIT 1;
 IF NOT FOUND THEN RETURN FALSE; END IF;
 IF issuance.operation_id NOT IN('openrouter.chat','openrouter.responses')
  OR jsonb_typeof(issuance.entry_doc->'action_sequence') IS DISTINCT FROM 'number'
  OR coalesce(issuance.entry_doc->>'action_sequence','') !~ '^[0-9]+$'
  OR jsonb_typeof(issuance.entry_doc->'request_hash') IS DISTINCT FROM 'string'
  OR issuance.call_identity IS NULL THEN
  IF NOT coalesce(is_accepted,TRUE) THEN RETURN FALSE; END IF;
  RAISE EXCEPTION 'Arena execution receipt issuance invalid' USING ERRCODE='22023'; END IF;
 SELECT * INTO terminal FROM public.lab_arena_ledger l WHERE l.call_identity=issuance.call_identity
  ORDER BY l.entry_id DESC LIMIT 1;
 IF terminal.entry_kind IS DISTINCT FROM 'settlement'
  OR (SELECT count(*) FROM public.lab_arena_ledger l WHERE l.call_identity=issuance.call_identity)>=20
  OR EXISTS(SELECT kind FROM unnest(ARRAY['reservation','dispatch','settlement'])kind
    WHERE (SELECT count(*) FROM public.lab_arena_ledger l WHERE l.call_identity=issuance.call_identity AND l.entry_kind=kind)<>1)
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.call_identity=issuance.call_identity
   AND l.entry_kind='dispatch' AND issuance.entry_id<l.entry_id AND l.entry_id<terminal.entry_id)
  OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.call_identity=issuance.call_identity
   AND (l.run_id,l.round_id,l.submission_id,l.provider,l.operation_id,l.funding_source) IS DISTINCT FROM
       (issuance.run_id,issuance.round_id,issuance.submission_id,issuance.provider,issuance.operation_id,issuance.funding_source))
  OR jsonb_typeof(terminal.terminal_response) IS DISTINCT FROM 'object'
  OR jsonb_typeof(terminal.terminal_response->'headers') IS DISTINCT FROM 'object'
  OR jsonb_typeof(terminal.terminal_response->'body_b64') IS DISTINCT FROM 'string'
  OR EXISTS(SELECT 1 FROM jsonb_each(CASE WHEN jsonb_typeof(terminal.terminal_response->'headers')='object'
      THEN terminal.terminal_response->'headers' ELSE '{}'::JSONB END) header WHERE jsonb_typeof(header.value) IS DISTINCT FROM 'string')
  OR jsonb_typeof(terminal.terminal_response->'status') IS DISTINCT FROM 'number'
  OR coalesce(terminal.terminal_response->>'status','') !~ '^[1-5][0-9][0-9]$' THEN
  IF NOT coalesce(is_accepted,TRUE) THEN RETURN FALSE; END IF;
  RAISE EXCEPTION 'Arena execution receipt binding invalid' USING ERRCODE='22023'; END IF;
 IF terminal.terminal_response->'status' IS DISTINCT FROM '404'::JSONB THEN RETURN FALSE; END IF;
 BEGIN
  body:=convert_from(decode(terminal.terminal_response->>'body_b64','base64'),'UTF8')::JSON;
 EXCEPTION WHEN invalid_text_representation OR character_not_in_repertoire OR invalid_parameter_value THEN
  IF NOT coalesce(is_accepted,TRUE) THEN RETURN FALSE; END IF;
  RAISE;
 END;
 IF json_typeof(body#>'{error,code}') IS DISTINCT FROM 'number'
  OR (body#>'{error,code}')::TEXT IS DISTINCT FROM '404' OR coalesce(body#>>'{error,message}','') NOT IN(
  'No endpoints found matching your data policy.',
  'No endpoints found matching your data policy (Zero data retention).',
  'No endpoints found matching your data policy. Configure: https://openrouter.ai/settings/privacy',
  'No endpoints found matching your data policy (Zero data retention). Configure: https://openrouter.ai/settings/privacy') THEN
  RETURN FALSE; END IF;
 IF terminal.terminal_response->'call_succeeded' IS DISTINCT FROM 'false'::JSONB THEN
  IF NOT coalesce(is_accepted,TRUE) THEN RETURN FALSE; END IF;
  RAISE EXCEPTION 'Arena execution receipt outcome invalid' USING ERRCODE='22023'; END IF;
 RETURN TRUE;
END;
$body540$;
ALTER FUNCTION public.lab_arena__execution_privacy_route_unavailable_v1(text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__execution_privacy_route_unavailable_v1(text) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $preflight_lab_arena__participant_incomplete_reason_v1$
DECLARE p RECORD;
BEGIN
 SELECT a.*,l.lanname INTO p FROM pg_proc a JOIN pg_language l ON l.oid=a.prolang
 WHERE a.oid=to_regprocedure('public.lab_arena__participant_incomplete_reason_v1(text,text)');
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'e33ae122311cdb01f42082c0b6b21ab62464797209eb6d2fc1a4f9b56574d1c0'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Arena incomplete helper source/security differs'; END IF;
END;
$preflight_lab_arena__participant_incomplete_reason_v1$;
CREATE OR REPLACE FUNCTION public.lab_arena__participant_incomplete_reason_v1(p_round_id TEXT,p_submission_id TEXT) RETURNS TEXT
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $body540$
DECLARE r public.lab_arena_rounds; source public.lab_arena_runs; judge public.lab_arena_runs;
 position INTEGER; total INTEGER; reason TEXT; cause TEXT; route BOOLEAN;
BEGIN
 SELECT * INTO r FROM public.lab_arena_rounds WHERE round_id=p_round_id;
 IF NOT FOUND OR r.status='cancelled'
  OR r.configuration_doc->>'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1'
  OR jsonb_typeof(r.configuration_doc->'sourcing_cost_eligibility_policy') IS DISTINCT FROM 'string'
  OR r.configuration_doc->>'sourcing_cost_eligibility_policy' NOT IN('successful_calls_per_icp_v1','successful_calls_flat_per_icp_v1')
  OR jsonb_typeof(r.configuration_doc#>'{schedule,publication_deadline}') IS DISTINCT FROM 'string'
  OR now()<(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants)p WHERE p->>'submission_id'=p_submission_id
   AND NOT coalesce((p->>'is_king')::BOOLEAN,FALSE)) THEN RETURN NULL; END IF;
 -- The immutable publication is the finalized authority for promotion and
 -- reward gates. Historical publications without this typed outcome get no
 -- new omission authority; late receipts cannot change a finalized outcome.
 IF r.status='published' THEN
  IF jsonb_typeof(r.publication_doc->'final_ranking') IS DISTINCT FROM 'array' THEN RETURN NULL; END IF;
  SELECT CASE WHEN count(*)=1 AND bool_and(
    ranked->'final_score'='null'::JSONB AND ranked->'cost_summary'='null'::JSONB
    AND ranked->'eligible'='false'::JSONB AND ranked->>'eligibility_reason'='execution_incomplete'
    AND ranked->>'evaluation'='not_evaluated'
    AND ranked->>'evaluation_reason' IN('openrouter_zdr_route_unavailable',
      'stage_closed','judge_error','judge_timeout','lease_expired','worker_lost','provider_error',
      'credential_error','result_rejected','judge_missing','judge_pending','execution_missing')
    AND ranked->>'evaluation_status'=CASE WHEN ranked->>'evaluation_reason'='openrouter_zdr_route_unavailable'
       THEN 'provider_blocked' ELSE 'evaluation_incomplete' END)
   THEN min(ranked->>'evaluation_reason') END INTO reason
   FROM jsonb_array_elements(r.publication_doc->'final_ranking')ranked
   WHERE ranked->>'submission_id'=p_submission_id;
  RETURN reason;
 END IF;
 total:=(r.configuration_doc->>'stage_1_icp_count')::INTEGER+(r.configuration_doc->>'stage_2_icp_count')::INTEGER;
 IF total NOT BETWEEN 2 AND 100 THEN RETURN NULL; END IF;
 FOR position IN 0..total-1 LOOP
  SELECT * INTO source FROM public.lab_arena_runs a WHERE a.round_id=p_round_id
   AND a.submission_id=p_submission_id AND a.kind='execute' AND a.icp_position=position
   ORDER BY (a.status='accepted') DESC,a.attempt DESC,a.run_id DESC LIMIT 1;
  IF NOT FOUND THEN reason:=coalesce(reason,'execution_missing'); CONTINUE; END IF;
  IF source.status IN('leased','submitted') THEN RETURN NULL; END IF;
  -- Validate every retained scored ICP, including the complete portion of an
  -- incomplete participant. An omission never masks malformed scored evidence.
  IF source.per_icp_score IS NOT NULL AND NOT coalesce((public.lab_arena__integrity_submission_summary(
    p_round_id,p_submission_id,ARRAY[position])->>'valid')::BOOLEAN,FALSE) THEN RETURN NULL; END IF;
  SELECT * INTO judge FROM public.lab_arena_runs j WHERE j.round_id=p_round_id
   AND j.kind='score' AND j.scoring_generation_floor=r.scoring_generation_floor
   AND (j.scored_run_id=source.run_id OR j.assignment_id=source.assignment_id||':score')
   ORDER BY (j.status='accepted') DESC,j.stage_generation DESC,j.attempt DESC,j.run_id DESC LIMIT 1;
  IF source.status='accepted' THEN
   IF source.terminal_cause IS DISTINCT FROM 'accepted' OR source.output_ref IS NULL THEN RETURN NULL; END IF;
   IF judge.status='accepted' THEN
    -- The independent finality helper validates current accepted binding.
    -- Its materialized score is required even for a route-blocked empty result.
    IF source.per_icp_score IS NULL OR judge.terminal_cause IS DISTINCT FROM 'accepted'
     OR judge.output_ref IS NULL OR judge.scored_run_id IS DISTINCT FROM source.run_id
     OR (judge.submission_id,judge.miner_hotkey,judge.stage,judge.icp_position) IS DISTINCT FROM
        (source.submission_id,source.miner_hotkey,source.stage,source.icp_position)
     OR NOT public.lab_arena__qualification_doc_valid(source.qualification_doc)
     OR (CASE WHEN public.lab_arena__company_judgment_policy_v1(r) THEN
       public.lab_arena__company_judgment_refs_v1(r,judge.company_judgment_refs)
       AND judge.judgment_cache_key IS NULL AND judge.judgment_input_hash IS NULL AND judge.judgment_scope_doc IS NULL
      ELSE judge.judgment_scope_doc->>'scorer_image_digest'=r.configuration_doc->>'scorer_image_digest'
       AND judge.judgment_scope_doc->>'scorer_image_reference'=r.configuration_doc->>'scorer_image_reference'
     END) IS NOT TRUE THEN RETURN NULL; END IF;
   ELSIF source.per_icp_score IS NOT NULL THEN RETURN NULL;
   END IF;
   route:=FALSE;
   IF (source.qualification_doc->'companies'='[]'::JSONB AND public.lab_arena__company_judgment_policy_v1(r)
     AND judge.company_judgment_refs='[]'::JSONB) OR judge.status IS DISTINCT FROM 'accepted' THEN
    route:=public.lab_arena__execution_privacy_route_unavailable_v1(source.run_id);
   END IF;
   IF route THEN reason:='openrouter_zdr_route_unavailable';
   ELSIF judge.status='accepted' THEN CONTINUE;
   ELSE
    IF judge.run_id IS NOT NULL AND judge.status NOT IN('pending','failed') THEN RETURN NULL; END IF;
    cause:=coalesce(judge.terminal_cause,CASE WHEN judge.run_id IS NULL THEN 'judge_missing' ELSE 'judge_pending' END);
    IF cause NOT IN('stage_closed','judge_error','judge_timeout','lease_expired','worker_lost','provider_error','credential_error','result_rejected','judge_missing','judge_pending') THEN RETURN NULL; END IF;
    reason:=coalesce(reason,cause);
   END IF;
  ELSE
   IF judge.status='accepted' THEN RETURN NULL; END IF;
   cause:=coalesce(source.terminal_cause,'');
   IF source.status IS DISTINCT FROM 'failed' OR cause NOT IN(
     'lease_expired','worker_lost','result_rejected','provider_error','judge_error','judge_timeout','stage_closed') THEN
    -- A genuine miner-caused terminal zero must be materialized normally.
    IF source.per_icp_score IS NOT NULL THEN CONTINUE; END IF;
    RETURN NULL;
   END IF;
   -- A recorded miner-caused failure on a prior attempt retains its zero.
   IF EXISTS(SELECT 1 FROM public.lab_arena_runs earlier WHERE earlier.assignment_id=source.assignment_id
     AND earlier.terminal_cause IN('model_timeout','invalid_output','budget_exhausted','credential_error','model_error')) THEN
    IF source.per_icp_score IS NOT NULL THEN CONTINUE; END IF;
    RETURN NULL; END IF;
   IF cause IN('provider_error','stage_closed') AND public.lab_arena__execution_privacy_route_unavailable_v1(source.run_id) THEN
    reason:='openrouter_zdr_route_unavailable';
   ELSE reason:=coalesce(reason,cause); END IF;
  END IF;
 END LOOP;
 RETURN reason;
END;
$body540$;
ALTER FUNCTION public.lab_arena__participant_incomplete_reason_v1(text,text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__participant_incomplete_reason_v1(text,text) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $patch_lab_arena__publication_execution_incomplete_v1_540$
DECLARE target OID:=to_regprocedure('public.lab_arena__publication_execution_incomplete_v1(text,text)'); d TEXT; metadata JSONB; expected TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL OR pg_get_userbyid((metadata->>'proowner')::OID)<>'lab_arena_owner' THEN
  RAISE EXCEPTION 'Arena participant isolation function missing/owner differs'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
     FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target)
    IS DISTINCT FROM $security540$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]$security540$::JSONB THEN
  RAISE EXCEPTION 'Arena participant isolation security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='7e75295c727e3c62243f9de6c2dcfb422718fe3bec539871d1bb8ebc62075c2e' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'5a276ceaafe5f9de1da40d4aae39df36f5f5e17c0e9157375216e4b88c13ac72' THEN
  RAISE EXCEPTION 'Arena participant isolation preimage differs: lab_arena__publication_execution_incomplete_v1'; END IF;
 IF (length(d)-length(replace(d,$old540_0$  v_count := (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER$old540_0$,'')))<>length($old540_0$  v_count := (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER$old540_0$) THEN
  RAISE EXCEPTION 'Arena participant isolation anchor differs'; END IF;
 d:=replace(d,$old540_0$  v_count := (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER$old540_0$,$new540_0$  -- Participant isolation540 uses the same NULL publication proof.
  IF public.lab_arena__participant_incomplete_reason_v1(p_round_id,p_submission_id) IS NOT NULL THEN
    RETURN TRUE;
  END IF;
  v_count := (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER$new540_0$);
 expected:=d;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR pg_get_functiondef(target) IS DISTINCT FROM expected THEN
  RAISE EXCEPTION 'Arena participant isolation readback differs'; END IF;
END;
$patch_lab_arena__publication_execution_incomplete_v1_540$;
DO $patch_lab_arena__all_judgments_ready_v1_540$
DECLARE target OID:=to_regprocedure('public.lab_arena__all_judgments_ready_v1(text)'); d TEXT; metadata JSONB; expected TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL OR pg_get_userbyid((metadata->>'proowner')::OID)<>'lab_arena_owner' THEN
  RAISE EXCEPTION 'Arena participant isolation function missing/owner differs'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
     FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target)
    IS DISTINCT FROM $security540$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]$security540$::JSONB THEN
  RAISE EXCEPTION 'Arena participant isolation security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='5188eedfb5b60116979b9a81a203b972384df208a21cf301749240e5a9f4e071' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'b873cbbadd22825ddf4515b2c10248d558c92405d3112418452562d7ff2b728a' THEN
  RAISE EXCEPTION 'Arena participant isolation preimage differs: lab_arena__all_judgments_ready_v1'; END IF;
 IF (length(d)-length(replace(d,$old540_0$  SELECT NOT EXISTS (SELECT 1 FROM public.lab_arena_runs source$old540_0$,'')))<>length($old540_0$  SELECT NOT EXISTS (SELECT 1 FROM public.lab_arena_runs source$old540_0$) THEN
  RAISE EXCEPTION 'Arena participant isolation anchor differs'; END IF;
 d:=replace(d,$old540_0$  SELECT NOT EXISTS (SELECT 1 FROM public.lab_arena_runs source$old540_0$,$new540_0$  WITH incomplete540 AS MATERIALIZED (
    SELECT participant->>'submission_id' AS submission_id,
      public.lab_arena__participant_incomplete_reason_v1(p_round_id,participant->>'submission_id') AS reason
    FROM public.lab_arena_rounds r,jsonb_array_elements(r.participants)participant WHERE r.round_id=p_round_id)
  SELECT NOT EXISTS (SELECT 1 FROM public.lab_arena_runs source$new540_0$);
 IF (length(d)-length(replace(d,$old540_1$WHERE source.round_id=p_round_id AND source.kind='execute'$old540_1$,'')))<>length($old540_1$WHERE source.round_id=p_round_id AND source.kind='execute'$old540_1$) THEN
  RAISE EXCEPTION 'Arena participant isolation anchor differs'; END IF;
 d:=replace(d,$old540_1$WHERE source.round_id=p_round_id AND source.kind='execute'$old540_1$,$new540_1$WHERE source.round_id=p_round_id AND source.kind='execute'
      AND NOT EXISTS(SELECT 1 FROM incomplete540 i WHERE i.submission_id=source.submission_id AND i.reason IS NOT NULL)$new540_1$);
 expected:=d;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR pg_get_functiondef(target) IS DISTINCT FROM expected THEN
  RAISE EXCEPTION 'Arena participant isolation readback differs'; END IF;
END;
$patch_lab_arena__all_judgments_ready_v1_540$;
DO $patch_lab_arena_transition_round_540$
DECLARE target OID:=to_regprocedure('public.lab_arena_transition_round(text,text,text,jsonb)'); d TEXT; metadata JSONB; expected TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL OR pg_get_userbyid((metadata->>'proowner')::OID)<>'lab_arena_owner' THEN
  RAISE EXCEPTION 'Arena participant isolation function missing/owner differs'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
     FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target)
    IS DISTINCT FROM $security540$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public","statement_timeout=600s"],"plpgsql"]$security540$::JSONB THEN
  RAISE EXCEPTION 'Arena participant isolation security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='7fd92dc70dfad3a16090d9ad40ac739c9e94d74db3f81b6ac6c311baeac2c4cd' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'1ab1ebf04fbc1920ca9667681f385c7e48e6e55b82ac19aa364249c8c9decb07' THEN
  RAISE EXCEPTION 'Arena participant isolation preimage differs: lab_arena_transition_round'; END IF;
 IF (length(d)-length(replace(d,$old540_0$  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;$old540_0$,'')))<>length($old540_0$  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;$old540_0$) THEN
  RAISE EXCEPTION 'Arena participant isolation anchor differs'; END IF;
 d:=replace(d,$old540_0$  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;$old540_0$,$new540_0$  -- Use the canonical claim/reservation lock order for publication540.
  IF p_expected_status='scored' AND p_next_status='published' THEN
    PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;$new540_0$);
 IF (length(d)-length(replace(d,$old540_1$    v_allowed := ARRAY['publication_doc', 'published_at'];$old540_1$,'')))<>length($old540_1$    v_allowed := ARRAY['publication_doc', 'published_at'];$old540_1$) THEN
  RAISE EXCEPTION 'Arena participant isolation anchor differs'; END IF;
 d:=replace(d,$old540_1$    v_allowed := ARRAY['publication_doc', 'published_at'];$old540_1$,$new540_1$    -- Atomic participant finalization540. Native accounting and active leases
    -- drain before the immutable publication becomes the final authority.
    PERFORM 1 FROM public.lab_arena_runs a WHERE a.round_id=p_round_id ORDER BY a.run_id FOR UPDATE;
    IF EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.status IN('leased','submitted'))
      OR EXISTS(WITH local_heads AS MATERIALIZED(
        SELECT DISTINCT ON(l.call_identity) l.call_identity,l.entry_id,l.entry_kind
        FROM public.lab_arena_ledger l WHERE l.round_id=p_round_id AND l.call_identity IS NOT NULL
        ORDER BY l.call_identity,l.entry_id DESC)
        SELECT 1 FROM local_heads h WHERE h.entry_kind IN('reservation','dispatch')
         AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger later WHERE later.call_identity=h.call_identity AND later.entry_id>h.entry_id)) THEN
      RETURN jsonb_build_object('status','retry','reason','evaluation_draining');
    END IF;
    IF EXISTS(SELECT 1 FROM public.lab_arena_runs judge JOIN public.lab_arena_runs source ON source.run_id=judge.scored_run_id
      WHERE judge.round_id=p_round_id AND judge.kind='score' AND judge.scoring_generation_floor=v_round.scoring_generation_floor
       AND judge.status='accepted' AND source.status<>'accepted'
       AND EXISTS(SELECT 1 FROM jsonb_array_elements(v_round.participants)p WHERE p->>'submission_id'=source.submission_id)) THEN
      RAISE EXCEPTION 'lab_arena_accepted_judgment_source_invalid' USING ERRCODE='22023';
    END IF;
    -- Every SQL-proved gap must appear once as NOT EVALUATED, with its actual
    -- cause. Caller labels cannot authorize omission of a completed score.
    IF EXISTS(SELECT 1 FROM jsonb_array_elements(v_round.participants)p
      CROSS JOIN LATERAL(SELECT public.lab_arena__participant_incomplete_reason_v1(p_round_id,p->>'submission_id') AS reason) proof
      WHERE proof.reason IS NOT NULL AND (SELECT count(*) FROM jsonb_array_elements(v_patch#>'{publication_doc,final_ranking}')ranked
       WHERE ranked->>'submission_id'=p->>'submission_id' AND ranked->'final_score'='null'::JSONB
        AND ranked->'cost_summary'='null'::JSONB AND ranked->'eligible'='false'::JSONB
        AND ranked->>'eligibility_reason'='execution_incomplete' AND ranked->>'evaluation'='not_evaluated'
        AND ranked->>'evaluation_reason'=proof.reason
        AND ranked->>'evaluation_status'=CASE WHEN proof.reason='openrouter_zdr_route_unavailable' THEN 'provider_blocked' ELSE 'evaluation_incomplete' END)<>1)
      OR EXISTS(SELECT 1 FROM jsonb_array_elements(v_patch#>'{publication_doc,final_ranking}')ranked
       WHERE ranked ? 'evaluation' AND (ranked->>'evaluation' IS DISTINCT FROM 'not_evaluated'
        OR public.lab_arena__participant_incomplete_reason_v1(p_round_id,ranked->>'submission_id') IS NULL)) THEN
      RAISE EXCEPTION 'lab_arena_incomplete_participant_publication_invalid' USING ERRCODE='22023';
    END IF;
    v_allowed := ARRAY['publication_doc', 'published_at'];$new540_1$);
 expected:=d;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR pg_get_functiondef(target) IS DISTINCT FROM expected THEN
  RAISE EXCEPTION 'Arena participant isolation readback differs'; END IF;
END;
$patch_lab_arena_transition_round_540$;
DO $patch_lab_arena_record_run_scores_540$
DECLARE target OID:=to_regprocedure('public.lab_arena_record_run_scores(text,smallint,jsonb)'); d TEXT; metadata JSONB; expected TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL OR pg_get_userbyid((metadata->>'proowner')::OID)<>'lab_arena_owner' THEN
  RAISE EXCEPTION 'Arena participant isolation function missing/owner differs'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
     FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target)
    IS DISTINCT FROM $security540$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]$security540$::JSONB THEN
  RAISE EXCEPTION 'Arena participant isolation security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='9d8e6f4576dbc7ba83853405e00117ea901152213610617b347f60d1d6cad684' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'678461f0d511beb5714fd05fdadd515b25af0aa5010a6059c6b228242dfda6e8' THEN
  RAISE EXCEPTION 'Arena participant isolation preimage differs: lab_arena_record_run_scores'; END IF;
 IF (length(d)-length(replace(d,$old540_0$          AND judged.judgment_scope_doc->>'scorer_image_digest'=
            v_round.configuration_doc->>'scorer_image_digest'
          AND judged.judgment_scope_doc->>'scorer_image_reference'=
            v_round.configuration_doc->>'scorer_image_reference'$old540_0$,'')))<>length($old540_0$          AND judged.judgment_scope_doc->>'scorer_image_digest'=
            v_round.configuration_doc->>'scorer_image_digest'
          AND judged.judgment_scope_doc->>'scorer_image_reference'=
            v_round.configuration_doc->>'scorer_image_reference'$old540_0$) THEN
  RAISE EXCEPTION 'Arena participant isolation anchor differs'; END IF;
 d:=replace(d,$old540_0$          AND judged.judgment_scope_doc->>'scorer_image_digest'=
            v_round.configuration_doc->>'scorer_image_digest'
          AND judged.judgment_scope_doc->>'scorer_image_reference'=
            v_round.configuration_doc->>'scorer_image_reference'$old540_0$,$new540_0$          -- Late company-policy decisions use their original ordered refs.
          AND judged.scoring_generation_floor=v_round.scoring_generation_floor
          AND (CASE WHEN public.lab_arena__company_judgment_policy_v1(v_round) THEN
            public.lab_arena__company_judgment_refs_v1(v_round,judged.company_judgment_refs)
            AND judged.judgment_cache_key IS NULL AND judged.judgment_input_hash IS NULL
            AND judged.judgment_scope_doc IS NULL
          ELSE NOT v_round.configuration_doc ? 'company_judgment_policy'
            AND NOT (v_round.configuration_doc->'scorer_policy') ? 'company_judgment_policy'
            AND judged.judgment_scope_doc->>'scorer_image_digest'=
              v_round.configuration_doc->>'scorer_image_digest'
            AND judged.judgment_scope_doc->>'scorer_image_reference'=
              v_round.configuration_doc->>'scorer_image_reference'
          END)$new540_0$);
 expected:=d;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR pg_get_functiondef(target) IS DISTINCT FROM expected THEN
  RAISE EXCEPTION 'Arena participant isolation readback differs'; END IF;
END;
$patch_lab_arena_record_run_scores_540$;
DO $restore540$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena540_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.arena540_schema_acl) THEN
  RAISE EXCEPTION 'Arena participant isolation schema ACL changed'; END IF;
END;
$restore540$;
NOTIFY pgrst,'reload schema';
COMMIT;
