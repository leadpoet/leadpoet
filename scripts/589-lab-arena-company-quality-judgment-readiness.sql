-- Quality refs use their existing immutable company evidence, not a full-group
-- scope. Frozen S3 input/artifact replay remains the gateway's responsibility.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $guard589$
DECLARE p pg_proc; d text;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid='public.lab_arena__all_judgments_ready_v1(text)'::regprocedure;
 d:=pg_get_functiondef(p.oid);
 IF encode(extensions.digest(d,'sha256'),'hex') NOT IN (
   '39e4ca21bb87af27fd0114b5f2d562a445ca93c2ef6ae3a515e1445dc0d6d1dc','fc3f2bf436135078c83351e3cb6f8045ba77caa35819db8944f4bc16fbb7dfec') THEN
   RAISE EXCEPTION 'Quality readiness preimage differs' USING ERRCODE='55000'; END IF;
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner'
   OR p.proacl::text IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
   OR NOT p.prosecdef OR p.provolatile<>'s'
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[] THEN
   RAISE EXCEPTION 'Quality readiness security differs' USING ERRCODE='42501'; END IF;
 SELECT * INTO STRICT p FROM pg_proc WHERE oid='public.lab_arena_record_run_scores(text,smallint,jsonb)'::regprocedure;
 IF encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex') NOT IN (
      '9d8e6f4576dbc7ba83853405e00117ea901152213610617b347f60d1d6cad684','68e0fe2b383b0295e8f121902a8d429e7a0b43b233d4357f0f242779da9fa9f8')
   OR pg_get_userbyid(p.proowner)<>'lab_arena_owner'
   OR p.proacl::text IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
   OR NOT p.prosecdef OR p.provolatile<>'v'
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[] THEN
   RAISE EXCEPTION 'Quality late-judge preimage/security differs' USING ERRCODE='55000'; END IF;
 IF to_regprocedure('public.lab_arena__quality_judgment_refs_v1(public.lab_arena_rounds,public.lab_arena_runs,public.lab_arena_runs)') IS NOT NULL THEN
   SELECT * INTO p FROM pg_proc WHERE oid='public.lab_arena__quality_judgment_refs_v1(public.lab_arena_rounds,public.lab_arena_runs,public.lab_arena_runs)'::regprocedure;
   IF encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'812d8b76d2d3579e733a2708e20fc7918e565e5c9c84062586f24d19d121cc62'
     OR pg_get_userbyid(p.proowner)<>'lab_arena_owner'
     OR p.proacl::text IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
     OR NOT p.prosecdef OR p.provolatile<>'s'
     OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[] THEN
     RAISE EXCEPTION 'Quality readiness helper differs' USING ERRCODE='55000'; END IF;
 ELSIF encode(extensions.digest(d,'sha256'),'hex')='fc3f2bf436135078c83351e3cb6f8045ba77caa35819db8944f4bc16fbb7dfec' THEN
   RAISE EXCEPTION 'Quality readiness helper missing' USING ERRCODE='55000';
 END IF;
 SELECT * INTO STRICT p FROM pg_proc WHERE oid='public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)'::regprocedure;
 IF encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex') NOT IN (
      '3a0ff0eb9209d66f766daaee4eebab92a81f5f1d05e5d189130d7ec1dc69b0af','c1dd9446626ef01989a5ae53ce4f4e28d4d1692ab2d9fb96747f388281e5649f')
   OR pg_get_userbyid(p.proowner)<>'lab_arena_owner'
   OR p.proacl::text IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
   OR NOT p.prosecdef OR p.provolatile<>'s'
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[] THEN
   RAISE EXCEPTION 'Quality continuation preimage/security differs' USING ERRCODE='55000'; END IF;
 IF to_regprocedure('public.lab_arena__quality_continuation_refs_v1(public.lab_arena_rounds,public.lab_arena_runs)') IS NOT NULL THEN
   SELECT * INTO p FROM pg_proc WHERE oid='public.lab_arena__quality_continuation_refs_v1(public.lab_arena_rounds,public.lab_arena_runs)'::regprocedure;
   IF encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'b94ec8c6ebbd7a94a2b2e3e34d31226f9fe3931e256985b9375f6504e27fa946'
     OR pg_get_userbyid(p.proowner)<>'lab_arena_owner'
     OR p.proacl::text IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
     OR NOT p.prosecdef OR p.provolatile<>'s'
     OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[] THEN
     RAISE EXCEPTION 'Quality continuation helper differs' USING ERRCODE='55000'; END IF;
 ELSIF encode(extensions.digest(pg_get_functiondef('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)'::regprocedure),'sha256'),'hex')='c1dd9446626ef01989a5ae53ce4f4e28d4d1692ab2d9fb96747f388281e5649f' THEN
   RAISE EXCEPTION 'Quality continuation helper missing' USING ERRCODE='55000';
 END IF;
 PERFORM set_config('lab_arena.quality_readiness_owner_create',
   has_schema_privilege('lab_arena_owner','public','CREATE')::text,true);
 IF NOT has_schema_privilege('lab_arena_owner','public','CREATE') THEN
   GRANT CREATE ON SCHEMA public TO lab_arena_owner;
 END IF;
END $guard589$;

CREATE OR REPLACE FUNCTION public.lab_arena__quality_judgment_refs_v1(
 p_round public.lab_arena_rounds,p_judge public.lab_arena_runs,p_source public.lab_arena_runs)
RETURNS boolean LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog, public AS $refs589$
DECLARE ref jsonb; ordinal bigint; scope jsonb; context jsonb;
BEGIN
 IF p_round.configuration_doc->>'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1'
   OR p_round.configuration_doc->>'company_quality_policy' IS DISTINCT FROM 'company_quality_v1'
   OR p_round.configuration_doc#>>'{scorer_policy,company_quality_policy}' IS DISTINCT FROM 'company_quality_v1'
   OR p_round.configuration_doc ? 'company_judgment_policy'
   OR (p_round.configuration_doc->'scorer_policy') ? 'company_judgment_policy'
   OR coalesce(p_round.configuration_doc#>>'{scorer_policy,scoring_adapter_version}','') NOT IN
       ('qualification_integrity_v2','qualification_contacts_v3')
   -- Contacts are encoded by the adapter, never a nested policy field.
   OR (p_round.configuration_doc->'scorer_policy') ? 'contact_policy'
   OR (CASE WHEN p_round.configuration_doc ? 'contact_policy' THEN
         p_round.configuration_doc->>'contact_policy' IS DISTINCT FROM 'contacts_v1'
         OR p_round.configuration_doc#>>'{scorer_policy,scoring_adapter_version}' IS DISTINCT FROM 'qualification_contacts_v3'
       ELSE p_round.configuration_doc#>>'{scorer_policy,scoring_adapter_version}' IS DISTINCT FROM 'qualification_integrity_v2' END)
   OR jsonb_typeof(p_judge.company_judgment_refs) IS DISTINCT FROM 'array'
   OR jsonb_array_length(p_judge.company_judgment_refs)>5
   OR p_judge.judgment_cache_key IS NOT NULL OR p_judge.judgment_input_hash IS NOT NULL
   OR p_judge.judgment_scope_doc IS NOT NULL
   OR p_judge.scoring_generation_floor IS DISTINCT FROM p_round.scoring_generation_floor
   OR (p_judge.round_id,p_judge.submission_id,p_judge.miner_hotkey,p_judge.stage,p_judge.icp_position,p_judge.scored_run_id)
       IS DISTINCT FROM (p_round.round_id,p_source.submission_id,p_source.miner_hotkey,p_source.stage,p_source.icp_position,p_source.run_id)
   OR p_judge.kind IS DISTINCT FROM 'score' OR p_judge.status IS DISTINCT FROM 'accepted' OR p_judge.terminal_cause IS DISTINCT FROM 'accepted'
   OR p_judge.output_ref IS NULL OR p_source.kind IS DISTINCT FROM 'execute'
   OR p_source.round_id IS DISTINCT FROM p_round.round_id OR p_source.status IS DISTINCT FROM 'accepted'
   OR p_source.terminal_cause IS DISTINCT FROM 'accepted' OR p_source.output_ref IS NULL
 THEN RETURN FALSE; END IF;
 context:=p_judge.claim_response->'company_judgment_cache';
 IF jsonb_typeof(context) IS DISTINCT FROM 'object'
   OR context->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.company_judgment_lease.v1'
   OR jsonb_typeof(context->'hits') IS DISTINCT FROM 'array'
   OR jsonb_typeof(context->'misses') IS DISTINCT FROM 'array' THEN RETURN FALSE; END IF;
 IF jsonb_array_length(p_judge.company_judgment_refs) <>
    jsonb_array_length(context->'hits')+jsonb_array_length(context->'misses')
 THEN RETURN FALSE; END IF;
 FOR ref,ordinal IN SELECT value,ordinality FROM jsonb_array_elements(p_judge.company_judgment_refs) WITH ORDINALITY LOOP
   IF jsonb_typeof(ref) IS DISTINCT FROM 'object'
     OR coalesce(ref->>'cache_key','') !~ '^sha256:[0-9a-f]{64}$'
     OR coalesce(ref->>'company_input_hash','') !~ '^sha256:[0-9a-f]{64}$'
   THEN RETURN FALSE; END IF;
   scope:=jsonb_build_object('schema_version','leadpoet.lab_arena.company_judgment_scope.v1',
     'integrity_policy','arena_integrity_v1','company_quality_policy','company_quality_v1',
     'round_id',p_round.round_id,'network_name',p_round.arena_network_name,'netuid',p_round.arena_netuid,
     'scorer_image_digest',p_round.configuration_doc->>'scorer_image_digest',
     'scorer_image_reference',p_round.configuration_doc->>'scorer_image_reference',
     'evaluation_date',p_round.evaluation_date,'company_input_hash',ref->>'company_input_hash',
     'cache_key',ref->>'cache_key');
   IF ref IS DISTINCT FROM jsonb_build_object('schema_version','leadpoet.lab_arena.company_judgment_ref.v1',
       'company_index',ordinal-1,'cache_key',ref->>'cache_key',
       'company_input_hash',ref->>'company_input_hash','scope_doc',scope)
   THEN RETURN FALSE; END IF;
   IF NOT EXISTS (
     SELECT 1 FROM public.lab_arena_company_judgments cache
     JOIN public.lab_arena_runs producer ON producer.run_id=cache.source_score_run_id
     JOIN public.lab_arena_runs execution ON execution.run_id=cache.source_scored_run_id
     WHERE cache.cache_key=ref->>'cache_key' AND cache.company_input_hash=ref->>'company_input_hash'
       AND cache.scope_doc=scope
       AND cache.evidence_hash ~ '^sha256:[0-9a-f]{64}$'
       AND cache.evidence_doc->>'schema_version'='leadpoet.lab_arena.company_judgment_evidence.v1'
       AND cache.evidence_doc->>'cache_key'=cache.cache_key
       AND cache.evidence_doc->>'company_input_hash'=cache.company_input_hash
       AND cache.evidence_doc->'authority_slot'=to_jsonb(cache.authority_slot)
       AND jsonb_typeof(cache.evidence_doc->'raw_judgment')='object'
       AND NOT (cache.evidence_doc->'raw_judgment') ? 'company_judgment_policy'
       AND coalesce(cache.evidence_doc->>'source_output_hash','') ~ '^sha256:[0-9a-f]{64}$'
       AND coalesce(cache.evidence_doc->>'source_completion_request_hash','') ~ '^sha256:[0-9a-f]{64}$'
       AND EXISTS(SELECT 1 FROM jsonb_array_elements(producer.claim_response#>'{company_judgment_cache,misses}') miss
         WHERE miss->>'cache_key'=cache.cache_key AND miss->>'company_input_hash'=cache.company_input_hash
           AND miss->'authority_slot'=to_jsonb(cache.authority_slot))
       AND (EXISTS(SELECT 1 FROM jsonb_array_elements(p_judge.claim_response#>'{company_judgment_cache,hits}') hit
           WHERE hit->'company_index'=ref->'company_index' AND hit->>'cache_key'=cache.cache_key
             AND hit->>'company_input_hash'=cache.company_input_hash
             AND hit->'authority_slot'=to_jsonb(cache.authority_slot)
             AND hit->>'evidence_hash'=cache.evidence_hash AND hit->'evidence_doc'=cache.evidence_doc)
         OR p_judge.run_id=producer.run_id AND EXISTS(
           SELECT 1 FROM jsonb_array_elements(p_judge.claim_response#>'{company_judgment_cache,misses}') miss
           WHERE miss->'company_index'=ref->'company_index' AND miss->>'cache_key'=cache.cache_key
             AND miss->>'company_input_hash'=cache.company_input_hash
             AND miss->'authority_slot'=to_jsonb(cache.authority_slot)))
       AND producer.kind='score' AND producer.scored_run_id=execution.run_id
       AND execution.kind='execute' AND execution.status='accepted' AND execution.terminal_cause='accepted'
       AND (producer.round_id,producer.submission_id,producer.miner_hotkey,producer.stage,producer.icp_position)
          IS NOT DISTINCT FROM (execution.round_id,execution.submission_id,execution.miner_hotkey,execution.stage,execution.icp_position)
       AND producer.round_id=p_round.round_id
       AND producer.runner_hotkey=cache.source_runner_hotkey
       AND cache.evidence_doc->>'source_score_run_id'=producer.run_id
       AND cache.evidence_doc->>'source_scored_run_id'=execution.run_id
       AND cache.evidence_doc->>'source_runner_hotkey'=producer.runner_hotkey
       AND cache.evidence_doc->>'source_output_ref'=producer.output_ref
       AND cache.evidence_doc->>'source_claim_request_id'=producer.claim_request_id
       AND cache.evidence_doc->>'source_claim_request_hash'=producer.claim_request_hash
       AND cache.evidence_doc->'source_lease_generation'=to_jsonb(producer.lease_generation)
       AND cache.evidence_doc->'runner_authority_exclusions'=producer.claim_response->'runner_authority_exclusions'
       AND (EXISTS(SELECT 1 FROM jsonb_array_elements(p_round.participants) participant
             WHERE participant->>'submission_id'=p_source.submission_id AND participant->'is_king'='true'::jsonb)
         OR NOT cache.evidence_doc->'runner_authority_exclusions' @> jsonb_build_array(p_source.miner_hotkey))
       AND EXISTS(SELECT 1 FROM jsonb_array_elements(producer.company_judgment_refs) original
         WHERE original->>'cache_key'=cache.cache_key AND original->>'company_input_hash'=cache.company_input_hash
           AND original->'scope_doc'=scope)
       AND (producer.status='accepted' AND producer.terminal_cause='accepted'
         OR producer.status='failed' AND producer.terminal_cause IN ('judge_error','judge_timeout','credential_error')
           AND (producer.result_doc->>'terminal_status'=producer.terminal_cause
             OR producer.terminal_cause='judge_error' AND producer.result_doc->>'terminal_status'='judge_timeout')
           AND producer.result_doc->>'company_judgment_completion_request_hash'=
               cache.evidence_doc->>'source_completion_request_hash')
   ) THEN RETURN FALSE; END IF;
 END LOOP;
 RETURN TRUE;
END $refs589$;
ALTER FUNCTION public.lab_arena__quality_judgment_refs_v1(public.lab_arena_rounds,public.lab_arena_runs,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__quality_judgment_refs_v1(public.lab_arena_rounds,public.lab_arena_runs,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Continuation validates the frozen reference inventory, not accepted results.
-- A failed partial attempt legitimately has missing cache evidence. Acceptance
-- still uses the separate immutable-evidence helper above and gateway replay.
CREATE OR REPLACE FUNCTION public.lab_arena__quality_continuation_refs_v1(
 p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS boolean LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog, public AS $continuation_refs589$
DECLARE c jsonb:=p_round.configuration_doc; p_refs jsonb:=p_run.company_judgment_refs;
 ref jsonb; ordinal bigint; scope jsonb; context jsonb;
BEGIN
 IF jsonb_typeof(c) IS DISTINCT FROM 'object'
   OR jsonb_typeof(c->'scorer_policy') IS DISTINCT FROM 'object'
   OR c->>'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1'
   OR c->>'company_quality_policy' IS DISTINCT FROM 'company_quality_v1'
   OR c#>>'{scorer_policy,company_quality_policy}' IS DISTINCT FROM 'company_quality_v1'
   OR c ? 'company_judgment_policy' OR (c->'scorer_policy') ? 'company_judgment_policy'
   OR coalesce(c#>>'{scorer_policy,scoring_adapter_version}','') NOT IN
       ('qualification_integrity_v2','qualification_contacts_v3')
   OR (c->'scorer_policy') ? 'contact_policy'
   OR (CASE WHEN c ? 'contact_policy' THEN
         c->>'contact_policy' IS DISTINCT FROM 'contacts_v1'
         OR c#>>'{scorer_policy,scoring_adapter_version}' IS DISTINCT FROM 'qualification_contacts_v3'
       ELSE c#>>'{scorer_policy,scoring_adapter_version}' IS DISTINCT FROM 'qualification_integrity_v2' END)
   OR p_round.round_id IS NULL OR p_round.arena_network_name IS NULL
   OR p_round.arena_netuid IS NULL OR p_round.evaluation_date IS NULL
   OR jsonb_typeof(c->'scorer_image_digest') IS DISTINCT FROM 'string'
   OR jsonb_typeof(c->'scorer_image_reference') IS DISTINCT FROM 'string'
   OR jsonb_typeof(p_refs) IS DISTINCT FROM 'array' THEN RETURN FALSE; END IF;
 IF jsonb_array_length(p_refs)>5 THEN RETURN FALSE; END IF;
 -- Unclaimed continuation rows have no claim yet; their exact inventory must
 -- match the failed parent in the outer predicate. A retained claim, including
 -- a failed partial completion, must prove its own complete ordered inventory.
 context:=p_run.claim_response->'company_judgment_cache';
 IF p_run.claim_response IS NOT NULL AND (
   jsonb_typeof(context) IS DISTINCT FROM 'object'
   OR context->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.company_judgment_lease.v1'
   OR jsonb_typeof(context->'hits') IS DISTINCT FROM 'array'
   OR jsonb_typeof(context->'misses') IS DISTINCT FROM 'array') THEN RETURN FALSE; END IF;
 IF context IS NOT NULL AND jsonb_array_length(p_refs) <>
    jsonb_array_length(context->'hits')+jsonb_array_length(context->'misses')
 THEN RETURN FALSE; END IF;
 FOR ref,ordinal IN SELECT value,ordinality FROM jsonb_array_elements(p_refs) WITH ORDINALITY LOOP
   IF jsonb_typeof(ref) IS DISTINCT FROM 'object'
     OR jsonb_typeof(ref->'cache_key') IS DISTINCT FROM 'string'
     OR jsonb_typeof(ref->'company_input_hash') IS DISTINCT FROM 'string'
     OR coalesce(ref->>'cache_key','') !~ '^sha256:[0-9a-f]{64}$'
     OR coalesce(ref->>'company_input_hash','') !~ '^sha256:[0-9a-f]{64}$'
   THEN RETURN FALSE; END IF;
   scope:=jsonb_build_object('schema_version','leadpoet.lab_arena.company_judgment_scope.v1',
     'integrity_policy','arena_integrity_v1','company_quality_policy','company_quality_v1',
     'round_id',p_round.round_id,'network_name',p_round.arena_network_name,'netuid',p_round.arena_netuid,
     'scorer_image_digest',c->>'scorer_image_digest','scorer_image_reference',c->>'scorer_image_reference',
     'evaluation_date',p_round.evaluation_date,'company_input_hash',ref->>'company_input_hash',
     'cache_key',ref->>'cache_key');
   IF ref IS DISTINCT FROM jsonb_build_object('schema_version','leadpoet.lab_arena.company_judgment_ref.v1',
       'company_index',ordinal-1,'cache_key',ref->>'cache_key',
       'company_input_hash',ref->>'company_input_hash','scope_doc',scope)
   THEN RETURN FALSE; END IF;
   IF context IS NOT NULL AND NOT EXISTS (
     SELECT 1 FROM jsonb_array_elements((context->'hits')||(context->'misses')) entry
     WHERE entry->'company_index'=ref->'company_index'
       AND entry->'cache_key'=ref->'cache_key'
       AND entry->'company_input_hash'=ref->'company_input_hash')
   THEN RETURN FALSE; END IF;
 END LOOP;
 RETURN TRUE;
END $continuation_refs589$;
ALTER FUNCTION public.lab_arena__quality_continuation_refs_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__quality_continuation_refs_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $restore589$
BEGIN
 IF current_setting('lab_arena.quality_readiness_owner_create')='false' THEN
   REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
END $restore589$;

DO $patch589$
DECLARE d text; old text:=$old589$                ELSE (
                  NOT (validation.round_row).configuration_doc ? 'company_judgment_policy'$old589$;
 new text:=$new589$                WHEN (validation.round_row).configuration_doc->>'company_quality_policy'='company_quality_v1' THEN
                  public.lab_arena__quality_judgment_refs_v1(validation.round_row,judge,source)
                ELSE (
                  NOT (validation.round_row).configuration_doc ? 'company_judgment_policy'$new589$;
BEGIN
 SELECT pg_get_functiondef('public.lab_arena__all_judgments_ready_v1(text)'::regprocedure) INTO d;
 IF strpos(d,new)>0 THEN RETURN; END IF;
 IF (length(d)-length(replace(d,old,'')))<>length(old) THEN
   RAISE EXCEPTION 'Quality readiness anchor differs' USING ERRCODE='55000'; END IF;
 EXECUTE replace(d,old,new);
END $patch589$;
-- Only the existing late_judge proof gets a quality alternative. Preserve
-- supersession and late_baseline admission, row identity and all write fences.
DO $late589$
DECLARE d text; old text:=$old589$          ELSE NOT v_round.configuration_doc ? 'company_judgment_policy'$old589$;
 new text:=$new589$          WHEN v_round.configuration_doc->>'company_quality_policy'='company_quality_v1' THEN
            public.lab_arena__quality_judgment_refs_v1(v_round,judged,v_run)
          ELSE NOT v_round.configuration_doc ? 'company_judgment_policy'$new589$;
BEGIN
 SELECT pg_get_functiondef('public.lab_arena_record_run_scores(text,smallint,jsonb)'::regprocedure) INTO d;
 IF strpos(d,new)>0 THEN RETURN; END IF;
 IF (length(d)-length(replace(d,old,'')))<>length(old) THEN
   RAISE EXCEPTION 'Quality late-judge anchor differs' USING ERRCODE='55000'; END IF;
 EXECUTE replace(d,old,new);
END $late589$;

-- Only replace the scope representation check. The original continuation
-- source/plan/participant/generation/parent checks and every caller's existing
-- cost, lease, deadline and real-clock backoff fences remain unchanged.
DO $continuation589$
DECLARE d text; old text:=$old589$    OR (CASE WHEN public.lab_arena__company_judgment_policy_v1(p_round) THEN$old589$;
 new text:=$new589$    OR (CASE WHEN p_round.configuration_doc->>'company_quality_policy'='company_quality_v1'
      OR p_round.configuration_doc#>>'{scorer_policy,company_quality_policy}'='company_quality_v1' THEN
      NOT public.lab_arena__quality_continuation_refs_v1(p_round,p_run)
      OR p_run.judgment_cache_key IS NOT NULL OR p_run.judgment_input_hash IS NOT NULL
      OR p_run.judgment_scope_doc IS NOT NULL
     WHEN public.lab_arena__company_judgment_policy_v1(p_round) THEN$new589$;
BEGIN
 SELECT pg_get_functiondef('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)'::regprocedure) INTO d;
 IF strpos(d,new)>0 THEN RETURN; END IF;
 IF (length(d)-length(replace(d,old,'')))<>length(old) THEN
   RAISE EXCEPTION 'Quality continuation anchor differs' USING ERRCODE='55000'; END IF;
 EXECUTE replace(d,old,new);
END $continuation589$;

DO $verify589$
DECLARE item record; p pg_proc;
BEGIN
 FOR item IN SELECT * FROM (VALUES
   ('public.lab_arena__all_judgments_ready_v1(text)',
    'fc3f2bf436135078c83351e3cb6f8045ba77caa35819db8944f4bc16fbb7dfec','sql'),
   ('public.lab_arena__quality_judgment_refs_v1(public.lab_arena_rounds,public.lab_arena_runs,public.lab_arena_runs)',
    '812d8b76d2d3579e733a2708e20fc7918e565e5c9c84062586f24d19d121cc62','plpgsql'),
   ('public.lab_arena_record_run_scores(text,smallint,jsonb)','68e0fe2b383b0295e8f121902a8d429e7a0b43b233d4357f0f242779da9fa9f8','plpgsql'),
   ('public.lab_arena__quality_continuation_refs_v1(public.lab_arena_rounds,public.lab_arena_runs)','b94ec8c6ebbd7a94a2b2e3e34d31226f9fe3931e256985b9375f6504e27fa946','plpgsql'),
   ('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)','c1dd9446626ef01989a5ae53ce4f4e28d4d1692ab2d9fb96747f388281e5649f','plpgsql')) AS expected(signature,hash,language)
 LOOP
   SELECT * INTO STRICT p FROM pg_proc WHERE oid=item.signature::regprocedure;
   IF encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>item.hash
     OR pg_get_userbyid(p.proowner)<>'lab_arena_owner'
     OR p.proacl::text IS DISTINCT FROM (CASE WHEN item.signature='public.lab_arena_record_run_scores(text,smallint,jsonb)'
       THEN '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' ELSE '{lab_arena_owner=X/lab_arena_owner}' END)
     OR NOT p.prosecdef OR p.provolatile<>(CASE WHEN item.signature='public.lab_arena_record_run_scores(text,smallint,jsonb)' THEN 'v' ELSE 's' END)
     OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[]
     OR (SELECT lanname FROM pg_language WHERE oid=p.prolang)<>item.language THEN
     RAISE EXCEPTION 'Quality readiness postimage/security differs: %',item.signature USING ERRCODE='55000';
   END IF;
 END LOOP;
END $verify589$;
COMMIT;
