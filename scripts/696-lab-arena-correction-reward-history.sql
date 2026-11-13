-- Keep the exact signed authority of an audited champion correction available
-- to historical weight readers. Return the existing reward-reader projection;
-- never expose the private round, parent manifest or operational receipt.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
CREATE TEMP TABLE arena696_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $installed696$
DECLARE p RECORD;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure('public.lab_arena_correction_reward_history_v1(text,text,integer,integer,integer,boolean)');
 IF FOUND AND(pg_get_userbyid(p.proowner)<>'lab_arena_owner'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u'
  OR encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'9d377d88142dd0c38f46590508aaf737b8c811478c66be2eab5df0081644daa7')THEN
  RAISE EXCEPTION 'correction696 installed reader source/security differs';END IF;
END;
$installed696$;

CREATE OR REPLACE FUNCTION public.lab_arena_correction_reward_history_v1(
 p_mode TEXT DEFAULT NULL,p_network_name TEXT DEFAULT NULL,p_netuid INTEGER DEFAULT NULL,
 p_limit INTEGER DEFAULT 200,p_offset INTEGER DEFAULT 0,p_public_only BOOLEAN DEFAULT FALSE)
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $history696$
DECLARE h RECORD;r public.lab_arena_rounds;n JSONB;basis JSONB;item JSONB;items JSONB:='[]';
 field TEXT;
BEGIN
 IF (p_network_name IS NULL)<>(p_netuid IS NULL) OR p_limit IS NULL OR p_limit NOT BETWEEN 1 AND 200
  OR p_offset IS NULL OR p_offset<0 OR p_public_only IS NULL THEN
  RAISE EXCEPTION 'lab_arena_correction_reward_history_input_invalid' USING ERRCODE='22023';
 END IF;
 FOR h IN SELECT history.* FROM public.lab_arena_scoring_supersessions history
 JOIN public.lab_arena_rounds current_round ON current_round.round_id=history.round_id
 WHERE history.recovery_doc->>'recovery_kind'='operator_postpublication_champion_correction_v1'
  AND(p_mode IS NULL OR current_round.configuration_doc->>'mode'=p_mode)
  AND(p_network_name IS NULL OR(current_round.arena_network_name=p_network_name AND current_round.arena_netuid=p_netuid))
 ORDER BY history.round_id,history.generation_floor LOOP
  n:=h.new_round_doc;
  IF jsonb_typeof(h.old_round_doc)IS DISTINCT FROM 'object' OR jsonb_typeof(n)IS DISTINCT FROM 'object'
   OR h.recovery_doc->>'operation'IS DISTINCT FROM 'round_operational_update_v1'
   OR h.request_hash IS DISTINCT FROM 'sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
   OR h.recovery_doc->>'round_sha256'IS DISTINCT FROM 'sha256:'||encode(extensions.digest(h.old_round_doc::TEXT,'sha256'),'hex')
   OR h.old_round_doc->>'round_id'IS DISTINCT FROM h.round_id OR n->>'round_id'IS DISTINCT FROM h.round_id
   OR h.old_round_doc->>'status'IS DISTINCT FROM 'published' OR n->>'status'IS DISTINCT FROM 'stage2_judged'
   OR n->'status_generation'IS DISTINCT FROM to_jsonb((h.old_round_doc->>'status_generation')::BIGINT+1)
   OR n->'published_at'IS DISTINCT FROM 'null'::JSONB OR n->'publication_doc'IS DISTINCT FROM 'null'::JSONB
   OR coalesce(length(btrim(h.recovery_doc->>'approval_ref')),0)=0
   OR coalesce(h.recovery_doc->>'runtime_image_digest','')!~'^sha256:[0-9a-f]{64}$'
   OR jsonb_typeof(h.recovery_doc->'requests')IS DISTINCT FROM 'array'
   OR(h.old_round_doc-ARRAY['status','status_generation','published_at','publication_doc','configuration_doc',
     'reward_basis_doc','reward_basis_hash','signing_key_doc','effective_reward_epoch','reward_activated_at',
     'king_start_epoch','king_outcome','king_hotkey','promotion_doc','baseline_promoted_at'])
     IS DISTINCT FROM(n-ARRAY['status','status_generation','published_at','publication_doc','configuration_doc',
     'reward_basis_doc','reward_basis_hash','signing_key_doc','effective_reward_epoch','reward_activated_at',
     'king_start_epoch','king_outcome','king_hotkey','promotion_doc','baseline_promoted_at'])
   OR((h.old_round_doc->'configuration_doc')-'schedule')IS DISTINCT FROM((n->'configuration_doc')-'schedule')
   OR((h.old_round_doc#>'{configuration_doc,schedule}')-ARRAY['final_scoring_close','publication_deadline'])
     IS DISTINCT FROM((n#>'{configuration_doc,schedule}')-ARRAY['final_scoring_close','publication_deadline']) THEN
   RAISE EXCEPTION 'lab_arena_correction_reward_history_receipt_invalid' USING ERRCODE='22023';
  END IF;
  FOREACH field IN ARRAY ARRAY['reward_basis_doc','reward_basis_hash','signing_key_doc','effective_reward_epoch',
    'reward_activated_at','king_start_epoch'] LOOP
   IF n->field IS DISTINCT FROM 'null'::JSONB THEN
    RAISE EXCEPTION 'lab_arena_correction_reward_history_receipt_invalid' USING ERRCODE='22023';
   END IF;
  END LOOP;
  r:=jsonb_populate_record(NULL::public.lab_arena_rounds,h.old_round_doc);
  basis:=r.reward_basis_doc;
  IF r.rewards_enabled IS DISTINCT FROM TRUE OR r.published_at IS NULL OR r.reward_activated_at IS NULL
   OR r.effective_reward_epoch IS NULL OR r.effective_reward_epoch<0
   OR jsonb_typeof(r.publication_doc)IS DISTINCT FROM 'object'
   OR r.publication_doc#>>'{king_decision,outcome}'IS DISTINCT FROM 'no_king'
   OR jsonb_typeof(basis)IS DISTINCT FROM 'object' OR jsonb_typeof(r.signing_key_doc)IS DISTINCT FROM 'object'
   OR coalesce(basis->>'schema_version','')NOT IN('leadpoet.lab_arena.reward_basis.v1','leadpoet.lab_arena.reward_basis.v2')
   OR basis->>'round_id'IS DISTINCT FROM r.round_id
   OR basis->'effective_reward_epoch'IS DISTINCT FROM to_jsonb(r.effective_reward_epoch)
   OR coalesce(r.reward_basis_hash,'')!~'^sha256:[0-9a-f]{64}$'
   OR basis->>'reward_basis_hash'IS DISTINCT FROM r.reward_basis_hash
   OR jsonb_typeof(basis->'signature')IS DISTINCT FROM 'object'
   OR basis#>>'{signature,public_key_hash}'IS DISTINCT FROM r.signing_key_doc->>'public_key_hash'
   OR coalesce(basis#>>'{signature,public_key_hash}','')!~'^sha256:[0-9a-f]{64}$'
   OR basis#>>'{signature,algorithm}'IS DISTINCT FROM 'ECDSA_SHA_256'
   OR r.signing_key_doc->>'algorithm'IS DISTINCT FROM 'ECDSA_SHA_256'
   OR basis->>'published_at'IS NULL OR(basis->>'published_at')::TIMESTAMPTZ IS DISTINCT FROM r.published_at
   OR r.arena_network_name IS DISTINCT FROM n->>'arena_network_name'
   OR to_jsonb(r.arena_netuid)IS DISTINCT FROM n->'arena_netuid' THEN
   RAISE EXCEPTION 'lab_arena_correction_reward_history_basis_invalid' USING ERRCODE='22023';
  END IF;
  item:=jsonb_build_object('round_id',r.round_id,'status',r.status,'arena_network_name',r.arena_network_name,
   'arena_netuid',r.arena_netuid,'rewards_enabled',r.rewards_enabled,'effective_reward_epoch',r.effective_reward_epoch,
   'king_outcome',r.king_outcome,'king_hotkey',r.king_hotkey,'king_start_epoch',r.king_start_epoch,
   'reward_basis_hash',r.reward_basis_hash,'reward_basis_doc',basis,'signing_key_doc',r.signing_key_doc,
   'reward_activated_at',r.reward_activated_at,'published_at',r.published_at,
   'configuration_doc',CASE WHEN p_public_only THEN public.lab_arena_public_configuration_v1(r)ELSE r.configuration_doc END);
  items:=items||jsonb_build_array(item);
 END LOOP;
 RETURN(SELECT coalesce(jsonb_agg(value ORDER BY(value->>'effective_reward_epoch')::BIGINT DESC,value->>'reward_basis_hash'),'[]')
 FROM(SELECT value FROM(SELECT DISTINCT ON(value->>'reward_basis_hash',(value->>'effective_reward_epoch')::BIGINT)value
  FROM jsonb_array_elements(items)entries(value)
  ORDER BY value->>'reward_basis_hash',(value->>'effective_reward_epoch')::BIGINT)unique_bases
 ORDER BY(value->>'effective_reward_epoch')::BIGINT DESC,value->>'reward_basis_hash'
 LIMIT p_limit OFFSET p_offset)page);
EXCEPTION WHEN data_exception THEN
 RAISE EXCEPTION 'lab_arena_correction_reward_history_receipt_invalid' USING ERRCODE='22023';
END;
$history696$;
ALTER FUNCTION public.lab_arena_correction_reward_history_v1(TEXT,TEXT,INTEGER,INTEGER,INTEGER,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_correction_reward_history_v1(TEXT,TEXT,INTEGER,INTEGER,INTEGER,BOOLEAN)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_correction_reward_history_v1(TEXT,TEXT,INTEGER,INTEGER,INTEGER,BOOLEAN) TO lab_arena_service;
DO $activation696$
DECLARE target OID:='public.lab_arena_activate_reward(text,jsonb,jsonb)'::regprocedure;
 definition TEXT;metadata JSONB;source_hash TEXT;updated TEXT;
 old_downgrade TEXT:=$old696$         AND prior.reward_basis_doc ->> 'schema_version' = 'leadpoet.lab_arena.reward_basis.v2'
     ) THEN$old696$;
 new_downgrade TEXT:=$new696$         AND prior.reward_basis_doc ->> 'schema_version' = 'leadpoet.lab_arena.reward_basis.v2'
       UNION ALL
       SELECT 1 FROM jsonb_array_elements(public.lab_arena_correction_reward_history_v1(
        NULL,v_round.arena_network_name,v_round.arena_netuid::INTEGER,200,0,FALSE))history(value)
       WHERE value#>>'{reward_basis_doc,schema_version}'='leadpoet.lab_arena.reward_basis.v2'
     ) THEN$new696$;
 old_previous TEXT:=$old696$    FROM public.lab_arena_rounds
    WHERE reward_activated_at IS NOT NULL$old696$;
 new_previous TEXT:=$new696$    FROM (
      SELECT reward_basis_doc,signing_key_doc,configuration_doc,effective_reward_epoch,
       reward_activated_at,rewards_enabled,arena_network_name,arena_netuid FROM public.lab_arena_rounds
      UNION ALL
      SELECT archived.reward_basis_doc,archived.signing_key_doc,archived.configuration_doc,archived.effective_reward_epoch,
       archived.reward_activated_at,archived.rewards_enabled,archived.arena_network_name,archived.arena_netuid
      FROM jsonb_array_elements(public.lab_arena_correction_reward_history_v1(
       'live',v_round.arena_network_name,v_round.arena_netuid::INTEGER,200,0,FALSE))history(value)
      CROSS JOIN LATERAL jsonb_populate_record(NULL::public.lab_arena_rounds,history.value)archived
    ) reward_sources
    WHERE reward_activated_at IS NOT NULL$new696$;
 old_max TEXT:=$old696$  SELECT pg_catalog.max(effective_reward_epoch)
  INTO v_maximum
  FROM public.lab_arena_rounds
  WHERE reward_activated_at IS NOT NULL
    AND arena_network_name = v_round.arena_network_name
    AND arena_netuid = v_round.arena_netuid;$old696$;
 new_max TEXT:=$new696$  -- correction696 retains the archived activation floor.
  SELECT pg_catalog.max(epoch) INTO v_maximum FROM(
   SELECT effective_reward_epoch AS epoch FROM public.lab_arena_rounds
   WHERE reward_activated_at IS NOT NULL AND arena_network_name=v_round.arena_network_name AND arena_netuid=v_round.arena_netuid
   UNION ALL
   SELECT(value->>'effective_reward_epoch')::BIGINT FROM jsonb_array_elements(
    public.lab_arena_correction_reward_history_v1(NULL,v_round.arena_network_name,v_round.arena_netuid::INTEGER,200,0,FALSE))history(value)
  )authorities;$new696$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',encode(extensions.digest(p.prosrc,'sha256'),'hex')
 INTO definition,metadata,source_hash FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((metadata->>'proowner')::OID)<>'lab_arena_owner'
  OR metadata->>'proacl'IS DISTINCT FROM '["lab_arena_owner=X/lab_arena_owner", "lab_arena_service=X/lab_arena_owner"]'
  OR(metadata->>'prosecdef')::BOOLEAN IS DISTINCT FROM TRUE THEN
  RAISE EXCEPTION 'correction696 activation security differs';END IF;
 IF source_hash='c1e3c1aa7d848c06bf9f49fcef03d99e642af8dd6ec1530447387fb1bbc398c1' THEN RETURN;END IF;
 IF length(definition)-length(replace(definition,old_downgrade,''))<>length(old_downgrade)THEN
  RAISE EXCEPTION 'correction696 downgrade preimage differs';END IF;
 IF source_hash='1c24e7386dd132068d8f9f1bc75d37f117f1377e869b3407475ab9af2f5a03ed' THEN
  updated:=definition;
 ELSE
 IF source_hash<>'03a0274b66a4ec6d83901e4a0765bb59d193fdb459a96c5842e5ee2def5f70d1'
  OR length(definition)-length(replace(definition,old_previous,''))<>length(old_previous)
  OR length(definition)-length(replace(definition,old_max,''))<>length(old_max) THEN
  RAISE EXCEPTION 'correction696 activation preimage differs';END IF;
 updated:=replace(replace(definition,old_previous,new_previous),old_max,new_max);
 END IF;
 updated:=replace(updated,old_downgrade,new_downgrade);
 EXECUTE updated;
 IF(SELECT to_jsonb(p)-'prosrc'FROM pg_proc p WHERE p.oid=target)IS DISTINCT FROM metadata
  OR(SELECT encode(extensions.digest(p.prosrc,'sha256'),'hex')FROM pg_proc p WHERE p.oid=target)<>'c1e3c1aa7d848c06bf9f49fcef03d99e642af8dd6ec1530447387fb1bbc398c1' THEN
  RAISE EXCEPTION 'correction696 activation metadata differs';END IF;
END;
$activation696$;

DO $restore696$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena696_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena696_acl)THEN
  RAISE EXCEPTION 'correction696 schema ACL differs';END IF;
END;
$restore696$;
NOTIFY pgrst,'reload schema';
COMMIT;
