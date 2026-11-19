-- Owner-run recovery for a published no-king round whose challengers were all
-- rejected because the review provider was unavailable. Reopen archives the
-- published authority (never deletes it), returns the round to stage1_scored
-- with the frozen baseline score kept, and returns only those rejected rows to
-- review on the current routes. Admission then freezes the passing entrants
-- and the normal driver runs stage 2, scoring, judging and publication.
-- Inactive until the owner calls it. No retry limit, score, cost or reward
-- rule changes. begin/finish review functions are not edited (PR 1272 owns
-- them); the existing late-review deadline hook is extended instead.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE arena702_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- 1. Exact receipt shape. Shared by the reward-history reader, the review
-- deadline hook and the admission hold, so every reader applies one rule.
CREATE OR REPLACE FUNCTION public.lab_arena__review_outage_receipt_valid_v1(h public.lab_arena_scoring_supersessions)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public SET TimeZone='UTC' AS $valid702$
DECLARE o JSONB:=h.old_round_doc;n JSONB:=h.new_round_doc;field TEXT;item RECORD;
 cleared TEXT[]:=ARRAY['status','status_generation','configuration_doc','published_at','publication_doc',
  'reward_basis_doc','reward_basis_hash','signing_key_doc','effective_reward_epoch','reward_activated_at',
  'king_start_epoch','king_outcome','king_hotkey','promotion_doc','baseline_promoted_at',
  'stage2_scoring_plan_doc','finalists'];
 moved TEXT[]:=ARRAY['stage_2_start','stage_2_close','final_scoring_close','publication_deadline'];
BEGIN
 IF h.recovery_doc->>'recovery_kind' IS DISTINCT FROM 'operator_review_outage_reopen_v1'
  OR h.recovery_doc->>'operation' IS DISTINCT FROM 'round_operational_update_v1'
  OR h.request_hash IS DISTINCT FROM 'sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
  OR jsonb_typeof(o) IS DISTINCT FROM 'object' OR jsonb_typeof(n) IS DISTINCT FROM 'object'
  OR h.recovery_doc->>'round_sha256' IS DISTINCT FROM 'sha256:'||encode(extensions.digest(o::TEXT,'sha256'),'hex')
  OR o->>'round_id' IS DISTINCT FROM h.round_id OR n->>'round_id' IS DISTINCT FROM h.round_id
  OR o->>'status' IS DISTINCT FROM 'published' OR n->>'status' IS DISTINCT FROM 'stage1_scored'
  OR n->'status_generation' IS DISTINCT FROM to_jsonb((o->>'status_generation')::BIGINT+1)
  OR o#>>'{publication_doc,king_decision,outcome}' IS DISTINCT FROM 'no_king'
  OR coalesce(length(btrim(h.recovery_doc->>'approval_ref')),0)=0
  OR jsonb_typeof(h.recovery_doc->'submissions') IS DISTINCT FROM 'object'
  OR NOT EXISTS(SELECT 1 FROM jsonb_object_keys(h.recovery_doc->'submissions'))
  OR (o-cleared) IS DISTINCT FROM (n-cleared)
  OR ((o->'configuration_doc')-'schedule') IS DISTINCT FROM ((n->'configuration_doc')-'schedule')
  OR ((o#>'{configuration_doc,schedule}')-moved) IS DISTINCT FROM ((n#>'{configuration_doc,schedule}')-moved)
  OR n#>>'{configuration_doc,schedule,stage_2_start}' IS DISTINCT FROM h.recovery_doc->>'review_deadline'
  OR coalesce(h.recovery_doc->>'republish_by','')!~'^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'
  OR (h.recovery_doc->>'republish_by')::TIMESTAMPTZ<=(n#>>'{configuration_doc,schedule,publication_deadline}')::TIMESTAMPTZ THEN
  RETURN FALSE;
 END IF;
 FOREACH field IN ARRAY cleared[4:17] LOOP
  IF n->field IS DISTINCT FROM 'null'::JSONB THEN RETURN FALSE; END IF;
 END LOOP;
 FOR item IN SELECT key,value FROM jsonb_each(h.recovery_doc->'submissions') LOOP
  IF item.value->'old'->>'submission_id' IS DISTINCT FROM item.key
   OR item.value->'old'->>'round_id' IS DISTINCT FROM h.round_id
   OR item.value->'old'->>'status' IS DISTINCT FROM 'rejected'
   OR item.value->'old'->>'rejection_rule' IS DISTINCT FROM 'code_review_incomplete'
   OR item.value->'old'#>>'{code_review_doc,error_code}' IS DISTINCT FROM 'code_review_provider_unavailable'
   OR item.value->'new' IS DISTINCT FROM (item.value->'old')||'{"status":"accepted","rejection_rule":null}'::JSONB THEN
   RETURN FALSE;
  END IF;
 END LOOP;
 RETURN TRUE;
EXCEPTION WHEN data_exception THEN RETURN FALSE;
END;
$valid702$;

-- The latest operational receipt for the round, or NULL.
CREATE OR REPLACE FUNCTION public.lab_arena__review_outage_open_receipt_v1(p_round_id TEXT)
RETURNS public.lab_arena_scoring_supersessions LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public SET TimeZone='UTC' AS $open702$
 SELECT h.* FROM public.lab_arena_scoring_supersessions h
 WHERE h.round_id=p_round_id AND h.recovery_doc->>'recovery_kind'='operator_review_outage_reopen_v1'
  AND h.generation_floor=(SELECT max(l.generation_floor) FROM public.lab_arena_scoring_supersessions l WHERE l.round_id=p_round_id)
  AND public.lab_arena__review_outage_receipt_valid_v1(h)
 LIMIT 1;
$open702$;

-- 2. Review window for exactly the listed rows while the reopened round is
-- unchanged and not yet admitted. NULL everywhere else.
CREATE OR REPLACE FUNCTION public.lab_arena__review_outage_review_deadline_v1(
 p_round public.lab_arena_rounds,p_submission public.lab_arena_submissions)
RETURNS TIMESTAMPTZ LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public SET TimeZone='UTC' AS $deadline702$
 SELECT (h.recovery_doc->>'review_deadline')::TIMESTAMPTZ
 FROM public.lab_arena__review_outage_open_receipt_v1(p_round.round_id) h
 WHERE h.round_id IS NOT NULL AND p_round.status='stage1_scored'
  AND (to_jsonb(p_round)-'updated_at')=h.new_round_doc
  AND p_submission.round_id=p_round.round_id AND p_submission.status='accepted'
  AND NOT p_submission.is_king AND p_submission.frozen_at IS NULL
  AND h.recovery_doc->'submissions' ? p_submission.submission_id
  AND (to_jsonb(p_submission)-ARRAY['updated_at','code_review_status','code_review_doc','code_review_claim',
    'code_review_started_at','code_review_expires_at','code_review_attempts']::TEXT[])=
   ((h.recovery_doc->'submissions'->p_submission.submission_id->'new')-ARRAY['updated_at','code_review_status',
    'code_review_doc','code_review_claim','code_review_started_at','code_review_expires_at','code_review_attempts']::TEXT[]);
$deadline702$;

DO $owners702$
DECLARE signature TEXT;
BEGIN
 FOREACH signature IN ARRAY ARRAY[
  'public.lab_arena__review_outage_receipt_valid_v1(public.lab_arena_scoring_supersessions)',
  'public.lab_arena__review_outage_open_receipt_v1(text)',
  'public.lab_arena__review_outage_review_deadline_v1(public.lab_arena_rounds,public.lab_arena_submissions)'] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',signature);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',signature);
 END LOOP;
END;
$owners702$;

-- 3. Extend the existing late-review hook used by begin_v2 and finish. The
-- Nov 6 result always wins; the new window applies only when it is NULL.
DO $hook702$
DECLARE target OID:=to_regprocedure('public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)');
 d TEXT;src TEXT;ident JSONB;metadata JSONB;body TEXT;updated TEXT;
 pre TEXT:='37f4106388a4683a297d46c470684dc17548f7431db6c72016cbae043836530f';
 post TEXT:='8af2cf1f7c29812d4f30491ea3e605eac5f02fde7b1ece7f8c66d7afbccd8d06';
BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc,to_jsonb(p)-'prosrc',
  jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,src,metadata,ident FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
   ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN RAISE EXCEPTION 'review702 hook security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')=post THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>pre THEN RAISE EXCEPTION 'review702 hook preimage differs'; END IF;
 body:=rtrim(src,E' \n');
 IF right(body,1)<>';' OR position('$review702$' IN d)>0 THEN RAISE EXCEPTION 'review702 hook body differs'; END IF;
 body:=left(body,length(body)-1);
 updated:=replace(d,src,E'\n SELECT coalesce((SELECT nov06.deadline FROM ('||body||E'\n ) nov06(deadline) LIMIT 1),\n'
  ||E'  public.lab_arena__review_outage_review_deadline_v1(p_round,p_submission));\n');
 IF updated=d THEN RAISE EXCEPTION 'review702 hook anchor differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>post
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'review702 hook readback differs'; END IF;
END;
$hook702$;

-- 4. Let only the reopen transaction return its exact rejected rows to
-- accepted. Every other rejected row stays frozen by the existing rule.
DO $frozen702$
DECLARE target OID:=to_regprocedure('public.lab_arena_submissions_frozen_v1()');
 d TEXT;ident JSONB;metadata JSONB;updated TEXT;
 pre TEXT:='a225ce8971679ea2d389789c7d3b940a871b202fff46a3859562efb8d49650cb';
 post TEXT:='363033bad5edd76c0685afbe8dc6e790b29e135998f5c50c2415992f661b8b2e';
 anchor TEXT:=E'AS $function$\nBEGIN\n';
 insert TEXT:=$insert702$  IF TG_OP='UPDATE' AND OLD.status='rejected' AND NEW.status='accepted'
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->>'recovery_kind'='operator_review_outage_reopen_v1'
       AND h.recovery_doc->'submissions'->OLD.submission_id->'old'=to_jsonb(OLD)-'updated_at'
       AND h.recovery_doc->'submissions'->OLD.submission_id->'new'=to_jsonb(NEW)-'updated_at') THEN
    -- review702: owner reopen of an exact provider-outage rejection.
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
$insert702$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',FALSE,'v',
   ARRAY['search_path=pg_catalog']::TEXT[]) THEN RAISE EXCEPTION 'review702 frozen security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')=post THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>pre THEN RAISE EXCEPTION 'review702 frozen preimage differs'; END IF;
 IF length(d)-length(replace(d,anchor,''))<>length(anchor) THEN RAISE EXCEPTION 'review702 frozen anchor differs'; END IF;
 updated:=replace(d,anchor,anchor||insert);
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>post
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'review702 frozen readback differs'; END IF;
END;
$frozen702$;

-- 5. Hold the reopened round at stage1_scored until the owner admits the
-- reviewed rows or the review window ends. Cancel is never held. Other
-- rounds, including the next day's round, are untouched.
CREATE OR REPLACE FUNCTION public.lab_arena_review_outage_hold_v1()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET TimeZone='UTC' AS $hold702$
DECLARE h public.lab_arena_scoring_supersessions;
BEGIN
 IF OLD.status='stage1_scored' AND NEW.status IS DISTINCT FROM OLD.status AND NEW.status<>'cancelled' THEN
  h:=public.lab_arena__review_outage_open_receipt_v1(OLD.round_id);
  IF h.round_id IS NOT NULL AND clock_timestamp()<(h.recovery_doc->>'review_deadline')::TIMESTAMPTZ THEN
   RAISE EXCEPTION 'lab_arena_review_outage_admission_pending' USING ERRCODE='55000';
  END IF;
 END IF;
 RETURN NEW;
END;
$hold702$;
ALTER FUNCTION public.lab_arena_review_outage_hold_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_review_outage_hold_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_review_outage_hold ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_review_outage_hold BEFORE UPDATE ON public.lab_arena_rounds
 FOR EACH ROW WHEN (OLD.status='stage1_scored' AND NEW.status IS DISTINCT FROM OLD.status)
 EXECUTE FUNCTION public.lab_arena_review_outage_hold_v1();

-- 6. Keep the archived signed no-king basis visible to every reward reader
-- while the round reruns, exactly as for a champion correction.
DO $installed702$
DECLARE p RECORD;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure('public.lab_arena_correction_reward_history_v1(text,text,integer,integer,integer,boolean)');
 IF NOT FOUND OR pg_get_userbyid(p.proowner)<>'lab_arena_owner'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR encode(extensions.digest(p.prosrc,'sha256'),'hex') NOT IN
   ('9d377d88142dd0c38f46590508aaf737b8c811478c66be2eab5df0081644daa7','f89d89f50e37b4ce78a6a865ead856f2b9449bd7c75ad4ff79b45a39935ac33a') THEN
  RAISE EXCEPTION 'review702 installed reader source/security differs';END IF;
END;
$installed702$;

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
 WHERE history.recovery_doc->>'recovery_kind' IN('operator_postpublication_champion_correction_v1','operator_review_outage_reopen_v1')
  AND(p_mode IS NULL OR current_round.configuration_doc->>'mode'=p_mode)
  AND(p_network_name IS NULL OR(current_round.arena_network_name=p_network_name AND current_round.arena_netuid=p_netuid))
 ORDER BY history.round_id,history.generation_floor LOOP
  n:=h.new_round_doc;
  IF h.recovery_doc->>'recovery_kind'='operator_review_outage_reopen_v1' THEN
   -- review702: the shared exact receipt rule.
   IF NOT coalesce((SELECT public.lab_arena__review_outage_receipt_valid_v1(x) FROM public.lab_arena_scoring_supersessions x
     WHERE x.round_id=h.round_id AND x.request_id=h.request_id),FALSE) THEN
    RAISE EXCEPTION 'lab_arena_correction_reward_history_receipt_invalid' USING ERRCODE='22023';
   END IF;
  ELSIF jsonb_typeof(h.old_round_doc)IS DISTINCT FROM 'object' OR jsonb_typeof(n)IS DISTINCT FROM 'object'
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
DO $reader702$
BEGIN
 IF encode(extensions.digest((SELECT prosrc FROM pg_proc WHERE oid='public.lab_arena_correction_reward_history_v1(text,text,integer,integer,integer,boolean)'::regprocedure),'sha256'),'hex')
  <>'f89d89f50e37b4ce78a6a865ead856f2b9449bd7c75ad4ff79b45a39935ac33a' THEN RAISE EXCEPTION 'review702 reader postimage differs'; END IF;
END;
$reader702$;

-- 7. Reopen. Dry run (the default) changes nothing and returns the exact
-- field and row changes plus the hashes the real call must repeat.
CREATE OR REPLACE FUNCTION public.lab_arena_reopen_review_outage_v1(
 p_round_id TEXT,p_round_sha256 TEXT,p_submissions JSONB,p_review_deadline TIMESTAMPTZ,p_stage_2_close TIMESTAMPTZ,
 p_final_scoring_close TIMESTAMPTZ,p_publication_deadline TIMESTAMPTZ,p_request_id TEXT,p_actor_ref TEXT,
 p_approval_ref TEXT,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET TimeZone='UTC' AS $reopen702$
DECLARE r public.lab_arena_rounds;n public.lab_arena_rounds;s public.lab_arena_submissions;base public.lab_arena_submissions;
 nxt public.lab_arena_rounds;prior public.lab_arena_scoring_supersessions;head public.lab_arena_ledger;
 proof JSONB;rows JSONB:='{}';listed JSONB:='[]';changes JSONB:='{}';schedule JSONB;baseline_id TEXT;round_hash TEXT;
 digest TEXT;revision BIGINT;key TEXT;fmt TEXT:='YYYY-MM-DD"T"HH24:MI:SS"Z"';republish_by TIMESTAMPTZ;
BEGIN
 IF p_round_id IS NULL OR p_request_id IS NULL OR length(p_request_id) NOT BETWEEN 1 AND 160
  OR p_actor_ref IS NULL OR length(btrim(p_actor_ref)) NOT BETWEEN 1 AND 200
  OR p_approval_ref IS NULL OR length(btrim(p_approval_ref)) NOT BETWEEN 1 AND 500
  OR p_review_deadline IS NULL OR p_stage_2_close IS NULL OR p_final_scoring_close IS NULL
  OR p_publication_deadline IS NULL OR p_dry_run IS NULL
  OR (NOT p_dry_run AND (p_round_sha256 IS NULL OR p_submissions IS NULL))
  OR (p_round_sha256 IS NOT NULL AND p_round_sha256!~'^sha256:[0-9a-f]{64}$')
  OR (p_submissions IS NOT NULL AND (jsonb_typeof(p_submissions) IS DISTINCT FROM 'array'
   OR EXISTS(SELECT 1 FROM jsonb_array_elements(p_submissions) x WHERE jsonb_typeof(x) IS DISTINCT FROM 'object'
    OR (SELECT array_agg(k ORDER BY k) FROM jsonb_object_keys(x) k) IS DISTINCT FROM ARRAY['row_sha256','submission_id']::TEXT[]
    OR coalesce(x->>'row_sha256','')!~'^sha256:[0-9a-f]{64}$')
   OR (SELECT count(DISTINCT x->>'submission_id') FROM jsonb_array_elements(p_submissions) x)<>jsonb_array_length(p_submissions))) THEN
  RAISE EXCEPTION 'review_outage_reopen: request invalid';
 END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.reward_epoch',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
 IF encode(extensions.digest((SELECT prosrc FROM pg_proc WHERE oid=to_regprocedure(
   'public.lab_arena_correction_reward_history_v1(text,text,integer,integer,integer,boolean)')),'sha256'),'hex')
   IS DISTINCT FROM 'f89d89f50e37b4ce78a6a865ead856f2b9449bd7c75ad4ff79b45a39935ac33a' THEN RAISE EXCEPTION 'review_outage_reopen: reward history reader unavailable'; END IF;
 SELECT * INTO prior FROM public.lab_arena_scoring_supersessions WHERE round_id=p_round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF prior.recovery_doc->>'recovery_kind' IS DISTINCT FROM 'operator_review_outage_reopen_v1'
   OR NOT public.lab_arena__review_outage_receipt_valid_v1(prior)
   OR (p_round_sha256 IS NOT NULL AND prior.recovery_doc->>'round_sha256' IS DISTINCT FROM p_round_sha256)
   OR (p_submissions IS NOT NULL AND prior.recovery_doc->'submission_hashes' IS DISTINCT FROM p_submissions) THEN
   RAISE EXCEPTION 'review_outage_reopen: replay differs';
  END IF;
  -- A repeat (dry run or apply) reports the first reopen and changes nothing.
  RETURN jsonb_build_object('status','reopened','idempotent',TRUE,'round_id',p_round_id,'request_id',p_request_id,
   'round_sha256',prior.recovery_doc->'round_sha256','review_deadline',prior.recovery_doc->'review_deadline',
   'republish_by',prior.recovery_doc->'republish_by','submissions',prior.recovery_doc->'submission_hashes');
 END IF;
 round_hash:='sha256:'||encode(extensions.digest((to_jsonb(r)-'updated_at')::TEXT,'sha256'),'hex');
 baseline_id:='baseline-'||regexp_replace(r.round_id,'^arena-','');
 IF current_setting('session_replication_role')<>'origin' THEN RAISE EXCEPTION 'review_outage_reopen: replication role differs'; END IF;
 IF p_round_sha256 IS NOT NULL AND round_hash IS DISTINCT FROM p_round_sha256 THEN RAISE EXCEPTION 'review_outage_reopen: round hash differs'; END IF;
 IF r.configuration_doc->>'mode' IS DISTINCT FROM 'live' OR r.rewards_enabled IS DISTINCT FROM TRUE
  OR r.status<>'published' OR r.published_at IS NULL OR r.publication_doc IS NULL OR r.reward_activated_at IS NULL
  OR r.reward_basis_doc IS NULL OR r.reward_basis_hash IS NULL OR r.effective_reward_epoch IS NULL OR r.signing_key_doc IS NULL
  OR r.promotion_doc IS NOT NULL OR r.baseline_promoted_at IS NOT NULL OR r.cancel_reason IS NOT NULL
  OR r.king_outcome IS DISTINCT FROM 'no_king' OR r.publication_doc#>>'{king_decision,outcome}' IS DISTINCT FROM 'no_king'
  OR coalesce(r.king_hotkey,'')<>'' OR coalesce(r.publication_doc#>>'{king_decision,king_hotkey}','')<>''
  OR coalesce(r.publication_doc#>>'{king_decision,king_submission_id}','')<>''
  OR coalesce(r.publication_doc#>>'{king_decision,winner_submission_id}','')<>''
  OR r.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
  OR coalesce((r.configuration_doc->>'parallel_twenty_icp_execution')::BOOLEAN,FALSE)
  OR r.stage3_scoring_plan_doc IS NOT NULL OR r.confirmation_cohort IS NOT NULL
  OR r.confirmation_bank_ref IS NOT NULL OR r.confirmation_bank_hash IS NOT NULL OR r.stage1_scoring_plan_doc IS NULL
  OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=r.round_id
   AND h.recovery_doc->>'recovery_kind' IN('operator_postpublication_champion_correction_v1',
    'operator_postpublication_judge_recovery_v1','operator_review_outage_reopen_v1')) THEN
  RAISE EXCEPTION 'review_outage_reopen: published no-king authority differs';
 END IF;
 -- Participants are exactly the frozen daily baseline; it is never rerun.
 SELECT * INTO base FROM public.lab_arena_submissions WHERE submission_id=baseline_id AND round_id=r.round_id;
 IF jsonb_typeof(r.participants) IS DISTINCT FROM 'array' OR jsonb_array_length(r.participants)<>1
  OR r.participants->0->>'submission_id' IS DISTINCT FROM baseline_id OR r.participants->0->'is_king' IS DISTINCT FROM 'true'::JSONB
  OR base.submission_id IS NULL OR base.status<>'frozen' OR NOT base.is_king
  OR base.miner_hotkey IS DISTINCT FROM r.configuration_doc->>'baseline_hotkey'
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=r.round_id
   AND (a.stage=2 OR a.submission_id IS DISTINCT FROM baseline_id OR a.status IN('pending','leased','submitted')))
  OR EXISTS(SELECT 1 FROM public.lab_arena_submissions o WHERE o.round_id=r.round_id AND NOT o.is_king AND o.status IN('accepted','frozen')) THEN
  RAISE EXCEPTION 'review_outage_reopen: participants or runs are not baseline-only';
 END IF;
 IF NOT EXISTS(SELECT 1 FROM public.lab_arena_restart_claim_control WHERE singleton AND NOT operator_paused AND guard_commitment='') THEN
  RAISE EXCEPTION 'review_outage_reopen: claim control is paused or guarded';
 END IF;
 -- Later rounds must not have frozen funding, benchmark or work yet.
 IF EXISTS(SELECT 1 FROM public.lab_arena_rounds newer WHERE newer.round_id<>r.round_id
   AND (newer.evaluation_date>r.evaluation_date OR (newer.evaluation_date IS NULL AND newer.created_at>r.created_at))
   AND newer.arena_network_name=r.arena_network_name AND newer.arena_netuid=r.arena_netuid
   AND newer.configuration_doc->>'mode'='live' AND newer.status<>'cancelled' AND (newer.status<>'open' OR newer.benchmark_ref IS NOT NULL
    OR newer.champion_funding_frozen OR newer.reward_activated_at IS NOT NULL
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=newer.round_id))) THEN
  RAISE EXCEPTION 'review_outage_reopen: a later round has already frozen work';
 END IF;
 SELECT newer.* INTO nxt FROM public.lab_arena_rounds newer WHERE newer.round_id<>r.round_id
  AND (newer.evaluation_date>r.evaluation_date OR (newer.evaluation_date IS NULL AND newer.created_at>r.created_at))
  AND newer.arena_network_name=r.arena_network_name AND newer.arena_netuid=r.arena_netuid AND newer.configuration_doc->>'mode'='live' AND newer.status<>'cancelled'
 ORDER BY (newer.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ LIMIT 1;
 -- The next round's baseline-freshness gate (627/640) blocks its baseline
 -- freeze and claims while this round is unpublished. Republication must
 -- therefore land, with a promotion margin, before the next cutoff. The
 -- driver cancels the reopened round at republish_by (advance below), so a
 -- slow rerun can never stall the next round.
 IF nxt.round_id IS NULL THEN RAISE EXCEPTION 'review_outage_reopen: next round unknown; cannot bound the baseline gate'; END IF;
 republish_by:=(nxt.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ-INTERVAL '60 minutes';
 IF p_publication_deadline IS NOT NULL AND p_publication_deadline>republish_by-INTERVAL '60 minutes' THEN
  RAISE EXCEPTION 'review_outage_reopen: publication would collide with next round % baseline gate (publication deadline must be at or before %)',
   nxt.round_id,to_char((republish_by-INTERVAL '60 minutes') AT TIME ZONE 'UTC',fmt);
 END IF;
 IF NOT(clock_timestamp()<p_review_deadline AND p_review_deadline<=clock_timestamp()+INTERVAL '6 hours'
  AND p_review_deadline<p_stage_2_close AND p_stage_2_close<p_final_scoring_close AND p_final_scoring_close<p_publication_deadline
  AND p_publication_deadline<=clock_timestamp()+INTERVAL '24 hours'
  AND date_trunc('second',p_review_deadline)=p_review_deadline AND date_trunc('second',p_stage_2_close)=p_stage_2_close
  AND date_trunc('second',p_final_scoring_close)=p_final_scoring_close
  AND date_trunc('second',p_publication_deadline)=p_publication_deadline) THEN
  RAISE EXCEPTION 'review_outage_reopen: deadline invalid (need now < review <= now+6h < ... strict order, publication <= now+24h, whole seconds)';
 END IF;
 -- Exactly every provider-unavailable rejection, each with an unchanged row.
 FOR s IN SELECT * FROM public.lab_arena_submissions WHERE round_id=r.round_id AND status='rejected'
   AND code_review_doc->>'error_code'='code_review_provider_unavailable' ORDER BY accepted_at,submission_id FOR UPDATE LOOP
  IF s.is_king OR s.rejection_rule IS DISTINCT FROM 'code_review_incomplete' OR s.code_review_status IS DISTINCT FROM 'error'
   OR s.accepted_at IS NULL OR s.frozen_at IS NOT NULL OR s.code_review_claim IS NULL OR s.code_review_expires_at IS NOT NULL
   OR s.code_review_attempts<1 THEN RAISE EXCEPTION 'review_outage_reopen: rejected row differs: %',s.submission_id; END IF;
  head:=public.lab_arena__ledger_head('sha256:'||encode(extensions.digest(convert_to('lab_arena.code_review:'||s.submission_id||':'||
   s.code_review_attempts::TEXT,'UTF8'),'sha256'),'hex'));
  IF head.entry_kind IS NULL OR head.entry_kind NOT IN('settlement','uncertain') THEN
   RAISE EXCEPTION 'review_outage_reopen: review spend is not terminal: %',s.submission_id; END IF;
  listed:=listed||jsonb_build_array(jsonb_build_object('submission_id',s.submission_id,
   'row_sha256','sha256:'||encode(extensions.digest((to_jsonb(s)-'updated_at')::TEXT,'sha256'),'hex')));
  rows:=rows||jsonb_build_object(s.submission_id,jsonb_build_object('old',to_jsonb(s)-'updated_at',
   'new',(to_jsonb(s)-'updated_at')||'{"status":"accepted","rejection_rule":null}'::JSONB));
 END LOOP;
 IF jsonb_array_length(listed)=0 OR jsonb_array_length(listed)>coalesce((r.configuration_doc->>'max_challengers')::INTEGER,256) THEN
  RAISE EXCEPTION 'review_outage_reopen: provider-outage rejections absent or above capacity'; END IF;
 IF p_submissions IS NOT NULL AND (SELECT jsonb_agg(x ORDER BY x->>'submission_id') FROM jsonb_array_elements(p_submissions) x)
   IS DISTINCT FROM (SELECT jsonb_agg(x ORDER BY x->>'submission_id') FROM jsonb_array_elements(listed) x) THEN
  RAISE EXCEPTION 'review_outage_reopen: listed submissions differ from the provider-outage rejections'; END IF;
 schedule:=r.configuration_doc->'schedule'||jsonb_build_object(
  'stage_2_start',to_char(p_review_deadline AT TIME ZONE 'UTC',fmt),'stage_2_close',to_char(p_stage_2_close AT TIME ZONE 'UTC',fmt),
  'final_scoring_close',to_char(p_final_scoring_close AT TIME ZONE 'UTC',fmt),
  'publication_deadline',to_char(p_publication_deadline AT TIME ZONE 'UTC',fmt));
 n:=r;n.status:='stage1_scored';n.status_generation:=r.status_generation+1;
 n.configuration_doc:=jsonb_set(r.configuration_doc,'{schedule}',schedule);
 n.published_at:=NULL;n.publication_doc:=NULL;n.reward_basis_doc:=NULL;n.reward_basis_hash:=NULL;n.signing_key_doc:=NULL;
 n.reward_activated_at:=NULL;n.effective_reward_epoch:=NULL;n.king_start_epoch:=NULL;n.king_outcome:=NULL;n.king_hotkey:=NULL;
 n.promotion_doc:=NULL;n.baseline_promoted_at:=NULL;n.stage2_scoring_plan_doc:=NULL;n.finalists:=NULL;
 FOR key IN SELECT k FROM jsonb_object_keys(to_jsonb(r)-'updated_at') k ORDER BY k LOOP
  IF (to_jsonb(r)->key) IS DISTINCT FROM (to_jsonb(n)->key) THEN
   changes:=changes||jsonb_build_object(key,jsonb_build_object('old',CASE WHEN key='configuration_doc'
    THEN r.configuration_doc->'schedule' ELSE to_jsonb(r)->key END,'new',CASE WHEN key='configuration_doc'
    THEN n.configuration_doc->'schedule' ELSE to_jsonb(n)->key END));
  END IF;
 END LOOP;
 IF p_dry_run THEN
  RETURN jsonb_build_object('status','dry_run','round_id',r.round_id,'round_sha256',round_hash,'request_id',p_request_id,
   'next_round_id',nxt.round_id,'review_deadline',to_char(p_review_deadline AT TIME ZONE 'UTC',fmt),
   'republish_by',to_char(republish_by AT TIME ZONE 'UTC',fmt),'round_changes',changes,'submissions',listed,'submission_changes',
   (SELECT jsonb_agg(jsonb_build_object('submission_id',x->>'submission_id','miner_hotkey',rows->(x->>'submission_id')->'old'->>'miner_hotkey',
     'status',jsonb_build_object('old','rejected','new','accepted'),
     'rejection_rule',jsonb_build_object('old','code_review_incomplete','new',NULL))) FROM jsonb_array_elements(listed) x));
 END IF;
 proof:=jsonb_build_object('operation','round_operational_update_v1','recovery_kind','operator_review_outage_reopen_v1',
  'round_sha256',round_hash,'approval_ref',p_approval_ref,'review_deadline',to_char(p_review_deadline AT TIME ZONE 'UTC',fmt),
  'next_round_id',nxt.round_id,'republish_by',to_char(republish_by AT TIME ZONE 'UTC',fmt),
  'submission_hashes',p_submissions,'submissions',rows);
 digest:='sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex');
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,
  old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,p_request_id,digest,p_actor_ref,txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',proof);
 IF NOT EXISTS(SELECT 1 FROM jsonb_array_elements(public.lab_arena_correction_reward_history_v1(
   'live',r.arena_network_name,r.arena_netuid::INTEGER,200,0,FALSE)) archived
  WHERE archived->>'round_id'=r.round_id AND archived->>'reward_basis_hash'=r.reward_basis_hash
   AND (archived->>'effective_reward_epoch')::BIGINT=r.effective_reward_epoch
   AND archived->'reward_basis_doc'=r.reward_basis_doc AND archived->'signing_key_doc'=r.signing_key_doc) THEN
  RAISE EXCEPTION 'review_outage_reopen: archived reward authority unavailable';
 END IF;
 UPDATE public.lab_arena_rounds SET status=n.status,status_generation=n.status_generation,configuration_doc=n.configuration_doc,
  published_at=NULL,publication_doc=NULL,reward_basis_doc=NULL,reward_basis_hash=NULL,signing_key_doc=NULL,
  reward_activated_at=NULL,effective_reward_epoch=NULL,king_start_epoch=NULL,king_outcome=NULL,king_hotkey=NULL,
  promotion_doc=NULL,baseline_promoted_at=NULL,stage2_scoring_plan_doc=NULL,finalists=NULL WHERE round_id=r.round_id;
 FOR key IN SELECT x->>'submission_id' FROM jsonb_array_elements(listed) x LOOP
  UPDATE public.lab_arena_submissions SET status='accepted',rejection_rule=NULL
   WHERE submission_id=key AND status='rejected' AND round_id=r.round_id;
  IF NOT FOUND THEN RAISE EXCEPTION 'review_outage_reopen: row update differs: %',key; END IF;
 END LOOP;
 -- Postcondition: exact round, exact rows, and an open review window for each.
 SELECT * INTO STRICT n FROM public.lab_arena_rounds WHERE round_id=r.round_id;
 IF (to_jsonb(n)-'updated_at') IS DISTINCT FROM (SELECT new_round_doc FROM public.lab_arena_scoring_supersessions
   WHERE round_id=r.round_id AND request_id=p_request_id) THEN RAISE EXCEPTION 'review_outage_reopen: round postcondition differs'; END IF;
 FOR s IN SELECT * FROM public.lab_arena_submissions WHERE round_id=r.round_id AND rows ? submission_id LOOP
  IF (to_jsonb(s)-'updated_at') IS DISTINCT FROM rows->s.submission_id->'new'
   OR public.lab_arena__nov06_late_review635_v1(n,s) IS DISTINCT FROM p_review_deadline THEN
   RAISE EXCEPTION 'review_outage_reopen: row postcondition differs: %',s.submission_id; END IF;
 END LOOP;
 RETURN jsonb_build_object('status','reopened','idempotent',FALSE,'round_id',r.round_id,'request_id',p_request_id,
  'next_round_id',nxt.round_id,'review_deadline',proof->'review_deadline','republish_by',proof->'republish_by',
  'submissions',listed,'round_changes',changes);
END;
$reopen702$;
ALTER FUNCTION public.lab_arena_reopen_review_outage_v1(TEXT,TEXT,JSONB,TIMESTAMPTZ,TIMESTAMPTZ,TIMESTAMPTZ,TIMESTAMPTZ,TEXT,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_reopen_review_outage_v1(TEXT,TEXT,JSONB,TIMESTAMPTZ,TIMESTAMPTZ,TIMESTAMPTZ,TIMESTAMPTZ,TEXT,TEXT,TEXT,BOOLEAN)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- 8. Admit. Freeze each passing, eligible row in acceptance order, reject the
-- rest with the normal freeze rules, and release the hold. Zero passes is a
-- valid outcome: the round then republishes with the baseline only.
CREATE OR REPLACE FUNCTION public.lab_arena_admit_review_outage_v1(
 p_round_id TEXT,p_reopen_request_id TEXT,p_request_id TEXT,p_actor_ref TEXT,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET TimeZone='UTC' AS $admit702$
DECLARE r public.lab_arena_rounds;n public.lab_arena_rounds;s public.lab_arena_submissions;
 h public.lab_arena_scoring_supersessions;prior public.lab_arena_scoring_supersessions;
 admitted JSONB:='[]';excluded JSONB:='{}';proof JSONB;revision BIGINT;deadline TIMESTAMPTZ;cap INTEGER;
 eligible BOOLEAN;rule TEXT;key TEXT;full_thinking BOOLEAN;
BEGIN
 IF p_round_id IS NULL OR p_reopen_request_id IS NULL OR p_request_id IS NULL OR length(p_request_id) NOT BETWEEN 1 AND 160
  OR p_request_id=p_reopen_request_id OR p_actor_ref IS NULL OR length(btrim(p_actor_ref)) NOT BETWEEN 1 AND 200 OR p_dry_run IS NULL THEN
  RAISE EXCEPTION 'review_outage_admit: request invalid';
 END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
 SELECT * INTO prior FROM public.lab_arena_scoring_supersessions WHERE round_id=p_round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF prior.recovery_doc->>'recovery_kind' IS DISTINCT FROM 'operator_review_outage_admit_v1'
   OR prior.recovery_doc->>'reopen_request_id' IS DISTINCT FROM p_reopen_request_id OR prior.actor_ref IS DISTINCT FROM p_actor_ref THEN
   RAISE EXCEPTION 'review_outage_admit: replay differs'; END IF;
  RETURN jsonb_build_object('status','admitted','idempotent',TRUE,'round_id',p_round_id,
   'admitted',prior.recovery_doc->'admitted','excluded',prior.recovery_doc->'excluded');
 END IF;
 h:=public.lab_arena__review_outage_open_receipt_v1(p_round_id);
 IF h.round_id IS NULL OR h.request_id IS DISTINCT FROM p_reopen_request_id OR r.status<>'stage1_scored'
  OR (to_jsonb(r)-'updated_at') IS DISTINCT FROM h.new_round_doc OR current_setting('session_replication_role')<>'origin' THEN
  RAISE EXCEPTION 'review_outage_admit: reopened round differs or is already admitted';
 END IF;
 deadline:=(h.recovery_doc->>'review_deadline')::TIMESTAMPTZ;
 cap:=coalesce((r.configuration_doc->>'max_challengers')::INTEGER,256);
 full_thinking:=coalesce(r.configuration_doc->>'reasoning_capture_policy'='full_generated_thinking_v1',FALSE);
 n:=r;
 FOR s IN SELECT * FROM public.lab_arena_submissions WHERE round_id=r.round_id AND h.recovery_doc->'submissions' ? submission_id
   ORDER BY accepted_at,submission_id FOR UPDATE LOOP
  IF s.status<>'accepted' OR s.is_king OR s.frozen_at IS NOT NULL
   OR (to_jsonb(s)-ARRAY['updated_at','code_review_status','code_review_doc','code_review_claim','code_review_started_at',
     'code_review_expires_at','code_review_attempts']::TEXT[])
    IS DISTINCT FROM ((h.recovery_doc->'submissions'->s.submission_id->'new')-ARRAY['updated_at','code_review_status','code_review_doc',
     'code_review_claim','code_review_started_at','code_review_expires_at','code_review_attempts']::TEXT[]) THEN
   RAISE EXCEPTION 'review_outage_admit: reopened row differs: %',s.submission_id; END IF;
  IF s.code_review_status='reviewing' OR s.code_review_expires_at IS NOT NULL
   OR (clock_timestamp()<deadline AND s.code_review_status NOT IN('passed','rejected')) THEN
   RAISE EXCEPTION 'review_outage_admit: review still pending: %',s.submission_id; END IF;
  -- Mirror freeze_participants: passed plus, under full thinking, the exact
  -- complete-source research proof (training_eligibility.assess_source).
  eligible:=s.code_review_status='passed' AND s.code_review_doc->'passed'='true'::JSONB
   AND (NOT full_thinking OR (s.source_archive_sha256 ~ '^sha256:[0-9a-f]{64}$'
    AND s.code_review_doc#>>'{research_control,policy_id}'='model_directed_research_v1'
    AND s.code_review_doc#>>'{research_control,policy_hash}'='sha256:1d30b4c2d838e84692b3cd9aa0fe0da3ff908f37b3836863b3937a3d93c9a73c'
    AND s.code_review_doc#>>'{research_control,reviewer_version}'='full_source_reviewer_research_v1'
    AND s.code_review_doc#>>'{research_control,source_archive_sha256}'=s.source_archive_sha256
    AND s.code_review_doc#>>'{research_control,classification}'='model_directed'
    AND jsonb_typeof(s.code_review_doc#>'{research_control,evidence_claims}')='array'
    AND (SELECT array_agg(DISTINCT c ORDER BY c) FROM jsonb_array_elements_text(s.code_review_doc#>'{research_control,evidence_claims}') c)
     =ARRAY['finish_continue','observation_to_action']::TEXT[]));
  IF NOT coalesce(eligible,FALSE) THEN
   rule:=CASE WHEN s.code_review_status='rejected' THEN 'code_review_rejected' ELSE 'code_review_incomplete' END;
  ELSIF jsonb_array_length(n.participants)-1>=cap THEN rule:='capacity.round_full';
  ELSE rule:=NULL;
  END IF;
  IF rule IS NULL THEN
   n.participants:=n.participants||jsonb_build_array(jsonb_build_object('submission_id',s.submission_id,
    'miner_hotkey',s.miner_hotkey,'source_ref',s.source_ref,'source_size_bytes',s.source_size_bytes,'is_king',FALSE));
   admitted:=admitted||jsonb_build_array(s.submission_id);
  ELSE
   excluded:=excluded||jsonb_build_object(s.submission_id,rule);
  END IF;
 END LOOP;
 IF jsonb_array_length(admitted)+(SELECT count(*) FROM jsonb_object_keys(excluded))
   <>(SELECT count(*) FROM jsonb_object_keys(h.recovery_doc->'submissions')) THEN
  RAISE EXCEPTION 'review_outage_admit: reopened rows missing'; END IF;
 -- Same finalists rule as lab_arena_open_stage under baseline-first (572).
 n.finalists:=(SELECT coalesce(jsonb_agg(x ORDER BY x),'[]'::JSONB) FROM jsonb_array_elements_text(admitted) x);
 IF p_dry_run THEN
  RETURN jsonb_build_object('status','dry_run','round_id',r.round_id,'admitted',admitted,'excluded',excluded,
   'participants',jsonb_build_object('old',r.participants,'new',n.participants),'review_deadline',h.recovery_doc->'review_deadline');
 END IF;
 proof:=jsonb_build_object('operation','round_operational_update_v1','recovery_kind','operator_review_outage_admit_v1',
  'reopen_request_id',p_reopen_request_id,'round_sha256','sha256:'||encode(extensions.digest((to_jsonb(r)-'updated_at')::TEXT,'sha256'),'hex'),
  'admitted',admitted,'excluded',excluded,'reviews',(SELECT jsonb_object_agg(x.submission_id,
   jsonb_build_object('code_review_status',x.code_review_status,'code_review_attempts',x.code_review_attempts,
    'code_review_doc_sha256','sha256:'||encode(extensions.digest(coalesce(x.code_review_doc,'null'::JSONB)::TEXT,'sha256'),'hex')))
   FROM public.lab_arena_submissions x WHERE x.round_id=r.round_id AND h.recovery_doc->'submissions' ? x.submission_id));
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,
  old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,p_request_id,'sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex'),p_actor_ref,
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',proof);
 UPDATE public.lab_arena_rounds SET participants=n.participants,finalists=n.finalists WHERE round_id=r.round_id;
 FOR key IN SELECT jsonb_array_elements_text(admitted) LOOP
  UPDATE public.lab_arena_submissions SET status='frozen',frozen_at=clock_timestamp(),rejection_rule=NULL
   WHERE submission_id=key AND status='accepted' AND code_review_status='passed';
  IF NOT FOUND THEN RAISE EXCEPTION 'review_outage_admit: freeze differs: %',key; END IF;
 END LOOP;
 FOR key,rule IN SELECT k,v FROM jsonb_each_text(excluded) e(k,v) LOOP
  UPDATE public.lab_arena_submissions SET status='rejected',rejection_rule=rule WHERE submission_id=key AND status='accepted';
  IF NOT FOUND THEN RAISE EXCEPTION 'review_outage_admit: exclusion differs: %',key; END IF;
 END LOOP;
 IF (public.lab_arena__review_outage_open_receipt_v1(p_round_id)).round_id IS NOT NULL THEN
  RAISE EXCEPTION 'review_outage_admit: hold postcondition differs'; END IF;
 RETURN jsonb_build_object('status','admitted','idempotent',FALSE,'round_id',r.round_id,'admitted',admitted,'excluded',excluded);
END;
$admit702$;
ALTER FUNCTION public.lab_arena_admit_review_outage_v1(TEXT,TEXT,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_admit_review_outage_v1(TEXT,TEXT,TEXT,TEXT,BOOLEAN)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- 9. Driver step, called each tick for a round at stage1_scored or past its
-- publication deadline. NULL-safe for every other round (not_reopened).
-- Admits once every listed review is final (or the review window ends), and
-- cancels a reopened round that has not republished by republish_by, so the
-- next round's baseline gate is never left waiting on it. Cancellation keeps
-- the archived no-king basis as the reward authority (reader above).
CREATE OR REPLACE FUNCTION public.lab_arena_advance_review_outage_v1(p_round_id TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET TimeZone='UTC' AS $advance702$
DECLARE r public.lab_arena_rounds;h public.lab_arena_scoring_supersessions;open_h public.lab_arena_scoring_supersessions;
 result JSONB;
BEGIN
 SELECT x.* INTO h FROM public.lab_arena_scoring_supersessions x WHERE x.round_id=p_round_id
  AND x.recovery_doc->>'recovery_kind'='operator_review_outage_reopen_v1' ORDER BY x.generation_floor DESC LIMIT 1;
 IF h.round_id IS NULL OR NOT public.lab_arena__review_outage_receipt_valid_v1(h) THEN
  RETURN jsonb_build_object('status','not_reopened');
 END IF;
 -- Admission and cancellation use the same claim-control -> round order
 -- as ordinary claims. Never hold the round while waiting on claim-control.
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
 IF r.status IN('published','cancelled') THEN
  RETURN jsonb_build_object('status','settled','round_status',r.status);
 END IF;
 IF clock_timestamp()>=(h.recovery_doc->>'republish_by')::TIMESTAMPTZ THEN
  result:=public.lab_arena_cancel_round(p_round_id,'review_outage_reopen_expired');
  RETURN jsonb_build_object('status','cancelled','reason','review_outage_reopen_expired',
   'republish_by',h.recovery_doc->'republish_by','next_round_id',h.recovery_doc->'next_round_id','cancel',result);
 END IF;
 open_h:=public.lab_arena__review_outage_open_receipt_v1(p_round_id);
 IF open_h.round_id IS NULL OR r.status<>'stage1_scored' THEN
  RETURN jsonb_build_object('status','running','round_status',r.status);
 END IF;
 IF clock_timestamp()<(open_h.recovery_doc->>'review_deadline')::TIMESTAMPTZ AND EXISTS(
   SELECT 1 FROM public.lab_arena_submissions s WHERE s.round_id=p_round_id
    AND open_h.recovery_doc->'submissions' ? s.submission_id
    AND (s.code_review_status NOT IN('passed','rejected') OR s.code_review_expires_at IS NOT NULL)) THEN
  RETURN jsonb_build_object('status','waiting','reason','review_pending','review_deadline',open_h.recovery_doc->'review_deadline');
 END IF;
 IF EXISTS(SELECT 1 FROM public.lab_arena_submissions s WHERE s.round_id=p_round_id
   AND open_h.recovery_doc->'submissions' ? s.submission_id AND s.code_review_status='reviewing') THEN
  -- An in-flight claim past the window settles before admission.
  RETURN jsonb_build_object('status','waiting','reason','review_in_flight');
 END IF;
 RETURN public.lab_arena_admit_review_outage_v1(p_round_id,open_h.request_id,'admit:'||open_h.request_id,'driver:review_outage',FALSE);
END;
$advance702$;
ALTER FUNCTION public.lab_arena_advance_review_outage_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_advance_review_outage_v1(TEXT) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_advance_review_outage_v1(TEXT) TO lab_arena_service;
-- The owner rerun trigger (scripts/lab_arena_admin.py review-hold, PR 1272)
-- runs as the Arena service, as 701's decide function does.
GRANT EXECUTE ON FUNCTION public.lab_arena_reopen_review_outage_v1(TEXT,TEXT,JSONB,TIMESTAMPTZ,TIMESTAMPTZ,TIMESTAMPTZ,TIMESTAMPTZ,TEXT,TEXT,TEXT,BOOLEAN)
 TO lab_arena_service;

DO $restore702$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena702_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena702_acl)THEN
  RAISE EXCEPTION 'review702 schema ACL differs';END IF;
END;
$restore702$;
NOTIFY pgrst,'reload schema';
COMMIT;
