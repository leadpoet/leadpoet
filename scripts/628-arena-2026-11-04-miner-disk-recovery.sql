-- Recover only the proved infrastructure-damaged miner ICP7 execution.
-- This is ordinary miner work; no king/source/model/scoring identity changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE arena628_schema_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $helpers628$ BEGIN
 IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions WHERE round_id='arena-2026-11-04' AND request_id='nov04-miner-disk628-v1') THEN
 IF NOT EXISTS(SELECT 1 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
 WHERE p.oid=to_regprocedure('public.lab_arena__miner_disk628_v1(public.lab_arena_runs)') AND prosrc=$body628$
 SELECT COALESCE(p_run.run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:3' AND p_run.assignment_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7'
  AND p_run.round_id='arena-2026-11-04' AND p_run.submission_id='sub-2246da743ed59dcfe8b248e9d0ddfe50' AND p_run.kind='execute'
  AND p_run.stage=2 AND p_run.icp_position=7 AND p_run.attempt=3 AND p_run.scoring_generation_floor=0
  AND NOT p_run.champion_restart_required AND EXISTS(
   SELECT 1 FROM public.lab_arena_scoring_supersessions h JOIN public.lab_arena_rounds r ON r.round_id=h.round_id
    JOIN public.lab_arena_runs original ON original.run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:2'
    JOIN public.lab_arena_submissions s ON s.submission_id=p_run.submission_id
   WHERE h.round_id=p_run.round_id AND h.request_id='nov04-miner-disk628-v1'
    AND h.recovery_doc->>'operation'='miner_infrastructure_recovery628_v1'
    AND h.new_round_doc->'configuration_doc'=r.configuration_doc
    AND h.new_round_doc->'participants'=r.participants
    AND s.status='frozen' AND NOT s.is_king AND s.miner_hotkey=p_run.miner_hotkey
    AND encode(extensions.digest((to_jsonb(s)-'updated_at')::TEXT,'sha256'),'hex')=h.recovery_doc->>'submission_hash'
    AND original.status='failed' AND original.terminal_cause='stage_closed'
    AND original.terminal_doc=jsonb_build_object('operation','miner_infrastructure_recovery628_v1',
     'request_id',h.request_id,'new_execution_run_id','arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:3')
    AND (to_jsonb(original)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
     ((h.recovery_doc->'superseded_runs'->original.run_id)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])),FALSE);
$body628$
 AND pg_get_userbyid(proowner)='lab_arena_owner' AND proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
 AND prosecdef AND provolatile='s' AND proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
 AND NOT proisstrict AND NOT proleakproof AND proparallel='u' AND prokind='f' AND NOT proretset
 AND l.lanname='sql') THEN RAISE EXCEPTION 'Miner628 helper replay differs: lab_arena__miner_disk628_v1'; END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
 WHERE p.oid=to_regprocedure('public.lab_arena__miner_disk_cost628_v1(public.lab_arena_runs)') AND prosrc=$body628$
 SELECT COALESCE(p_run.run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:2' AND p_run.kind='execute' AND p_run.stage=2
  AND p_run.submission_id='sub-2246da743ed59dcfe8b248e9d0ddfe50' AND p_run.icp_position=7 AND p_run.attempt=2
  AND p_run.status='failed' AND p_run.terminal_cause='stage_closed' AND EXISTS(
   SELECT 1 FROM public.lab_arena_scoring_supersessions h JOIN public.lab_arena_rounds r ON r.round_id=h.round_id
   WHERE h.round_id=p_run.round_id AND h.request_id='nov04-miner-disk628-v1'
    AND h.recovery_doc->>'operation'='miner_infrastructure_recovery628_v1'
    AND h.new_round_doc->'configuration_doc'=r.configuration_doc
    AND (to_jsonb(p_run)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
     ((h.recovery_doc->'superseded_runs'->p_run.run_id)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])),FALSE);
$body628$
 AND pg_get_userbyid(proowner)='lab_arena_owner' AND proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
 AND prosecdef AND provolatile='s' AND proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
 AND NOT proisstrict AND NOT proleakproof AND proparallel='u' AND prokind='f' AND NOT proretset
 AND l.lanname='sql') THEN RAISE EXCEPTION 'Miner628 helper replay differs: lab_arena__miner_disk_cost628_v1'; END IF;
 RETURN; END IF;
 IF to_regprocedure('public.lab_arena__miner_disk628_v1(public.lab_arena_runs)') IS NOT NULL OR to_regprocedure('public.lab_arena__miner_disk_cost628_v1(public.lab_arena_runs)') IS NOT NULL THEN RAISE EXCEPTION 'Miner628 unexpected helper'; END IF;
 EXECUTE $helpers_body628$CREATE OR REPLACE FUNCTION public.lab_arena__miner_disk628_v1(p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $miner628$
 SELECT COALESCE(p_run.run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:3' AND p_run.assignment_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7'
  AND p_run.round_id='arena-2026-11-04' AND p_run.submission_id='sub-2246da743ed59dcfe8b248e9d0ddfe50' AND p_run.kind='execute'
  AND p_run.stage=2 AND p_run.icp_position=7 AND p_run.attempt=3 AND p_run.scoring_generation_floor=0
  AND NOT p_run.champion_restart_required AND EXISTS(
   SELECT 1 FROM public.lab_arena_scoring_supersessions h JOIN public.lab_arena_rounds r ON r.round_id=h.round_id
    JOIN public.lab_arena_runs original ON original.run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:2'
    JOIN public.lab_arena_submissions s ON s.submission_id=p_run.submission_id
   WHERE h.round_id=p_run.round_id AND h.request_id='nov04-miner-disk628-v1'
    AND h.recovery_doc->>'operation'='miner_infrastructure_recovery628_v1'
    AND h.new_round_doc->'configuration_doc'=r.configuration_doc
    AND h.new_round_doc->'participants'=r.participants
    AND s.status='frozen' AND NOT s.is_king AND s.miner_hotkey=p_run.miner_hotkey
    AND encode(extensions.digest((to_jsonb(s)-'updated_at')::TEXT,'sha256'),'hex')=h.recovery_doc->>'submission_hash'
    AND original.status='failed' AND original.terminal_cause='stage_closed'
    AND original.terminal_doc=jsonb_build_object('operation','miner_infrastructure_recovery628_v1',
     'request_id',h.request_id,'new_execution_run_id','arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:3')
    AND (to_jsonb(original)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
     ((h.recovery_doc->'superseded_runs'->original.run_id)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])),FALSE);
$miner628$;
CREATE OR REPLACE FUNCTION public.lab_arena__miner_disk_cost628_v1(p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $cost628$
 SELECT COALESCE(p_run.run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:2' AND p_run.kind='execute' AND p_run.stage=2
  AND p_run.submission_id='sub-2246da743ed59dcfe8b248e9d0ddfe50' AND p_run.icp_position=7 AND p_run.attempt=2
  AND p_run.status='failed' AND p_run.terminal_cause='stage_closed' AND EXISTS(
   SELECT 1 FROM public.lab_arena_scoring_supersessions h JOIN public.lab_arena_rounds r ON r.round_id=h.round_id
   WHERE h.round_id=p_run.round_id AND h.request_id='nov04-miner-disk628-v1'
    AND h.recovery_doc->>'operation'='miner_infrastructure_recovery628_v1'
    AND h.new_round_doc->'configuration_doc'=r.configuration_doc
    AND (to_jsonb(p_run)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
     ((h.recovery_doc->'superseded_runs'->p_run.run_id)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])),FALSE);
$cost628$;
ALTER FUNCTION public.lab_arena__miner_disk628_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__miner_disk628_v1(public.lab_arena_runs) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
ALTER FUNCTION public.lab_arena__miner_disk_cost628_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__miner_disk_cost628_v1(public.lab_arena_runs) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
$helpers_body628$;
END; $helpers628$;
DO $patch628_0$ DECLARE target OID:='public.lab_arena_runs_terminal_v1()'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":false,"provolatile":"v","proconfig":["search_path=pg_catalog"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"plpgsql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena_runs_terminal_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='32408a885643ba776835268b91b9c160eb4a90a4faac3e33f4b297d84b4759c9' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'acfa209c4731f6c88e62a9ef961f1ce4341831cb9e73981c95b3ba21545aa1a1' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena_runs_terminal_v1'; END IF;
 IF length(d)-length(replace(d,$old628_0_1$BEGIN
$old628_0_1$,''))<>length($old628_0_1$BEGIN
$old628_0_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_0_1$BEGIN
$old628_0_1$,$new628_0_1$BEGIN
  IF TG_OP='UPDATE' AND OLD.round_id='arena-2026-11-04' AND OLD.submission_id='sub-2246da743ed59dcfe8b248e9d0ddfe50'
   AND OLD.stage=2 AND OLD.icp_position=7 AND OLD.kind IN('execute','score')
   AND OLD.status='accepted' AND OLD.terminal_cause='accepted' AND OLD.scoring_generation_floor=0
   AND NEW.status='failed' AND NEW.terminal_cause='stage_closed'
   AND (to_jsonb(NEW)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
    (to_jsonb(OLD)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])
   AND NEW.terminal_doc=jsonb_build_object('operation','miner_infrastructure_recovery628_v1',
    'request_id','nov04-miner-disk628-v1','new_execution_run_id','arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:3')
   AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=OLD.round_id
    AND h.request_id='nov04-miner-disk628-v1' AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
    AND h.recovery_doc->>'operation'='miner_infrastructure_recovery628_v1'
    AND h.recovery_doc->'superseded_runs'->OLD.run_id=to_jsonb(OLD)) THEN
   NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
$new628_0_1$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'32408a885643ba776835268b91b9c160eb4a90a4faac3e33f4b297d84b4759c9' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_0$;
DO $patch628_1$ DECLARE target OID:='public.lab_arena__baseline_recovery_phase_v1(lab_arena_rounds,lab_arena_runs)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"sql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena__baseline_recovery_phase_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='a865dab9a2282a6f869ff10e7d8262fa273568e85bc2b0a72967092442cb66b5' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'4acbf6eba8b497455270d1f7b1ea4e407e3dbf175d136005e38fe81af90a9428' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena__baseline_recovery_phase_v1'; END IF;
 IF length(d)-length(replace(d,$old628_1_1$ SELECT COALESCE(public.lab_arena_judge_continuation_enabled_v1(p_run.round_id)
$old628_1_1$,''))<>length($old628_1_1$ SELECT COALESCE(public.lab_arena_judge_continuation_enabled_v1(p_run.round_id)
$old628_1_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_1_1$ SELECT COALESCE(public.lab_arena_judge_continuation_enabled_v1(p_run.round_id)
$old628_1_1$,$new628_1_1$ -- The shared late-work phase gate also admits this exact miner repair.
 -- Miner identity, funding, source and scoring remain ordinary miner paths.
 SELECT (public.lab_arena__miner_disk628_v1(p_run) AND p_round.round_id=p_run.round_id
  AND p_round.status='stage2_scoring' AND p_round.scoring_generation_floor=0
  AND clock_timestamp()<(p_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ)
 OR COALESCE(public.lab_arena_judge_continuation_enabled_v1(p_run.round_id)
$new628_1_1$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'a865dab9a2282a6f869ff10e7d8262fa273568e85bc2b0a72967092442cb66b5' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_1$;
DO $patch628_2$ DECLARE target OID:='public.lab_arena_close_scoring(text,smallint)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}","prosecdef":true,"provolatile":"v","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"plpgsql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena_close_scoring'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='fe5436748cfd1bef4d904742b67ead53834c67c3df66fb0cf74cb0e0d77d2b58' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'10093875bac014642baa5056bc1f578611c46a327b9b7e870a9a2ca454c5d636' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena_close_scoring'; END IF;
 IF length(d)-length(replace(d,$old628_2_1$          WHERE judge.scored_run_id=recovery.run_id AND judge.kind='score'))))
$old628_2_1$,''))<>length($old628_2_1$          WHERE judge.scored_run_id=recovery.run_id AND judge.kind='score'))))
$old628_2_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_2_1$          WHERE judge.scored_run_id=recovery.run_id AND judge.kind='score'))))
$old628_2_1$,$new628_2_1$          WHERE judge.scored_run_id=recovery.run_id AND judge.kind='score'))))
        OR (public.lab_arena__miner_disk628_v1(recovery)
          AND (recovery.status IN('pending','leased','submitted')
           OR (recovery.status='accepted' AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs judge
            WHERE judge.scored_run_id=recovery.run_id AND judge.kind='score'))))
$new628_2_1$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'fe5436748cfd1bef4d904742b67ead53834c67c3df66fb0cf74cb0e0d77d2b58' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_2$;
DO $patch628_3$ DECLARE target OID:='public.lab_arena_execution_attempt_count(lab_arena_runs)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"sql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena_execution_attempt_count'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='1da63d39594cc688d2a450070410469266d5a9501cab460099afcbea7db35879' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'7017065b833239221c74bba24788d97dc09558edd7fb86f5d4ae1ffa6b7b546d' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena_execution_attempt_count'; END IF;
 IF length(d)-length(replace(d,$old628_3_1$ SELECT GREATEST(0, p_run.attempt - (SELECT count(*)::INTEGER
$old628_3_1$,''))<>length($old628_3_1$ SELECT GREATEST(0, p_run.attempt - (SELECT count(*)::INTEGER
$old628_3_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_3_1$ SELECT GREATEST(0, p_run.attempt - (SELECT count(*)::INTEGER
$old628_3_1$,$new628_3_1$ SELECT CASE WHEN public.lab_arena__miner_disk628_v1(p_run) THEN 2 ELSE GREATEST(0, p_run.attempt - (SELECT count(*)::INTEGER
$new628_3_1$);
 IF length(d)-length(replace(d,$old628_3_3$     ))))::INTEGER;
$old628_3_3$,''))<>length($old628_3_3$     ))))::INTEGER;
$old628_3_3$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_3_3$     ))))::INTEGER;
$old628_3_3$,$new628_3_3$     ))))::INTEGER END;
$new628_3_3$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'1da63d39594cc688d2a450070410469266d5a9501cab460099afcbea7db35879' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_3$;
DO $patch628_4$ DECLARE target OID:='public.lab_arena__execute_infrastructure_cost_exempt_v1(lab_arena_runs)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"sql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena__execute_infrastructure_cost_exempt_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='74d938f6e44693680c31e18c6a2d36e6a62ab6223808787c3a159f250f77229e' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'b04dc290f505d83f4fe840b118212484327d4bf379669c51b0f913e66d0c8010' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena__execute_infrastructure_cost_exempt_v1'; END IF;
 IF length(d)-length(replace(d,$old628_4_1$ SELECT public.lab_arena__disk_infrastructure626_v1(p_run) OR COALESCE(r.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1',FALSE)
$old628_4_1$,''))<>length($old628_4_1$ SELECT public.lab_arena__disk_infrastructure626_v1(p_run) OR COALESCE(r.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1',FALSE)
$old628_4_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_4_1$ SELECT public.lab_arena__disk_infrastructure626_v1(p_run) OR COALESCE(r.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1',FALSE)
$old628_4_1$,$new628_4_1$ SELECT public.lab_arena__miner_disk_cost628_v1(p_run) OR public.lab_arena__disk_infrastructure626_v1(p_run) OR COALESCE(r.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1',FALSE)
$new628_4_1$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'74d938f6e44693680c31e18c6a2d36e6a62ab6223808787c3a159f250f77229e' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_4$;
DO $patch628_5$ DECLARE target OID:='public.lab_arena__execute_assignment_calls_v1(text,text)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"sql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena__execute_assignment_calls_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='370ca13a8faea65c45598d59cc51a8402c454d3f80612a2d617fa3994024b525' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'8401ce81b39375032556b6799e4229d27e48503fc5e3fc4e3ae4df6b1263d2d9' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena__execute_assignment_calls_v1'; END IF;
 IF length(d)-length(replace(d,$old628_5_1$    AND NOT public.lab_arena__retired_source621_v1(r) AND NOT public.lab_arena__disk_infrastructure626_v1(r)
$old628_5_1$,''))<>length($old628_5_1$    AND NOT public.lab_arena__retired_source621_v1(r) AND NOT public.lab_arena__disk_infrastructure626_v1(r)
$old628_5_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_5_1$    AND NOT public.lab_arena__retired_source621_v1(r) AND NOT public.lab_arena__disk_infrastructure626_v1(r)
$old628_5_1$,$new628_5_1$    AND NOT public.lab_arena__retired_source621_v1(r) AND NOT public.lab_arena__disk_infrastructure626_v1(r) AND NOT public.lab_arena__miner_disk_cost628_v1(r)
$new628_5_1$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'370ca13a8faea65c45598d59cc51a8402c454d3f80612a2d617fa3994024b525' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_5$;
DO $patch628_6$ DECLARE target OID:='public.lab_arena__submission_stage_consumed(text,smallint,text,text)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"sql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena__submission_stage_consumed'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='3717de967920ea690b218fc5b9a4e615a788bcac7c332b4d084e61ba54ee2f1f' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'370e200ab664b10a6ac6b55e5f16c4aae8097acfa8ba617caba81ee9c6b5aa97' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena__submission_stage_consumed'; END IF;
 IF length(d)-length(replace(d,$old628_6_1$    WHERE runs.kind = p_kind AND NOT public.lab_arena__retired_source621_v1(runs) AND NOT public.lab_arena__disk_infrastructure626_v1(runs)
$old628_6_1$,''))<>length($old628_6_1$    WHERE runs.kind = p_kind AND NOT public.lab_arena__retired_source621_v1(runs) AND NOT public.lab_arena__disk_infrastructure626_v1(runs)
$old628_6_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_6_1$    WHERE runs.kind = p_kind AND NOT public.lab_arena__retired_source621_v1(runs) AND NOT public.lab_arena__disk_infrastructure626_v1(runs)
$old628_6_1$,$new628_6_1$    WHERE runs.kind = p_kind AND NOT public.lab_arena__retired_source621_v1(runs) AND NOT public.lab_arena__disk_infrastructure626_v1(runs) AND NOT public.lab_arena__miner_disk_cost628_v1(runs)
$new628_6_1$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'3717de967920ea690b218fc5b9a4e615a788bcac7c332b4d084e61ba54ee2f1f' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_6$;
DO $patch628_7$ DECLARE target OID:='public.lab_arena_scoring_policy_history_v1(text)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"sql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena_scoring_policy_history_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='0845419611380e6489d26ec6736a81be238a5551b2b0c9a210c69c942be339d6' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'b929362f1c5d8a26b79921ccfe920142683fad44b9818658cf53e163764b5536' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena_scoring_policy_history_v1'; END IF;
 IF length(d)-length(replace(d,$old628_7_1$  'generation_floor',CASE WHEN h.recovery_doc->>'operation' IN('nov03_forward_only_verifier_v1','nov03_admission_review605_v1','nov03_admit_reviewed_originals606_v1','nov03_baseline8_corrected_judge607_v1','nov03_three_unfinished_executions609_v1','nov03_baseline8_checkpoint_continuation610_v1','nov03_g3_natural_expiry_executions611_v1','nov03_mychn_checkpoint_continuation612_v1','nov03_filson_checkpoint_continuation613_v1','nov03_captify_checkpoint_continuation614_v1','baseline_source_supersession_v1','baseline_source_replacement_plan_v1','baseline_disk_recovery626_v1') THEN (h.new_round_doc->>'scoring_generation_floor')::BIGINT ELSE h.generation_floor END,
$old628_7_1$,''))<>length($old628_7_1$  'generation_floor',CASE WHEN h.recovery_doc->>'operation' IN('nov03_forward_only_verifier_v1','nov03_admission_review605_v1','nov03_admit_reviewed_originals606_v1','nov03_baseline8_corrected_judge607_v1','nov03_three_unfinished_executions609_v1','nov03_baseline8_checkpoint_continuation610_v1','nov03_g3_natural_expiry_executions611_v1','nov03_mychn_checkpoint_continuation612_v1','nov03_filson_checkpoint_continuation613_v1','nov03_captify_checkpoint_continuation614_v1','baseline_source_supersession_v1','baseline_source_replacement_plan_v1','baseline_disk_recovery626_v1') THEN (h.new_round_doc->>'scoring_generation_floor')::BIGINT ELSE h.generation_floor END,
$old628_7_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_7_1$  'generation_floor',CASE WHEN h.recovery_doc->>'operation' IN('nov03_forward_only_verifier_v1','nov03_admission_review605_v1','nov03_admit_reviewed_originals606_v1','nov03_baseline8_corrected_judge607_v1','nov03_three_unfinished_executions609_v1','nov03_baseline8_checkpoint_continuation610_v1','nov03_g3_natural_expiry_executions611_v1','nov03_mychn_checkpoint_continuation612_v1','nov03_filson_checkpoint_continuation613_v1','nov03_captify_checkpoint_continuation614_v1','baseline_source_supersession_v1','baseline_source_replacement_plan_v1','baseline_disk_recovery626_v1') THEN (h.new_round_doc->>'scoring_generation_floor')::BIGINT ELSE h.generation_floor END,
$old628_7_1$,$new628_7_1$  'generation_floor',CASE WHEN h.recovery_doc->>'operation' IN('nov03_forward_only_verifier_v1','nov03_admission_review605_v1','nov03_admit_reviewed_originals606_v1','nov03_baseline8_corrected_judge607_v1','nov03_three_unfinished_executions609_v1','nov03_baseline8_checkpoint_continuation610_v1','nov03_g3_natural_expiry_executions611_v1','nov03_mychn_checkpoint_continuation612_v1','nov03_filson_checkpoint_continuation613_v1','nov03_captify_checkpoint_continuation614_v1','baseline_source_supersession_v1','baseline_source_replacement_plan_v1','baseline_disk_recovery626_v1','miner_infrastructure_recovery628_v1','miner_infrastructure_plan629_v1') THEN (h.new_round_doc->>'scoring_generation_floor')::BIGINT ELSE h.generation_floor END,
$new628_7_1$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'0845419611380e6489d26ec6736a81be238a5551b2b0c9a210c69c942be339d6' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_7$;
DO $patch628_8$ DECLARE target OID:='public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}","prosecdef":true,"provolatile":"v","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"plpgsql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena_claim_signal_judgment_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='e056d6b5ef627d5bc93c47cb7f33ad581ffec56e34007d79f49c8fd3ffe61a3f' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'2f24a829a2115e6414947614efef8d6a5e0c4654a1ececbdcb1f75fd85336f5a' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena_claim_signal_judgment_v1'; END IF;
 IF length(d)-length(replace(d,$old628_8_1$        AND retired621.recovery_doc->>'operation'='baseline_disk_recovery626_v1'))
$old628_8_1$,''))<>length($old628_8_1$        AND retired621.recovery_doc->>'operation'='baseline_disk_recovery626_v1'))
$old628_8_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_8_1$        AND retired621.recovery_doc->>'operation'='baseline_disk_recovery626_v1'))
$old628_8_1$,$new628_8_1$        AND retired621.recovery_doc->>'operation'='baseline_disk_recovery626_v1')
       OR (retired621.request_id='nov04-miner-disk628-v1'
        AND retired621.recovery_doc->>'operation'='miner_infrastructure_recovery628_v1'))
$new628_8_1$);
 IF length(d)-length(replace(d,$old628_8_3$      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'submission_id'='baseline-2026-11-04')
$old628_8_3$,''))<>length($old628_8_3$      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'submission_id'='baseline-2026-11-04')
$old628_8_3$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_8_3$      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'submission_id'='baseline-2026-11-04')
$old628_8_3$,$new628_8_3$      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'submission_id'=
       CASE retired621.request_id WHEN 'nov04-miner-disk628-v1' THEN 'sub-2246da743ed59dcfe8b248e9d0ddfe50' ELSE 'baseline-2026-11-04' END)
$new628_8_3$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'e056d6b5ef627d5bc93c47cb7f33ad581ffec56e34007d79f49c8fd3ffe61a3f' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_8$;
DO $patch628_9$ DECLARE target OID:='public.lab_arena__prepare_company_judgment_claim_v1(text,text,bigint,timestamp with time zone)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}","prosecdef":true,"provolatile":"v","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"plpgsql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena__prepare_company_judgment_claim_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='3d03e8352f59bfd0555ef09c2c8d4500c2463c801d533be3167193a524c585a9' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'5724f1c2b7a15fb3a4a1a7b1a130e518eb2823735acbe799467f909c3ef268df' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena__prepare_company_judgment_claim_v1'; END IF;
 IF length(d)-length(replace(d,$old628_9_1$        AND retired621.recovery_doc->>'operation'='baseline_disk_recovery626_v1'))
$old628_9_1$,''))<>length($old628_9_1$        AND retired621.recovery_doc->>'operation'='baseline_disk_recovery626_v1'))
$old628_9_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_9_1$        AND retired621.recovery_doc->>'operation'='baseline_disk_recovery626_v1'))
$old628_9_1$,$new628_9_1$        AND retired621.recovery_doc->>'operation'='baseline_disk_recovery626_v1')
       OR (retired621.request_id='nov04-miner-disk628-v1'
        AND retired621.recovery_doc->>'operation'='miner_infrastructure_recovery628_v1'))
$new628_9_1$);
 IF length(d)-length(replace(d,$old628_9_3$      AND retired621.recovery_doc->'superseded_runs'->accepted.source_scored_run_id->>'submission_id'='baseline-2026-11-04')
$old628_9_3$,''))<>length($old628_9_3$      AND retired621.recovery_doc->'superseded_runs'->accepted.source_scored_run_id->>'submission_id'='baseline-2026-11-04')
$old628_9_3$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_9_3$      AND retired621.recovery_doc->'superseded_runs'->accepted.source_scored_run_id->>'submission_id'='baseline-2026-11-04')
$old628_9_3$,$new628_9_3$      AND retired621.recovery_doc->'superseded_runs'->accepted.source_scored_run_id->>'submission_id'=
       CASE retired621.request_id WHEN 'nov04-miner-disk628-v1' THEN 'sub-2246da743ed59dcfe8b248e9d0ddfe50' ELSE 'baseline-2026-11-04' END)
$new628_9_3$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'3d03e8352f59bfd0555ef09c2c8d4500c2463c801d533be3167193a524c585a9' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_9$;
DO $patch628_10$ DECLARE target OID:='public.lab_arena__judge_continuation_v1(lab_arena_rounds,lab_arena_runs)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proowner),'acl',proacl::TEXT,'prosecdef',prosecdef,
 'provolatile',provolatile,'proconfig',proconfig,'proisstrict',proisstrict,'proleakproof',proleakproof,
 'proparallel',proparallel,'prokind',prokind,'proretset',proretset,'lanname',l.lanname)
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security628${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","proretset":false,"lanname":"plpgsql"}$security628$::JSONB THEN RAISE EXCEPTION 'Miner628 security differs: lab_arena__judge_continuation_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='4f3a15162da6a90cd54f96ae6d97d53550b3a04d9cf31f3966d0209ef5b64efd' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'f3f1b5b3ed605168a3fec329d29e08a0599a2e3f58c3c903d4e41ef450b0deb3' THEN RAISE EXCEPTION 'Miner628 preimage differs: lab_arena__judge_continuation_v1'; END IF;
 IF length(d)-length(replace(d,$old628_10_1$       WHERE later.assignment_id=p_run.assignment_id AND later.attempt>p_run.attempt)) THEN RETURN TRUE; END IF;
$old628_10_1$,''))<>length($old628_10_1$       WHERE later.assignment_id=p_run.assignment_id AND later.attempt>p_run.attempt)) THEN RETURN TRUE; END IF;
$old628_10_1$) THEN RAISE EXCEPTION 'Miner628 anchor differs'; END IF;
 d:=replace(d,$old628_10_1$       WHERE later.assignment_id=p_run.assignment_id AND later.attempt>p_run.attempt)) THEN RETURN TRUE; END IF;
$old628_10_1$,$new628_10_1$       WHERE later.assignment_id=p_run.assignment_id AND later.attempt>p_run.attempt)) THEN RETURN TRUE; END IF;
  IF p_run.run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:score:2' AND p_run.kind='score'
   AND p_run.scored_run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:3' AND p_round.scoring_generation_floor=0
   AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=p_round.round_id
    AND h.request_id='nov04-miner-disk629-plan-v1'
    AND h.recovery_doc->>'operation'='miner_infrastructure_plan629_v1'
    AND h.new_round_doc->'configuration_doc'=p_round.configuration_doc
    AND h.recovery_doc#>>'{work_item,scored_run_id}'=p_run.scored_run_id
    AND h.recovery_doc#>'{work_item,company_judgment_refs}'=p_run.company_judgment_refs) THEN RETURN TRUE; END IF;
$new628_10_1$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'4f3a15162da6a90cd54f96ae6d97d53550b3a04d9cf31f3966d0209ef5b64efd' THEN RAISE EXCEPTION 'Miner628 readback differs'; END IF;
END; $patch628_10$;
DO $data628$
DECLARE manifest CONSTANT JSONB:=$manifest628${"submission_hash":"2da537b3e64036b321faaecd75cc6037bbc39bd208ec4f8f643fcbe7a61b1672","miner_run_hashes":{"arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:0:1":"e3f7307ac627617ec3b497e993155c71ebfe355258e32c44d0da3a8253f750c8","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:0:score:1":"38cce388f8ceeb9412e8f484a544107100aff7dc9e38d79cb12f5bbef56612b1","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:1:1":"5bb1867cf6060bebfcb7757b64736e065e113c1109ccc367b3e71f5dc8805ed7","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:1:score:1":"a140bf594d67ac733822b7485b32ed5404dd0d26d8155e75683d082bdacaa8ba","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:2:1":"9ea18228fbd25278f994847f61bed6b15c42abc492d478831d2c842b3c6da04c","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:2:score:1":"b5fe0dfc7c5f2bbc92b6860f1659f846cd095371eddb1a74fb3d0943ebdca404","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:3:1":"c5656aa9d8dbd38b38f822e67c236570791a8606aad06bc752ce43884884ca9a","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:3:score:1":"8dffb0b5dcadc77969a173e14664a951a15f5aa3213f6548de2087bce3e6244b","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:3:score:2":"4ee9d856dae489b2f6f3833b375f98764bd5cadf4085a63df512ff7f5e419f74","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:4:1":"c69c06aee22fc00682993308cade3df289b6c86ae369d38cda02615d3f1d40e7","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:4:score:1":"d60234b45e026d6131bf0baf346bc0e235c8a39c424058040f256312d19f16c6","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:5:1":"5253242358e50d9330eab20994f03a37c57180a9201b9591e8b95977d349092f","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:5:score:1":"ace362c248e2a0ce75445d035c9f5d9a1a9d7e0ce6ed38e0d4ff0b0e45ed9c31","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:6:1":"6551d8302eb529ab07de43e403b1ac9aed1b57bf5b37830626d86a9db50a0794","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:6:score:1":"df138c1fc792bf3c8808e3b79ce7629e9dc0dfefd9da5a390f659959e87b9720","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:6:score:2":"8a01f3460bb10a059ccaf41760737e12b75f2d34fef3aec1fa3db79cd51c4666","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:7:1":"5bb3cb1ea5084d33523ad666e11533954eeae2c79886b47d142bdfa6f88f09b5","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:7:score:1":"7f0a1d6124a62e0ab3690d90e0ffdbbfc93be25b47bc633dab32d362fd801314","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:8:1":"90d81b88bc22d2ae22233a4e3419590c10bda2642a81d500373c4214e6db8f09","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:8:score:1":"33c8b45a0fd148a04c3d79c885bb091bd4f59a4317d5f09f5dd98c6183171f0d","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:9:1":"3b577e1404173fffc38128604b061704543cce8522a36f4393a80b98b40c402c","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:9:score:1":"366f2fd49f3b74b7b02845c3b586d3e96be1a3b98a4b26cb1497ec4f1ae86dfa","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:0:1":"f892260ae275336fe354d9beb7476cfdf0875a0b7e4a212e12e6f82852917cef","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:0:2":"6dd3ff2b506b49a1414fe5c79b40c9a6159931a23dc955ca7b2cadaca3f9c4c0","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:0:score:1":"d81418d8b3afce8709afbaa1dffb98c143da68b4a72922048ca5155d7992e546","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:0:score:2":"a9abb8f08cbbf93fd514452f51ea42454690222a40b7a407f1d7e03026e53034","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:1:1":"725f459fb0ce4c350c9365bced0f731f2c6094b1b30889bb26af7b7b5d9c2405","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:1:2":"6ad4309139072aae35ced24ccca74e946e471643460427a85f9ab28582b34d2c","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:1:score:1":"de02850716ad4960218176053c65b106a3b57443d34c02a80d99feca93f7aa96","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:2:1":"7bf6388c6e4f6d09c6061aba1a2801e79ecb6e914fd3fa3065c8d4be97e895b2","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:2:2":"7c140e38f280ed7da5ae0d4600de60dfcb6bd683b2e4c14d8c5c4eb8ba695869","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:2:score:1":"855106a7bfe6b4f8e8804db03b0634974ecc797e972cbcf8dc207405ed7ad672","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:3:1":"8b2fe1a6e2832082a27d01e53a03d1939f84ea508ccb605a3ae7ada6203b9bc7","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:3:2":"a835d4dd2c43409f9da1d17c6755c074490afd439f3430fe11a812b98f1b68d7","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:3:score:1":"b39ef8e17ca989769a2f9eaf80ed19ec096654de0399389fc3276df9801582a9","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:3:score:2":"8ea0ce827168eda0f1db34ff8b7abbdc935943efbb0c6310e81737f2acdbc48d","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:4:1":"87b3d499f34732ecfb07f75e46d5dbc6ea8d434b8045461553660bdb9ead0dfa","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:4:2":"469a394d6b38fa42082cecb60a9b5527a111301846d938efa8f33184b55b64e8","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:4:score:1":"b2e8138959c60e5a8a03e29b148b81145c08066fc7be601e498e60b95bfecdd2","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:5:1":"8678c49b7e4edb1f5437d3e0e369a04c050eec7a85fbacd610659d9bd9377bed","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:5:2":"640c30a242ffb245f89a2baf79d7842e3dcd7bd5b9669b0827d0c35a266bd73a","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:5:score:1":"46c9f883045452af0f41adadd6f833d5e8b9aa5ca8f5dc34b7f25bb327d9863b","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:6:1":"957f6cce9c3a42d4b74185aa6101cff212e9430aa486fb02e9780764ac17e712","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:6:2":"174704e6c1b8c26c611f86b71a7876eaf1e823f494b8b164c2840099e0a54b79","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:6:score:1":"59fc3a2306780a10a305e5e3bb839592bbeb9c62d657b821d5e0c77d70d84f72","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:7:1":"b77f5b1e0e270e7ebb940676537a69a0a339fcc4a61896d24a90fb4945c90a65","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:7:2":"fcb4935ab650f1aeea0df994adaa821f07762867907b61c91c2207330a6c8f69","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:7:score:1":"4f74e289511944f42a9aa61ca169d4ec127725849f1e41a49af96a9c43891743","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:8:1":"a8b8cab9362876f4f664d455962de1b101bf5165479c76b75d3dbc9ed8304030","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:8:2":"1af31d6a8bab0e7d5d7e343494726a799c036002dc5fb9b25cc6b746fd21dfd6","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:8:score:1":"9a8efb80280f1bd4458bf44bafbeab24c40dea391200553e0b3cf6de7b978ee7","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:9:1":"3abf7ae9d7f6bd781091b0e26430278d13c84f707b0bc70c9e4b2eef3d8dce7b","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:9:2":"76dfc70854839a94d476feb5d1820ce091dc259be69f367100cb4f450589897d","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:9:score:1":"0fa65c62b262158baf755506e179c0d7bef614a796ccd183a770e6d2d04165a1","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:0:1":"0fc1bfe0b7def6af446a11dddd2ba76fd2c8117df7b222965ad6c86a243a43ac","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:0:2":"69bf7032de2e8e199e5fc14f559b3bdfd6fc29303c3130e69f667a5bd07be0e5","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:0:score:1":"82255747ccfbb2cd2dfaaaa72bd8b31193758c74601a20ccaf4c2e25e9680f2a","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:1:1":"496fae5a7c4981c06e5dc62bf8adcc17f5bad7f0a29146cfe80ecab3b09bbd93","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:1:2":"5fcce7e1f302aa1a3743723d222947f73f312fdc3269bee47d71a5dd92382f94","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:1:score:1":"3b1cbab642a8a2461c5062ba8651c2da4829938ea7e06ee2c4f0a0603a1d5c0c","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:2:1":"11d1656fbcf656566f659d1d247fd361cba3788a813846aedb4a715eaf1171fa","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:2:2":"dc6f551132330023a00ac99d2688f0ba5916b8d4541b1273403bcae5669bc870","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:2:score:1":"5358291ddabd288c6fa1b253f22e39980bb0633d29cc644c8f20f6a334f2bfd0","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:3:1":"de3b81ba1040c3593548db685565ca3f327a3edf771e49e44192c7c0631fee61","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:3:2":"d50b4f4896d5ed8a38f246f608d29ded2a38946fb4de88d3e94f279e6d79f70c","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:3:score:1":"e261bb8edac19bc7aac95d9e16c9a711f97bb11cb5083ce6c0895caf0aae28ee","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:4:1":"60822e928ec2a54d56470289f59d7e56fe4584f82fcb6090c5f5c38cc20d4fb8","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:4:2":"99a9a4c1de9b28398d9fde8d149e555052eb33eb0506f7a656a85509e2f6b64f","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:4:score:1":"081083cbfb4e2c87d748bab639492d9b67c791f799fca7835e794bcaec1dc189","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:4:score:2":"34bce7bc1ceffd5635459447130aeb45f09391ee8157890c5c3a0a90a94793ce","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:5:1":"ace7bfcd65abf5c000df293b6e2f2869e9bd06f3274340071d32b302dff42a1c","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:5:2":"57ac434730560fa91ffda6c1cd46ac6fd3b906a0c6b1dc8ebf8637449a48f9f5","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:5:score:1":"26b194dfd8e8131599291b244f370c9a63868785a88817f949192ff26752a79f","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:5:score:2":"3bb0987fec4adbcc9e6c6104dff21f272919a48f256652980c5a2c6f4bf69155","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:6:1":"11949cbfdd65fb265e9d1f9495de9316981cd8f0c52c64a5260c192da6949773","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:6:2":"ca2ad62ddaf9df41132a7e2dc0f1f268b307175653822290ef7724f2e67e718f","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:6:score:1":"117860fd998d65710a2db0eb4b2252bf666de71e66d7334fea03e9d7d14057ae","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:1":"4dbd2216a2b1b6a2017b091ed339c0b8e582712f9e444a3e7b83656b335ed88e","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:2":"da81635a5e2686798266f88eab2204ed03f55cf396a0d3b4aab7664eea701715","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:score:1":"dfa91b91177c4b84fc695bb49c9474be5fb35e7587499454161d129ca985fc93","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:8:1":"1842a1e9c196c45813d1027ba8f07427702f56b747dc086a86650da8ee7c9992","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:8:2":"f18b0122e7624f7d44b696a7e81a7bfffafed65a3ad5373f6b36ff969d32b43d","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:8:score:1":"49e6d18cbeedfd86c356dee7c3c3dcbc92fb9b55249ce6b8e5dab5f294c1f990","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:9:1":"980d65eef8ab8d393219dd63200a6bb4a5c874af3e8cf57c8233c7a125cd2722","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:9:score:1":"5444c69ca1572d0bf95c0931c751491bb1da10367bb9f4ed6a2b10d2e3c6092f","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:0:1":"3a5018474c4fd33d31049c26234c930575fff08cadf85ee7798926c9dbd01b5f","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:0:score:1":"2609ffb668e32b68c350fd105109edec1e6659a6d3642082d15f4fa788bc708a","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:1:1":"296198550a6ed1f52eb70f73beab002034e9eacbbac304ad89ed3d6515cf99bc","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:1:score:1":"2d4c57132379ec0a22e7ee4cae21be92955075130ec9c58bee7cd057733cc1eb","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:2:1":"99eaff1d6b28e09a1f6cfc66078462745c526152a51260bf6cfeb0fd4aca2dba","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:2:score:1":"e9ffbfc6daa90dfde946a88e4c2c8c96904e7d9eda8dfd8b1dc1083870bf1168","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:3:1":"5df0685c0d82fbc6a7fe4813d582aa99f9746b9b4f4437d0648da80e005c72bd","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:3:score:1":"568ce58f0b7c9592933223b1f16f7fab49ed60610f5fdcb3a43f87e46c9c9788","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:4:1":"9d9fd650744103fd520ecf90c46cc523271fb1f55d2a0dc02c86e4f06efc8bbd","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:4:score:1":"d3d8bc6d7bf2ae969fb61d0c63369ebcad699240861ed81aba3d30384c4bad56","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:4:score:2":"7d687835c0b59a63e30190534a838f6dde00438261bed3febac0a008c1cc384f","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:5:1":"56089dd4c7bc7a329f25e1be25241a807ee5aacf32ebade06366a081a5e7020a","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:5:score:1":"89edf98f346ec544be24cfd3666321a522676c25d34e4e12cf2afb6ed72fe416","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:6:1":"ed6e3cf39eb802623c73e1e01bb78ad884891b7442fef291afaab0d2e0e6a3f8","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:6:score:1":"4ef43c5ce5392e391518d95aaabde5dc7cfc4f6fc7c642a46da87771d0146fa0","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:7:1":"886178fdabfaa86ab92e13c3c11c198154faa6c29c49a2087aea58b348869054","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:7:score:1":"393f54188c21d110a261e109d93cbbb3716b67fee086fcdb506f9c76fd8c6539","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:8:1":"12cbdb2bba098758d0ba93d5e6db4b21adf02e40fd0380547c4e00fad0870044","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:8:score:1":"e9f8d4e541002ff5852b5aac668573be6114acfb8d6b978b810d70af82e5e979","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:9:1":"95338948e5888b0f478b02b1380a5bf983097b11b2f50c230082d5b99749bcdb","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:9:score:1":"4e2bb5f170ef509c059191775b3a9575c831753928f2fa2d861fa55d1485a813","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:0:1":"13b0c28bd06ceb077e5523937f266e531649137060b91ce80b870ac14a60bc1c","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:0:2":"beb191f67ad696e2c2330bd68830b3913ec10f35039374bb897d5f1de8919994","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:0:score:1":"9c99cc9fb39df47879b21ba0222ea51a6f243b694ccc5afa523145f941822e5d","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:1:1":"9af7c38c4f654ca07b6b06f6fdb3213d1ad6641625ce06b94be2336628c16f3a","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:1:2":"677a26b88e9f9471271b34829a2028061fd11ab62b2db8bb41c23a26240f62f3","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:1:score:1":"163d2343d37864df65e887045d0cd02fe0f1c92e54e2f7b52fa0a5562cda8dac","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:2:1":"be29fef4e68df262c125cfd87f97bc9d6144390d7ff6b0493e88ad80d3ec4838","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:2:2":"e36d665b2d08e0fa6bd3c5071074d8419d22a50ac4e43a7e20ad02425a451cc1","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:2:score:1":"7acd6b0b9adc7c1418f23276e9894689db5e25b9a90aa588fd159b721ed4cc42","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:3:1":"0c38cdd4627a149944c07afd8846399781155ca80e71def7ed1a78261bd8db42","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:3:2":"ccc2c59f9fa901edac4dc38adfcc62ce20d29ed8fb8ac2d57d0212aadf8693ce","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:3:score:1":"b98f2088bdc55e1a70e3a8862afa0af9b42f2cf3aaebfd0bf6daaae5dc502253","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:4:1":"3c8f7373c83fe8c42c166267b22cc0938054ed9a16812703bb78b3f63a60ca0b","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:4:2":"79362393258fe32c9106fbd66d46a588da1208a4384056d14c454b3a8eed8c03","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:4:score:1":"f11661fa281bdd7cec60d1bbe25b553b1fc9ea7b90715bec64ed5914ccd02145","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:4:score:2":"d6a960a74e8982bd413231bf8bc3cb82eb975de4b4d75212a20af76e6486fd57","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:5:1":"82c02e958b503caf2cd2cef67e422cc07d463d975a0fb9bc685d18782afa1a41","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:5:2":"f12fa288d6967ce874d47d958f209dd6defc525a51f555f0e7a97077770e0237","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:5:score:1":"677441e316777cb5e15a0b47b86aebad5c97f366665767f3da08be00e0d5ef27","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:6:1":"c0a5d1be6b5ec3aa5f25d38e94184e28c2a0448159221afcf301a7611ead450a","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:6:2":"e78453716fde1b5f6e0169f011c449c2d03474c267ea110c2b7aff3d904fe620","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:6:score:1":"50c1cc1c0aa740086c020e7361a4e5998aea963bfca06e02a7819e9403efd183","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:7:1":"b4aebd2a65e99ecca6935d494a3919cdeba0704be61d061ba6043d7aa3522b75","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:7:2":"8939bcd6b6b0dff86d52b8b46aafb14859a88ad4320bb1b7433147013e7b9abc","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:7:score:1":"fcc675b31f98c2b4f3480b333840437d81221b5a328d7da515106f6039963f3e","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:8:1":"3c9d87c9ea81c7a11ab83773984013da4d579c58e67816344a52393819aaa84c","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:8:2":"b72b9f165d00b45379d1595a57f3e7b55f9b9492a368fada154b11029d77bea7","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:8:score:1":"2707c606c292b38fabc309e3bd5251901071f9a95ec237d85fbb46965de36222","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:9:1":"7a9caf5eb81e01b6d44d6126fac10b029d7187c6de9d318380ac872679c491d4","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:9:2":"531cebe1575831843537a108faea005391e40f0aab765824de65817a37e2987f","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:9:score:1":"ddd5b23d069830e44da0779bc2bb42ce1fe5598bdc2101344138c330613bdde6","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:0:1":"0bcb21e59a611550278ebd50218d91439bbe28416ed5c5494c0a0aa78b7cb5f7","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:0:score:1":"5f6c746905fc5cc4efa0351012138abbceea122f67933493bf13b5792493bad0","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:1:1":"e4d82a860882f637c86bb8b12dfbd8432776524d71d5a6e023cc86dae34c984a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:1:2":"8fa8c721b68e07e3692e90db871b2090f2a961c3796b2e84b45b0d5d594d7729","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:1:score:1":"3220cc6882728e0d3e82f63ea6ddb4c6680e76745d2cbb4a32049e48f31c6a9e","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:1:score:2":"1d2800618e653494e5a2938a7963774e93ec106ae73a02391545161fddb20239","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:2:1":"8c6530bf91ba2c8e74d4587f4bb92d3b12170ed76b9c9d1449c393b1c7ef54d2","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:2:score:1":"14be0e4ed29d0f779caeddf5305d3e2e02435371a32a71421ecbb5b43103316a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:3:1":"f4f77e47bf2cfc3d6da907ff6dd13805ce3a54682df4181ac6e71fdbd7f296be","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:3:score:1":"299b2b1210604095eebcdd53261c9d13a88908b16d08c92948c1e360656a494f","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:3:score:2":"c7b7870f73d199967bb78796f2ae01450e3d88cc03644f773010947a6282512b","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:4:1":"d6376c4af19609735830677bf896f4f99a1d24cdb7356d878e7fcfcfe7e62374","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:4:2":"1d232b7ae5dafab4534d72943860df6a115d918660d6f46cdd29ee894d02687a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:4:score:1":"451a0bdb6bc2bdabff91a058b48cc3c6f921e14210bce4bb45062ba5d7a9f47b","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:5:1":"709aaf8aff5716cb63ae3e9c8771806a674f865b4c88400e02085c36855f5779","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:5:2":"7aaa48d5ad93c6d12fbf83d8c2464338e5e93ed122ba7a17840d8132cfabb914","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:5:score:1":"b81a5e47e2d6d75aa331a8d086b492b28141c0e54fdbd4f50b3ed99a22b19194","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:6:1":"dfcb1ee2cb5750bdc98ef9b6e6db49802c475d2009eab3d378c02fcd6db493ce","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:6:2":"d472f1fbbd899f37960d71bee5f634780294f103605be567f9114af236318d97","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:6:score:1":"d7e83c751477487d151ef8de56815e95d8eb86b520ce8289999f9a9caaab8c0f","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:7:1":"cd4e0de1d740fd78fb14af9270652e06809a481963ee14db51cf3d76a41091de","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:7:2":"55d8bad8c69a856bf397d07f560c3250a0ac713ffe3ecc4198f20df74d93a2c8","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:7:score:1":"f9cf79ffcb3afbe56395f0dd28f53c9489413ccf222172cc434102734772b1f8","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:8:1":"979aae1e83870bafab5afc78a41571410f849d21cd5ef31924ec41dc150a1036","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:8:2":"ca1c960c80be55dbcb8688fbf76f4ede2c6e2f621fb1fcb8299bbbd5a1571452","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:8:score:1":"ddc63bc8d4171c03245aca0cf68a0e114319e1923a99dd49e4842864c5cda18c","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:9:1":"d27d2d8e9cc03b8b70e18bf4cd2754b8e0cec400d42c2877faaa1adfbdbe6614","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:9:2":"6a95000d52da3f4a83bd9b61a06816309f4b4ffc142c95aefdb79211133b3a2a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:9:score:1":"c679ac34f92c6aa71f7982672b1ff55d68bd1fea6c9ae20e968d76712c0ef263","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:0:1":"72345062a5eda8f1002c031a89c80a6d624ffc92571169746a7f4baa64613eb6","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:0:score:1":"e77bcef5d6375101d92fa50e2ea49d3c13ca867c65b7ba0b1e477b72496145a5","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:1:1":"d80fb6858d35acef4374a9c2c9b50bb6371a3988157009b6d807890de1366a68","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:1:score:1":"e8c45b0a7463accf37496766eb463004db3b19d90d31e829a1d47095ab4f8573","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:2:1":"32ebd3a25489b00bae2c2e295d92c9c62f9e6aa4809649b918ba72d42b89bef9","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:2:score:1":"c4524fe788a6a0b550a93bca1369428a30b42df98ae2d5b35cd15ee85763c160","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:3:1":"2d6440d80a55a0e824e3808e644b84f1da6fc6b56b6d8713f02d6a943887fa99","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:3:score:1":"af6d0b2a298b381551e974b07fcb540f686dbc9a49a57446bdd15fe3b2290b44","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:4:1":"72092fd0c50ec25804fb6653590282186fa828df1806917fb37363cfdaa0882e","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:4:score:1":"85893364a0c82cb891ff78ffd216dce0165ea55196b3bfdf8df403afab6002f4","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:5:1":"4d06288fd871052d06b403467038bcf46d0d9fb09304c1ccffb4842aa6b60323","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:5:2":"6457a984cb70552aae9980873f12e9f9894396e09e97e98924ea64d16ec18431","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:5:score:1":"3b0770c4dc27ec4a878b7dbe5ebd2bfecb50a4e4cdae016feeb750e4761a8a80","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:5:score:2":"1aac0a8df06b7111dccad01e17ea7df457d5e7b297978e31b3be057d28f4f408","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:6:1":"725554ad6e1c330fd08e735935850d729b9f74bf0a4699c5568b132e29fe95de","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:6:score:1":"05240863dc482143f58e5f1958fc4531d62c53e8f3292107830058acd86a42a7","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:7:1":"1cbdbbfa8f820c4ed76831a5012db8fb33b5b07e89fc2eaf09b89b37a88b09d5","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:7:score:1":"cf20c855960208fa23ab18aad393699ee3c7d3e10a7e0dc7beb5f1a103421571","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:8:1":"8a9786ccee46e404633392c39c7aec178a4878b11a67fb6ff35660aa43e35b77","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:8:score:1":"e8558098e2c5a4facc8283a0a8afa97a636fbe499f510817cdc469b506e58231","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:9:1":"0a6d9577cbc30ad2be3135cfb55b81bad358919142f76fa4aedafd1a8000d30d","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:9:score:1":"d30848602d9beb885566befacc038b63adc6a4480ab8ab877139674d01c9f8ee","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:0:1":"ebbb7072c49a9235e6905ecab0572b4f956f559b21e612409c7bf968a6358a7e","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:0:score:1":"7f48f29b8d2bc312565fc3147ef1b6fbd3b4d2ba936d361a5ce632fc68cc405f","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:1":"6f2b46eb2a81dcb09d4bbba2b52c76cb85db3aaa89cc56c730a1ec75b67350a0","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:score:1":"4d1dc0d8e939c38d35b4b92c77db54e5a561fcd74e891405fc357183f655a124","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:score:2":"f756283f3608e3f03d1f0332d8ba173ee3ebbc8e10cf033b283fd8b597da3a2f","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:score:3":"e1b7d37f9ac1c84f101003bd5c3eff6d7938c8047c7dc9d01a2c8bef157d083a","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:score:4":"e72dcccde665db701ace3f0f9ec16161d95ee58a667a352ddedff05125711bf7","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:2:1":"e9f8906fbf02048a0ae08a9b67c2a00571829b05aff8fda3b7621325fbdfb236","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:2:score:1":"42bf40a75576951c950187b9039063cdbacb4137b38166028b96bee3fc64c2ca","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:2:score:2":"9f3ee6a0ee77d569551d720062d9f19f11608af8beb5bcde674c20cc0ac13832","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:3:1":"5c4a462ee6197fc29b0709794eb76d1f81becca4b56483d31206bb3c9cb065f3","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:3:score:1":"cce62acef0f179ca6ad8b7ba0a454d82fc3190b3e316dcdf5ada17a58b8a93e7","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:3:score:2":"abcc1094d0416a4a691311a0b0fa67bb36399107ed9c2d8572bd29dd7d0ed046","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:4:1":"1f354b43d71a17932751c5606384a156510fe4a4a1c7857cd360e3db67e148c7","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:4:score:1":"212c95a6b5d6ea00dfdfd7eb443cf070a22f71eca211e9c000f6c042de477c31","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:4:score:2":"7ce90d0fb3634cc833a1902fca542fc31ec8d565560ff50f06799160d748a3c4","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:5:1":"5d95bfe5eee06871f87c4ae7fd1d07da102b9a87d57e2caca04d19ee361a3023","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:5:score:1":"3d9bb151fc22b98b5c3cff4a94955ef33ba3cee8711b4a4ce65d8f424d766015","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:5:score:2":"117ba598e8e33f897bc441665920b2d6ff0455464f2707188882435203c50816","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:6:1":"d2c7002ddaac2fea5336428430e9729fa029d68bd12ae46aa5d8db14fca8b300","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:6:score:1":"20a150db1984208058d9479bc7d34e4fd7b0d9545ee8e17085cb9d831b52b6da","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:7:1":"42e49e8d933625a2a39da2f43928e41f2c6da252de4e0dbc1b87894893f380e1","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:7:score:1":"6e3df1e43453f3916bd5f9be10da8a959a3c3bc40ae229c48a58def416be0e91","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:8:1":"06d668b9bad8d0d36fd0836de7ae246dfa768ddd7688ec577fb751aa8b030d02","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:8:score:1":"6624ec3bbff2f7bbf7955a2a79e8bb25390986591fe73a6f3bff0339cf2778e0","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:9:1":"7c925b5e63438385de95c576b19cf50c7fac52d170267a1de700482eaec86d71","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:9:score:1":"0aa3ade417f05a5f03c9450fd82ef4c4e74df72326d7e1ab0f8d6333fd64cb1b"}}$manifest628$::JSONB;
 r public.lab_arena_rounds; n public.lab_arena_rounds; h public.lab_arena_scoring_supersessions;
 source public.lab_arena_runs; judge public.lab_arena_runs; s public.lab_arena_submissions;
 originals JSONB; hashes JSONB; revision BIGINT;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-04' FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id AND request_id='nov04-miner-disk628-v1';
 IF FOUND THEN
  IF h.recovery_doc->>'operation'<>'miner_infrastructure_recovery628_v1'
   OR h.recovery_doc->'manifest' IS DISTINCT FROM manifest
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc THEN
   RAISE EXCEPTION 'Miner628 replay differs'; END IF;
  RETURN;
 END IF;
 IF r.status NOT IN('stage2_scoring','stage2_judged','scored') OR r.scoring_generation_floor<>0
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR r.baseline_promoted_at IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions prior WHERE prior.round_id=r.round_id
   AND prior.request_id='nov04-baseline-disk626-v1' AND prior.recovery_doc->>'operation'='baseline_disk_recovery626_v1'
   AND prior.new_round_doc->'configuration_doc'=r.configuration_doc AND prior.new_round_doc->'participants'=r.participants)
  OR (SELECT count(*) FROM public.lab_arena_restart_claim_control WHERE singleton)<>1
  OR EXISTS(SELECT 1 FROM public.lab_arena_restart_claim_control WHERE operator_paused OR guard_commitment<>''
   OR owner_commitment<>'' OR guard_expires_at IS NOT NULL OR candidate_commit<>'' OR restart_scope<>'' OR restart_phase<>'') THEN
  RAISE EXCEPTION 'Miner628 round or control differs'; END IF;
 SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id='sub-2246da743ed59dcfe8b248e9d0ddfe50' FOR SHARE;
 IF s.is_king OR s.status<>'frozen' OR s.round_id<>r.round_id
  OR encode(extensions.digest((to_jsonb(s)-'updated_at')::TEXT,'sha256'),'hex')<>manifest->>'submission_hash' THEN
  RAISE EXCEPTION 'Miner628 frozen source differs'; END IF;
 SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO hashes
 FROM public.lab_arena_runs a WHERE round_id=r.round_id AND stage=2;
 IF hashes IS DISTINCT FROM manifest->'miner_run_hashes' THEN RAISE EXCEPTION 'Miner628 original work differs'; END IF;
 SELECT * INTO STRICT source FROM public.lab_arena_runs
 WHERE run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:2' FOR UPDATE;
 SELECT * INTO STRICT judge FROM public.lab_arena_runs
 WHERE run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:score:1' FOR UPDATE;
 IF source.kind<>'execute' OR source.stage<>2 OR source.icp_position<>7 OR source.attempt<>2
  OR source.status<>'accepted' OR source.terminal_cause<>'accepted' OR source.output_ref IS NULL
  OR source.miner_hotkey<>s.miner_hotkey OR source.champion_restart_required
  OR judge.kind<>'score' OR judge.status<>'accepted' OR judge.terminal_cause<>'accepted'
  OR judge.scored_run_id<>source.run_id OR judge.scoring_generation_floor<>0
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE (a.assignment_id=source.assignment_id AND a.attempt>2)
   OR (a.assignment_id=judge.assignment_id AND a.attempt>1))
  OR NOT public.lab_arena__judge_calls_clear_v1(source.assignment_id)
  OR NOT public.lab_arena__judge_calls_clear_v1(judge.assignment_id)
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_trajectory_events e WHERE e.run_id=source.run_id
   AND e.event_kind='runtime.stderr' AND e.content->>'text' LIKE '%TYCHE runtime blocked: [Errno 28] No space left on device%')
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_trajectory_events e WHERE e.run_id=source.run_id
   AND e.event_kind='runtime.finished' AND e.content->'exit_code'='0'::JSONB AND e.content->'company_count'='4'::JSONB)
 THEN RAISE EXCEPTION 'Miner628 damaged execution proof differs'; END IF;
 originals:=jsonb_build_object(source.run_id,to_jsonb(source),judge.run_id,to_jsonb(judge));
 n:=r; n.status:='stage2_scoring'; n.status_generation:=r.status_generation+1;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,'nov04-miner-disk628-v1','sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex'),
  'nov04-miner-disk628',txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',
  jsonb_build_object('operation','miner_infrastructure_recovery628_v1','scoring_generation_floor',0,'manifest',manifest,
   'submission_hash',manifest->>'submission_hash','superseded_runs',originals,
   'new_execution_run_id',source.assignment_id||':3'));
 UPDATE public.lab_arena_rounds SET status=n.status,status_generation=n.status_generation WHERE round_id=r.round_id;
 UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',
  terminal_doc=jsonb_build_object('operation','miner_infrastructure_recovery628_v1','request_id','nov04-miner-disk628-v1',
   'new_execution_run_id',source.assignment_id||':3') WHERE run_id IN(source.run_id,judge.run_id);
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,
  attempt,status,stage_generation,kind)
 VALUES(source.assignment_id||':3',source.assignment_id,r.round_id,source.submission_id,source.miner_hotkey,2,7,
  3,'pending',r.stage_generation,'execute');
END;
$data628$;
DO $restore628$ BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena628_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena628_schema_acl) THEN RAISE EXCEPTION 'Miner628 schema ACL differs'; END IF;
END; $restore628$;
NOTIFY pgrst,'reload schema';
COMMIT;
