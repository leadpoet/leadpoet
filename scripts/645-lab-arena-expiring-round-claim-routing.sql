-- General expiring, round-scoped control of new lease ownership only.
-- Existing leases, exact request replay and ordinary retry rules remain intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
ALTER TABLE public.lab_arena_restart_claim_control
 ADD COLUMN IF NOT EXISTS new_claim_round_id TEXT,
 ADD COLUMN IF NOT EXISTS new_claim_allowed_runner_hotkey TEXT,
 ADD COLUMN IF NOT EXISTS new_claim_until TIMESTAMPTZ,
 ADD COLUMN IF NOT EXISTS new_claim_actor_ref TEXT;
DO $shape645$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='public.lab_arena_restart_claim_control'::regclass AND conname='lab_arena_new_claim_route_complete_v1') THEN
  ALTER TABLE public.lab_arena_restart_claim_control ADD CONSTRAINT lab_arena_new_claim_route_complete_v1 CHECK (
   num_nonnulls(new_claim_round_id,new_claim_allowed_runner_hotkey,new_claim_until,new_claim_actor_ref)=0
   OR (num_nonnulls(new_claim_round_id,new_claim_allowed_runner_hotkey,new_claim_until,new_claim_actor_ref)=4
    AND new_claim_round_id ~ '^arena-[0-9]{4}-[0-9]{2}-[0-9]{2}$'
    AND new_claim_allowed_runner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'
    AND isfinite(new_claim_until) AND length(btrim(new_claim_actor_ref)) BETWEEN 1 AND 200));
 END IF;
 IF (SELECT jsonb_object_agg(attname,jsonb_build_array(format_type(atttypid,atttypmod),attnotnull,atthasdef,attgenerated::text,attidentity::text))
  FROM pg_attribute WHERE attrelid='public.lab_arena_restart_claim_control'::regclass AND NOT attisdropped
   AND attname=ANY(ARRAY['new_claim_round_id','new_claim_allowed_runner_hotkey','new_claim_until','new_claim_actor_ref']))
  IS DISTINCT FROM $columns645${"new_claim_round_id":["text",false,false,"",""],"new_claim_allowed_runner_hotkey":["text",false,false,"",""],"new_claim_until":["timestamp with time zone",false,false,"",""],"new_claim_actor_ref":["text",false,false,"",""]}$columns645$::jsonb THEN RAISE EXCEPTION 'Route645 column shape differs'; END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='public.lab_arena_restart_claim_control'::regclass
  AND conname='lab_arena_new_claim_route_complete_v1' AND contype='c' AND convalidated AND NOT connoinherit
  AND encode(extensions.digest(pg_get_constraintdef(oid),'sha256'),'hex')='ff39ad4c4540fd59e1268812bc53dbc5d3dde8501b2605768bbc5d1fc7fa806b') THEN RAISE EXCEPTION 'Route645 constraint shape differs'; END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='public.lab_arena_runs'::regclass
  AND tgname='lab_arena_runs_restart_claim_gate' AND NOT tgisinternal AND tgenabled='O' AND tgtype=19
  AND tgfoid='public.lab_arena_restart_claim_gate_v1()'::regprocedure AND tgnargs=0 AND tgqual IS NULL
  AND tgattr::text=(SELECT attnum::text FROM pg_attribute WHERE attrelid='public.lab_arena_runs'::regclass AND attname='status' AND NOT attisdropped)) THEN RAISE EXCEPTION 'Route645 trigger attachment differs'; END IF;
END;
$shape645$;
DO $patch645$
DECLARE target OID; d TEXT; metadata JSONB; identity JSONB; expected TEXT;
BEGIN
 target:='public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure;
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',jsonb_build_array(pg_get_userbyid(proowner),proacl::TEXT,prosecdef,provolatile,proconfig,l.lanname)
 INTO d,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
 IF identity IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::jsonb THEN RAISE EXCEPTION 'Route645 claim security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex') NOT IN ('6aabab38cf3474db8e97c0b2aaffd1f071541b694020b0411a023acd780fe637','2acc6ff6c64ed797299cd18f203b3d62459720e2ae1ab9cb1f03a7850e62dd82') THEN
  expected:=CASE encode(extensions.digest(d,'sha256'),'hex') WHEN 'a5dc7e6f4370dbffa6e390609a584e5d317cdba33445e58b5bf58bd8ec9f1f6c' THEN '6aabab38cf3474db8e97c0b2aaffd1f071541b694020b0411a023acd780fe637' WHEN 'e0715a0a01f53c68d5ec9c020d90bc13848e12580c49e5de41e30b0e536ed575' THEN '2acc6ff6c64ed797299cd18f203b3d62459720e2ae1ab9cb1f03a7850e62dd82' ELSE NULL END;
  IF expected IS NULL OR length(d)-length(replace(d,$old_gate645$  SELECT operator_paused OR guard_commitment <> '' INTO v_claims_paused$old_gate645$,''))<>length($old_gate645$  SELECT operator_paused OR guard_commitment <> '' INTO v_claims_paused$old_gate645$)
   OR length(d)-length(replace(d,$old_alt645$             AND NOT EXISTS (
               SELECT 1 FROM round_quarantined_runners AS quarantined$old_alt645$,''))<>length($old_alt645$             AND NOT EXISTS (
               SELECT 1 FROM round_quarantined_runners AS quarantined$old_alt645$) THEN RAISE EXCEPTION 'Route645 claim preimage differs'; END IF;
  d:=replace(replace(d,$old_gate645$  SELECT operator_paused OR guard_commitment <> '' INTO v_claims_paused$old_gate645$,$new_gate645$  SELECT operator_paused OR guard_commitment <> ''
    OR coalesce(new_claim_round_id=p_round_id AND new_claim_until>clock_timestamp()
      AND new_claim_allowed_runner_hotkey IS DISTINCT FROM p_runner_hotkey,FALSE)
    INTO v_claims_paused$new_gate645$),$old_alt645$             AND NOT EXISTS (
               SELECT 1 FROM round_quarantined_runners AS quarantined$old_alt645$,$new_alt645$             -- A routed-out runner cannot prevent this allowed runner's retry.
             AND NOT EXISTS (
               SELECT 1 FROM public.lab_arena_restart_claim_control route
               WHERE route.singleton AND route.new_claim_round_id=p_round_id
                 AND route.new_claim_until>clock_timestamp()
                 AND route.new_claim_allowed_runner_hotkey IS DISTINCT FROM others.runner_hotkey)
             AND NOT EXISTS (
               SELECT 1 FROM round_quarantined_runners AS quarantined$new_alt645$);
  IF encode(extensions.digest(d,'sha256'),'hex')<>expected THEN RAISE EXCEPTION 'Route645 claim postimage differs'; END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>expected THEN RAISE EXCEPTION 'Route645 claim metadata differs'; END IF;
 END IF;
 target:='public.lab_arena_restart_claim_gate_v1()'::regprocedure;
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',jsonb_build_array(pg_get_userbyid(proowner),proacl::TEXT,prosecdef,provolatile,proconfig,l.lanname)
 INTO d,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
 IF identity IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::jsonb THEN RAISE EXCEPTION 'Route645 trigger security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'dc47402c37e2f9fb4220d8bba3b7baa89805aca1cb6f73c08dc3175972bb652f' THEN
  IF encode(extensions.digest(d,'sha256'),'hex')<>'0a1b50f622830caa8b362f114e69da121354ba13fa5315672ac30bb5f94dfa82' OR length(d)-length(replace(d,$old_trigger645$    SELECT operator_paused OR guard_commitment <> '' INTO v_paused$old_trigger645$,''))<>length($old_trigger645$    SELECT operator_paused OR guard_commitment <> '' INTO v_paused$old_trigger645$) THEN RAISE EXCEPTION 'Route645 trigger preimage differs'; END IF;
  d:=replace(d,$old_trigger645$    SELECT operator_paused OR guard_commitment <> '' INTO v_paused$old_trigger645$,$new_trigger645$    SELECT operator_paused OR guard_commitment <> ''
      OR coalesce(new_claim_round_id=NEW.round_id AND new_claim_until>clock_timestamp()
        AND new_claim_allowed_runner_hotkey IS DISTINCT FROM NEW.runner_hotkey,FALSE)
      INTO v_paused$new_trigger645$);
  IF encode(extensions.digest(d,'sha256'),'hex')<>'dc47402c37e2f9fb4220d8bba3b7baa89805aca1cb6f73c08dc3175972bb652f' THEN RAISE EXCEPTION 'Route645 trigger postimage differs'; END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'dc47402c37e2f9fb4220d8bba3b7baa89805aca1cb6f73c08dc3175972bb652f' THEN RAISE EXCEPTION 'Route645 trigger metadata differs'; END IF;
 END IF;
END;
$patch645$;
-- Exact authorized Nov5 primary-only activation; no lease is moved or expired.
DO $activate645$
DECLARE c public.lab_arena_restart_claim_control; r public.lab_arena_rounds;
 primary_runner CONSTANT TEXT:='5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9';
 until_time CONSTANT TIMESTAMPTZ:='2026-11-06T09:35:00Z';
BEGIN
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-05' FOR SHARE;
 IF current_setting('session_replication_role')<>'origin' OR clock_timestamp()>=until_time
  OR r.status IN('published','cancelled') OR r.evaluation_date<>'2026-11-05' OR r.arena_netuid<>71 OR r.arena_network_name<>'finney'
  OR (r.configuration_doc->'runner_hotkeys' ? primary_runner) IS DISTINCT FROM TRUE OR coalesce(r.configuration_doc->'banned_hotkeys' ? primary_runner,FALSE)
  OR c.operator_paused OR c.guard_commitment<>'' OR c.owner_commitment<>'' OR c.guard_expires_at IS NOT NULL
  OR c.candidate_commit<>'' OR c.restart_scope<>'' OR c.restart_phase<>'' OR c.captured_leases<>'[]'::jsonb THEN RAISE EXCEPTION 'Route645 exact scope, clock or restart state differs'; END IF;
 IF c.new_claim_round_id IS NOT NULL AND (c.new_claim_round_id,c.new_claim_allowed_runner_hotkey,c.new_claim_until,c.new_claim_actor_ref)
  IS DISTINCT FROM ('arena-2026-11-05'::text,primary_runner,until_time,'nov05-primary-new-claims645'::text) THEN RAISE EXCEPTION 'Route645 existing route differs'; END IF;
 UPDATE public.lab_arena_restart_claim_control SET new_claim_round_id='arena-2026-11-05',
  new_claim_allowed_runner_hotkey=primary_runner,new_claim_until=until_time,new_claim_actor_ref='nov05-primary-new-claims645' WHERE singleton;
 IF (SELECT to_jsonb(a)-ARRAY['new_claim_round_id','new_claim_allowed_runner_hotkey','new_claim_until','new_claim_actor_ref'] FROM public.lab_arena_restart_claim_control a WHERE singleton)
  IS DISTINCT FROM to_jsonb(c)-ARRAY['new_claim_round_id','new_claim_allowed_runner_hotkey','new_claim_until','new_claim_actor_ref'] THEN RAISE EXCEPTION 'Route645 existing control changed'; END IF;
END;
$activate645$;
NOTIFY pgrst,'reload schema';
COMMIT;
