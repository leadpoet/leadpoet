-- Preserve the independent operator hold owner across canonical guard changes.
-- Apply under the canonical release lock after every old guard RPC finishes.
-- No active guard or operator hold may be present at installation. This does
-- not reconstruct an owner already overwritten before this migration.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));

DO $control647$
DECLARE c public.lab_arena_restart_claim_control;
BEGIN
 IF current_setting('session_replication_role') IS DISTINCT FROM 'origin' THEN
  RAISE EXCEPTION 'Guard647 origin triggers required' USING ERRCODE='55000'; END IF;
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 IF c.operator_paused OR c.guard_commitment<>'' OR c.owner_commitment<>''
  OR c.guard_expires_at IS NOT NULL OR c.candidate_commit<>''
  OR c.restart_scope<>'' OR c.restart_phase<>'' OR c.captured_leases<>'[]'::JSONB THEN
  RAISE EXCEPTION 'Guard647 requires clear canonical control and released operator hold' USING ERRCODE='55000';
 END IF;
END;
$control647$;

CREATE TEMP TABLE arena647_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner',oid,'CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
DO $grant647$
BEGIN
 IF NOT (SELECT had_create FROM arena647_schema_acl) THEN
  GRANT CREATE ON SCHEMA public TO lab_arena_owner;
 END IF;
END;
$grant647$;

DO $owner647$
DECLARE
 target RECORD; target_oid OID; definition TEXT; updated TEXT;
 metadata JSONB; identity JSONB;
 old_actor CONSTANT TEXT:='actor_ref = left(p_actor_ref, 200)';
 new_actor CONSTANT TEXT:='actor_ref = CASE WHEN operator_paused THEN actor_ref ELSE left(p_actor_ref, 200) END';
BEGIN
 FOR target IN SELECT * FROM (VALUES
  ('public.lab_arena_acquire_restart_guard_v1(text,text,bigint,integer,text,text,text)',
   '2c9f6a26b3ba26734442a4972459087d219a105560fddd6db718cd39d814d345',
   '83dba0a5c7970dd456e78451e9ed3b7fd8bf1c77b2f4281a90a95b222ce95a0a',2),
  ('public.lab_arena_retarget_restart_guard_v1(text,text,bigint,text,text,text,integer,text)',
   'dc6689529530bd50eeb7b9af4dcaa7621ff0174e646b6bfc99d7c42d7c079e96',
   '763cea56276682120e1fdb0720eb2b376612864e0afe44efa49bb401cfef6bc6',1),
  ('public.lab_arena_abort_restart_guard_v1(text,text,bigint,text)',
   'f5fcff465cc4c3e060c3012d09ece395f1e126aa759e182e72f159ca41367bef',
   '1c2527706fce107f3fc4c1d131f62d205bbd9d2b47d514df5d2e88dc4bf411e3',1),
  ('public.lab_arena_release_restart_guard_v1(text,text,bigint,text)',
   '8bf0cf88f0530a60375195b499bb2538b57f1d6dcac161274a5ce951b7729266',
   'deab57267c9a7832f06848d374141a3794f570d21be8de68fcebb96b9d18d1c8',1)
 ) AS reviewed(signature,preimage,postimage,anchors) LOOP
  target_oid:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
   jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
    p.prosecdef,p.provolatile,p.proconfig,l.lanname)
  INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target_oid;
  IF definition IS NULL OR identity IS DISTINCT FROM jsonb_build_array(
   'lab_arena_owner',
   '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner,service_role=X/lab_arena_owner}',
   TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql') THEN
   RAISE EXCEPTION 'Guard647 security differs: %',target.signature USING ERRCODE='55000'; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM target.preimage
   OR length(definition)-length(replace(definition,old_actor,''))<>length(old_actor)*target.anchors THEN
   RAISE EXCEPTION 'Guard647 source differs: %',target.signature USING ERRCODE='55000'; END IF;
  updated:=replace(definition,old_actor,new_actor);
  IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM target.postimage THEN
   RAISE EXCEPTION 'Guard647 postimage differs: %',target.signature USING ERRCODE='55000'; END IF;
  EXECUTE updated;
  IF encode(extensions.digest(pg_get_functiondef(target_oid),'sha256'),'hex') IS DISTINCT FROM target.postimage
   OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target_oid) IS DISTINCT FROM metadata THEN
   RAISE EXCEPTION 'Guard647 readback differs: %',target.signature USING ERRCODE='55000'; END IF;
 END LOOP;
END;
$owner647$;

DO $restore647$
BEGIN
 IF NOT (SELECT had_create FROM arena647_schema_acl) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')
  IS DISTINCT FROM (SELECT nspacl FROM arena647_schema_acl) THEN
  RAISE EXCEPTION 'Guard647 schema ACL changed' USING ERRCODE='55000'; END IF;
END;
$restore647$;
COMMIT;
