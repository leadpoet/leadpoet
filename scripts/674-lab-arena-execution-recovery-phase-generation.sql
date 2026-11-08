-- Phase transitions advance stage_generation; they do not change grading authority.
-- Recover an older expired execution only within the same frozen grading floor,
-- and bind its finite successor to the current phase in both journal and row.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena674_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $generation674$
DECLARE target OID:='public.lab_arena_grant_execution_infrastructure_recovery_v1(text,text,text,text,text,boolean)'::regprocedure;
 d TEXT; updated TEXT; metadata JSONB; defaults TEXT; identity JSONB; wanted RECORD; actual RECORD;
 protected_before JSONB; protected_after JSONB;
BEGIN
 FOR wanted IN SELECT * FROM (VALUES
 ('public.lab_arena__operator_execute_recovery_child_v1(public.lab_arena_runs)','4dee08783888087e296654ded754ad15813eb9db1fe11755dd077fa6ca350f92','s','sql'),
 ('public.lab_arena__operator_execute_recovery_valid_v1(public.lab_arena_rounds,public.lab_arena_runs)','9d0ab4381036326f06060889c2b02573c8c81d5891b621319ebeb4c130097081','s','plpgsql')
 ) expected(signature,source_hash,volatility,language) LOOP
  SELECT pg_get_userbyid(p.proowner) AS owner,p.proacl::TEXT AS acl,p.prosecdef,p.provolatile,
   p.proconfig,l.lanname,encode(extensions.digest(p.prosrc,'sha256'),'hex') AS source_hash
  INTO actual FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(wanted.signature);
  IF NOT FOUND OR actual.owner<>'lab_arena_owner' OR actual.acl IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
   OR actual.prosecdef IS DISTINCT FROM TRUE OR actual.provolatile::TEXT<>wanted.volatility
   OR actual.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
   OR actual.lanname<>wanted.language OR actual.source_hash<>wanted.source_hash THEN
   RAISE EXCEPTION 'Execution recovery674 helper security or source differs: %',wanted.signature; END IF;
 END LOOP;
 -- PostgreSQL 15 stores parser source positions inside proargdefaults. Compare
 -- its deparsed expression so CREATE OR REPLACE preserves the actual default.
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-ARRAY['prosrc','proargdefaults'],
  pg_get_expr(p.proargdefaults,0),jsonb_build_array(
  pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
 INTO d,metadata,defaults,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
 IF d IS NULL OR defaults IS DISTINCT FROM 'true'
  OR identity IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
  RAISE EXCEPTION 'Execution recovery674 grant security differs'; END IF;
 IF (SELECT encode(extensions.digest(prosrc,'sha256'),'hex') FROM pg_proc WHERE oid=target)='ccc1ea4ecb54f6ae8cc0ac0ea545be793d75eee24e36630f907c212b9cba7b4b' THEN RETURN; END IF;
 IF (SELECT encode(extensions.digest(prosrc,'sha256'),'hex') FROM pg_proc WHERE oid=target)<>'532d14d6356b218a6366a656e5a4a46563d9cb179c4af147d6aaa924ff1d32f2'
  OR length(d)-length(replace(d,$old_generation674$  OR parent.stage_generation<>r.stage_generation OR parent.scoring_generation_floor<>r.scoring_generation_floor$old_generation674$,''))<>length($old_generation674$  OR parent.stage_generation<>r.stage_generation OR parent.scoring_generation_floor<>r.scoring_generation_floor$old_generation674$)
  OR length(d)-length(replace(d,$old_child674$ child:=parent; child.run_id:=parent.assignment_id||':'||(parent.attempt+1)::TEXT; child.attempt:=parent.attempt+1;$old_child674$,''))<>length($old_child674$ child:=parent; child.run_id:=parent.assignment_id||':'||(parent.attempt+1)::TEXT; child.attempt:=parent.attempt+1;$old_child674$) THEN
  RAISE EXCEPTION 'Execution recovery674 grant preimage differs'; END IF;
 SELECT jsonb_object_agg(p.oid::TEXT,to_jsonb(p)) INTO protected_before FROM pg_proc p
 WHERE p.pronamespace='public'::regnamespace AND (p.proname IN(
  'lab_arena_claim_assignment','lab_arena_reserve_call','lab_arena_complete_attempt','lab_arena_expire_leases',
  'lab_arena_execution_attempt_count','lab_arena_provider_funding') OR p.proname LIKE 'lab_arena__operator_%');
 updated:=replace(replace(d,$old_generation674$  OR parent.stage_generation<>r.stage_generation OR parent.scoring_generation_floor<>r.scoring_generation_floor$old_generation674$,$new_generation674$  OR parent.stage_generation>r.stage_generation OR parent.scoring_generation_floor<>r.scoring_generation_floor$new_generation674$),
  $old_child674$ child:=parent; child.run_id:=parent.assignment_id||':'||(parent.attempt+1)::TEXT; child.attempt:=parent.attempt+1;$old_child674$,$new_child674$ child:=parent; child.run_id:=parent.assignment_id||':'||(parent.attempt+1)::TEXT; child.attempt:=parent.attempt+1;
 child.stage_generation:=r.stage_generation;$new_child674$);
 EXECUTE updated;
 IF (SELECT to_jsonb(p)-ARRAY['prosrc','proargdefaults'] FROM pg_proc p WHERE oid=target) IS DISTINCT FROM metadata
  OR (SELECT pg_get_expr(p.proargdefaults,0) FROM pg_proc p WHERE oid=target) IS DISTINCT FROM defaults
  OR (SELECT encode(extensions.digest(prosrc,'sha256'),'hex') FROM pg_proc WHERE oid=target)<>'ccc1ea4ecb54f6ae8cc0ac0ea545be793d75eee24e36630f907c212b9cba7b4b' THEN
  RAISE EXCEPTION 'Execution recovery674 grant postimage or metadata differs'; END IF;
 SELECT jsonb_object_agg(p.oid::TEXT,to_jsonb(p)) INTO protected_after FROM pg_proc p
 WHERE p.pronamespace='public'::regnamespace AND (p.proname IN(
  'lab_arena_claim_assignment','lab_arena_reserve_call','lab_arena_complete_attempt','lab_arena_expire_leases',
  'lab_arena_execution_attempt_count','lab_arena_provider_funding') OR p.proname LIKE 'lab_arena__operator_%');
 IF protected_after IS DISTINCT FROM protected_before THEN RAISE EXCEPTION 'Execution recovery674 shared functions changed'; END IF;
END;
$generation674$;
DO $restore674$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena674_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.arena674_schema_acl) THEN
  RAISE EXCEPTION 'Execution recovery674 schema ACL differs'; END IF;
END;
$restore674$;
COMMIT;
