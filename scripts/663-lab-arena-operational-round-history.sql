-- An operational audit revision is not a new grading generation.
-- One generic tag covers truthful owner-audited round state/schedule changes.
-- Existing operations, restoration semantics, data and write authority stay exact.
BEGIN;
SET LOCAL lock_timeout='3s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $history663$
DECLARE target OID:=to_regprocedure('public.lab_arena_scoring_policy_history_v1(text)');
 definition TEXT; updated TEXT; metadata JSONB; identity JSONB; digest TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
   jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
     p.provolatile,p.proconfig,l.lanname)
 INTO definition,metadata,identity
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
 IF definition IS NULL OR identity IS DISTINCT FROM
   '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB THEN
  RAISE EXCEPTION 'Operational history663 security differs' USING ERRCODE='55000';
 END IF;
 digest:=encode(pg_catalog.sha256(pg_catalog.convert_to(definition,'UTF8')),'hex');
 IF digest='c132d7014d7d85ad82a7c94b9d93e0e35f79233771815e245443633b5a157dca' THEN RETURN; END IF;
 IF digest<>'34123c2c2c2e68d4056a94c2f9f5bba4889d158ee8ac38869f6e68da88a2eb84'
   OR length(definition)-length(replace(definition,$old663$'nov05_b66_post_repair639_v1') THEN$old663$,''))<>length($old663$'nov05_b66_post_repair639_v1') THEN$old663$) THEN
  RAISE EXCEPTION 'Operational history663 preimage differs' USING ERRCODE='55000';
 END IF;
 updated:=replace(definition,$old663$'nov05_b66_post_repair639_v1') THEN$old663$,
   $new663$'nov05_b66_post_repair639_v1','round_operational_update_v1') THEN$new663$);
 IF encode(pg_catalog.sha256(pg_catalog.convert_to(updated,'UTF8')),'hex')<>'c132d7014d7d85ad82a7c94b9d93e0e35f79233771815e245443633b5a157dca' THEN
  RAISE EXCEPTION 'Operational history663 postimage differs' USING ERRCODE='55000';
 END IF;
 EXECUTE updated;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
   OR encode(pg_catalog.sha256(pg_catalog.convert_to(pg_get_functiondef(target),'UTF8')),'hex')<>'c132d7014d7d85ad82a7c94b9d93e0e35f79233771815e245443633b5a157dca' THEN
  RAISE EXCEPTION 'Operational history663 readback differs' USING ERRCODE='55000';
 END IF;
END;
$history663$;
COMMIT;
