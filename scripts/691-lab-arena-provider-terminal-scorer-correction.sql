-- Admit a separately reviewed scorer correction after a proved provider terminal.
-- Reuse the exact retained transient/unhealed proof; selected runtime must change.
-- Preserve the one-scorer-correction, budget/provider caps, ancestry, costs and work.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $patch691$
DECLARE p RECORD;proof RECORD;d TEXT;definition TEXT;metadata JSONB;
 old_text TEXT:=$old691$   OR ((p_recovery_kind='operator_judge_budget_exception_v1' AND$old691$;
 new_text TEXT:=$new691$   OR ((p_recovery_kind IN ('operator_judge_budget_exception_v1','operator_scorer_bug_recovery_v1') AND$new691$;
BEGIN
 SELECT * INTO STRICT proof FROM pg_proc
  WHERE oid='public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)'::regprocedure;
 IF encode(extensions.digest(proof.prosrc,'sha256'),'hex')<>'4d359779d99e492d325f8d2e55cd71c00e9156dee6cfa566ea1db34431e4b57f'
  OR pg_get_userbyid(proof.proowner)<>'lab_arena_owner'
  OR proof.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT proof.prosecdef OR proof.provolatile<>'s'
  OR proof.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR proof.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR proof.proisstrict OR proof.proleakproof OR proof.proparallel<>'u' THEN
  RAISE EXCEPTION 'provider-terminal correction691 retained proof differs'; END IF;
 SELECT * INTO STRICT p FROM pg_proc WHERE oid='public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)'::regprocedure;
 metadata:=to_jsonb(p)-'prosrc';d:=p.prosrc;
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v'
  OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN
  RAISE EXCEPTION 'provider-terminal correction691 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='1f2b05d010a26698a2aa9b7c402868582884376efc6533857178d813e8b3fc85' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'5befa705e76c54757dea017657c7a60a8cd184b7f27419153c5157671740c77e'
  OR (length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN
  RAISE EXCEPTION 'provider-terminal correction691 preimage differs'; END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'1f2b05d010a26698a2aa9b7c402868582884376efc6533857178d813e8b3fc85' THEN
  RAISE EXCEPTION 'provider-terminal correction691 postimage differs'; END IF;
 definition:=pg_get_functiondef(p.oid);
 IF (length(definition)-length(replace(definition,p.prosrc,'')))/length(p.prosrc)<>1 THEN
  RAISE EXCEPTION 'provider-terminal correction691 definition differs'; END IF;
 EXECUTE replace(definition,p.prosrc,d);
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid) IS DISTINCT FROM metadata
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex') FROM pg_proc q WHERE q.oid=p.oid)<>'1f2b05d010a26698a2aa9b7c402868582884376efc6533857178d813e8b3fc85' THEN
  RAISE EXCEPTION 'provider-terminal correction691 readback differs'; END IF;
END;
$patch691$;
NOTIFY pgrst,'reload schema';COMMIT;
