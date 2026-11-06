-- Prospective 23:50 UTC replacement/review freeze for midnight daily rounds.
-- Legacy NULL retains 3600 seconds. Signed configuration, first intake cutoff,
-- disclosure, evaluation windows, source, charges and frozen outcomes are unchanged.
-- Apply after the exact November 6 recovery is finalized: adding a column also
-- changes native whole-row JSON used by its immutable receipt guards.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
LOCK TABLE public.lab_arena_rounds,public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
DO $recovery639$
BEGIN
 IF EXISTS(SELECT 1 FROM public.lab_arena_submissions WHERE submission_id='sub-43a929eb0e2e487f68014f48e7d80e6b')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions WHERE round_id='arena-2026-11-06'
   AND request_id='nov06-admit-ninth-reviewed638-v1') THEN
  RAISE EXCEPTION 'Arena cutoff639 requires completed ninth recovery';
 END IF;
 IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
  WHERE h.request_id IN('nov06-late-review635-v1','nov06-ninth-review637-v1')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions f WHERE f.round_id=h.round_id
   AND f.request_id=CASE h.request_id WHEN 'nov06-late-review635-v1' THEN 'nov06-admit-late-reviewed636-v1'
    ELSE 'nov06-admit-ninth-reviewed638-v1' END)) THEN
  RAISE EXCEPTION 'Arena cutoff639 requires completed November 6 recovery';
 END IF;
END;
$recovery639$;
-- Add without a default first, so existing frozen rounds retain NULL/legacy.
DO $precolumn639$
BEGIN
 IF EXISTS(SELECT 1 FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
  WHERE a.attrelid='public.lab_arena_rounds'::regclass AND a.attname='replacement_freeze_seconds'
  AND (a.atttypid<>'integer'::regtype OR a.attnotnull OR a.attisdropped
   OR pg_get_expr(d.adbin,d.adrelid) IS DISTINCT FROM '600')) THEN
  RAISE EXCEPTION 'Arena cutoff639 existing column differs'; END IF;
END;
$precolumn639$;
ALTER TABLE public.lab_arena_rounds ADD COLUMN IF NOT EXISTS replacement_freeze_seconds INTEGER;
ALTER TABLE public.lab_arena_rounds ALTER COLUMN replacement_freeze_seconds SET DEFAULT 600;
DO $column639$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_attribute WHERE attrelid='public.lab_arena_rounds'::regclass
  AND attname='replacement_freeze_seconds' AND atttypid='integer'::regtype AND NOT attnotnull AND NOT attisdropped) THEN
  RAISE EXCEPTION 'Arena cutoff639 column differs'; END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='public.lab_arena_rounds'::regclass
  AND conname='lab_arena_round_replacement_freeze639_check') THEN
  ALTER TABLE public.lab_arena_rounds ADD CONSTRAINT lab_arena_round_replacement_freeze639_check
   CHECK(replacement_freeze_seconds IN(600,3600));
 ELSIF NOT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='public.lab_arena_rounds'::regclass
  AND conname='lab_arena_round_replacement_freeze639_check' AND convalidated
  AND pg_get_constraintdef(oid)='CHECK ((replacement_freeze_seconds = ANY (ARRAY[600, 3600])))') THEN
  RAISE EXCEPTION 'Arena cutoff639 constraint differs';
 END IF;
END;
$column639$;
CREATE TEMP TABLE arena639_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $guard639$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena_round_replacement_freeze639_v1()');
 IF FOUND AND (p.prosrc IS DISTINCT FROM $expected639$BEGIN
 IF TG_OP='INSERT' THEN
  IF NEW.replacement_freeze_seconds IS DISTINCT FROM 600 THEN
   RAISE EXCEPTION 'lab_arena_replacement_policy_invalid'; END IF;
 ELSIF NEW.replacement_freeze_seconds IS DISTINCT FROM OLD.replacement_freeze_seconds THEN
  IF OLD.replacement_freeze_seconds IS NOT NULL OR NEW.replacement_freeze_seconds IS DISTINCT FROM 600
   OR OLD.status IS DISTINCT FROM 'open' OR OLD.benchmark_ref IS NOT NULL
   OR COALESCE(OLD.participants,'[]'::JSONB)<>'[]'::JSONB
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=OLD.round_id)
   OR EXISTS(SELECT 1 FROM public.lab_arena_submissions WHERE round_id=OLD.round_id
    AND (code_review_attempts>0 OR code_review_started_at IS NOT NULL OR code_review_status<>'pending')) THEN
   RAISE EXCEPTION 'lab_arena_replacement_policy_frozen'; END IF;
 END IF;
 RETURN NEW;
END;
$expected639$
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'v'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Arena cutoff639 field guard differs'; END IF;
END;
$guard639$;
CREATE OR REPLACE FUNCTION public.lab_arena_round_replacement_freeze639_v1()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $field639$BEGIN
 IF TG_OP='INSERT' THEN
  IF NEW.replacement_freeze_seconds IS DISTINCT FROM 600 THEN
   RAISE EXCEPTION 'lab_arena_replacement_policy_invalid'; END IF;
 ELSIF NEW.replacement_freeze_seconds IS DISTINCT FROM OLD.replacement_freeze_seconds THEN
  IF OLD.replacement_freeze_seconds IS NOT NULL OR NEW.replacement_freeze_seconds IS DISTINCT FROM 600
   OR OLD.status IS DISTINCT FROM 'open' OR OLD.benchmark_ref IS NOT NULL
   OR COALESCE(OLD.participants,'[]'::JSONB)<>'[]'::JSONB
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=OLD.round_id)
   OR EXISTS(SELECT 1 FROM public.lab_arena_submissions WHERE round_id=OLD.round_id
    AND (code_review_attempts>0 OR code_review_started_at IS NOT NULL OR code_review_status<>'pending')) THEN
   RAISE EXCEPTION 'lab_arena_replacement_policy_frozen'; END IF;
 END IF;
 RETURN NEW;
END;
$field639$;
ALTER FUNCTION public.lab_arena_round_replacement_freeze639_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_round_replacement_freeze639_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $trigger639$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass
  AND tgname='lab_arena_round_replacement_freeze639') THEN
  CREATE TRIGGER lab_arena_round_replacement_freeze639 BEFORE INSERT OR UPDATE ON public.lab_arena_rounds
   FOR EACH ROW EXECUTE FUNCTION public.lab_arena_round_replacement_freeze639_v1();
 ELSIF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass
  AND tgname='lab_arena_round_replacement_freeze639' AND tgtype=23 AND tgenabled='O'
  AND tgfoid='public.lab_arena_round_replacement_freeze639_v1()'::regprocedure AND tgqual IS NULL) THEN
  RAISE EXCEPTION 'Arena cutoff639 trigger differs'; END IF;
END;
$trigger639$;
UPDATE public.lab_arena_rounds r SET replacement_freeze_seconds=600
 WHERE r.replacement_freeze_seconds IS NULL AND r.status='open' AND r.benchmark_ref IS NULL
 AND COALESCE(r.participants,'[]'::JSONB)='[]'::JSONB
 AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=r.round_id)
 AND NOT EXISTS(SELECT 1 FROM public.lab_arena_submissions WHERE round_id=r.round_id
  AND (code_review_attempts>0 OR code_review_started_at IS NOT NULL OR code_review_status<>'pending'));
DO $patch639_0$
DECLARE target OID:=to_regprocedure('public.lab_arena__register_submission_v2(text,text,text,jsonb,text,bigint,text)'); d TEXT; updated TEXT; metadata JSONB; ident JSONB;
 old TEXT:=$old639_0$- INTERVAL '1 hour'$old639_0$;
 new TEXT:=$new639_0$- pg_catalog.make_interval(secs => COALESCE(v_round.replacement_freeze_seconds,3600))$new639_0$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Arena cutoff639 function 0 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='f491226a0d74efcb450881004a3b7507dc1e8768d0c4dd3f12f7f97c0dee8c8e' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'ea517cb1876cccae0e92c601d68a34a8ff90ecf4409080f739b196c4974b1a92' THEN
  RAISE EXCEPTION 'Arena cutoff639 function 0 preimage differs'; END IF;
 IF length(d)-length(replace(d,old,''))<>length(old) THEN
  RAISE EXCEPTION 'Arena cutoff639 function 0 anchor differs'; END IF;
 updated:=replace(d,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'f491226a0d74efcb450881004a3b7507dc1e8768d0c4dd3f12f7f97c0dee8c8e' THEN
  RAISE EXCEPTION 'Arena cutoff639 function 0 postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'f491226a0d74efcb450881004a3b7507dc1e8768d0c4dd3f12f7f97c0dee8c8e'
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Arena cutoff639 function 0 readback differs'; END IF;
END;
$patch639_0$;
DO $patch639_1$
DECLARE target OID:=to_regprocedure('public.lab_arena_accept_submission_with_credentials(text,text,text,jsonb)'); d TEXT; updated TEXT; metadata JSONB; ident JSONB;
 old TEXT:=$old639_1$- INTERVAL '1 hour'$old639_1$;
 new TEXT:=$new639_1$- pg_catalog.make_interval(secs => COALESCE(v_round.replacement_freeze_seconds,3600))$new639_1$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Arena cutoff639 function 1 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='85b28093923024ae546bbd7ad11e8185cc1993f8c4a2d3606acc992ac40b2d1c' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'3cd386766ce8277fac755906c37df5bca7b9d69b046ee964a4587e39f73c938d' THEN
  RAISE EXCEPTION 'Arena cutoff639 function 1 preimage differs'; END IF;
 IF length(d)-length(replace(d,old,''))<>length(old) THEN
  RAISE EXCEPTION 'Arena cutoff639 function 1 anchor differs'; END IF;
 updated:=replace(d,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'85b28093923024ae546bbd7ad11e8185cc1993f8c4a2d3606acc992ac40b2d1c' THEN
  RAISE EXCEPTION 'Arena cutoff639 function 1 postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'85b28093923024ae546bbd7ad11e8185cc1993f8c4a2d3606acc992ac40b2d1c'
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Arena cutoff639 function 1 readback differs'; END IF;
END;
$patch639_1$;
DO $patch639_2$
DECLARE target OID:=to_regprocedure('public.lab_arena_accept_submission_source_with_credentials(text,text,text,jsonb,text,text)'); d TEXT; updated TEXT; metadata JSONB; ident JSONB;
 old TEXT:=$old639_2$- INTERVAL '1 hour'$old639_2$;
 new TEXT:=$new639_2$- pg_catalog.make_interval(secs => COALESCE(v_round.replacement_freeze_seconds,3600))$new639_2$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Arena cutoff639 function 2 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='65c8c7495aaaec54c9cc2875553161198edef064ec3f581ce22cf47458c7e6cb' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'6b239f34cda2af8b2b8e625d4971242d021ed36ba462b8b99b80acc946f6c4cb' THEN
  RAISE EXCEPTION 'Arena cutoff639 function 2 preimage differs'; END IF;
 IF length(d)-length(replace(d,old,''))<>length(old) THEN
  RAISE EXCEPTION 'Arena cutoff639 function 2 anchor differs'; END IF;
 updated:=replace(d,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'65c8c7495aaaec54c9cc2875553161198edef064ec3f581ce22cf47458c7e6cb' THEN
  RAISE EXCEPTION 'Arena cutoff639 function 2 postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'65c8c7495aaaec54c9cc2875553161198edef064ec3f581ce22cf47458c7e6cb'
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Arena cutoff639 function 2 readback differs'; END IF;
END;
$patch639_2$;
DO $patch639_3$
DECLARE target OID:=to_regprocedure('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'); d TEXT; updated TEXT; metadata JSONB; ident JSONB;
 old TEXT:=$old639_3$- INTERVAL '1 hour'$old639_3$;
 new TEXT:=$new639_3$- pg_catalog.make_interval(secs => COALESCE(v_round.replacement_freeze_seconds,3600))$new639_3$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Arena cutoff639 function 3 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='d101aa5610e1c64aa92a975754a45cfdbfb70b1d7fff20832ddb1ea72dc148ff' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'44e1e4125cd373913cb1f02419ba17b7ffb1f45e8b55e7190aa256e85f5a6b1c' THEN
  RAISE EXCEPTION 'Arena cutoff639 function 3 preimage differs'; END IF;
 IF length(d)-length(replace(d,old,''))<>length(old) THEN
  RAISE EXCEPTION 'Arena cutoff639 function 3 anchor differs'; END IF;
 updated:=replace(d,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'d101aa5610e1c64aa92a975754a45cfdbfb70b1d7fff20832ddb1ea72dc148ff' THEN
  RAISE EXCEPTION 'Arena cutoff639 function 3 postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'d101aa5610e1c64aa92a975754a45cfdbfb70b1d7fff20832ddb1ea72dc148ff'
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Arena cutoff639 function 3 readback differs'; END IF;
END;
$patch639_3$;
DO $patch639_4$
DECLARE target OID:=to_regprocedure('public.lab_arena_submission_replacement_schema_v1()'); d TEXT; updated TEXT; metadata JSONB; ident JSONB;
 old TEXT:=$old639_4$    'replacement_freeze_seconds', 3600,$old639_4$;
 new TEXT:=$new639_4$    'replacement_freeze_seconds', 3600,
    'per_round_replacement_freeze_seconds', TRUE,
    'default_replacement_freeze_seconds', 600,$new639_4$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,metadata,ident FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=target;
 IF d IS NULL OR ident IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Arena cutoff639 function 4 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='42d80fc942a8c8a931804be622328e8672cb8a2c1703b384d6c8b61c0c266ff2' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'2f555038f80d9d89f4db2cae6fe585320366915ee8e7303f968386d3334024db' THEN
  RAISE EXCEPTION 'Arena cutoff639 function 4 preimage differs'; END IF;
 IF length(d)-length(replace(d,old,''))<>length(old) THEN
  RAISE EXCEPTION 'Arena cutoff639 function 4 anchor differs'; END IF;
 updated:=replace(d,old,new);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'42d80fc942a8c8a931804be622328e8672cb8a2c1703b384d6c8b61c0c266ff2' THEN
  RAISE EXCEPTION 'Arena cutoff639 function 4 postimage differs'; END IF;
 EXECUTE updated;
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'42d80fc942a8c8a931804be622328e8672cb8a2c1703b384d6c8b61c0c266ff2'
  OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'Arena cutoff639 function 4 readback differs'; END IF;
END;
$patch639_4$;
DO $restore639$
BEGIN
 IF NOT (SELECT had_create FROM arena639_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM arena639_schema_acl) THEN
  RAISE EXCEPTION 'Arena cutoff639 schema ACL differs'; END IF;
END;
$restore639$;
NOTIFY pgrst,'reload schema';
COMMIT;
