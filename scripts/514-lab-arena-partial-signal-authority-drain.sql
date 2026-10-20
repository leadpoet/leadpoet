-- Use the same canonical authority exclusions as signed v3 completion.
-- Reused signal authorities stay part of the immutable company receipt.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $partial_signal_authority_drain514$
DECLARE
  target OID:=to_regprocedure('public.lab_arena__restart_drain_state_v1()');
  helper OID:=to_regprocedure('public.lab_arena_signal_authority_exclusions_v1(text)');
  definition TEXT;
  helper_definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  old_gate CONSTANT TEXT := $old$                  AND partial_company.evidence_doc -> 'runner_authority_exclusions' =
                    runs.claim_response -> 'runner_authority_exclusions'$old$;
  new_gate CONSTANT TEXT := $new$                  AND partial_company.evidence_doc -> 'runner_authority_exclusions' =
                    public.lab_arena_signal_authority_exclusions_v1(runs.run_id)$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname)
  INTO helper_definition,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=helper;
  IF helper_definition IS NULL OR identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB
    OR encode(extensions.digest(helper_definition,'sha256'),'hex')<>'a2dcb05a80bf6b9d45b047eb6b9bbb1d3db88c09e587cc476d9d6b8190352a47' THEN
    RAISE EXCEPTION 'Partial signal authority helper differs' USING ERRCODE='55000';
  END IF;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname)
  INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target;
  IF definition IS NULL OR identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Partial signal drain security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='a38782b4b6555c455fae64a2434d0bbd394f2f970bc42326475f187a4a1756b8' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')<>'49f2d7713419d20c182bf1548f9e731cb369611b5121ab6b116239ba2a8f77f2'
    OR (length(definition)-length(replace(definition,old_gate,'')))<>length(old_gate) THEN
    RAISE EXCEPTION 'Partial signal drain preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=replace(definition,old_gate,new_gate);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>'a38782b4b6555c455fae64a2434d0bbd394f2f970bc42326475f187a4a1756b8' THEN
    RAISE EXCEPTION 'Partial signal drain postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'a38782b4b6555c455fae64a2434d0bbd394f2f970bc42326475f187a4a1756b8' THEN
    RAISE EXCEPTION 'Partial signal drain readback differs' USING ERRCODE='55000';
  END IF;
END;
$partial_signal_authority_drain514$;
COMMIT;
