-- A settlement is the final ledger head for a closed provider call. A paid
-- non-2xx response can omit call_succeeded; that omission is not in-flight
-- work. Keep reservation, dispatch, and unresolved uncertain heads blocked.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $settled_score_call_continuation477$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena__judge_calls_clear_v1(text)';
  v_preimage CONSTANT TEXT := 'bc1e769a81daacdcb6fac4c6d1e094b0ad288a075ca5a1e1a65433fd38c3e550';
  v_postimage CONSTANT TEXT := 'aa8250cbfe7b62e0ba7802eb1b8c2a38c5b4a012a3e0044a04cb09692318bae7';
  v_old_gate CONSTANT TEXT := $old$      OR (heads.entry_kind='settlement' AND
          jsonb_typeof(heads.terminal_response->'call_succeeded') IS DISTINCT FROM 'boolean')
$old$;
  v_definition TEXT;
  v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_signature::REGPROCEDURE;
  IF v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array('lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
    ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena settled score call security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_old_gate,'')))<>pg_catalog.length(v_old_gate) THEN
    RAISE EXCEPTION 'Arena settled score call preimage differs' USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,v_old_gate,'');
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena settled score call postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena settled score call readback differs' USING ERRCODE='55000';
  END IF;
END;
$settled_score_call_continuation477$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
