-- Permit a completed scored round to publish during an automatic restart drain.
-- Preserve explicit operator holds, all other progression holds and score holds.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $publication_restart_drain586$
DECLARE
  v_target OID := 'public.lab_arena_operator_hold_transition_guard_v1()'::pg_catalog.regprocedure;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
  v_preimage CONSTANT TEXT := 'a1180771aa74f37a92fedd26d35a6e3fadc6fa474a50eda5da08df26b5e00268';
  v_postimage CONSTANT TEXT := '754b81efbc162250f062f03b920a732d4439ce0ecc6aae590edae5c30218dffc';
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
  INTO v_definition,v_metadata,v_identity
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  JOIN pg_catalog.pg_language l ON l.oid=p.prolang
  WHERE p.oid=v_target;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'v',
    ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql')
    OR NOT EXISTS (
      SELECT 1 FROM pg_catalog.pg_trigger t
      WHERE t.tgrelid='public.lab_arena_rounds'::pg_catalog.regclass
        AND t.tgname='lab_arena_rounds_operator_hold_transition_guard'
        AND t.tgfoid=v_target AND t.tgenabled='O' AND NOT t.tgisinternal
    ) THEN
    RAISE EXCEPTION 'Arena publication586 security or trigger shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') IS DISTINCT FROM v_preimage THEN
    RAISE EXCEPTION 'Arena publication586 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_definition,
    $old586a$  v_paused BOOLEAN;
$old586a$,
    $new586a$  v_paused BOOLEAN;
  v_operator_paused BOOLEAN;
$new586a$);
  v_updated:=pg_catalog.replace(v_updated,
    $old586b$  SELECT operator_paused OR guard_commitment <> '' INTO v_paused
  FROM public.lab_arena_restart_claim_control WHERE singleton;
  IF COALESCE(v_paused, TRUE) THEN
$old586b$,
    $new586b$  SELECT operator_paused, operator_paused OR guard_commitment <> ''
    INTO v_operator_paused, v_paused
  FROM public.lab_arena_restart_claim_control WHERE singleton;
  -- A completed score may publish during an automatic restart drain.
  -- An explicit operator pause still stops publication.
  IF COALESCE(v_paused, TRUE) AND NOT (
    OLD.status = 'scored' AND NEW.status = 'published'
    AND v_operator_paused IS FALSE
  ) THEN
$new586b$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex') IS DISTINCT FROM v_postimage THEN
    RAISE EXCEPTION 'Arena publication586 target differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_target)
      IS DISTINCT FROM v_metadata
     OR pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_target),'sha256'),'hex')
      IS DISTINCT FROM v_postimage THEN
    RAISE EXCEPTION 'Arena publication586 readback differs' USING ERRCODE='55000';
  END IF;
END;
$publication_restart_drain586$;
COMMIT;
