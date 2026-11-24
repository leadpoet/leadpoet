-- An abandoned restart guard must stop withholding paid review work once its
-- lease expires. Migration 705 fixed the review loop's own check; the paid
-- reservation gate still refuses every entry while any commitment is present,
-- so a guard left behind by a dead restart controller keeps review stopped
-- even after the loop starts asking for work again. Read the lease here too.
-- The control CHECK keeps guard_expires_at NOT NULL whenever a guard is set,
-- so an expired lease is always an abandoned restart and never an operator
-- pause. A live drain still refuses reservations exactly as before, and
-- nothing about ledger rules, verdicts, deadlines or stored rows changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena706_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create FROM pg_namespace WHERE nspname='public';
DO $prerequisites706$
BEGIN
  IF pg_catalog.to_regproc('public.lab_arena_review_guard_state_v1') IS NULL THEN
    RAISE EXCEPTION 'apply the Arena review guard row read before migration 706';
  END IF;
END;
$prerequisites706$;
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Anchored rewrite of the exact deployed reservation gate, then of the
-- readiness probe that pins the gate body it accepts. Each is skipped when
-- already at its post-image, so a partial apply is safe to repeat.
DO $patch706$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text)';
  v_probe CONSTANT TEXT := 'public.lab_arena_submission_replacement_schema_v1()';
  v_acl CONSTANT TEXT := '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}';
  v_preimage CONSTANT TEXT := 'd5bf3a7c86e97dfb1b7080f5d6e959b05c0fa3463a10ecf88f93e67fddbaba5a';
  v_postimage CONSTANT TEXT := '7153f9bf3687dd149f92c0239045e9970bda1522276a0307d0173962a803f791';
  v_probe_preimage CONSTANT TEXT := 'e39f8dd8162715b476dd02f7a6535512500fa0d29e8ccf9f43b2903361d3adbc';
  v_decl CONSTANT TEXT := $d706$  v_review_guard_present BOOLEAN;$d706$;
  v_decl_new CONSTANT TEXT := $d706$  v_review_guard_active BOOLEAN;$d706$;
  v_read CONSTANT TEXT := $r706$  SELECT guard_commitment <> '' INTO STRICT v_review_guard_present
  FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;$r706$;
  v_read_new CONSTANT TEXT := $r706$  -- lab_arena_restart_guard_lease706: a retained commitment records that a
  -- restart was begun, but only a live lease may withhold paid review. The
  -- control CHECK keeps guard_expires_at NOT NULL whenever a guard is set,
  -- so an expired lease is an abandoned restart, never an operator pause.
  SELECT guard_commitment <> ''
           AND guard_expires_at IS NOT NULL
           AND guard_expires_at > pg_catalog.clock_timestamp()
    INTO STRICT v_review_guard_active
  FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;$r706$;
  v_use CONSTANT TEXT := $u706$  IF v_review_guard_present AND NOT ($u706$;
  v_use_new CONSTANT TEXT := $u706$  IF v_review_guard_active AND NOT ($u706$;
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
  v_anchor TEXT;
  v_index INTEGER;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity FROM pg_catalog.pg_proc p WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner',v_acl,TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena lease706 security differs: %',v_signature USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') IS DISTINCT FROM v_preimage THEN
      RAISE EXCEPTION 'Arena lease706 preimage differs: %',v_signature USING ERRCODE='55000';
    END IF;
    v_updated:=v_definition;
    FOR v_index IN 1..3 LOOP
      v_anchor:=(ARRAY[v_decl,v_read,v_use])[v_index];
      IF length(v_updated)-length(pg_catalog.replace(v_updated,v_anchor,''))<>length(v_anchor) THEN
        RAISE EXCEPTION 'Arena lease706 anchor differs: %',v_index USING ERRCODE='55000';
      END IF;
      v_updated:=pg_catalog.replace(v_updated,v_anchor,
        (ARRAY[v_decl_new,v_read_new,v_use_new])[v_index]);
    END LOOP;
    IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
      RAISE EXCEPTION 'Arena lease706 postimage differs: %',
        pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex') USING ERRCODE='55000';
    END IF;
    EXECUTE v_updated;
    IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
       OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
         IS DISTINCT FROM v_metadata THEN
      RAISE EXCEPTION 'Arena lease706 readback differs: %',v_signature USING ERRCODE='55000';
    END IF;
  END IF;

  -- The readiness probe is an accept-list of exactly one gate body. Swap the
  -- hash it pins in place: the edit is a fixed-length substitution of a
  -- literal this migration already knows, and the readback below proves the
  -- stored definition is byte-identical to what we executed.
  v_oid:=pg_catalog.to_regprocedure(v_probe);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity FROM pg_catalog.pg_proc p WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner',v_acl,TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena lease706 security differs: %',v_probe USING ERRCODE='55000';
  END IF;
  IF pg_catalog.strpos(v_definition,v_postimage)=0 THEN
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') IS DISTINCT FROM v_probe_preimage THEN
      RAISE EXCEPTION 'Arena lease706 preimage differs: %',v_probe USING ERRCODE='55000';
    END IF;
    v_anchor:=pg_catalog.quote_literal(v_preimage);
    IF length(v_definition)-length(pg_catalog.replace(v_definition,v_anchor,''))<>length(v_anchor) THEN
      RAISE EXCEPTION 'Arena lease706 anchor differs: %',v_probe USING ERRCODE='55000';
    END IF;
    v_updated:=pg_catalog.replace(v_definition,v_anchor,pg_catalog.quote_literal(v_postimage));
    EXECUTE v_updated;
    IF pg_catalog.pg_get_functiondef(v_oid) IS DISTINCT FROM v_updated
       OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
         IS DISTINCT FROM v_metadata THEN
      RAISE EXCEPTION 'Arena lease706 readback differs: %',v_probe USING ERRCODE='55000';
    END IF;
  END IF;
END;
$patch706$;

DO $restore706$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena706_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
      (SELECT nspacl FROM pg_temp.arena706_acl) THEN
    RAISE EXCEPTION 'Arena lease706 schema ACL changed';
  END IF;
END;
$restore706$;
NOTIFY pgrst, 'reload schema';
COMMIT;
