-- Match publication: an assessed baseline remains a score comparison when its
-- training capture is incomplete. Challenger eligibility and all score, identity,
-- promotion, slot, decay and historical-reward checks stay unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $baseline_training698$
DECLARE
  v_signature REGPROCEDURE := 'public.lab_arena_reward_slot_snapshot(text,jsonb)'::REGPROCEDURE;
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_schema_acl ACLITEM[];
  v_hash TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT,
      p.prosecdef, p.provolatile, p.proconfig, l.lanname)
  INTO v_definition, v_identity FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_language AS l ON l.oid = p.prolang WHERE p.oid = v_signature;
  SELECT nspacl INTO v_schema_acl FROM pg_catalog.pg_namespace WHERE nspname = 'public';
  IF v_identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION '698 reward slot security differs' USING ERRCODE = '55000';
  END IF;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = 'd212eef055af33f3eb30a57e04d6bfa8b1e4e37fe83b95850d66ca7fc4fa32f2' THEN RETURN; END IF;
  IF v_hash <> '71700c2885e479b2a367cdcb27729d083525095731ee74b52f71408daac30956' THEN
    RAISE EXCEPTION '698 reward slot preimage differs' USING ERRCODE = '55000';
  END IF;
  v_updated := pg_catalog.replace(v_definition, $old$      AND (baseline.entry -> 'eligible' = 'true'::JSONB
$old$, $new$      AND (baseline.entry -> 'eligible' = 'true'::JSONB
        OR (baseline.entry -> 'eligible' = 'false'::JSONB
          AND baseline.entry ->> 'eligibility_reason' = 'training_data_incomplete'
          AND baseline.entry #> '{training_data_eligibility,eligible}' = 'false'::JSONB
          AND baseline.entry #> '{training_data_eligibility,source_review,eligible}' = 'true'::JSONB
          AND baseline.entry #>> '{training_data_eligibility,source_review,exemption}' = 'organizer_baseline')
$new$);
  IF pg_catalog.encode(extensions.digest(v_updated, 'sha256'), 'hex') <> 'd212eef055af33f3eb30a57e04d6bfa8b1e4e37fe83b95850d66ca7fc4fa32f2' THEN
    RAISE EXCEPTION '698 reward slot postimage differs' USING ERRCODE = '55000';
  END IF;
  EXECUTE v_updated;
  IF (SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT,
       p.prosecdef, p.provolatile, p.proconfig, l.lanname)
      FROM pg_catalog.pg_proc AS p JOIN pg_catalog.pg_language AS l ON l.oid = p.prolang
      WHERE p.oid = v_signature) IS DISTINCT FROM v_identity
     OR pg_catalog.pg_get_functiondef(v_signature) IS DISTINCT FROM v_updated
     OR (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public') IS DISTINCT FROM v_schema_acl THEN
    RAISE EXCEPTION '698 reward slot readback differs' USING ERRCODE = '55000';
  END IF;
END;
$baseline_training698$;
NOTIFY pgrst, 'reload schema';
COMMIT;
