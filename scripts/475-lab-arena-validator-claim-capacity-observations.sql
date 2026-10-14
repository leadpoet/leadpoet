-- Allow only bounded claim decisions and live worker capacity in private
-- authenticated validator observations. These diagnostics grant no job authority.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $claim_capacity_observations$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_append_validator_events_v1(text,text,integer,jsonb,text)';
  v_preimage CONSTANT TEXT :=
    '148651f416b4bef86cee95ca037925867de3a9765f21b43cd0e28c3fa037ee4c';
  v_postimage CONSTANT TEXT :=
    '7bd9bfdeb5042891dadd6ee761abbbc8cd7c4901d05e0eed8ac92dd8fdd9a670';
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
  v_old_status CONSTANT TEXT := $old_status$      ELSIF v_pair.key IN ('retryable','launch_timed_out','launch_stderr_truncated') THEN$old_status$;
  v_new_status CONSTANT TEXT := $new_status$      ELSIF v_pair.key = 'last_claim_status' THEN
        IF jsonb_typeof(v_pair.value) NOT IN ('string','null')
           OR (jsonb_typeof(v_pair.value) = 'string'
               AND (v_pair.value #>> '{}') NOT IN (
                 'leased','no_pending','no_open_round','no_free_slot','paused',
                 'stage_closed','cache_busy','request_id_reused','rejected',
                 'request_failed','no_local_capacity','other')) THEN
          RAISE EXCEPTION 'lab_arena_validator_event_content_invalid' USING ERRCODE = '22023';
        END IF;
      ELSIF v_pair.key = 'last_claim_round_id' THEN
        IF jsonb_typeof(v_pair.value) NOT IN ('string','null')
           OR (jsonb_typeof(v_pair.value) = 'string'
               AND (v_pair.value #>> '{}') !~ '^[A-Za-z0-9._:-]{1,200}$') THEN
          RAISE EXCEPTION 'lab_arena_validator_event_content_invalid' USING ERRCODE = '22023';
        END IF;
      ELSIF v_pair.key IN ('retryable','launch_timed_out','launch_stderr_truncated') THEN$new_status$;
  v_old_slots CONSTANT TEXT := $old_slots$      ELSIF v_pair.key IN ('http_status','attempt','active_runs','ready_slots','proxy_count','launch_exit_code') THEN
        IF jsonb_typeof(v_pair.value) <> 'null' THEN
          IF jsonb_typeof(v_pair.value) <> 'number' OR (v_pair.value #>> '{}') !~ '^-?[0-9]+$' THEN
            RAISE EXCEPTION 'lab_arena_validator_event_content_invalid' USING ERRCODE = '22023';
          END IF;
          PERFORM (v_pair.value #>> '{}')::INTEGER;
        END IF;$old_slots$;
  v_new_slots CONSTANT TEXT := $new_slots$      ELSIF v_pair.key IN ('http_status','attempt','active_runs','ready_slots','proxy_count','launch_exit_code','usable_slots','quarantined_slots') THEN
        IF jsonb_typeof(v_pair.value) <> 'null' THEN
          IF jsonb_typeof(v_pair.value) <> 'number' OR (v_pair.value #>> '{}') !~ '^-?[0-9]+$' THEN
            RAISE EXCEPTION 'lab_arena_validator_event_content_invalid' USING ERRCODE = '22023';
          END IF;
          PERFORM (v_pair.value #>> '{}')::INTEGER;
          IF v_pair.key IN ('usable_slots','quarantined_slots')
             AND (v_pair.value #>> '{}')::INTEGER NOT BETWEEN 0 AND 251 THEN
            RAISE EXCEPTION 'lab_arena_validator_event_content_invalid' USING ERRCODE = '22023';
          END IF;
        END IF;$new_slots$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(owner.rolname, p.proacl::TEXT,
           p.prosecdef, p.provolatile, p.proconfig)
  INTO v_definition, v_identity
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
       pg_catalog.jsonb_build_array('lab_arena_owner',
         '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
         TRUE, 'v', ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena validator event security shape differs'
      USING ERRCODE = '55000';
  END IF;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = v_postimage THEN
    RETURN;
  END IF;
  IF v_hash <> v_preimage
     OR pg_catalog.strpos(v_definition, v_old_status) = 0
     OR pg_catalog.strpos(v_definition, v_old_slots) = 0 THEN
    RAISE EXCEPTION 'Arena validator event preimage differs'
      USING ERRCODE = '55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old_status, v_new_status);
  v_definition := pg_catalog.replace(v_definition, v_old_slots, v_new_slots);
  IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex')
       <> v_postimage THEN
    RAISE EXCEPTION 'Arena validator event postimage differs'
      USING ERRCODE = '55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(
       pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE), 'sha256'), 'hex')
       <> v_postimage THEN
    RAISE EXCEPTION 'Arena validator event readback differs'
      USING ERRCODE = '55000';
  END IF;
END;
$claim_capacity_observations$;
COMMIT;
