-- Isolate bounded validator readiness events from ordinary diagnostic volume.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $readiness_allowance$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_append_validator_events_v1(text,text,integer,jsonb,text)';
  v_preimage CONSTANT TEXT :=
    '7bd9bfdeb5042891dadd6ee761abbbc8cd7c4901d05e0eed8ac92dd8fdd9a670';
  v_postimage CONSTANT TEXT :=
    '209e26af62bf268ba616fc4c5d3edb03c736830dcac70ec6cc177955586f37ac';
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
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
  IF v_hash <> v_preimage THEN
    RAISE EXCEPTION 'Arena validator event preimage differs' USING ERRCODE = '55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition, $old0$  v_day INTEGER;$old0$, $new0$  v_day INTEGER;
  v_readiness_minute INTEGER;
  v_readiness_day INTEGER;
  v_readiness_new INTEGER;$new0$);
  v_definition := pg_catalog.replace(v_definition, $old1$  SELECT count(*) FILTER (WHERE created_at >= v_now - INTERVAL '1 minute'), count(*)
  INTO v_minute, v_day FROM public.lab_arena_validator_events
  WHERE validator_hotkey = p_validator_hotkey AND created_at >= v_now - INTERVAL '1 day';$old1$, $new1$  -- Operational error floods must not consume restart readiness capacity.
  -- Each lane retains the existing 60/minute and 1000/day bound per hotkey.
  SELECT count(*) FILTER (WHERE NOT readiness AND created_at >= v_now - INTERVAL '1 minute'),
         count(*) FILTER (WHERE NOT readiness),
         count(*) FILTER (WHERE readiness AND created_at >= v_now - INTERVAL '1 minute'),
         count(*) FILTER (WHERE readiness)
  INTO v_minute, v_day, v_readiness_minute, v_readiness_day
  FROM (
    SELECT created_at, COALESCE((event_kind IN ('validator.startup','validator.ready','validator.stopping') OR (event_kind IN ('validator.error','validator.recovered') AND content->>'phase' IN ('startup','weights','scoring_setup','scoring_cycle'))), FALSE) AS readiness
    FROM public.lab_arena_validator_events
    WHERE validator_hotkey = p_validator_hotkey AND created_at >= v_now - INTERVAL '1 day'
  ) recent;$new1$);
  v_definition := pg_catalog.replace(v_definition, $old2$  SELECT count(*) INTO v_inserted FROM (
    SELECT DISTINCT (value ->> 'event_id')::UUID AS id FROM jsonb_array_elements(p_events)
  ) incoming WHERE NOT EXISTS (
    SELECT 1 FROM public.lab_arena_validator_events old WHERE old.validator_hotkey = p_validator_hotkey AND old.event_id = incoming.id
  );
  IF v_minute + v_inserted > 60 OR v_day + v_inserted > 1000 THEN$old2$, $new2$  SELECT count(*) FILTER (WHERE NOT readiness), count(*) FILTER (WHERE readiness)
  INTO v_inserted, v_readiness_new FROM (
    -- The first occurrence is the one the insertion loop retains.
    SELECT DISTINCT ON ((value->>'event_id')::UUID) (value->>'event_id')::UUID AS id,
           COALESCE((value->>'kind' IN ('validator.startup','validator.ready','validator.stopping') OR (value->>'kind' IN ('validator.error','validator.recovered') AND value->'content'->>'phase' IN ('startup','weights','scoring_setup','scoring_cycle'))), FALSE) AS readiness
    FROM jsonb_array_elements(p_events) WITH ORDINALITY AS item(value, position)
    ORDER BY (value->>'event_id')::UUID, position
  ) incoming WHERE NOT EXISTS (
    SELECT 1 FROM public.lab_arena_validator_events old WHERE old.validator_hotkey = p_validator_hotkey AND old.event_id = incoming.id
  );
  IF v_minute + v_inserted > 60 OR v_day + v_inserted > 1000
     OR v_readiness_minute + v_readiness_new > 60
     OR v_readiness_day + v_readiness_new > 1000 THEN$new2$);
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
$readiness_allowance$;
COMMIT;
