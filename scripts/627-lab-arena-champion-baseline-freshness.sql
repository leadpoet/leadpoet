-- Fail closed when a prior day's promotion supersedes a frozen baseline.
-- Preserve every source, accepted result, existing dispatch and historical round.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE arena627_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $helper627$
DECLARE p RECORD;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure('public.lab_arena__baseline_source_current_v1(public.lab_arena_rounds)');
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'9f6d44f3be0463154a4fa2e1db64c930a7035e8ac4623b235679c5fe58af8153'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'s' OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Baseline freshness627 helper differs'; END IF;
END;
$helper627$;
CREATE OR REPLACE FUNCTION public.lab_arena__baseline_source_current_v1(p_round public.lab_arena_rounds)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $current627$
DECLARE prior public.lab_arena_rounds; winner public.lab_arena_submissions; evaluation_day TEXT;
BEGIN
  -- Historical results and non-full-thinking rounds retain their saved authority.
  IF p_round.configuration_doc->>'mode' IS DISTINCT FROM 'live'
    OR p_round.configuration_doc->>'reasoning_capture_policy' IS DISTINCT FROM 'full_generated_thinking_v1'
    OR p_round.status IN ('published','cancelled') THEN RETURN TRUE; END IF;
  -- create_round leaves evaluation_date NULL until the later commit RPC.
  evaluation_day:=coalesce(p_round.evaluation_date,
    to_char((p_round.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ AT TIME ZONE 'UTC','YYYY-MM-DD'));
  IF evaluation_day IS NULL THEN RETURN FALSE; END IF;
  IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    JOIN public.lab_arena_rounds correction ON correction.round_id=h.round_id
    WHERE h.old_round_doc->>'status'='published'
      AND correction.status NOT IN ('published','cancelled')
      AND correction.arena_network_name=p_round.arena_network_name
      AND correction.arena_netuid=p_round.arena_netuid
      AND correction.evaluation_date<evaluation_day) THEN RETURN FALSE; END IF;
  -- A published crown must finish its existing promotion before a comparison.
  -- A later completed prior day retains the existing superseded-day policy.
  IF EXISTS(SELECT 1 FROM public.lab_arena_rounds pending
    WHERE pending.arena_network_name=p_round.arena_network_name
      AND pending.arena_netuid=p_round.arena_netuid
      AND pending.configuration_doc->>'mode'='live' AND pending.rewards_enabled
      AND pending.evaluation_date<evaluation_day
      AND pending.status='published' AND pending.promotion_required
      AND pending.publication_doc#>>'{king_decision,outcome}'='crowned'
      AND pending.baseline_promoted_at IS NULL
      AND NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds newer
        WHERE newer.arena_network_name=p_round.arena_network_name
          AND newer.arena_netuid=p_round.arena_netuid
          AND newer.configuration_doc->>'mode'='live' AND newer.status='published'
          AND newer.evaluation_date>pending.evaluation_date
          AND newer.evaluation_date<evaluation_day)) THEN RETURN FALSE; END IF;
  SELECT r.* INTO prior FROM public.lab_arena_rounds r
  WHERE r.arena_network_name=p_round.arena_network_name AND r.arena_netuid=p_round.arena_netuid
    AND r.configuration_doc->>'mode'='live' AND r.rewards_enabled
    AND r.evaluation_date<evaluation_day AND r.status='published'
    AND r.baseline_promoted_at IS NOT NULL
    AND r.publication_doc#>>'{king_decision,outcome}'='crowned'
  ORDER BY r.evaluation_date DESC,r.baseline_promoted_at DESC,r.round_id DESC LIMIT 1;
  IF NOT FOUND THEN
    -- Bootstrap is allowed only when no eligible prior champion exists.
    RETURN NOT p_round.champion_funding_frozen OR
      (p_round.champion_submission_id IS NULL AND p_round.champion_hotkey IS NULL);
  END IF;
  SELECT s.* INTO winner FROM public.lab_arena_submissions s
  WHERE s.submission_id=prior.publication_doc#>>'{king_decision,winner_submission_id}'
    AND s.round_id=prior.round_id
    AND s.miner_hotkey=prior.publication_doc#>>'{king_decision,king_hotkey}'
    AND s.status='frozen' AND NOT s.is_king;
  IF NOT FOUND OR coalesce(prior.promotion_doc->>'commit','') !~ '^[0-9a-f]{40}$' THEN RETURN FALSE; END IF;
  IF NOT p_round.champion_funding_frozen THEN RETURN TRUE; END IF;
  RETURN coalesce(p_round.champion_submission_id=winner.submission_id
    AND p_round.champion_hotkey=winner.miner_hotkey
    AND p_round.configuration_doc->>'baseline_source_url'=
      'https://github.com/leadpoet/leadpoet-sales-agent/archive/'||
        (prior.promotion_doc->>'commit')||'.tar.gz'
    AND p_round.configuration_doc->>'reasoning_baseline_source_sha256' ~ '^sha256:[0-9a-f]{64}$'
    AND p_round.configuration_doc->'baseline_reasoning_configuration'=
      (prior.configuration_doc - ARRAY(SELECT key FROM jsonb_object_keys(prior.configuration_doc) key
        WHERE key NOT IN ('reasoning_capture_policy','reasoning_route_registry'))),FALSE);
END;
$current627$;
ALTER FUNCTION public.lab_arena__baseline_source_current_v1(public.lab_arena_rounds) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__baseline_source_current_v1(public.lab_arena_rounds)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TEMP TABLE arena627_patches(signature TEXT,preimage TEXT,postimage TEXT,ordinal INTEGER,old_text TEXT,new_text TEXT) ON COMMIT DROP;
INSERT INTO arena627_patches VALUES
('public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)','caf315c8c6c4c15d946ba70b00b0a7c4f893f42ada5e77b3af22fe4fdea8575b','17619139dd59bcadcfc170ad677c5caeda47fb329ec833668759b63385f64cbd',0,$old627$  current_round public.lab_arena_rounds;$old627$,$new627$  current_round public.lab_arena_rounds;
  baseline_evaluation_day TEXT;$new627$),
('public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)','caf315c8c6c4c15d946ba70b00b0a7c4f893f42ada5e77b3af22fe4fdea8575b','17619139dd59bcadcfc170ad677c5caeda47fb329ec833668759b63385f64cbd',1,$old627$  -- The promotion lock and target row lock cover this fresh receipt read.
  IF EXISTS(
    SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_rounds correction ON correction.round_id=h.round_id
    WHERE h.old_round_doc->>'status'='published'
      AND correction.status NOT IN('published','cancelled')
      AND correction.arena_network_name=current_round.arena_network_name
      AND correction.arena_netuid=current_round.arena_netuid
      AND correction.evaluation_date<current_round.evaluation_date) THEN
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;$old627$,$new627$  baseline_evaluation_day:=coalesce(current_round.evaluation_date,
    to_char((current_round.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ AT TIME ZONE 'UTC','YYYY-MM-DD'));
  IF current_round.configuration_doc->>'mode'='live' AND current_round.configuration_doc->>'reasoning_capture_policy'='full_generated_thinking_v1' THEN
  IF NOT public.lab_arena__baseline_source_current_v1(current_round) THEN
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;
  ELSE
  -- The promotion lock and target row lock cover this fresh receipt read.
  IF EXISTS(
    SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_rounds correction ON correction.round_id=h.round_id
    WHERE h.old_round_doc->>'status'='published'
      AND correction.status NOT IN('published','cancelled')
      AND correction.arena_network_name=current_round.arena_network_name
      AND correction.arena_netuid=current_round.arena_netuid
      AND correction.evaluation_date<current_round.evaluation_date) THEN
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;
  END IF;$new627$),
('public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)','caf315c8c6c4c15d946ba70b00b0a7c4f893f42ada5e77b3af22fe4fdea8575b','17619139dd59bcadcfc170ad677c5caeda47fb329ec833668759b63385f64cbd',2,$old627$    AND prior.round_id <> p_round_id$old627$,$new627$    AND prior.round_id <> p_round_id
    AND prior.evaluation_date < baseline_evaluation_day$new627$),
('public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)','caf315c8c6c4c15d946ba70b00b0a7c4f893f42ada5e77b3af22fe4fdea8575b','17619139dd59bcadcfc170ad677c5caeda47fb329ec833668759b63385f64cbd',3,$old627$  ORDER BY prior.baseline_promoted_at DESC, prior.round_id DESC LIMIT 1;$old627$,$new627$  ORDER BY prior.evaluation_date DESC, prior.baseline_promoted_at DESC, prior.round_id DESC LIMIT 1;$new627$),
('public.lab_arena_freeze_champion_funding(text)','7633073e713c6969e550c1316d8531426337ada442e563f6bdc8d399bc4c3151','ac3168201636f6256b88d7b8bc83234527859a9c09ae2a5c64fcbec662561df7',0,$old627$  v_round public.lab_arena_rounds;$old627$,$new627$  v_round public.lab_arena_rounds;
  baseline_evaluation_day TEXT;$new627$),
('public.lab_arena_freeze_champion_funding(text)','7633073e713c6969e550c1316d8531426337ada442e563f6bdc8d399bc4c3151','ac3168201636f6256b88d7b8bc83234527859a9c09ae2a5c64fcbec662561df7',1,$old627$  -- The promotion lock and target row lock cover this fresh receipt read.
  IF EXISTS(
    SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_rounds correction ON correction.round_id=h.round_id
    WHERE h.old_round_doc->>'status'='published'
      AND correction.status NOT IN('published','cancelled')
      AND correction.arena_network_name=v_round.arena_network_name
      AND correction.arena_netuid=v_round.arena_netuid
      AND correction.evaluation_date<v_round.evaluation_date) THEN
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;$old627$,$new627$  baseline_evaluation_day:=coalesce(v_round.evaluation_date,
    to_char((v_round.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ AT TIME ZONE 'UTC','YYYY-MM-DD'));
  IF v_round.configuration_doc->>'mode'='live' AND v_round.configuration_doc->>'reasoning_capture_policy'='full_generated_thinking_v1' THEN
  IF NOT public.lab_arena__baseline_source_current_v1(v_round) THEN
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;
  ELSE
  -- The promotion lock and target row lock cover this fresh receipt read.
  IF EXISTS(
    SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_rounds correction ON correction.round_id=h.round_id
    WHERE h.old_round_doc->>'status'='published'
      AND correction.status NOT IN('published','cancelled')
      AND correction.arena_network_name=v_round.arena_network_name
      AND correction.arena_netuid=v_round.arena_netuid
      AND correction.evaluation_date<v_round.evaluation_date) THEN
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;
  END IF;$new627$),
('public.lab_arena_freeze_champion_funding(text)','7633073e713c6969e550c1316d8531426337ada442e563f6bdc8d399bc4c3151','ac3168201636f6256b88d7b8bc83234527859a9c09ae2a5c64fcbec662561df7',2,$old627$    AND prior.round_id <> p_round_id$old627$,$new627$    AND prior.round_id <> p_round_id
    AND (v_round.configuration_doc->>'mode' IS DISTINCT FROM 'live'
      OR v_round.configuration_doc->>'reasoning_capture_policy' IS DISTINCT FROM 'full_generated_thinking_v1'
      OR prior.evaluation_date < baseline_evaluation_day)$new627$),
('public.lab_arena_freeze_champion_funding(text)','7633073e713c6969e550c1316d8531426337ada442e563f6bdc8d399bc4c3151','ac3168201636f6256b88d7b8bc83234527859a9c09ae2a5c64fcbec662561df7',3,$old627$  ORDER BY prior.baseline_promoted_at DESC, prior.round_id DESC LIMIT 1;$old627$,$new627$  ORDER BY CASE WHEN v_round.configuration_doc->>'mode'='live'
      AND v_round.configuration_doc->>'reasoning_capture_policy'='full_generated_thinking_v1'
      THEN prior.evaluation_date END DESC, prior.baseline_promoted_at DESC, prior.round_id DESC LIMIT 1;$new627$),
('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','1866128216b9f8ebf4e37ebbca9de4aebf025422085a06c1bb2d5b1e95ffde6e','a5dc7e6f4370dbffa6e390609a584e5d317cdba33445e58b5bf58bd8ec9f1f6c',0,$old627$  v_publication_deadline TEXT;$old627$,$new627$  v_publication_deadline TEXT;
  v_baseline_source_current BOOLEAN;$new627$),
('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','1866128216b9f8ebf4e37ebbca9de4aebf025422085a06c1bb2d5b1e95ffde6e','a5dc7e6f4370dbffa6e390609a584e5d317cdba33445e58b5bf58bd8ec9f1f6c',1,$old627$  -- Only lab_arena_service can call this RPC. The gateway verifies the signed$old627$,$new627$  -- Promotion/correction, then claim control, then round/run.
  PERFORM pg_advisory_xact_lock_shared(hashtextextended('lab_arena.baseline_promotion',0));
  -- Only lab_arena_service can call this RPC. The gateway verifies the signed$new627$),
('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','1866128216b9f8ebf4e37ebbca9de4aebf025422085a06c1bb2d5b1e95ffde6e','a5dc7e6f4370dbffa6e390609a584e5d317cdba33445e58b5bf58bd8ec9f1f6c',2,$old627$  SELECT * INTO v_existing FROM public.lab_arena_runs$old627$,$new627$  v_baseline_source_current:=public.lab_arena__baseline_source_current_v1(v_round);
  SELECT * INTO v_existing FROM public.lab_arena_runs$new627$),
('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','1866128216b9f8ebf4e37ebbca9de4aebf025422085a06c1bb2d5b1e95ffde6e','a5dc7e6f4370dbffa6e390609a584e5d317cdba33445e58b5bf58bd8ec9f1f6c',3,$old627$    IF v_existing.claim_request_hash = p_request_hash THEN$old627$,$new627$    IF v_existing.claim_request_hash = p_request_hash THEN
      IF NOT v_baseline_source_current AND v_existing.kind IN('execute','score') AND EXISTS(
        SELECT 1 FROM public.lab_arena_submissions baseline
        WHERE baseline.submission_id=v_existing.submission_id AND baseline.round_id=p_round_id AND baseline.is_king) THEN
        RETURN jsonb_build_object('status','promotion_pending');
      END IF;$new627$),
('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','1866128216b9f8ebf4e37ebbca9de4aebf025422085a06c1bb2d5b1e95ffde6e','a5dc7e6f4370dbffa6e390609a584e5d317cdba33445e58b5bf58bd8ec9f1f6c',4,$old627$    SELECT runs.run_id FROM public.lab_arena_runs AS runs
  WHERE runs.round_id = p_round_id$old627$,$new627$    SELECT runs.run_id FROM public.lab_arena_runs AS runs
  WHERE runs.round_id = p_round_id
    AND (v_baseline_source_current OR runs.kind NOT IN ('execute','score') OR NOT EXISTS(
      SELECT 1 FROM public.lab_arena_submissions baseline
      WHERE baseline.submission_id=runs.submission_id AND baseline.round_id=p_round_id AND baseline.is_king))$new627$),
('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','1866128216b9f8ebf4e37ebbca9de4aebf025422085a06c1bb2d5b1e95ffde6e','a5dc7e6f4370dbffa6e390609a584e5d317cdba33445e58b5bf58bd8ec9f1f6c',5,$old627$  IF NOT FOUND THEN
    RETURN pg_catalog.jsonb_build_object('status', 'no_pending');$old627$,$new627$  IF NOT FOUND THEN
    IF NOT v_baseline_source_current AND EXISTS(SELECT 1 FROM public.lab_arena_runs pending
      JOIN public.lab_arena_submissions baseline ON baseline.submission_id=pending.submission_id
      WHERE pending.round_id=p_round_id AND pending.status='pending' AND pending.kind IN('execute','score') AND baseline.is_king) THEN
      RETURN jsonb_build_object('status','promotion_pending');
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'no_pending');$new627$),
('public.lab_arena_transition_round(text,text,text,jsonb)','90fd40203320e777844a38c430aa7f12c12ed99e32df9beabebb50bb7a068a17','1972932e35517f1261ec3f1b782b481c83941916412deedb2745827bea03f029',0,$old627$  -- Keep claims and progression in one order: control, then round/run.$old627$,$new627$  -- Publication must serialize with prior promotion and source correction.
  IF p_expected_status='scored' AND p_next_status='published' THEN
    PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
  END IF;
  -- Keep claims and progression in one order: control, then round/run.$new627$),
('public.lab_arena_transition_round(text,text,text,jsonb)','90fd40203320e777844a38c430aa7f12c12ed99e32df9beabebb50bb7a068a17','1972932e35517f1261ec3f1b782b481c83941916412deedb2745827bea03f029',1,$old627$  ELSIF p_expected_status = 'scored' AND p_next_status = 'published' THEN$old627$,$new627$  ELSIF p_expected_status = 'scored' AND p_next_status = 'published' THEN
  IF NOT public.lab_arena__baseline_source_current_v1(v_round) THEN
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;$new627$),
('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)','8740ff3538cdd0800178711ea8cda5af273b9ede0e8e319d4f1a1a66037fd4ab','175b9bf90d9b1aa5436d78dca6f9a724445c9fd689650f76ef92e1d42f3303ed',0,$old627$  BEGIN
  -- lab_arena_infrastructure_retry_cost_consumer_v1$old627$,$new627$  IF EXISTS(SELECT 1 FROM public.lab_arena_runs baseline
    WHERE baseline.run_id=p_run_id AND baseline.kind IN('execute','score')
      AND baseline.submission_id='baseline-'||regexp_replace(baseline.round_id,'^arena-','')) THEN
    PERFORM pg_advisory_xact_lock_shared(hashtextextended('lab_arena.baseline_promotion',0));
  END IF;
  BEGIN
  -- lab_arena_infrastructure_retry_cost_consumer_v1$new627$),
('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)','8740ff3538cdd0800178711ea8cda5af273b9ede0e8e319d4f1a1a66037fd4ab','175b9bf90d9b1aa5436d78dca6f9a724445c9fd689650f76ef92e1d42f3303ed',1,$old627$  -- A held recovered evaluation admits no fresh execution liability.$old627$,$new627$  -- Keep existing receipts available for completion and billing above.
  IF v_is_baseline AND v_run.kind IN('execute','score')
    AND NOT public.lab_arena__baseline_source_current_v1(v_round) THEN
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;
  -- A held recovered evaluation admits no fresh execution liability.$new627$),
('public.lab_arena_mark_dispatched(text,text,text)','0b7b0f601204f13e2a65fd218a1c6286066aa797dd2ccfeaaa24f2954eb436ab','107864d49bcd4bdeaa0885e51b5ad9a735a109bb482acea0e639cde2443b3e3b',0,$old627$  v_dispatch_entry_id BIGINT;$old627$,$new627$  v_dispatch_entry_id BIGINT;
  v_round public.lab_arena_rounds;$new627$),
('public.lab_arena_mark_dispatched(text,text,text)','0b7b0f601204f13e2a65fd218a1c6286066aa797dd2ccfeaaa24f2954eb436ab','107864d49bcd4bdeaa0885e51b5ad9a735a109bb482acea0e639cde2443b3e3b',1,$old627$BEGIN
  BEGIN
$old627$,$new627$BEGIN
  IF EXISTS(SELECT 1 FROM public.lab_arena_runs baseline
    WHERE baseline.run_id=p_run_id AND baseline.kind IN('execute','score')
      AND baseline.submission_id='baseline-'||regexp_replace(baseline.round_id,'^arena-','')) THEN
    PERFORM pg_advisory_xact_lock_shared(hashtextextended('lab_arena.baseline_promotion',0));
  END IF;
  BEGIN
$new627$),
('public.lab_arena_mark_dispatched(text,text,text)','0b7b0f601204f13e2a65fd218a1c6286066aa797dd2ccfeaaa24f2954eb436ab','107864d49bcd4bdeaa0885e51b5ad9a735a109bb482acea0e639cde2443b3e3b',2,$old627$  INSERT INTO public.lab_arena_ledger ($old627$,$new627$  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id=v_run.round_id;
  IF v_run.kind IN('execute','score') AND EXISTS(SELECT 1 FROM public.lab_arena_submissions baseline
    WHERE baseline.submission_id=v_run.submission_id AND baseline.round_id=v_run.round_id AND baseline.is_king)
    AND NOT public.lab_arena__baseline_source_current_v1(v_round) THEN
    -- The locked head is still a reservation: no dispatch can have occurred.
    -- Reuse recovery accounting so the held amount is released exactly once.
    INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
      call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)
    VALUES('recovery',v_head.miner_hotkey,v_head.round_id,v_head.submission_id,p_run_id,v_head.stage,
      p_call_identity,v_head.provider,v_head.operation_id,v_head.funding_source,0,
      jsonb_build_object('reason','baseline_promotion_pending','reserved_microusd',v_head.amount_microusd));
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;
  INSERT INTO public.lab_arena_ledger ($new627$);
DO $patch627$
DECLARE expected RECORD; h RECORD; d TEXT; target OID; metadata JSONB; security JSONB;
BEGIN
 FOR expected IN SELECT DISTINCT signature,preimage,postimage FROM pg_temp.arena627_patches LOOP
  target:=to_regprocedure(expected.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO d,metadata,security FROM pg_proc p WHERE p.oid=target;
  IF security IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,
    CASE WHEN expected.signature LIKE '%latest_promoted_champion%' THEN 's' ELSE 'v' END,
    CASE WHEN expected.signature LIKE '%transition_round%' THEN ARRAY['search_path=pg_catalog, public','statement_timeout=600s']
      ELSE ARRAY['search_path=pg_catalog, public'] END) THEN
    RAISE EXCEPTION 'Baseline freshness627 security differs: %',expected.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=expected.postimage THEN CONTINUE; END IF;
  IF d IS NULL OR encode(extensions.digest(d,'sha256'),'hex')<>expected.preimage THEN
    RAISE EXCEPTION 'Baseline freshness627 preimage differs: %',expected.signature; END IF;
  FOR h IN SELECT * FROM pg_temp.arena627_patches
  WHERE signature=expected.signature ORDER BY ordinal LOOP
    IF length(d)-length(replace(d,h.old_text,''))<>length(h.old_text) THEN
      RAISE EXCEPTION 'Baseline freshness627 anchor differs: %',expected.signature; END IF;
    d:=replace(d,h.old_text,h.new_text);
  END LOOP;
  IF encode(extensions.digest(d,'sha256'),'hex')<>expected.postimage THEN
    RAISE EXCEPTION 'Baseline freshness627 postimage differs: %',expected.signature; END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>expected.postimage THEN
    RAISE EXCEPTION 'Baseline freshness627 readback differs: %',expected.signature; END IF;
 END LOOP;
END;
$patch627$;
-- Keep the existing two-argument reader for older clients. New freezes
-- select within their own prior-day boundary before fetching any source.
DO $bound_reader627$
DECLARE p RECORD;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure('public.lab_arena_latest_promoted_champion_v1(text,integer,text)');
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'c5ef0f481424e226b79cb1722b86b9bd9099bdec76e983d955d85497a0940a8c'
   OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
   OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' OR p.prokind<>'f' OR p.proretset
   OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
   RAISE EXCEPTION 'Baseline freshness627 bounded reader differs'; END IF;
END;
$bound_reader627$;
CREATE OR REPLACE FUNCTION public.lab_arena_latest_promoted_champion_v1(
 p_network_name TEXT,p_netuid INTEGER,p_before_evaluation_date TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $prior627$
 SELECT jsonb_build_object('round_id',r.round_id,'winner_submission_id',s.submission_id,
   'commit',r.promotion_doc->>'commit','reasoning_capture_policy',r.configuration_doc->>'reasoning_capture_policy')
 FROM public.lab_arena_rounds r JOIN public.lab_arena_submissions s
   ON s.submission_id=r.publication_doc#>>'{king_decision,winner_submission_id}'
   AND s.round_id=r.round_id AND s.miner_hotkey=r.publication_doc#>>'{king_decision,king_hotkey}'
   AND s.status='frozen' AND NOT s.is_king
 WHERE r.arena_network_name=p_network_name AND r.arena_netuid=p_netuid
   AND r.configuration_doc->>'mode'='live' AND r.rewards_enabled
   AND r.status='published' AND r.baseline_promoted_at IS NOT NULL
   AND r.publication_doc#>>'{king_decision,outcome}'='crowned'
   AND r.evaluation_date<p_before_evaluation_date
 ORDER BY r.evaluation_date DESC,r.baseline_promoted_at DESC,r.round_id DESC LIMIT 1;
$prior627$;
ALTER FUNCTION public.lab_arena_latest_promoted_champion_v1(TEXT,INTEGER,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_latest_promoted_champion_v1(TEXT,INTEGER,TEXT)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_latest_promoted_champion_v1(TEXT,INTEGER,TEXT) TO lab_arena_service;
DO $restore627$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena627_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
  (SELECT nspacl FROM pg_temp.arena627_schema_acl) THEN RAISE EXCEPTION 'Baseline freshness627 schema ACL differs'; END IF;
END;
$restore627$;
NOTIFY pgrst,'reload schema';
COMMIT;
