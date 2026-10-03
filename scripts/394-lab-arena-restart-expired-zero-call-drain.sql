-- A restart guard must continue natural expiry of its captured, zero-ledger
-- score leases. The ordinary round driver stops before expiry under the guard.
-- Keep claims and round progression paused; preserve the original failed
-- attempt and its dormant retry through the existing expiry RPC.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $restart_expired_zero_call_drain$
DECLARE
  v_drain TEXT;
  v_quiescence TEXT;
  v_drain_owner NAME;
  v_quiescence_owner NAME;
  v_drain_security BOOLEAN;
  v_quiescence_security BOOLEAN;
  v_drain_volatility "char";
  v_quiescence_volatility "char";
  v_drain_config TEXT[];
  v_quiescence_config TEXT[];
  v_anchor TEXT;
  v_position INTEGER;
  v_drain_preimage CONSTANT TEXT := 'dbc37aedf894a8e26a7a6eb4fe3a72c6d1d6cdb1a71ee8557183670320494b9a';
  v_quiescence_preimage CONSTANT TEXT := 'd33ef97bf5fd0a95b61cd807d4e1102449024a3e0f9b02a044aea0740b5c1b46';
  v_drain_postimage CONSTANT TEXT := '183fbb7525dd7561451bb386960ef9e602efd9c205dce754683c9e60ead2ee47';
  v_quiescence_postimage CONSTANT TEXT := 'c3a87110def2cb47833ec4bc353c5c6a965c655f0f03c490019fd34d5de4dcfb';
  v_drain_anchors TEXT[] := ARRAY[
    'v_reported INTEGER := 0;',
    '        ELSE ''lost''',
    'COUNT(*) FILTER (WHERE outcome = ''reported'')::INTEGER,',
    'INTO v_accepted, v_reported, v_leased, v_lost',
    '|| v_accepted::TEXT || '':'' || v_reported::TEXT || '':''',
    '''reported_terminal_receipt_count'', v_reported,',
    'v_captured = v_accepted + v_reported'
  ];
  v_quiescence_anchors TEXT[] := ARRAY[
    '  v_drain JSONB;',
    '  v_drain := public.lab_arena__restart_drain_state_v1();'
  ];
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.prosecdef, p.provolatile, p.proconfig
  INTO v_drain, v_drain_owner, v_drain_security,
       v_drain_volatility, v_drain_config
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE n.nspname='public' AND p.proname='lab_arena__restart_drain_state_v1'
    AND p.pronargs=0;
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.prosecdef, p.provolatile, p.proconfig
  INTO v_quiescence, v_quiescence_owner, v_quiescence_security,
       v_quiescence_volatility, v_quiescence_config
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE n.nspname='public' AND p.proname='lab_arena_restart_quiescence_v1'
    AND p.pronargs=3;
  IF v_drain IS NULL OR v_quiescence IS NULL
     OR v_drain_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_quiescence_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_drain_security IS DISTINCT FROM TRUE
     OR v_quiescence_security IS DISTINCT FROM TRUE
     OR v_drain_volatility IS DISTINCT FROM 's'
     OR v_quiescence_volatility IS DISTINCT FROM 'v'
     OR v_drain_config IS NULL OR v_quiescence_config IS NULL
     OR NOT ('search_path=pg_catalog, public'=ANY(v_drain_config))
     OR NOT ('search_path=pg_catalog, public'=ANY(v_quiescence_config)) THEN
    RAISE EXCEPTION 'Arena restart expiry function security shape differs'
      USING ERRCODE='55000';
  END IF;

  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')=v_drain_postimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')=v_quiescence_postimage THEN
    IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_postimage
       OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_postimage THEN
      RAISE EXCEPTION 'Arena restart expiry replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_preimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_preimage THEN
    RAISE EXCEPTION 'Arena restart expiry preimage differs' USING ERRCODE='55000';
  END IF;
  FOREACH v_anchor IN ARRAY v_drain_anchors LOOP
    v_position := pg_catalog.strpos(v_drain,v_anchor);
    IF v_position=0 OR pg_catalog.strpos(
        pg_catalog.substr(v_drain,v_position+pg_catalog.length(v_anchor)),v_anchor)>0 THEN
      RAISE EXCEPTION 'Arena restart drain preimage differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
  FOREACH v_anchor IN ARRAY v_quiescence_anchors LOOP
    v_position := pg_catalog.strpos(v_quiescence,v_anchor);
    IF v_position=0 OR pg_catalog.strpos(
        pg_catalog.substr(v_quiescence,v_position+pg_catalog.length(v_anchor)),v_anchor)>0 THEN
      RAISE EXCEPTION 'Arena restart quiescence preimage differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
  IF pg_catalog.strpos(v_drain,'WHEN runs.status = ''failed''')=0
     OR pg_catalog.strpos(v_quiescence,'lab_arena_restart_guard_owner_or_generation_differs')=0 THEN
    RAISE EXCEPTION 'Arena restart receipt or owner check differs'
      USING ERRCODE='55000';
  END IF;

  v_drain := pg_catalog.replace(v_drain,
    'v_reported INTEGER := 0;',
    'v_reported INTEGER := 0;' || E'\n  ' || 'v_expired INTEGER := 0;');
  v_drain := pg_catalog.replace(v_drain,
    '        ELSE ''lost''',
    $expired_receipt$        -- lab_arena_captured_expired_zero_call_receipt_v1
        WHEN runs.status = 'failed'
          AND runs.terminal_cause = 'lease_expired'
          AND runs.kind = 'score'
          AND runs.result_doc IS NULL AND runs.output_ref IS NULL
          AND runs.lease_expires_at <= pg_catalog.clock_timestamp()
          AND runs.terminal_doc = pg_catalog.jsonb_build_object(
            'expired_at', runs.terminal_doc ->> 'expired_at')
          AND (runs.terminal_doc ->> 'expired_at')::TIMESTAMPTZ
            >= runs.lease_expires_at
          AND (runs.terminal_doc ->> 'expired_at')::TIMESTAMPTZ
            <= pg_catalog.clock_timestamp()
          AND NOT EXISTS (
            SELECT 1 FROM public.lab_arena_ledger AS expired_cost
            WHERE expired_cost.run_id = runs.run_id
          ) THEN 'expired'
        ELSE 'lost'$expired_receipt$);
  v_drain := pg_catalog.replace(v_drain,
    'COUNT(*) FILTER (WHERE outcome = ''reported'')::INTEGER,',
    'COUNT(*) FILTER (WHERE outcome = ''reported'')::INTEGER,' || E'\n    '
      || 'COUNT(*) FILTER (WHERE outcome = ''expired'')::INTEGER,');
  v_drain := pg_catalog.replace(v_drain,
    'INTO v_accepted, v_reported, v_leased, v_lost',
    'INTO v_accepted, v_reported, v_expired, v_leased, v_lost');
  v_drain := pg_catalog.replace(v_drain,
    $old_commit$|| v_accepted::TEXT || ':' || v_reported::TEXT || ':'$old_commit$,
    $new_commit$|| v_accepted::TEXT || ':' || v_reported::TEXT || ':'
      || v_expired::TEXT || ':'$new_commit$);
  v_drain := pg_catalog.replace(v_drain,
    '''reported_terminal_receipt_count'', v_reported,',
    '''reported_terminal_receipt_count'', v_reported,' || E'\n    '
      || '''expired_receipt_count'', v_expired,');
  v_drain := pg_catalog.replace(v_drain,
    'v_captured = v_accepted + v_reported',
    'v_captured = v_accepted + v_reported + v_expired');

  v_quiescence := pg_catalog.replace(v_quiescence,
    '  v_drain JSONB;',
    '  v_drain JSONB;' || E'\n  ' || 'v_round_id TEXT;');
  v_quiescence := pg_catalog.replace(v_quiescence,
    '  v_drain := public.lab_arena__restart_drain_state_v1();',
    $guarded_expiry$  -- lab_arena_guard_owned_natural_expiry_v1
  -- The owner check and claim-control advisory lock above serialize with
  -- claims. A live guard permits only overdue, captured, zero-ledger score
  -- leases to expire. Existing expiry writes the durable failed attempt and
  -- dormant retry, while the claim gate prevents any new work from starting.
  IF v_control.restart_phase = 'draining'
     AND v_control.guard_expires_at > pg_catalog.clock_timestamp() THEN
    FOR v_round_id IN
      SELECT DISTINCT runs.round_id
      FROM pg_catalog.jsonb_array_elements(v_control.captured_leases) AS item
      JOIN public.lab_arena_runs AS runs ON runs.run_id=item->>'run_id'
      WHERE runs.lease_generation=(item->>'lease_generation')::BIGINT
        AND runs.status='leased'
        AND runs.lease_expires_at<=pg_catalog.clock_timestamp()
    LOOP
      -- Match the ordinary expiry lock order. This blocks concurrent
      -- completion/provider writes and fixes the write set across the check
      -- and the existing expiry function's second clock read.
      PERFORM 1 FROM public.lab_arena_rounds
      WHERE round_id=v_round_id FOR UPDATE;
      IF NOT FOUND THEN
        RAISE EXCEPTION 'Arena restart captured expiry round disappeared'
          USING ERRCODE='55000';
      END IF;
      PERFORM 1 FROM public.lab_arena_runs
      WHERE round_id=v_round_id AND status='leased'
      ORDER BY run_id FOR UPDATE;
      -- Every row that could become overdue before the expiry RPC must be
      -- captured, in the current scoring generation, and without a result.
      -- lab_arena_expire_leases processes every overdue lease in a round.
      -- Prove its entire write set before calling it, or fail closed.
      IF EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS overdue
        JOIN public.lab_arena_rounds AS round
          ON round.round_id=overdue.round_id
        WHERE overdue.round_id=v_round_id AND overdue.status='leased'
          AND (overdue.kind<>'score'
            OR round.status NOT IN ('stage1_scoring','stage2_scoring')
            OR overdue.stage_generation<>round.stage_generation
            OR overdue.result_doc IS NOT NULL OR overdue.output_ref IS NOT NULL
            OR NOT EXISTS (
              SELECT 1
              FROM pg_catalog.jsonb_array_elements(v_control.captured_leases) AS item
              WHERE item->>'run_id'=overdue.run_id
                AND (item->>'lease_generation')::BIGINT=overdue.lease_generation
            ))
      ) THEN
        RAISE EXCEPTION 'Arena restart captured expiry write set differs'
          USING ERRCODE='55000';
      END IF;
      -- Cost-bearing work may still complete and settle. Never apply the
      -- whole-round expiry RPC while any of its leased rows has ledger data.
      IF EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS leased
        WHERE leased.round_id=v_round_id AND leased.status='leased'
          AND EXISTS (
            SELECT 1 FROM public.lab_arena_ledger AS cost
            WHERE cost.run_id=leased.run_id
          )
      ) THEN
        CONTINUE;
      END IF;
      PERFORM public.lab_arena_expire_leases(v_round_id);
    END LOOP;
  END IF;
  v_drain := public.lab_arena__restart_drain_state_v1();$guarded_expiry$);
  EXECUTE v_drain;
  EXECUTE v_quiescence;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__restart_drain_state_v1()'::REGPROCEDURE),
    pg_catalog.pg_get_functiondef(
    'public.lab_arena_restart_quiescence_v1(text,text,bigint)'::REGPROCEDURE)
  INTO v_drain, v_quiescence;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_postimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_postimage THEN
    RAISE EXCEPTION 'Arena restart expiry postimage differs' USING ERRCODE='55000';
  END IF;
END;
$restart_expired_zero_call_drain$;

NOTIFY pgrst, 'reload schema';
COMMIT;
