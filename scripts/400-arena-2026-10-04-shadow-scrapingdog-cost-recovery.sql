-- Record four observed Scrapingdog charges whose large successful replies could
-- not be stored. Both shadow rounds and their accepted results stay immutable.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $shadow_scrapingdog_cost_recovery$
DECLARE
  v_item RECORD;
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
  v_reservation public.lab_arena_ledger;
  v_dispatch public.lab_arena_ledger;
  v_uncertain public.lab_arena_ledger;
  v_head public.lab_arena_ledger;
  v_terminal JSONB := pg_catalog.jsonb_build_object(
    'status', 502,
    'headers', pg_catalog.jsonb_build_object(
      'content-type', 'application/json', 'content-length', '41'),
    'body_b64', 'eyJlcnJvciI6eyJjb2RlIjoicHJvdmlkZXJfdW5hdmFpbGFibGUifX0=',
    'call_succeeded', FALSE,
    'provider_cost', pg_catalog.jsonb_build_object(
      'basis', 'scrapingdog_legacy_endpoint_map',
      'units', '5', 'unit_name', 'credits',
      'operation', 'scrapingdog.scrape')
  );
  v_history_count INTEGER;
  v_new_entry_id BIGINT;
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_ledger_settlement_uq') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_ledger_nonsettlement_terminal_uq') IS NULL
     OR pg_catalog.to_regprocedure('public.lab_arena__ledger_head(text)') IS NULL THEN
    RAISE EXCEPTION 'shadow_scrapingdog_recovery_prerequisites_missing';
  END IF;

  -- Published round locks precede the ledger reads and inserts, matching the
  -- live reservation lock order. Check complete, reward-disabled shadow runs.
  FOR v_item IN
    SELECT * FROM (VALUES
      ('arena-2026-10-04-costprobe3f7051d', 2),
      ('arena-2026-10-04-unproven3f7051d', 3)
    ) AS expected(round_id, accepted_scores)
    ORDER BY round_id
  LOOP
    SELECT * INTO v_round FROM public.lab_arena_rounds
    WHERE round_id = v_item.round_id FOR UPDATE;
    IF NOT FOUND
       OR v_round.status IS DISTINCT FROM 'published'
       OR v_round.published_at IS NULL
       OR v_round.publication_doc IS NULL
       OR v_round.rewards_enabled IS DISTINCT FROM FALSE
       OR v_round.configuration_doc ->> 'round_id' IS DISTINCT FROM v_item.round_id
       OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'shadow'
       OR v_round.configuration_doc ->> 'rewards_enabled' IS DISTINCT FROM 'false'
       OR v_round.reward_basis_hash IS NOT NULL
       OR v_round.reward_basis_doc IS NOT NULL
       OR v_round.reward_activated_at IS NOT NULL
       OR (SELECT count(*) FROM public.lab_arena_runs AS r
           WHERE r.round_id = v_item.round_id
             AND r.kind = 'score' AND r.status = 'accepted')
          IS DISTINCT FROM v_item.accepted_scores::BIGINT
       OR (SELECT count(*) FROM public.lab_arena_runs AS r
           WHERE r.round_id = v_item.round_id
             AND r.kind = 'execute' AND r.status = 'accepted')
          IS DISTINCT FROM v_item.accepted_scores::BIGINT
       OR (SELECT count(*) FROM public.lab_arena_runs AS r
           WHERE r.round_id = v_item.round_id)
          IS DISTINCT FROM (2 * v_item.accepted_scores)::BIGINT
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_ledger AS l
         WHERE l.round_id = v_item.round_id
           AND l.call_identity IS NOT NULL
           AND l.entry_kind IN ('reservation', 'dispatch')
           AND NOT EXISTS (
             SELECT 1 FROM public.lab_arena_ledger AS later
             WHERE later.call_identity = l.call_identity
               AND later.entry_id > l.entry_id)) THEN
      RAISE EXCEPTION 'shadow_scrapingdog_round_guard_failed: %', v_item.round_id;
    END IF;
  END LOOP;

  -- Check every target before the first insert. A drifted head or receipt
  -- aborts the whole migration; replay accepts only these exact settlements.
  FOR v_item IN
    SELECT * FROM (VALUES
      ('arena-2026-10-04-unproven3f7051d',
       'arena-2026-10-04-unproven3f7051d:baseline-2026-10-04-unproven3f7051d:1:0:score:1',
       'sha256:c50ae556daa81714d581c98bd499b1a3e9439d6f7fb9b207daf4412990f37e65',
       1618548::BIGINT, 1618549::BIGINT, 1618550::BIGINT),
      ('arena-2026-10-04-unproven3f7051d',
       'arena-2026-10-04-unproven3f7051d:baseline-2026-10-04-unproven3f7051d:1:0:score:1',
       'sha256:b13d8eb23e96d3f35f7f8441281f347ad295fd04b714784967f7b0659e03ea3c',
       1618554::BIGINT, 1618555::BIGINT, 1618556::BIGINT),
      ('arena-2026-10-04-costprobe3f7051d',
       'arena-2026-10-04-costprobe3f7051d:baseline-2026-10-04-costprobe3f7051d:1:0:score:1',
       'sha256:a60ec5332268b7332d494235df83925f770cd037e2129a34f0d6195703c6afc8',
       1618824::BIGINT, 1618825::BIGINT, 1618826::BIGINT),
      ('arena-2026-10-04-costprobe3f7051d',
       'arena-2026-10-04-costprobe3f7051d:baseline-2026-10-04-costprobe3f7051d:1:0:score:1',
       'sha256:b3182d81d5cc65a3d17096802af8c2b865f2ed8f3daa72819f8bc68bb13fe592',
       1618830::BIGINT, 1618831::BIGINT, 1618832::BIGINT)
    ) AS expected(round_id, run_id, call_identity, reservation_id, dispatch_id, uncertain_id)
    ORDER BY round_id, uncertain_id
  LOOP
    SELECT * INTO v_run FROM public.lab_arena_runs
    WHERE run_id = v_item.run_id AND round_id = v_item.round_id;
    SELECT * INTO v_reservation FROM public.lab_arena_ledger
    WHERE entry_id = v_item.reservation_id;
    SELECT * INTO v_dispatch FROM public.lab_arena_ledger
    WHERE entry_id = v_item.dispatch_id;
    SELECT * INTO v_uncertain FROM public.lab_arena_ledger
    WHERE entry_id = v_item.uncertain_id;
    v_head := public.lab_arena__ledger_head(v_item.call_identity);
    SELECT count(*) INTO v_history_count FROM public.lab_arena_ledger
    WHERE call_identity = v_item.call_identity;

    IF v_run.run_id IS NULL OR v_run.kind IS DISTINCT FROM 'score'
       OR v_run.status IS DISTINCT FROM 'accepted'
       OR v_reservation.entry_kind IS DISTINCT FROM 'reservation'
       OR v_dispatch.entry_kind IS DISTINCT FROM 'dispatch'
       OR v_uncertain.entry_kind IS DISTINCT FROM 'uncertain'
       OR v_reservation.call_identity IS DISTINCT FROM v_item.call_identity
       OR v_dispatch.call_identity IS DISTINCT FROM v_item.call_identity
       OR v_uncertain.call_identity IS DISTINCT FROM v_item.call_identity
       OR v_reservation.run_id IS DISTINCT FROM v_item.run_id
       OR v_dispatch.run_id IS DISTINCT FROM v_item.run_id
       OR v_uncertain.run_id IS DISTINCT FROM v_item.run_id
       OR v_reservation.round_id IS DISTINCT FROM v_item.round_id
       OR v_dispatch.round_id IS DISTINCT FROM v_item.round_id
       OR v_uncertain.round_id IS DISTINCT FROM v_item.round_id
       OR v_reservation.submission_id IS DISTINCT FROM v_run.submission_id
       OR v_dispatch.submission_id IS DISTINCT FROM v_reservation.submission_id
       OR v_uncertain.submission_id IS DISTINCT FROM v_reservation.submission_id
       OR v_reservation.miner_hotkey IS DISTINCT FROM v_run.miner_hotkey
       OR v_dispatch.miner_hotkey IS DISTINCT FROM v_reservation.miner_hotkey
       OR v_uncertain.miner_hotkey IS DISTINCT FROM v_reservation.miner_hotkey
       OR v_reservation.stage IS DISTINCT FROM v_run.stage
       OR v_dispatch.stage IS DISTINCT FROM v_reservation.stage
       OR v_uncertain.stage IS DISTINCT FROM v_reservation.stage
       OR v_reservation.provider IS DISTINCT FROM 'scrapingdog'
       OR v_dispatch.provider IS DISTINCT FROM 'scrapingdog'
       OR v_uncertain.provider IS DISTINCT FROM 'scrapingdog'
       OR v_reservation.operation_id IS DISTINCT FROM 'scrapingdog.scrape'
       OR v_dispatch.operation_id IS DISTINCT FROM 'scrapingdog.scrape'
       OR v_uncertain.operation_id IS DISTINCT FROM 'scrapingdog.scrape'
       OR v_reservation.funding_source IS DISTINCT FROM 'host'
       OR v_dispatch.funding_source IS DISTINCT FROM 'host'
       OR v_uncertain.funding_source IS DISTINCT FROM 'host'
       OR v_reservation.amount_microusd IS DISTINCT FROM 0
       OR v_dispatch.amount_microusd IS DISTINCT FROM 0
       OR v_uncertain.amount_microusd IS DISTINCT FROM 0
       OR v_uncertain.entry_doc ->> 'reason' IS DISTINCT FROM 'worker_reported'
       OR v_uncertain.entry_doc #>> '{call,reason}' IS DISTINCT FROM 'settle_failure'
       OR v_uncertain.entry_doc #> '{call,call_succeeded}' IS DISTINCT FROM 'true'::JSONB
       OR v_uncertain.entry_doc #>> '{call,failure_stage}' IS DISTINCT FROM 'settlement'
       OR v_uncertain.entry_doc #>> '{call,error_class}' IS DISTINCT FROM 'ArenaStoreError'
       OR v_uncertain.entry_doc #> '{call,known_actual_microusd}' IS DISTINCT FROM '250'::JSONB
       OR v_uncertain.entry_doc #> '{call,provider_status}' IS DISTINCT FROM '200'::JSONB
       OR v_uncertain.entry_doc #> '{call,provider_cost}' IS DISTINCT FROM
          v_terminal -> 'provider_cost'
       OR v_head.entry_id IS NULL
       OR (v_head.entry_kind = 'uncertain'
           AND v_head.entry_id IS DISTINCT FROM v_item.uncertain_id)
       OR v_head.entry_kind NOT IN ('uncertain', 'settlement')
       OR (v_head.entry_kind = 'uncertain' AND v_history_count IS DISTINCT FROM 3)
       OR (v_head.entry_kind = 'settlement' AND v_history_count IS DISTINCT FROM 4) THEN
      RAISE EXCEPTION 'shadow_scrapingdog_call_guard_failed: %', v_item.call_identity;
    END IF;

    IF v_head.entry_kind = 'settlement' AND (
       v_head.call_identity IS DISTINCT FROM v_reservation.call_identity
       OR v_head.run_id IS DISTINCT FROM v_reservation.run_id
       OR v_head.round_id IS DISTINCT FROM v_reservation.round_id
       OR v_head.submission_id IS DISTINCT FROM v_reservation.submission_id
       OR v_head.miner_hotkey IS DISTINCT FROM v_reservation.miner_hotkey
       OR v_head.stage IS DISTINCT FROM v_reservation.stage
       OR v_head.provider IS DISTINCT FROM v_reservation.provider
       OR v_head.operation_id IS DISTINCT FROM v_reservation.operation_id
       OR v_head.funding_source IS DISTINCT FROM v_reservation.funding_source
       OR
       v_head.amount_microusd IS DISTINCT FROM 250
       OR v_head.terminal_response IS DISTINCT FROM v_terminal
       OR v_head.entry_doc ->> 'shadow_scrapingdog_cost_recovery' IS DISTINCT FROM 'true'
       OR v_head.entry_doc ->> 'reconciled_uncertainty_entry_id' IS DISTINCT FROM
          v_item.uncertain_id::TEXT
       OR v_head.entry_doc ->> 'reserved_microusd' IS DISTINCT FROM '0'
       OR v_head.entry_doc ->> 'released_microusd' IS DISTINCT FROM '-250'
       OR v_head.entry_doc ->> 'variance_microusd' IS DISTINCT FROM '250') THEN
      RAISE EXCEPTION 'shadow_scrapingdog_replay_conflict: %', v_item.call_identity;
    END IF;
  END LOOP;

  FOR v_item IN
    SELECT * FROM (VALUES
      ('sha256:c50ae556daa81714d581c98bd499b1a3e9439d6f7fb9b207daf4412990f37e65', 1618550::BIGINT),
      ('sha256:b13d8eb23e96d3f35f7f8441281f347ad295fd04b714784967f7b0659e03ea3c', 1618556::BIGINT),
      ('sha256:a60ec5332268b7332d494235df83925f770cd037e2129a34f0d6195703c6afc8', 1618826::BIGINT),
      ('sha256:b3182d81d5cc65a3d17096802af8c2b865f2ed8f3daa72819f8bc68bb13fe592', 1618832::BIGINT)
    ) AS expected(call_identity, uncertain_id)
  LOOP
    v_head := public.lab_arena__ledger_head(v_item.call_identity);
    IF v_head.entry_kind = 'settlement' THEN
      CONTINUE;
    END IF;
    SELECT * INTO v_reservation FROM public.lab_arena_ledger
    WHERE call_identity = v_item.call_identity AND entry_kind = 'reservation';
    INSERT INTO public.lab_arena_ledger (
      entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
      call_identity, provider, operation_id, funding_source,
      amount_microusd, entry_doc, terminal_response
    ) VALUES (
      'settlement', v_reservation.miner_hotkey, v_reservation.round_id,
      v_reservation.submission_id, v_reservation.run_id, v_reservation.stage,
      v_reservation.call_identity, v_reservation.provider,
      v_reservation.operation_id, v_reservation.funding_source,
      250,
      pg_catalog.jsonb_build_object(
        'reserved_microusd', 0, 'released_microusd', -250,
        'variance_microusd', 250, 'late_reconciliation', TRUE,
        'shadow_scrapingdog_cost_recovery', TRUE,
        'reconciled_uncertainty_entry_id', v_item.uncertain_id,
        'reconciled_uncertainty_reason', 'worker_reported'),
      v_terminal
    ) RETURNING entry_id INTO v_new_entry_id;
    IF v_new_entry_id <= v_item.uncertain_id THEN
      RAISE EXCEPTION 'shadow_scrapingdog_ledger_sequence_behind';
    END IF;
  END LOOP;
END;
$shadow_scrapingdog_cost_recovery$;

COMMIT;
