-- Reconcile two exact Sep16 generic_http_request calls whose retained gateway
-- diagnostics prove an authenticated HTTP 200 completed call without billing.
-- An authenticated GET /api/v2/tools at 2026-09-16T20:33:19Z reported this
-- exact tool as Free per call, at zero credits and zero USD. The provider did
-- not expose a billing row for either identity, so this migration uses only
-- the catalog price contract and never claims billing-ledger evidence.
-- Existing ledger rows stay immutable. Two zero settlements are appended.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_271_prerequisites$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena__ledger_head(text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__successful_call_cost_state(text,text,text)'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_ledger_settlement_uq'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_ledger_nonsettlement_terminal_uq'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply current Lab Arena migrations through 269 before 271';
  END IF;
END;
$lab_arena_271_prerequisites$;

LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;

DO $reconcile_sep16_generic_http_free_costs$
DECLARE
  v_expected RECORD;
  v_run public.lab_arena_runs%ROWTYPE;
  v_head public.lab_arena_ledger%ROWTYPE;
  v_reservation public.lab_arena_ledger%ROWTYPE;
  v_dispatch public.lab_arena_ledger%ROWTYPE;
  v_state JSONB;
  v_exact_active BIGINT;
  v_active_generic BIGINT;
  v_settled BIGINT := 0;
  v_inserted BIGINT;
  v_basis CONSTANT TEXT :=
    'deepline_authenticated_catalog_free_per_call_2026-09-16';
  v_pricing_proof CONSTANT TEXT :=
    'sha256:6be6c06a52fe7c3133d95325ef183a0896f1278129ce6a48fcf2ac0a2b4c6f73';
  v_tool_contract CONSTANT TEXT :=
    'sha256:e1ab8fdbe0b81511c7f2b8e9ab509ff952d640a257f0665a894a879d2e5e7d63';
  v_catalog_response CONSTANT TEXT :=
    'sha256:7b29488b6399ca8a445311a9ad93501727871a7f5f3920f0fbf66b3da3c4bd73';
BEGIN
  IF NOT EXISTS (
       SELECT 1
       FROM public.lab_arena_rounds AS round_row
       JOIN public.lab_arena_submissions AS submission
         ON submission.round_id = round_row.round_id
       WHERE round_row.round_id = 'arena-2026-09-16'
         AND round_row.status = 'cancelled'
         AND round_row.status_generation = 16
         AND round_row.stage_generation = 12
         AND round_row.cancel_reason = 'operator'
         AND round_row.configuration_doc
               ->> 'sourcing_cost_eligibility_policy'
             = 'successful_calls_v1'
         AND submission.submission_id = 'baseline-2026-09-16'
         AND submission.miner_hotkey =
             '5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9'
         AND submission.status = 'frozen'
         AND submission.is_king
     ) THEN
    RAISE EXCEPTION 'arena_20260916_generic_http_round_or_submission_mismatch';
  END IF;

  FOR v_expected IN
    SELECT * FROM (VALUES
      (
        445520::BIGINT,
        445500::BIGINT,
        445501::BIGINT,
        'arena-2026-09-16:baseline-2026-09-16:1:0:rerun269:1',
        'arena-2026-09-16:baseline-2026-09-16:1:0:rerun269',
        0::SMALLINT,
        1::SMALLINT,
        'accepted',
        'accepted',
        'sha256:04e80ee1554b329921dc21e603e96517e1e1cb44703efe503facefa54d73780e',
        'ctx-tool-04e80ee1554b329921dc21e603e96517',
        'iad1::9hjtk-1789588271601-58dfa6a48e84',
        168069::BIGINT
      ),
      (
        447528::BIGINT,
        447505::BIGINT,
        447506::BIGINT,
        'arena-2026-09-16:baseline-2026-09-16:1:5:rerun269:2',
        'arena-2026-09-16:baseline-2026-09-16:1:5:rerun269',
        5::SMALLINT,
        2::SMALLINT,
        'failed',
        'provider_error',
        'sha256:669b75511203146a5ff6316fcc1837d474185bd974fc87981279030c2e299efd',
        'ctx-tool-669b75511203146a5ff6316fcc1837d4',
        'iad1::5wn7l-1789590248594-ac5d12110a16',
        25723::BIGINT
      )
    ) AS expected(
      uncertain_entry_id, reservation_entry_id, dispatch_entry_id,
      run_id, assignment_id, icp_position, attempt, run_status,
      terminal_cause, call_identity, caller_request_id, provider_job_id,
      body_bytes
    )
  LOOP
    v_head := public.lab_arena__ledger_head(v_expected.call_identity);
    IF v_head.entry_kind = 'settlement'
       AND v_head.amount_microusd = 0
       AND v_head.entry_doc
             ->> 'sep16_generic_http_catalog_reconciliation' = 'true'
       AND (v_head.entry_doc
              ->> 'reconciled_uncertainty_entry_id')::BIGINT
             = v_expected.uncertain_entry_id
       AND v_head.entry_doc ->> 'deepline_request_id'
             = v_expected.caller_request_id
       AND v_head.entry_doc ->> 'deepline_job_id'
             = v_expected.provider_job_id
       AND v_head.entry_doc #>> '{catalog_evidence,pricing_proof_sha256}'
             = v_pricing_proof
       AND v_head.entry_doc #>> '{catalog_evidence,tool_contract_sha256}'
             = v_tool_contract
       AND v_head.entry_doc #>> '{catalog_evidence,response_sha256}'
             = v_catalog_response
       AND v_head.terminal_response ->> 'call_succeeded' = 'true'
       AND v_head.terminal_response #>> '{provider_cost,basis}' = v_basis
       AND v_head.terminal_response #>> '{provider_cost,units}' = '0'
       AND v_head.terminal_response #>> '{provider_cost,unit_name}' = 'credits'
       AND v_head.terminal_response #>> '{provider_cost,operation}'
             = 'generic_http_request'
       AND v_head.terminal_response #>> '{provider_cost,request_id}'
             = v_expected.provider_job_id THEN
      v_settled := v_settled + 1;
    ELSIF v_head.entry_kind = 'settlement' THEN
      RAISE EXCEPTION
        'arena_20260916_generic_http_settlement_conflict:%',
        v_expected.uncertain_entry_id;
    END IF;
  END LOOP;

  WITH heads AS (
    SELECT DISTINCT ON (ledger.call_identity) ledger.*
    FROM public.lab_arena_ledger AS ledger
    WHERE ledger.round_id = 'arena-2026-09-16'
      AND ledger.submission_id = 'baseline-2026-09-16'
      AND ledger.run_id LIKE
          'arena-2026-09-16:baseline-2026-09-16:1:%:rerun269:%'
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  )
  SELECT
    pg_catalog.count(*),
    pg_catalog.count(*) FILTER (
      WHERE (entry_id, call_identity) IN (
        (
          445520,
          'sha256:04e80ee1554b329921dc21e603e96517e1e1cb44703efe503facefa54d73780e'
        ),
        (
          447528,
          'sha256:669b75511203146a5ff6316fcc1837d474185bd974fc87981279030c2e299efd'
        )
      )
    )
  INTO v_active_generic, v_exact_active
  FROM heads
  WHERE entry_kind = 'uncertain'
    AND entry_doc ->> 'reason' = 'worker_reported'
    AND entry_doc #>> '{call,reason}' = 'missing_provider_cost'
    AND entry_doc #>> '{call,deepline_operation}' = 'generic_http_request'
    AND pg_catalog.jsonb_typeof(
          entry_doc #> '{call,call_succeeded}'
        ) = 'boolean'
    AND (entry_doc #>> '{call,call_succeeded}')::BOOLEAN;

  v_state := public.lab_arena__successful_call_cost_state(
    'baseline-2026-09-16', 'execute', NULL
  );
  IF v_settled = 2 THEN
    IF v_active_generic IS DISTINCT FROM 0
       OR (v_state ->> 'success_unresolved_calls')::BIGINT
            IS DISTINCT FROM 0 THEN
      RAISE EXCEPTION
        'arena_20260916_generic_http_replay_has_new_unresolved_calls';
    END IF;
    RETURN;
  ELSIF v_settled IS DISTINCT FROM 0 THEN
    RAISE EXCEPTION 'arena_20260916_generic_http_partial_settlement';
  END IF;
  IF v_active_generic IS DISTINCT FROM 2
     OR v_exact_active IS DISTINCT FROM 2
     OR (v_state ->> 'success_unresolved_calls')::BIGINT
          IS DISTINCT FROM 2 THEN
    RAISE EXCEPTION
      'arena_20260916_generic_http_unresolved_set_mismatch';
  END IF;

  FOR v_expected IN
    SELECT * FROM (VALUES
      (
        445520::BIGINT, 445500::BIGINT, 445501::BIGINT,
        'arena-2026-09-16:baseline-2026-09-16:1:0:rerun269:1',
        'arena-2026-09-16:baseline-2026-09-16:1:0:rerun269',
        0::SMALLINT, 1::SMALLINT, 'accepted', 'accepted',
        'sha256:04e80ee1554b329921dc21e603e96517e1e1cb44703efe503facefa54d73780e',
        'ctx-tool-04e80ee1554b329921dc21e603e96517',
        'iad1::9hjtk-1789588271601-58dfa6a48e84', 168069::BIGINT
      ),
      (
        447528::BIGINT, 447505::BIGINT, 447506::BIGINT,
        'arena-2026-09-16:baseline-2026-09-16:1:5:rerun269:2',
        'arena-2026-09-16:baseline-2026-09-16:1:5:rerun269',
        5::SMALLINT, 2::SMALLINT, 'failed', 'provider_error',
        'sha256:669b75511203146a5ff6316fcc1837d474185bd974fc87981279030c2e299efd',
        'ctx-tool-669b75511203146a5ff6316fcc1837d4',
        'iad1::5wn7l-1789590248594-ac5d12110a16', 25723::BIGINT
      )
    ) AS expected(
      uncertain_entry_id, reservation_entry_id, dispatch_entry_id,
      run_id, assignment_id, icp_position, attempt, run_status,
      terminal_cause, call_identity, caller_request_id, provider_job_id,
      body_bytes
    )
  LOOP
    SELECT * INTO v_run
    FROM public.lab_arena_runs
    WHERE run_id = v_expected.run_id
    FOR SHARE;
    IF v_run.run_id IS NULL
       OR v_run.round_id IS DISTINCT FROM 'arena-2026-09-16'
       OR v_run.submission_id IS DISTINCT FROM 'baseline-2026-09-16'
       OR v_run.miner_hotkey IS DISTINCT FROM
            '5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9'
       OR v_run.assignment_id IS DISTINCT FROM v_expected.assignment_id
       OR v_run.stage IS DISTINCT FROM 1
       OR v_run.icp_position IS DISTINCT FROM v_expected.icp_position
       OR v_run.attempt IS DISTINCT FROM v_expected.attempt
       OR v_run.kind IS DISTINCT FROM 'execute'
       OR v_run.status IS DISTINCT FROM v_expected.run_status
       OR v_run.terminal_cause IS DISTINCT FROM v_expected.terminal_cause THEN
      RAISE EXCEPTION 'arena_20260916_generic_http_run_mismatch:%',
        v_expected.uncertain_entry_id;
    END IF;

    v_head := public.lab_arena__ledger_head(v_expected.call_identity);
    IF v_head.entry_id IS DISTINCT FROM v_expected.uncertain_entry_id
       OR v_head.entry_kind IS DISTINCT FROM 'uncertain'
       OR v_head.miner_hotkey IS DISTINCT FROM v_run.miner_hotkey
       OR v_head.round_id IS DISTINCT FROM v_run.round_id
       OR v_head.submission_id IS DISTINCT FROM v_run.submission_id
       OR v_head.run_id IS DISTINCT FROM v_run.run_id
       OR v_head.stage IS DISTINCT FROM v_run.stage
       OR v_head.provider IS DISTINCT FROM 'deepline'
       OR v_head.operation_id IS DISTINCT FROM 'deepline.execute'
       OR v_head.funding_source IS DISTINCT FROM 'host'
       OR v_head.amount_microusd IS DISTINCT FROM 0
       OR v_head.entry_doc ->> 'reason'
            IS DISTINCT FROM 'worker_reported'
       OR v_head.entry_doc #>> '{call,reason}'
            IS DISTINCT FROM 'missing_provider_cost'
       OR v_head.entry_doc #>> '{call,deepline_request_id}'
            IS DISTINCT FROM v_expected.caller_request_id
       OR v_head.entry_doc #>> '{call,deepline_job_id}'
            IS DISTINCT FROM v_expected.provider_job_id
       OR v_head.entry_doc #>> '{call,deepline_operation}'
            IS DISTINCT FROM 'generic_http_request'
       OR v_head.entry_doc #>> '{call,credential_fingerprint}'
            IS DISTINCT FROM
            'sha256:6337fb7ec65645a89f951013ad29797ac8ef19a0542b666f741e77837e0743ee'
       OR pg_catalog.jsonb_typeof(
            v_head.entry_doc #> '{call,provider_status}'
          ) IS DISTINCT FROM 'number'
       OR v_head.entry_doc #>> '{call,provider_status}'
            IS DISTINCT FROM '200'
       OR v_head.entry_doc #>> '{call,top_level_job_status}'
            IS DISTINCT FROM 'completed'
       OR pg_catalog.jsonb_typeof(
            v_head.entry_doc #> '{call,call_succeeded}'
          ) IS DISTINCT FROM 'boolean'
       OR v_head.entry_doc #>> '{call,call_succeeded}'
            IS DISTINCT FROM 'true'
       OR pg_catalog.jsonb_typeof(
            v_head.entry_doc #> '{call,body_is_mapping}'
          ) IS DISTINCT FROM 'boolean'
       OR v_head.entry_doc #>> '{call,body_is_mapping}'
            IS DISTINCT FROM 'true'
       OR pg_catalog.jsonb_typeof(
            v_head.entry_doc #> '{call,billing_present}'
          ) IS DISTINCT FROM 'boolean'
       OR v_head.entry_doc #>> '{call,billing_present}'
            IS DISTINCT FROM 'false'
       OR pg_catalog.jsonb_typeof(
            v_head.entry_doc #> '{call,usage_present}'
          ) IS DISTINCT FROM 'boolean'
       OR v_head.entry_doc #>> '{call,usage_present}'
            IS DISTINCT FROM 'false'
       OR pg_catalog.jsonb_typeof(
            v_head.entry_doc #> '{call,body_bytes}'
          ) IS DISTINCT FROM 'number'
       OR (v_head.entry_doc #>> '{call,body_bytes}')::BIGINT
            IS DISTINCT FROM v_expected.body_bytes THEN
      RAISE EXCEPTION 'arena_20260916_generic_http_head_mismatch:%',
        v_expected.uncertain_entry_id;
    END IF;

    SELECT * INTO v_reservation
    FROM public.lab_arena_ledger
    WHERE entry_id = v_expected.reservation_entry_id
      AND call_identity = v_expected.call_identity
      AND entry_kind = 'reservation';
    SELECT * INTO v_dispatch
    FROM public.lab_arena_ledger
    WHERE entry_id = v_expected.dispatch_entry_id
      AND call_identity = v_expected.call_identity
      AND entry_kind = 'dispatch';
    IF v_reservation.entry_id IS NULL
       OR v_dispatch.entry_id IS NULL
       OR v_reservation.miner_hotkey IS DISTINCT FROM v_head.miner_hotkey
       OR v_reservation.round_id IS DISTINCT FROM v_head.round_id
       OR v_reservation.submission_id IS DISTINCT FROM v_head.submission_id
       OR v_reservation.run_id IS DISTINCT FROM v_head.run_id
       OR v_reservation.stage IS DISTINCT FROM v_head.stage
       OR v_reservation.provider IS DISTINCT FROM v_head.provider
       OR v_reservation.operation_id IS DISTINCT FROM v_head.operation_id
       OR v_reservation.funding_source IS DISTINCT FROM v_head.funding_source
       OR v_reservation.amount_microusd IS DISTINCT FROM 0
       OR v_reservation.entry_doc ->> 'deepline_request_id'
            IS DISTINCT FROM v_expected.caller_request_id
       OR v_reservation.entry_doc ->> 'tool'
            IS DISTINCT FROM 'generic_http_request'
       OR v_reservation.entry_doc ->> 'credential_fingerprint'
            IS DISTINCT FROM
            'sha256:6337fb7ec65645a89f951013ad29797ac8ef19a0542b666f741e77837e0743ee'
       OR v_dispatch.miner_hotkey IS DISTINCT FROM v_head.miner_hotkey
       OR v_dispatch.round_id IS DISTINCT FROM v_head.round_id
       OR v_dispatch.submission_id IS DISTINCT FROM v_head.submission_id
       OR v_dispatch.run_id IS DISTINCT FROM v_head.run_id
       OR v_dispatch.stage IS DISTINCT FROM v_head.stage
       OR v_dispatch.provider IS DISTINCT FROM v_head.provider
       OR v_dispatch.operation_id IS DISTINCT FROM v_head.operation_id
       OR v_dispatch.funding_source IS DISTINCT FROM v_head.funding_source
       OR v_dispatch.amount_microusd IS DISTINCT FROM 0 THEN
      RAISE EXCEPTION 'arena_20260916_generic_http_chain_mismatch:%',
        v_expected.uncertain_entry_id;
    END IF;
  END LOOP;

  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source,
    amount_microusd, entry_doc, terminal_response
  )
  SELECT
    'settlement', uncertainty.miner_hotkey, uncertainty.round_id,
    uncertainty.submission_id, uncertainty.run_id, uncertainty.stage,
    uncertainty.call_identity, uncertainty.provider,
    uncertainty.operation_id, uncertainty.funding_source, 0,
    pg_catalog.jsonb_build_object(
      'reserved_microusd', reservation.amount_microusd,
      'released_microusd', reservation.amount_microusd,
      'variance_microusd', -reservation.amount_microusd,
      'late_reconciliation', TRUE,
      'sep16_generic_http_catalog_reconciliation', TRUE,
      'reconciled_uncertainty_entry_id', uncertainty.entry_id,
      'reconciled_uncertainty_reason',
        uncertainty.entry_doc ->> 'reason',
      'deepline_request_id',
        reservation.entry_doc ->> 'deepline_request_id',
      'deepline_job_id',
        uncertainty.entry_doc #>> '{call,deepline_job_id}',
      'deepline_operation', 'generic_http_request',
      'credential_fingerprint',
        reservation.entry_doc ->> 'credential_fingerprint',
      'provider_response_payload_retained', FALSE,
      'catalog_evidence', pg_catalog.jsonb_build_object(
        'authenticated_endpoint', '/api/v2/tools',
        'checked_at', '2026-09-16T20:33:19Z',
        'tool_id', 'generic_http_request',
        'provider', 'generic_http',
        'operation', 'generic_http_request',
        'operation_id', 'generic_http_request',
        'operation_aliases',
          pg_catalog.jsonb_build_array('generic_http_request', 'request'),
        'credits_per_unit', 0,
        'usd_per_unit', 0,
        'currency', 'USD',
        'unit', 'call',
        'display_text', 'Free',
        'pricing_proof_sha256', v_pricing_proof,
        'tool_contract_sha256', v_tool_contract,
        'response_sha256', v_catalog_response
      )
    ),
    pg_catalog.jsonb_build_object(
      'status', 200,
      'headers', '{}'::JSONB,
      'body_b64', '',
      'call_succeeded', TRUE,
      'provider_cost', pg_catalog.jsonb_build_object(
        'basis', v_basis,
        'units', '0',
        'unit_name', 'credits',
        'operation', 'generic_http_request',
        'request_id',
          uncertainty.entry_doc #>> '{call,deepline_job_id}'
      )
    )
  FROM public.lab_arena_ledger AS uncertainty
  JOIN public.lab_arena_ledger AS reservation
    ON reservation.call_identity = uncertainty.call_identity
   AND reservation.entry_kind = 'reservation'
  WHERE uncertainty.entry_id IN (445520, 447528)
    AND uncertainty.entry_kind = 'uncertain'
  ORDER BY uncertainty.entry_id;
  GET DIAGNOSTICS v_inserted = ROW_COUNT;
  IF v_inserted IS DISTINCT FROM 2 THEN
    RAISE EXCEPTION 'arena_20260916_generic_http_insert_count_mismatch';
  END IF;

  v_state := public.lab_arena__successful_call_cost_state(
    'baseline-2026-09-16', 'execute', NULL
  );
  IF (v_state ->> 'success_unresolved_calls')::BIGINT IS DISTINCT FROM 0
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_ledger
         WHERE entry_kind = 'settlement'
           AND entry_doc
                 ->> 'sep16_generic_http_catalog_reconciliation' = 'true'
           AND (entry_doc
                  ->> 'reconciled_uncertainty_entry_id')::BIGINT
                 IN (445520, 447528)) IS DISTINCT FROM 2 THEN
    RAISE EXCEPTION 'arena_20260916_generic_http_postcondition_mismatch';
  END IF;
END;
$reconcile_sep16_generic_http_free_costs$;

COMMIT;
