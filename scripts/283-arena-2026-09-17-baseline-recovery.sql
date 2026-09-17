-- Exact same-round baseline recovery for the cancelled Sep17 daily evaluation.
-- Installing this migration is inert. Only the sealed owner RPC can archive
-- the failed baseline evidence and prepare the replacement execution.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $requires_sep17_recovery283$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena__successful_call_cost_state(text,text,text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_open_parallel_execution_v1(text,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep17_recovery282_archive_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep17_recovery282_nonbaseline_ledger_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_ledger') IS NULL THEN
    RAISE EXCEPTION 'apply current Arena schema before recovery283';
  END IF;
END;
$requires_sep17_recovery283$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE IF NOT EXISTS public.lab_arena_sep17_baseline_recovery283_authority (
  round_id TEXT PRIMARY KEY CHECK (round_id = 'arena-2026-09-17'),
  archive_round_id TEXT NOT NULL UNIQUE CHECK (
    archive_round_id = 'arena-2026-09-17-rerun282archive'
  ),
  archive_submission_id TEXT NOT NULL UNIQUE CHECK (
    archive_submission_id = 'baseline-2026-09-17-rerun282archive'
  ),
  execute_namespace TEXT NOT NULL CHECK (execute_namespace = 'rerun283'),
  terminal_source_ref TEXT NOT NULL CHECK (
    terminal_source_ref =
      'arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery282.tar.gz'
  ),
  terminal_source_size_bytes BIGINT NOT NULL CHECK (
    terminal_source_size_bytes = 580050
  ),
  terminal_source_sha256 TEXT NOT NULL CHECK (
    terminal_source_sha256 =
      'cfa4a9bf8093dd230e8643bec5508b675e6f4b0ded75ac3d0f4c2a23616b6e41'
  ),
  terminal_source_commit TEXT NOT NULL CHECK (
    terminal_source_commit = '5a89cee202416419ade064e85ff40499f51d34b0'
  ),
  recovery_source_ref TEXT NOT NULL CHECK (
    recovery_source_ref =
      'arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery283.tar.gz'
  ),
  recovery_source_size_bytes BIGINT NOT NULL CHECK (
    recovery_source_size_bytes = 581173
  ),
  recovery_source_sha256 TEXT NOT NULL CHECK (
    recovery_source_sha256 = '7df255f5db5267b13a65c9c3476cb47fd84837e00ce503bd1f588585ee8a37ae'
  ),
  recovery_source_commit TEXT NOT NULL CHECK (
    recovery_source_commit = 'e40484afde214a417128d1eadacaff716755784f'
  ),
  bank_sha256 TEXT NOT NULL CHECK (
    bank_sha256 =
      '7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871'
  ),
  terminal_round_hash TEXT NOT NULL CHECK (
    terminal_round_hash =
      'sha256:c6aeccb8904a305893b17615bd7d25216e6e3944142be6e9c9721b00d470e9f9'
  ),
  terminal_baseline_submission_hash TEXT NOT NULL CHECK (
    terminal_baseline_submission_hash =
      'sha256:9a7eac3fcc014727f766e89e5895ae402efc02787a72ce56877daeb4e51b9e59'
  ),
  terminal_baseline_runs_hash TEXT NOT NULL CHECK (
    terminal_baseline_runs_hash =
      'sha256:fe47691b39811d8cf1fb5c000e8a142619cd8b5d9f6ec7f47ba8afd4fae67f24'
  ),
  terminal_baseline_ledger_hash TEXT NOT NULL CHECK (
    terminal_baseline_ledger_hash =
      'sha256:38804211241b6ea168fb253e432cfc233905b84930e2147676d2dc7f9b1fb4d5'
  ),
  terminal_nonbaseline_ledger_hash TEXT NOT NULL CHECK (
    terminal_nonbaseline_ledger_hash =
      'sha256:8bd6df08e53b8fe17e846cc5ccb89535a40e42dcc89d87b57825d4f16e67cea2'
  ),
  terminal_submissions_hash TEXT NOT NULL CHECK (
    terminal_submissions_hash =
      'sha256:bab95364f338757f4664b7d60aed1e9ef6bf33229f489a903bed63e03d0fd43e'
  ),
  terminal_nonbaseline_submissions_hash TEXT NOT NULL CHECK (
    terminal_nonbaseline_submissions_hash =
      'sha256:7c0d8883f0984000ea0fa2a30726fb489f959b3ce484b8dd5d40d79596684be6'
  ),
  terminal_baseline_run_count BIGINT NOT NULL CHECK (
    terminal_baseline_run_count = 24
  ),
  terminal_baseline_ledger_count BIGINT NOT NULL CHECK (
    terminal_baseline_ledger_count = 5606
  ),
  terminal_nonbaseline_ledger_count BIGINT NOT NULL CHECK (
    terminal_nonbaseline_ledger_count = 15
  ),
  terminal_nonbaseline_ledger_max_entry_id BIGINT NOT NULL CHECK (
    terminal_nonbaseline_ledger_max_entry_id = 453599
  ),
  terminal_baseline_ledger_max_entry_id BIGINT NOT NULL CHECK (
    terminal_baseline_ledger_max_entry_id = 477810
  ),
  terminal_baseline_settled_microusd BIGINT NOT NULL CHECK (
    terminal_baseline_settled_microusd = 9418557
  ),
  terminal_baseline_uncertain_microusd BIGINT NOT NULL CHECK (
    terminal_baseline_uncertain_microusd = 12221290
  ),
  terminal_submission_count BIGINT NOT NULL CHECK (terminal_submission_count = 9),
  terminal_nonbaseline_submission_count BIGINT NOT NULL CHECK (
    terminal_nonbaseline_submission_count = 8
  ),
  terminal_execute_cost_doc JSONB NOT NULL CHECK (
    terminal_execute_cost_doc =
      $execute_cost${"call_count":1868,"inflight_calls":0,"refused_calls":0,"reserved_or_uncertain_microusd":11967325,"settled_microusd":9418557,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":1809,"successful_microusd":9415615,"uncertain_calls":41}$execute_cost$::JSONB
  ),
  terminal_score_cost_doc JSONB NOT NULL CHECK (
    terminal_score_cost_doc =
      $score_cost${"call_count":0,"inflight_calls":0,"refused_calls":0,"reserved_or_uncertain_microusd":0,"settled_microusd":0,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":0,"successful_microusd":0,"uncertain_calls":0}$score_cost$::JSONB
  ),
  terminal_status_generation BIGINT NOT NULL CHECK (
    terminal_status_generation = 7
  ),
  terminal_stage_generation BIGINT NOT NULL CHECK (
    terminal_stage_generation = 6
  ),
  terminal_cancel_reason TEXT NOT NULL CHECK (
    terminal_cancel_reason = 'operator'
  ),
  old_schedule JSONB NOT NULL CHECK (
    old_schedule = $old_schedule${"benchmark_deadline":"2026-09-17T10:00:00Z","final_scoring_close":"2026-09-17T20:20:00Z","publication_deadline":"2026-09-17T20:50:00Z","stage_1_close":"2026-09-17T16:00:00Z","stage_1_scoring_close":"2026-09-17T18:00:00Z","stage_1_start":"2026-09-17T10:00:01Z","stage_2_close":"2026-09-17T18:20:00Z","stage_2_start":"2026-09-17T18:00:01Z","submission_cutoff":"2026-09-17T00:00:00Z","submission_open":"2026-09-16T00:00:00Z"}$old_schedule$::JSONB
  ),
  forward_schedule JSONB NOT NULL CHECK (
    forward_schedule = $new_schedule${"benchmark_deadline":"2026-09-17T14:00:00Z","final_scoring_close":"2026-09-18T00:20:00Z","publication_deadline":"2026-09-18T00:50:00Z","stage_1_close":"2026-09-17T20:00:00Z","stage_1_scoring_close":"2026-09-17T22:00:00Z","stage_1_start":"2026-09-17T14:00:01Z","stage_2_close":"2026-09-17T22:20:00Z","stage_2_start":"2026-09-17T22:00:01Z","submission_cutoff":"2026-09-17T00:00:00Z","submission_open":"2026-09-16T00:00:00Z"}$new_schedule$::JSONB
  ),
  authorized_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.lab_arena_sep17_baseline_recovery283_authority
  OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_sep17_baseline_recovery283_authority
  ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_sep17_baseline_recovery283_authority
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE TABLE IF NOT EXISTS public.lab_arena_sep17_baseline_recovery283_audit (
  round_id TEXT PRIMARY KEY CHECK (round_id = 'arena-2026-09-17'),
  archive_round_id TEXT NOT NULL UNIQUE,
  archive_submission_id TEXT NOT NULL UNIQUE,
  execute_namespace TEXT NOT NULL,
  terminal_round_doc JSONB NOT NULL,
  terminal_baseline_submission_doc JSONB NOT NULL,
  terminal_round_hash TEXT NOT NULL,
  terminal_baseline_submission_hash TEXT NOT NULL,
  terminal_baseline_runs_hash TEXT NOT NULL,
  terminal_baseline_ledger_hash TEXT NOT NULL,
  terminal_nonbaseline_ledger_hash TEXT NOT NULL,
  terminal_submissions_hash TEXT NOT NULL,
  terminal_nonbaseline_submissions_hash TEXT NOT NULL,
  terminal_baseline_run_count BIGINT NOT NULL,
  terminal_baseline_ledger_count BIGINT NOT NULL,
  terminal_nonbaseline_ledger_count BIGINT NOT NULL,
  terminal_nonbaseline_ledger_max_entry_id BIGINT NOT NULL,
  terminal_baseline_ledger_max_entry_id BIGINT NOT NULL,
  terminal_baseline_settled_microusd BIGINT NOT NULL,
  terminal_baseline_uncertain_microusd BIGINT NOT NULL,
  terminal_submission_count BIGINT NOT NULL,
  terminal_nonbaseline_submission_count BIGINT NOT NULL,
  terminal_execute_cost_doc JSONB NOT NULL,
  terminal_score_cost_doc JSONB NOT NULL,
  terminal_source_ref TEXT NOT NULL,
  terminal_source_size_bytes BIGINT NOT NULL,
  terminal_source_sha256 TEXT NOT NULL,
  terminal_source_commit TEXT NOT NULL,
  recovery_source_ref TEXT NOT NULL,
  recovery_source_size_bytes BIGINT NOT NULL,
  recovery_source_sha256 TEXT NOT NULL,
  recovery_source_commit TEXT NOT NULL,
  bank_sha256 TEXT NOT NULL,
  forward_schedule JSONB NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.lab_arena_sep17_baseline_recovery283_audit
  OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_sep17_baseline_recovery283_audit
  ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_sep17_baseline_recovery283_audit
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

INSERT INTO public.lab_arena_sep17_baseline_recovery283_authority (
  round_id, archive_round_id, archive_submission_id, execute_namespace,
  terminal_source_ref, terminal_source_size_bytes, terminal_source_sha256,
  terminal_source_commit, recovery_source_ref, recovery_source_size_bytes,
  recovery_source_sha256, recovery_source_commit, bank_sha256,
  terminal_round_hash, terminal_baseline_submission_hash,
  terminal_baseline_runs_hash, terminal_baseline_ledger_hash,
  terminal_nonbaseline_ledger_hash,
  terminal_submissions_hash, terminal_nonbaseline_submissions_hash,
  terminal_baseline_run_count, terminal_baseline_ledger_count,
  terminal_nonbaseline_ledger_count,
  terminal_nonbaseline_ledger_max_entry_id,
  terminal_baseline_ledger_max_entry_id,
  terminal_baseline_settled_microusd, terminal_baseline_uncertain_microusd,
  terminal_submission_count, terminal_nonbaseline_submission_count,
  terminal_execute_cost_doc, terminal_score_cost_doc,
  terminal_status_generation, terminal_stage_generation,
  terminal_cancel_reason, old_schedule, forward_schedule
) VALUES (
  'arena-2026-09-17', 'arena-2026-09-17-rerun282archive',
  'baseline-2026-09-17-rerun282archive', 'rerun283',
  'arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery282.tar.gz',
  580050,
  'cfa4a9bf8093dd230e8643bec5508b675e6f4b0ded75ac3d0f4c2a23616b6e41',
  '5a89cee202416419ade064e85ff40499f51d34b0',
  'arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery283.tar.gz',
  581173, '7df255f5db5267b13a65c9c3476cb47fd84837e00ce503bd1f588585ee8a37ae',
  'e40484afde214a417128d1eadacaff716755784f',
  '7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871',
  'sha256:c6aeccb8904a305893b17615bd7d25216e6e3944142be6e9c9721b00d470e9f9',
  'sha256:9a7eac3fcc014727f766e89e5895ae402efc02787a72ce56877daeb4e51b9e59',
  'sha256:fe47691b39811d8cf1fb5c000e8a142619cd8b5d9f6ec7f47ba8afd4fae67f24',
  'sha256:38804211241b6ea168fb253e432cfc233905b84930e2147676d2dc7f9b1fb4d5',
  'sha256:8bd6df08e53b8fe17e846cc5ccb89535a40e42dcc89d87b57825d4f16e67cea2',
  'sha256:bab95364f338757f4664b7d60aed1e9ef6bf33229f489a903bed63e03d0fd43e',
  'sha256:7c0d8883f0984000ea0fa2a30726fb489f959b3ce484b8dd5d40d79596684be6',
  24, 5606,
  15,
  453599,
  477810,
  9418557,
  12221290,
  9, 8,
  $execute_cost${"call_count":1868,"inflight_calls":0,"refused_calls":0,"reserved_or_uncertain_microusd":11967325,"settled_microusd":9418557,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":1809,"successful_microusd":9415615,"uncertain_calls":41}$execute_cost$::JSONB,
  $score_cost${"call_count":0,"inflight_calls":0,"refused_calls":0,"reserved_or_uncertain_microusd":0,"settled_microusd":0,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":0,"successful_microusd":0,"uncertain_calls":0}$score_cost$::JSONB,
  7, 6,
  'operator',
  $old_schedule${"benchmark_deadline":"2026-09-17T10:00:00Z","final_scoring_close":"2026-09-17T20:20:00Z","publication_deadline":"2026-09-17T20:50:00Z","stage_1_close":"2026-09-17T16:00:00Z","stage_1_scoring_close":"2026-09-17T18:00:00Z","stage_1_start":"2026-09-17T10:00:01Z","stage_2_close":"2026-09-17T18:20:00Z","stage_2_start":"2026-09-17T18:00:01Z","submission_cutoff":"2026-09-17T00:00:00Z","submission_open":"2026-09-16T00:00:00Z"}$old_schedule$::JSONB,
  $new_schedule${"benchmark_deadline":"2026-09-17T14:00:00Z","final_scoring_close":"2026-09-18T00:20:00Z","publication_deadline":"2026-09-18T00:50:00Z","stage_1_close":"2026-09-17T20:00:00Z","stage_1_scoring_close":"2026-09-17T22:00:00Z","stage_1_start":"2026-09-17T14:00:01Z","stage_2_close":"2026-09-17T22:20:00Z","stage_2_start":"2026-09-17T22:00:01Z","submission_cutoff":"2026-09-17T00:00:00Z","submission_open":"2026-09-16T00:00:00Z"}$new_schedule$::JSONB
) ON CONFLICT (round_id) DO NOTHING;


-- Keep the already archived original attempt and its owner audit immutable.
CREATE OR REPLACE FUNCTION public.lab_arena_sep17_recovery283_prior_valid_v1()
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep17_recovery283_prior_valid$
  SELECT public.lab_arena_sep17_recovery282_archive_valid_v1() IS TRUE
    AND public.lab_arena_sep17_recovery282_nonbaseline_ledger_valid_v1() IS TRUE
    AND EXISTS (
      SELECT 1 FROM public.lab_arena_sep17_baseline_recovery282_authority AS prior
      WHERE round_id = 'arena-2026-09-17'
        AND 'sha256:' || pg_catalog.encode(extensions.digest(
          pg_catalog.to_jsonb(prior)::TEXT, 'sha256'), 'hex') =
            'sha256:1b1049120703801b3819a35d1406946cb434a736a606e9fa6f36b37bf0b01234'
    )
    AND EXISTS (
      SELECT 1 FROM public.lab_arena_sep17_baseline_recovery282_audit AS prior
      WHERE round_id = 'arena-2026-09-17'
        AND 'sha256:' || pg_catalog.encode(extensions.digest(
          pg_catalog.to_jsonb(prior)::TEXT, 'sha256'), 'hex') =
            'sha256:f32d631b3e3873b32b4f24d92686d4a73ca27136de340ce17aefb944c42925b6'
    );
$sep17_recovery283_prior_valid$;
ALTER FUNCTION public.lab_arena_sep17_recovery283_prior_valid_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep17_recovery283_prior_valid_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep17_recovery283_terminal_valid_v1()
RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep17_recovery283_terminal_valid$
DECLARE
  v_authority public.lab_arena_sep17_baseline_recovery283_authority%ROWTYPE;
  v_round public.lab_arena_rounds%ROWTYPE;
  v_baseline public.lab_arena_submissions%ROWTYPE;
  v_round_hash TEXT;
  v_baseline_hash TEXT;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_nonbaseline_ledger_hash TEXT;
  v_submissions_hash TEXT;
  v_nonbaseline_submissions_hash TEXT;
  v_execute_cost JSONB;
  v_score_cost JSONB;
  v_run_count BIGINT;
  v_ledger_count BIGINT;
  v_nonbaseline_ledger_count BIGINT;
  v_nonbaseline_ledger_max BIGINT;
  v_ledger_max BIGINT;
  v_settled BIGINT;
  v_uncertain BIGINT;
  v_submission_count BIGINT;
  v_nonbaseline_count BIGINT;
BEGIN
  SELECT * INTO v_authority
  FROM public.lab_arena_sep17_baseline_recovery283_authority
  WHERE round_id = 'arena-2026-09-17';
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-17';
  SELECT * INTO v_baseline FROM public.lab_arena_submissions
  WHERE round_id = 'arena-2026-09-17'
    AND submission_id = 'baseline-2026-09-17';
  IF v_authority.round_id IS NULL OR v_round.round_id IS NULL
     OR v_baseline.submission_id IS NULL THEN
    RETURN FALSE;
  END IF;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(v_round)::TEXT, 'sha256'), 'hex') INTO v_round_hash;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(v_baseline)::TEXT, 'sha256'), 'hex') INTO v_baseline_hash;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex'),
    pg_catalog.count(*)
  INTO v_runs_hash, v_run_count
  FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_round.round_id AND submission_id = v_baseline.submission_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex'),
    pg_catalog.count(*), COALESCE(pg_catalog.max(entry_id), 0),
    COALESCE(pg_catalog.sum(amount_microusd) FILTER (
      WHERE entry_kind = 'settlement'), 0),
    COALESCE(pg_catalog.sum(amount_microusd) FILTER (
      WHERE entry_kind = 'uncertain'), 0)
  INTO v_ledger_hash, v_ledger_count, v_ledger_max, v_settled, v_uncertain
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_round.round_id AND submission_id = v_baseline.submission_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex'),
    pg_catalog.count(*), COALESCE(pg_catalog.max(entry_id), 0)
  INTO v_nonbaseline_ledger_hash, v_nonbaseline_ledger_count,
       v_nonbaseline_ledger_max
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_round.round_id AND submission_id <> v_baseline.submission_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'), ''
      ORDER BY submission_id), ''), 'sha256'), 'hex'),
    pg_catalog.count(*)
  INTO v_submissions_hash, v_submission_count
  FROM public.lab_arena_submissions AS row_value
  WHERE round_id = v_round.round_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'), ''
      ORDER BY submission_id), ''), 'sha256'), 'hex'),
    pg_catalog.count(*)
  INTO v_nonbaseline_submissions_hash, v_nonbaseline_count
  FROM public.lab_arena_submissions AS row_value
  WHERE round_id = v_round.round_id AND submission_id <> v_baseline.submission_id;
  v_execute_cost := public.lab_arena__successful_call_cost_state(
    v_baseline.submission_id, 'execute', NULL
  );
  v_score_cost := public.lab_arena__successful_call_cost_state(
    v_baseline.submission_id, 'score', NULL
  );
  RETURN public.lab_arena_sep17_recovery283_prior_valid_v1() IS TRUE
     AND v_round_hash IS NOT DISTINCT FROM v_authority.terminal_round_hash
     AND v_baseline_hash IS NOT DISTINCT FROM
          v_authority.terminal_baseline_submission_hash
     AND v_runs_hash IS NOT DISTINCT FROM v_authority.terminal_baseline_runs_hash
     AND v_ledger_hash IS NOT DISTINCT FROM
          v_authority.terminal_baseline_ledger_hash
     AND v_nonbaseline_ledger_hash IS NOT DISTINCT FROM
          v_authority.terminal_nonbaseline_ledger_hash
     AND v_submissions_hash IS NOT DISTINCT FROM
          v_authority.terminal_submissions_hash
     AND v_nonbaseline_submissions_hash IS NOT DISTINCT FROM
          v_authority.terminal_nonbaseline_submissions_hash
     AND v_run_count IS NOT DISTINCT FROM v_authority.terminal_baseline_run_count
     AND v_ledger_count IS NOT DISTINCT FROM
          v_authority.terminal_baseline_ledger_count
     AND v_nonbaseline_ledger_count IS NOT DISTINCT FROM
          v_authority.terminal_nonbaseline_ledger_count
     AND v_nonbaseline_ledger_max IS NOT DISTINCT FROM
          v_authority.terminal_nonbaseline_ledger_max_entry_id
     AND v_ledger_max IS NOT DISTINCT FROM
          v_authority.terminal_baseline_ledger_max_entry_id
     AND v_settled IS NOT DISTINCT FROM
          v_authority.terminal_baseline_settled_microusd
     AND v_uncertain IS NOT DISTINCT FROM
          v_authority.terminal_baseline_uncertain_microusd
     AND v_submission_count IS NOT DISTINCT FROM
          v_authority.terminal_submission_count
     AND v_nonbaseline_count IS NOT DISTINCT FROM
          v_authority.terminal_nonbaseline_submission_count
     AND v_execute_cost IS NOT DISTINCT FROM v_authority.terminal_execute_cost_doc
     AND v_score_cost IS NOT DISTINCT FROM v_authority.terminal_score_cost_doc
     AND COALESCE((v_execute_cost ->> 'inflight_calls')::BIGINT, -1) = 0
     AND COALESCE((v_execute_cost ->> 'success_unresolved_calls')::BIGINT, -1) = 0
     AND COALESCE((v_score_cost ->> 'inflight_calls')::BIGINT, -1) = 0
     AND COALESCE((v_score_cost ->> 'success_unresolved_calls')::BIGINT, -1) = 0
     AND v_round.status = 'cancelled'
     AND v_round.status_generation = v_authority.terminal_status_generation
     AND v_round.stage_generation = v_authority.terminal_stage_generation
     AND v_round.cancel_reason = v_authority.terminal_cancel_reason
     AND v_round.benchmark_ref = 'arena/arena-2026-09-17/benchmark.json'
     AND v_round.evaluation_date = '2026-09-17'
     AND v_round.icp_set_date = DATE '2026-09-16'
     AND v_round.reward_basis_hash IS NULL
     AND v_round.king_outcome IS NULL
     AND v_round.effective_reward_epoch IS NULL
     AND v_round.participants = pg_catalog.jsonb_build_array(
       pg_catalog.to_jsonb(v_round.participants -> 0)
     )
     AND pg_catalog.jsonb_array_length(v_round.participants) = 1
     AND v_baseline.status = 'frozen' AND v_baseline.is_king
     AND v_baseline.source_ref = v_authority.terminal_source_ref
     AND v_baseline.source_size_bytes = v_authority.terminal_source_size_bytes
     AND v_round.configuration_doc -> 'schedule' = v_authority.old_schedule
     AND v_round.configuration_doc #>> '{call_quotas,openrouter}' = '200'
     AND v_round.configuration_doc #>> '{scoring_call_quotas,openrouter}' = '120'
     AND v_round.configuration_doc ->> 'benchmark_disclosure_policy' =
          'after_scoring_day2_v1'
     AND v_round.configuration_doc ->> 'intent_details_policy' =
          'intent_details_v1'
     AND v_round.configuration_doc ->> 'parallel_twenty_icp_execution' = 'true'
     AND (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER = 2700
     AND (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER = 20
     AND (SELECT pg_catalog.count(DISTINCT assignment_id)
          FROM public.lab_arena_runs
          WHERE round_id = v_round.round_id AND submission_id = v_baseline.submission_id
            AND kind = 'execute' AND assignment_id LIKE '%:rerun282') = 20
     AND NOT EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round.round_id AND submission_id = v_baseline.submission_id
         AND (kind <> 'execute' OR attempt NOT IN (1, 2)
           OR assignment_id <> v_round.round_id || ':' || v_baseline.submission_id || ':' ||
             stage::TEXT || ':' || icp_position::TEXT || ':rerun282'
           OR icp_position NOT BETWEEN 0 AND 19
           OR stage <> CASE WHEN icp_position < 10 THEN 1 ELSE 2 END)
     )
     AND NOT EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round.round_id
         AND (status IN ('pending', 'leased', 'submitted')
           OR submission_id <> v_baseline.submission_id)
     );
END;
$sep17_recovery283_terminal_valid$;
ALTER FUNCTION public.lab_arena_sep17_recovery283_terminal_valid_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep17_recovery283_terminal_valid_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION
public.lab_arena_sep17_recovery283_nonbaseline_ledger_valid_v1()
RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep17_recovery283_nonbaseline_ledger_valid$
DECLARE
  v_authority public.lab_arena_sep17_baseline_recovery283_authority%ROWTYPE;
  v_hash TEXT;
  v_count BIGINT;
BEGIN
  SELECT * INTO v_authority
  FROM public.lab_arena_sep17_baseline_recovery283_authority
  WHERE round_id = 'arena-2026-09-17';
  IF NOT FOUND THEN
    RETURN FALSE;
  END IF;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'), ''
      ORDER BY entry_id), ''), 'sha256'), 'hex'), pg_catalog.count(*)
  INTO v_hash, v_count
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_authority.round_id
    AND submission_id <> 'baseline-2026-09-17'
    AND entry_id <= v_authority.terminal_nonbaseline_ledger_max_entry_id;
  RETURN v_hash IS NOT DISTINCT FROM
           v_authority.terminal_nonbaseline_ledger_hash
     AND v_count IS NOT DISTINCT FROM
           v_authority.terminal_nonbaseline_ledger_count
     AND NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger AS late
       WHERE late.round_id = v_authority.round_id
         AND late.submission_id <> 'baseline-2026-09-17'
         AND late.entry_id > v_authority.terminal_nonbaseline_ledger_max_entry_id
         AND (
           late.entry_kind <> 'settlement'
           OR late.entry_doc ->> 'late_reconciliation' IS DISTINCT FROM 'true'
           OR NOT EXISTS (
             SELECT 1 FROM public.lab_arena_ledger AS original
             WHERE original.entry_id <=
                     v_authority.terminal_nonbaseline_ledger_max_entry_id
               AND original.entry_kind = 'uncertain'
               AND original.round_id = late.round_id
               AND original.submission_id = late.submission_id
               AND original.run_id = late.run_id
               AND original.miner_hotkey = late.miner_hotkey
               AND original.stage = late.stage
               AND original.call_identity = late.call_identity
               AND original.provider = late.provider
               AND original.operation_id = late.operation_id
               AND original.funding_source IS NOT DISTINCT FROM
                     late.funding_source
               AND original.entry_id =
                   (late.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
               AND (
                 (late.provider = 'deepline'
                  AND late.entry_doc ->> 'deepline_delayed_reconciliation' = 'true')
                 OR (late.provider = 'openrouter'
                  AND late.entry_doc ->> 'openrouter_delayed_reconciliation' = 'true')
               )
           )
         )
     );
END;
$sep17_recovery283_nonbaseline_ledger_valid$;
ALTER FUNCTION
public.lab_arena_sep17_recovery283_nonbaseline_ledger_valid_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION
public.lab_arena_sep17_recovery283_nonbaseline_ledger_valid_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep17_recovery283_archive_valid_v1()
RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep17_recovery283_archive_valid$
DECLARE
  v_audit public.lab_arena_sep17_baseline_recovery283_audit%ROWTYPE;
  v_archive_round public.lab_arena_rounds%ROWTYPE;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_submission_doc JSONB;
BEGIN
  SELECT * INTO v_audit
  FROM public.lab_arena_sep17_baseline_recovery283_audit
  WHERE round_id = 'arena-2026-09-17';
  SELECT * INTO v_archive_round FROM public.lab_arena_rounds
  WHERE round_id = v_audit.archive_round_id;
  IF v_audit.round_id IS NULL OR v_archive_round.round_id IS NULL THEN
    RETURN FALSE;
  END IF;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_audit.archive_round_id
    AND submission_id = v_audit.archive_submission_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_ledger_hash FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_audit.archive_round_id
    AND submission_id = v_audit.archive_submission_id
    AND entry_id <= v_audit.terminal_baseline_ledger_max_entry_id;
  SELECT pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id'
  INTO v_submission_doc FROM public.lab_arena_submissions AS row_value
  WHERE round_id = v_audit.archive_round_id
    AND submission_id = v_audit.archive_submission_id;
  RETURN public.lab_arena_sep17_recovery283_prior_valid_v1() IS TRUE
     AND v_runs_hash IS NOT DISTINCT FROM v_audit.terminal_baseline_runs_hash
     AND v_ledger_hash IS NOT DISTINCT FROM v_audit.terminal_baseline_ledger_hash
     AND v_submission_doc IS NOT DISTINCT FROM
          (v_audit.terminal_baseline_submission_doc - 'round_id' - 'submission_id')
     AND v_archive_round.status = 'cancelled'
     AND v_archive_round.rewards_enabled IS FALSE
     AND v_archive_round.cancel_reason =
          'authorized_sep17_recovery282_baseline_archive'
     AND v_archive_round.configuration_doc IS NOT DISTINCT FROM
          ((v_audit.terminal_round_doc -> 'configuration_doc') ||
            pg_catalog.jsonb_build_object(
              'round_id', v_audit.archive_round_id,
              'rewards_enabled', FALSE
            ))
     AND (pg_catalog.to_jsonb(v_archive_round) - ARRAY[
           'round_id', 'status', 'configuration_doc', 'rewards_enabled',
           'cancel_reason'
         ]::TEXT[]) IS NOT DISTINCT FROM
         (v_audit.terminal_round_doc - ARRAY[
           'round_id', 'status', 'configuration_doc', 'rewards_enabled',
           'cancel_reason'
         ]::TEXT[])
     AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
          WHERE round_id = v_audit.archive_round_id
            AND submission_id = v_audit.archive_submission_id)
         = v_audit.terminal_baseline_run_count
     AND (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
          WHERE round_id = v_audit.archive_round_id
            AND submission_id = v_audit.archive_submission_id
            AND entry_id <= v_audit.terminal_baseline_ledger_max_entry_id)
         = v_audit.terminal_baseline_ledger_count
     AND NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger AS late
       WHERE late.round_id = v_audit.archive_round_id
         AND late.submission_id = v_audit.archive_submission_id
         AND late.entry_id > v_audit.terminal_baseline_ledger_max_entry_id
         AND (
           late.entry_kind <> 'settlement'
           OR late.entry_doc ->> 'late_reconciliation' IS DISTINCT FROM 'true'
           OR NOT EXISTS (
             SELECT 1 FROM public.lab_arena_ledger AS original
             WHERE original.entry_id <=
                     v_audit.terminal_baseline_ledger_max_entry_id
               AND original.entry_kind = 'uncertain'
               AND original.round_id = late.round_id
               AND original.submission_id = late.submission_id
               AND original.run_id = late.run_id
               AND original.miner_hotkey = late.miner_hotkey
               AND original.stage = late.stage
               AND original.call_identity = late.call_identity
               AND original.provider = late.provider
               AND original.operation_id = late.operation_id
               AND original.funding_source IS NOT DISTINCT FROM
                     late.funding_source
               AND original.entry_id =
                   (late.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
               AND (
                 (late.provider = 'deepline'
                  AND late.entry_doc ->> 'deepline_delayed_reconciliation' = 'true')
                 OR (late.provider = 'openrouter'
                  AND late.entry_doc ->> 'openrouter_delayed_reconciliation' = 'true')
               )
           )
         )
     );
END;
$sep17_recovery283_archive_valid$;
ALTER FUNCTION public.lab_arena_sep17_recovery283_archive_valid_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep17_recovery283_archive_valid_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DO $verify_sep17_recovery283_terminal$
BEGIN
  IF EXISTS (
       SELECT 1 FROM public.lab_arena_sep17_baseline_recovery283_audit
       WHERE round_id = 'arena-2026-09-17'
     ) THEN
    IF public.lab_arena_sep17_recovery283_archive_valid_v1() IS NOT TRUE
       OR public.lab_arena_sep17_recovery283_nonbaseline_ledger_valid_v1()
            IS NOT TRUE THEN
      RAISE EXCEPTION 'Sep17 recovery283 archive seal differs'
        USING ERRCODE = '55000';
    END IF;
  ELSIF public.lab_arena_sep17_recovery283_terminal_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'Sep17 cancelled baseline terminal seal differs'
      USING ERRCODE = '55000';
  END IF;
END;
$verify_sep17_recovery283_terminal$;

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep17_baseline_recovery283_v1(
  p_source_size_bytes BIGINT,
  p_source_sha256 TEXT,
  p_source_commit TEXT,
  p_bank_sha256 TEXT,
  p_forward_schedule JSONB
)
RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $prepare_sep17_baseline_recovery283$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-17';
  v_baseline_id CONSTANT TEXT := 'baseline-2026-09-17';
  v_authority public.lab_arena_sep17_baseline_recovery283_authority%ROWTYPE;
  v_audit public.lab_arena_sep17_baseline_recovery283_audit%ROWTYPE;
  v_round public.lab_arena_rounds%ROWTYPE;
  v_baseline public.lab_arena_submissions%ROWTYPE;
  v_archive_configuration JSONB;
  v_new_configuration JSONB;
  v_new_participants JSONB;
  v_expected_configuration JSONB;
  v_nonbaseline_submissions_hash TEXT;
  v_position INTEGER;
  v_stage SMALLINT;
  v_assignment TEXT;
  v_count BIGINT;
BEGIN
  PERFORM pg_catalog.set_config('lock_timeout', '5s', TRUE);
  PERFORM pg_catalog.set_config('statement_timeout', '120s', TRUE);
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('arena-2026-09-17-baseline-recovery283', 0)
  );
  SELECT * INTO STRICT v_authority
  FROM public.lab_arena_sep17_baseline_recovery283_authority
  WHERE round_id = v_round_id FOR SHARE;
  IF p_source_size_bytes IS DISTINCT FROM
       v_authority.recovery_source_size_bytes
     OR p_source_sha256 IS DISTINCT FROM v_authority.recovery_source_sha256
     OR p_source_commit IS DISTINCT FROM v_authority.recovery_source_commit
     OR p_bank_sha256 IS DISTINCT FROM v_authority.bank_sha256
     OR p_forward_schedule IS DISTINCT FROM v_authority.forward_schedule THEN
    RAISE EXCEPTION 'Sep17 recovery283 source, bank, or schedule differs'
      USING ERRCODE = '22023';
  END IF;
  LOCK TABLE public.lab_arena_sep17_baseline_recovery283_audit
    IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
  SELECT * INTO v_audit
  FROM public.lab_arena_sep17_baseline_recovery283_audit
  WHERE round_id = v_round_id FOR UPDATE;
  IF FOUND THEN
    v_expected_configuration := pg_catalog.jsonb_set(
      v_audit.terminal_round_doc -> 'configuration_doc',
      '{schedule}', v_authority.forward_schedule, FALSE
    );
    SELECT pg_catalog.jsonb_agg(
      CASE WHEN item ->> 'submission_id' = v_baseline_id THEN
        item || pg_catalog.jsonb_build_object(
          'source_ref', v_authority.recovery_source_ref,
          'source_size_bytes', v_authority.recovery_source_size_bytes
        ) ELSE item END ORDER BY ordinal
    ) INTO v_new_participants
    FROM pg_catalog.jsonb_array_elements(
      v_audit.terminal_round_doc -> 'participants'
    ) WITH ORDINALITY AS entries(item, ordinal);
    SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
      COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
        pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'), ''
        ORDER BY submission_id), ''), 'sha256'), 'hex')
    INTO v_nonbaseline_submissions_hash
    FROM public.lab_arena_submissions AS row_value
    WHERE round_id = v_round_id AND submission_id <> v_baseline_id;
    IF public.lab_arena_sep17_recovery283_archive_valid_v1() IS NOT TRUE
       OR public.lab_arena_sep17_recovery283_nonbaseline_ledger_valid_v1()
            IS NOT TRUE
       OR v_audit.archive_round_id IS DISTINCT FROM v_authority.archive_round_id
       OR v_audit.archive_submission_id IS DISTINCT FROM
            v_authority.archive_submission_id
       OR v_audit.execute_namespace IS DISTINCT FROM v_authority.execute_namespace
       OR v_audit.terminal_round_hash IS DISTINCT FROM
            v_authority.terminal_round_hash
       OR v_audit.terminal_baseline_submission_hash IS DISTINCT FROM
            v_authority.terminal_baseline_submission_hash
       OR v_audit.terminal_baseline_runs_hash IS DISTINCT FROM
            v_authority.terminal_baseline_runs_hash
       OR v_audit.terminal_baseline_ledger_hash IS DISTINCT FROM
            v_authority.terminal_baseline_ledger_hash
       OR v_audit.terminal_nonbaseline_ledger_hash IS DISTINCT FROM
            v_authority.terminal_nonbaseline_ledger_hash
       OR v_audit.terminal_nonbaseline_ledger_count IS DISTINCT FROM
            v_authority.terminal_nonbaseline_ledger_count
       OR v_audit.terminal_nonbaseline_ledger_max_entry_id IS DISTINCT FROM
            v_authority.terminal_nonbaseline_ledger_max_entry_id
       OR v_audit.recovery_source_ref IS DISTINCT FROM
            v_authority.recovery_source_ref
       OR v_audit.recovery_source_size_bytes IS DISTINCT FROM
            v_authority.recovery_source_size_bytes
       OR v_audit.recovery_source_sha256 IS DISTINCT FROM
            v_authority.recovery_source_sha256
       OR v_audit.recovery_source_commit IS DISTINCT FROM
            v_authority.recovery_source_commit
       OR v_audit.bank_sha256 IS DISTINCT FROM v_authority.bank_sha256
       OR v_audit.forward_schedule IS DISTINCT FROM v_authority.forward_schedule
       OR (SELECT configuration_doc FROM public.lab_arena_rounds
           WHERE round_id = v_round_id) IS DISTINCT FROM
            v_expected_configuration
       OR NOT EXISTS (
         SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id = v_round_id
           AND status IN (
             'stage1', 'stage1_closed', 'stage1_scoring', 'stage1_judged',
             'stage1_scored', 'stage2', 'stage2_closed', 'stage2_scoring',
             'stage2_judged', 'scored', 'published'
           )
           AND status_generation >= v_authority.terminal_status_generation + 1
           AND stage_generation >= v_authority.terminal_stage_generation + 1
           AND participants IS NOT DISTINCT FROM v_new_participants
       )
       OR NOT EXISTS (
         SELECT 1 FROM public.lab_arena_submissions
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND source_ref = v_authority.recovery_source_ref
           AND source_size_bytes = v_authority.recovery_source_size_bytes
           AND submission_doc ->> 'source_sha256' =
                 v_authority.recovery_source_sha256
           AND submission_doc ->> 'source_commit' =
                 v_authority.recovery_source_commit
           AND submission_doc ->> 'source_ref' =
                 v_authority.recovery_source_ref
           AND (submission_doc ->> 'source_size_bytes')::BIGINT =
                 v_authority.recovery_source_size_bytes
       )
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id AND submission_id = v_baseline_id
             AND kind = 'execute'
             AND assignment_id LIKE '%:rerun283') <> 20
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'execute' AND (
             assignment_id <> v_round_id || ':' || v_baseline_id || ':' ||
               stage::TEXT || ':' || icp_position::TEXT || ':rerun283'
             OR icp_position NOT BETWEEN 0 AND 19
             OR stage <> CASE WHEN icp_position < 10 THEN 1 ELSE 2 END
           )
       )
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
           WHERE round_id = v_round_id AND submission_id <> v_baseline_id) <>
            v_authority.terminal_nonbaseline_submission_count
       OR v_nonbaseline_submissions_hash IS DISTINCT FROM
            v_authority.terminal_nonbaseline_submissions_hash
       THEN
      RAISE EXCEPTION 'Sep17 recovery283 replay differs' USING ERRCODE = '55000';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing', 'round_id', v_round_id,
      'baseline_execute_assignments', 20,
      'execute_namespace', v_authority.execute_namespace,
      'openrouter_calls_per_icp', 200
    );
  END IF;
  IF pg_catalog.jsonb_typeof(p_forward_schedule) IS DISTINCT FROM 'object'
     OR p_forward_schedule ->> 'submission_open' <>
          v_authority.old_schedule ->> 'submission_open'
     OR p_forward_schedule ->> 'submission_cutoff' <>
          v_authority.old_schedule ->> 'submission_cutoff'
     OR (p_forward_schedule ->> 'benchmark_deadline')::TIMESTAMPTZ <=
          pg_catalog.clock_timestamp()
     OR (p_forward_schedule ->> 'stage_1_start')::TIMESTAMPTZ <=
          (p_forward_schedule ->> 'benchmark_deadline')::TIMESTAMPTZ
     OR (p_forward_schedule ->> 'stage_1_close')::TIMESTAMPTZ <=
          (p_forward_schedule ->> 'stage_1_start')::TIMESTAMPTZ
     OR (p_forward_schedule ->> 'stage_1_scoring_close')::TIMESTAMPTZ <=
          (p_forward_schedule ->> 'stage_1_close')::TIMESTAMPTZ
     OR (p_forward_schedule ->> 'stage_2_start')::TIMESTAMPTZ <=
          (p_forward_schedule ->> 'stage_1_scoring_close')::TIMESTAMPTZ
     OR (p_forward_schedule ->> 'stage_2_close')::TIMESTAMPTZ <=
          (p_forward_schedule ->> 'stage_2_start')::TIMESTAMPTZ
     OR (p_forward_schedule ->> 'final_scoring_close')::TIMESTAMPTZ <=
          (p_forward_schedule ->> 'stage_2_close')::TIMESTAMPTZ
     OR (p_forward_schedule ->> 'publication_deadline')::TIMESTAMPTZ <=
          (p_forward_schedule ->> 'final_scoring_close')::TIMESTAMPTZ THEN
    RAISE EXCEPTION 'Sep17 recovery283 forward schedule invalid'
      USING ERRCODE = '22023';
  END IF;
  IF public.lab_arena_sep17_recovery283_terminal_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'Sep17 cancelled baseline terminal seal differs'
      USING ERRCODE = '55000';
  END IF;
  SELECT * INTO STRICT v_round FROM public.lab_arena_rounds
  WHERE round_id = v_round_id FOR UPDATE;
  SELECT * INTO STRICT v_baseline FROM public.lab_arena_submissions
  WHERE round_id = v_round_id AND submission_id = v_baseline_id FOR UPDATE;
  INSERT INTO public.lab_arena_sep17_baseline_recovery283_audit (
    round_id, archive_round_id, archive_submission_id, execute_namespace,
    terminal_round_doc, terminal_baseline_submission_doc,
    terminal_round_hash, terminal_baseline_submission_hash,
    terminal_baseline_runs_hash, terminal_baseline_ledger_hash,
    terminal_nonbaseline_ledger_hash,
    terminal_submissions_hash, terminal_nonbaseline_submissions_hash,
    terminal_baseline_run_count, terminal_baseline_ledger_count,
    terminal_nonbaseline_ledger_count,
    terminal_nonbaseline_ledger_max_entry_id,
    terminal_baseline_ledger_max_entry_id,
    terminal_baseline_settled_microusd, terminal_baseline_uncertain_microusd,
    terminal_submission_count, terminal_nonbaseline_submission_count,
    terminal_execute_cost_doc, terminal_score_cost_doc,
    terminal_source_ref, terminal_source_size_bytes, terminal_source_sha256,
    terminal_source_commit, recovery_source_ref, recovery_source_size_bytes,
    recovery_source_sha256, recovery_source_commit, bank_sha256,
    forward_schedule
  ) VALUES (
    v_round_id, v_authority.archive_round_id,
    v_authority.archive_submission_id, v_authority.execute_namespace,
    pg_catalog.to_jsonb(v_round), pg_catalog.to_jsonb(v_baseline),
    v_authority.terminal_round_hash,
    v_authority.terminal_baseline_submission_hash,
    v_authority.terminal_baseline_runs_hash,
    v_authority.terminal_baseline_ledger_hash,
    v_authority.terminal_nonbaseline_ledger_hash,
    v_authority.terminal_submissions_hash,
    v_authority.terminal_nonbaseline_submissions_hash,
    v_authority.terminal_baseline_run_count,
    v_authority.terminal_baseline_ledger_count,
    v_authority.terminal_nonbaseline_ledger_count,
    v_authority.terminal_nonbaseline_ledger_max_entry_id,
    v_authority.terminal_baseline_ledger_max_entry_id,
    v_authority.terminal_baseline_settled_microusd,
    v_authority.terminal_baseline_uncertain_microusd,
    v_authority.terminal_submission_count,
    v_authority.terminal_nonbaseline_submission_count,
    v_authority.terminal_execute_cost_doc,
    v_authority.terminal_score_cost_doc,
    v_authority.terminal_source_ref,
    v_authority.terminal_source_size_bytes,
    v_authority.terminal_source_sha256,
    v_authority.terminal_source_commit,
    v_authority.recovery_source_ref,
    v_authority.recovery_source_size_bytes,
    v_authority.recovery_source_sha256,
    v_authority.recovery_source_commit,
    v_authority.bank_sha256, v_authority.forward_schedule
  );
  v_archive_configuration := v_round.configuration_doc ||
    pg_catalog.jsonb_build_object(
      'round_id', v_authority.archive_round_id,
      'rewards_enabled', FALSE
    );
  v_new_configuration := pg_catalog.jsonb_set(
    v_round.configuration_doc,
    '{schedule}', v_authority.forward_schedule, FALSE
  );
  IF (v_new_configuration - 'schedule') IS DISTINCT FROM
       (v_round.configuration_doc - 'schedule')
     OR v_new_configuration #>> '{call_quotas,openrouter}' <> '200' THEN
    RAISE EXCEPTION 'Sep17 recovery283 changed frozen configuration';
  END IF;
  SELECT pg_catalog.jsonb_agg(
    CASE WHEN item ->> 'submission_id' = v_baseline_id THEN
      item || pg_catalog.jsonb_build_object(
        'source_ref', v_authority.recovery_source_ref,
        'source_size_bytes', v_authority.recovery_source_size_bytes
      ) ELSE item END ORDER BY ordinal
  ) INTO v_new_participants
  FROM pg_catalog.jsonb_array_elements(v_round.participants)
    WITH ORDINALITY AS entries(item, ordinal);
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
  INSERT INTO public.lab_arena_rounds (
    round_id, status, status_generation, stage_generation, configuration_doc,
    rewards_enabled, participants, benchmark_ref, evaluation_date,
    stage1_scoring_plan_doc, stage2_scoring_plan_doc, finalists,
    publication_doc, king_outcome, king_hotkey, king_start_epoch,
    effective_reward_epoch, reward_basis_hash, reward_basis_doc,
    signing_key_doc, reward_activated_at, cancel_reason, published_at,
    created_at, updated_at, promotion_required, promotion_doc,
    baseline_promoted_at, icp_set_date, confirmation_bank_ref,
    confirmation_bank_hash, confirmation_cohort, stage3_scoring_plan_doc,
    champion_funding_frozen, champion_submission_id, champion_hotkey,
    champion_fallback_providers
  ) SELECT
    row_value.round_id, row_value.status, row_value.status_generation,
    row_value.stage_generation, row_value.configuration_doc,
    row_value.rewards_enabled, row_value.participants, row_value.benchmark_ref,
    row_value.evaluation_date, row_value.stage1_scoring_plan_doc,
    row_value.stage2_scoring_plan_doc, row_value.finalists,
    row_value.publication_doc, row_value.king_outcome, row_value.king_hotkey,
    row_value.king_start_epoch, row_value.effective_reward_epoch,
    row_value.reward_basis_hash, row_value.reward_basis_doc,
    row_value.signing_key_doc, row_value.reward_activated_at,
    row_value.cancel_reason, row_value.published_at, row_value.created_at,
    row_value.updated_at, row_value.promotion_required,
    row_value.promotion_doc, row_value.baseline_promoted_at,
    row_value.icp_set_date, row_value.confirmation_bank_ref,
    row_value.confirmation_bank_hash, row_value.confirmation_cohort,
    row_value.stage3_scoring_plan_doc, row_value.champion_funding_frozen,
    row_value.champion_submission_id, row_value.champion_hotkey,
    row_value.champion_fallback_providers
  FROM pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_rounds,
    pg_catalog.to_jsonb(v_round) || pg_catalog.jsonb_build_object(
      'round_id', v_authority.archive_round_id,
      'status', 'cancelled',
      'configuration_doc', v_archive_configuration,
      'rewards_enabled', FALSE,
      'cancel_reason', 'authorized_sep17_recovery282_baseline_archive'
    )
  ) AS row_value;
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(v_baseline) || pg_catalog.jsonb_build_object(
      'submission_id', v_authority.archive_submission_id,
      'round_id', v_authority.archive_round_id
    )
  )).*;
  UPDATE public.lab_arena_runs
  SET round_id = v_authority.archive_round_id,
      submission_id = v_authority.archive_submission_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> v_authority.terminal_baseline_run_count THEN
    RAISE EXCEPTION 'Sep17 recovery283 run archive count differs';
  END IF;
  UPDATE public.lab_arena_ledger
  SET round_id = v_authority.archive_round_id,
      submission_id = v_authority.archive_submission_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> v_authority.terminal_baseline_ledger_count THEN
    RAISE EXCEPTION 'Sep17 recovery283 ledger archive count differs';
  END IF;
  UPDATE public.lab_arena_submissions
  SET source_ref = v_authority.recovery_source_ref,
      source_size_bytes = v_authority.recovery_source_size_bytes,
      submission_doc = COALESCE(submission_doc, '{}'::JSONB) ||
        pg_catalog.jsonb_build_object(
          'source_ref', v_authority.recovery_source_ref,
          'source_size_bytes', v_authority.recovery_source_size_bytes,
          'source_sha256', v_authority.recovery_source_sha256,
          'source_commit', v_authority.recovery_source_commit
        )
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep17 recovery283 baseline update differs';
  END IF;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1',
      status_generation = v_authority.terminal_status_generation + 1,
      stage_generation = v_authority.terminal_stage_generation + 1,
      configuration_doc = v_new_configuration,
      participants = v_new_participants,
      stage1_scoring_plan_doc = NULL,
      stage2_scoring_plan_doc = NULL,
      stage3_scoring_plan_doc = NULL,
      finalists = NULL,
      cancel_reason = NULL,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round_id;
  FOR v_position IN 0 .. 19 LOOP
    v_stage := CASE WHEN v_position < 10 THEN 1 ELSE 2 END;
    v_assignment := v_round_id || ':' || v_baseline_id || ':' ||
      v_stage::TEXT || ':' || v_position::TEXT || ':rerun283';
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey,
      stage, icp_position, attempt, kind, status, stage_generation
    ) VALUES (
      v_assignment || ':1', v_assignment, v_round_id, v_baseline_id,
      v_baseline.miner_hotkey, v_stage, v_position, 1, 'execute', 'pending',
      v_authority.terminal_stage_generation + 1
    );
  END LOOP;
  ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'), ''
      ORDER BY submission_id), ''), 'sha256'), 'hex')
  INTO v_nonbaseline_submissions_hash
  FROM public.lab_arena_submissions AS row_value
  WHERE round_id = v_round_id AND submission_id <> v_baseline_id;
  IF public.lab_arena_sep17_recovery283_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep17_recovery283_nonbaseline_ledger_valid_v1()
          IS NOT TRUE
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id = v_round_id AND submission_id <> v_baseline_id) <>
          v_authority.terminal_nonbaseline_submission_count
     OR v_nonbaseline_submissions_hash IS DISTINCT FROM
          v_authority.terminal_nonbaseline_submissions_hash
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'execute' AND status = 'pending'
           AND assignment_id LIKE '%:rerun283') <> 20
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND submission_id = v_baseline_id
         AND kind = 'execute' AND (
           assignment_id <> v_round_id || ':' || v_baseline_id || ':' ||
             stage::TEXT || ':' || icp_position::TEXT || ':rerun283'
           OR icp_position NOT BETWEEN 0 AND 19
           OR stage <> CASE WHEN icp_position < 10 THEN 1 ELSE 2 END
         )
     )
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = v_round_id AND submission_id = v_baseline_id) <> 0 THEN
    RAISE EXCEPTION 'Sep17 recovery283 postcondition differs';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'prepared', 'round_id', v_round_id,
    'baseline_execute_assignments', 20,
    'archived_runs', v_authority.terminal_baseline_run_count,
    'archived_ledger_entries', v_authority.terminal_baseline_ledger_count,
    'preserved_nonparticipant_submissions',
      v_authority.terminal_nonbaseline_submission_count,
    'execute_namespace', v_authority.execute_namespace,
    'openrouter_calls_per_icp', 200
  );
END;
$prepare_sep17_baseline_recovery283$;
ALTER FUNCTION public.lab_arena_prepare_sep17_baseline_recovery283_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep17_baseline_recovery283_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_sep17_baseline_recovery283_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
)
  TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep17_recovery283_archive_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep17_recovery283_archive_guard$
BEGIN
  IF NEW.round_id = 'arena-2026-09-17'
     AND EXISTS (
       SELECT 1 FROM public.lab_arena_sep17_baseline_recovery283_audit
       WHERE round_id = NEW.round_id
     )
     AND public.lab_arena_sep17_recovery283_archive_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'Sep17 recovery283 archive seal differs'
      USING ERRCODE = '55000';
  END IF;
  RETURN NEW;
END;
$sep17_recovery283_archive_guard$;
ALTER FUNCTION public.lab_arena_sep17_recovery283_archive_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep17_recovery283_archive_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep17_recovery283_archive_guard
  ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_sep17_recovery283_archive_guard
BEFORE UPDATE ON public.lab_arena_rounds
FOR EACH ROW EXECUTE FUNCTION
  public.lab_arena_sep17_recovery283_archive_guard_v1();

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
