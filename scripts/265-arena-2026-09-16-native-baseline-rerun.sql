-- Exact same-round baseline rerun authority for arena-2026-09-16.
-- This migration installs a guarded two-step operator procedure. It does not
-- run the procedure or change any competition row when applied.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $requires_sep16_schema$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_parallel_execution_schema_v1()') IS NULL
     OR pg_catalog.to_regprocedure('public.lab_arena_open_scoring_v2(text,smallint,jsonb)') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_judgment_cache') IS NULL THEN
    RAISE EXCEPTION 'apply current Arena migrations before 265';
  END IF;
END;
$requires_sep16_schema$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE IF NOT EXISTS public.lab_arena_sep16_baseline_rerun_audit (
  round_id TEXT PRIMARY KEY CHECK (round_id = 'arena-2026-09-16'),
  archive_round_id TEXT NOT NULL UNIQUE
    CHECK (archive_round_id = 'arena-2026-09-16-archive'),
  old_round_doc JSONB NOT NULL,
  old_baseline_submission_doc JSONB NOT NULL,
  old_baseline_runs_hash TEXT NOT NULL,
  old_baseline_ledger_hash TEXT NOT NULL,
  old_challenger_runs_hash TEXT NOT NULL,
  old_challenger_submissions_hash TEXT NOT NULL,
  old_challenger_ledger_hash TEXT NOT NULL,
  old_challenger_ledger_max_entry_id BIGINT NOT NULL,
  old_round_hash TEXT NOT NULL,
  old_baseline_submission_hash TEXT NOT NULL,
  bank_sha256 TEXT NOT NULL,
  new_source_ref TEXT NOT NULL,
  new_source_sha256 TEXT NOT NULL,
  new_source_commit TEXT NOT NULL,
  old_scorer_image_digest TEXT NOT NULL,
  new_scorer_image_digest TEXT NOT NULL,
  old_actual_microusd BIGINT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  stage1_scoring_opened_at TIMESTAMPTZ,
  stage2_scoring_opened_at TIMESTAMPTZ
);
ALTER TABLE public.lab_arena_sep16_baseline_rerun_audit OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_sep16_baseline_rerun_audit ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_sep16_baseline_rerun_audit
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

-- A later exact committed release migration inserts the reviewed GitHub
-- archive bytes/commit and forward schedule after the write-once S3 object
-- has been checked. Service callers cannot authorize or change that proof.
CREATE TABLE IF NOT EXISTS public.lab_arena_sep16_rerun_release_authority (
  round_id TEXT PRIMARY KEY CHECK (round_id = 'arena-2026-09-16'),
  source_ref TEXT NOT NULL CHECK (
    source_ref = 'arena/arena-2026-09-16/sources/baseline-2026-09-16-native-rerun265.tar.gz'
  ),
  source_size_bytes BIGINT NOT NULL CHECK (source_size_bytes BETWEEN 1 AND 10485760),
  source_sha256 TEXT NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
  source_commit TEXT NOT NULL CHECK (source_commit ~ '^[0-9a-f]{40}$'),
  champion_model_main_commit TEXT NOT NULL
    CHECK (champion_model_main_commit = source_commit),
  champion_model_lab_commit TEXT NOT NULL
    CHECK (champion_model_lab_commit = source_commit),
  bank_sha256 TEXT NOT NULL CHECK (
    bank_sha256 = '42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390'
  ),
  old_round_hash TEXT NOT NULL CHECK (old_round_hash ~ '^sha256:[0-9a-f]{64}$'),
  old_baseline_submission_hash TEXT NOT NULL
    CHECK (old_baseline_submission_hash ~ '^sha256:[0-9a-f]{64}$'),
  old_baseline_runs_hash TEXT NOT NULL
    CHECK (old_baseline_runs_hash ~ '^sha256:[0-9a-f]{64}$'),
  old_baseline_ledger_hash TEXT NOT NULL
    CHECK (old_baseline_ledger_hash ~ '^sha256:[0-9a-f]{64}$'),
  old_challenger_runs_hash TEXT NOT NULL
    CHECK (old_challenger_runs_hash ~ '^sha256:[0-9a-f]{64}$'),
  old_challenger_submissions_hash TEXT NOT NULL
    CHECK (old_challenger_submissions_hash ~ '^sha256:[0-9a-f]{64}$'),
  old_challenger_ledger_hash TEXT NOT NULL
    CHECK (old_challenger_ledger_hash ~ '^sha256:[0-9a-f]{64}$'),
  old_challenger_ledger_max_entry_id BIGINT NOT NULL
    CHECK (old_challenger_ledger_max_entry_id >= 0),
  old_baseline_actual_microusd BIGINT NOT NULL
    CHECK (old_baseline_actual_microusd >= 0),
  old_scorer_image_digest TEXT NOT NULL
    CHECK (old_scorer_image_digest =
      'sha256:ee84f274ba24b07fa204c03535b21aac3c030c72aff0fcf7918d88d3c2c8ddee'),
  new_scorer_image_reference TEXT NOT NULL CHECK (
    new_scorer_image_reference LIKE '%@' || new_scorer_image_digest
  ),
  new_scorer_image_digest TEXT NOT NULL
    CHECK (new_scorer_image_digest ~ '^sha256:[0-9a-f]{64}$'),
  scoring_tree_hash TEXT NOT NULL
    CHECK (scoring_tree_hash = 'f8452314cc8b1529fbfa8b7fc9345143c1b7d731'),
  native_runtime_commit TEXT NOT NULL
    CHECK (native_runtime_commit = '9d97e2ae295f715209b8fdfef2b6ddebdc46622e'),
  verified_parallel_runner_slots SMALLINT NOT NULL
    CHECK (verified_parallel_runner_slots = 10),
  forward_schedule JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(forward_schedule) = 'object'),
  authorized_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.lab_arena_sep16_rerun_release_authority OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_sep16_rerun_release_authority ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_sep16_rerun_release_authority
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

-- The signed activation governs epochs independently of the competition's
-- current status. Expose the signed document's original timestamp, not a
-- later competition override timestamp.
CREATE OR REPLACE VIEW public.lab_arena_reward_basis_v1 AS
  SELECT round_id, effective_reward_epoch, reward_basis_hash, reward_basis_doc,
         signing_key_doc,
         reward_basis_doc ->> 'king_outcome' AS king_outcome,
         NULLIF(reward_basis_doc ->> 'king_hotkey', '') AS king_hotkey,
         (reward_basis_doc ->> 'king_start_epoch')::BIGINT AS king_start_epoch,
         (reward_basis_doc ->> 'published_at')::TIMESTAMPTZ AS published_at,
         arena_network_name, arena_netuid
  FROM public.lab_arena_rounds
  WHERE configuration_doc ->> 'mode' = 'live'
    AND rewards_enabled
    AND reward_activated_at IS NOT NULL
    AND reward_basis_doc IS NOT NULL
    AND signing_key_doc IS NOT NULL
    AND (
      reward_basis_doc ->> 'king_outcome' = 'no_king'
      OR (
        reward_basis_doc ->> 'king_outcome' IN ('crowned', 'defended')
        AND reward_basis_doc ->> 'king_hotkey'
            IS DISTINCT FROM configuration_doc ->> 'baseline_hotkey'
      )
    );
ALTER VIEW public.lab_arena_reward_basis_v1 OWNER TO lab_arena_owner;
REVOKE ALL ON public.lab_arena_reward_basis_v1 FROM PUBLIC;
GRANT SELECT ON public.lab_arena_reward_basis_v1 TO lab_arena_service;
DO $reward_basis_service_acl$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'GRANT SELECT ON public.lab_arena_reward_basis_v1 TO service_role';
  END IF;
END;
$reward_basis_service_acl$;

-- Reuse the exact current integrity scorer validation and cache handling.
-- Only the new baseline score assignment namespace differs. This private
-- clone is called by the exact Sep16 wrapper below, never by the daily driver.
DO $clone_current_scoring$
DECLARE
  v_definition TEXT;
  v_old_name TEXT := 'public.lab_arena_open_scoring_v2(';
  v_new_name TEXT := 'public.lab_arena_open_scoring_sep16_baseline_only_v1(';
  v_old_assignment TEXT := $old$
    v_assignment := p_round_id || ':' || v_scored.submission_id || ':' ||
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score';
$old$;
  v_new_assignment TEXT := $new$
    v_assignment := p_round_id || ':' || v_scored.submission_id || ':' ||
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun265';
$new$;
BEGIN
  v_definition := pg_catalog.pg_get_functiondef(
    'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure
  );
  IF pg_catalog.strpos(v_definition, v_old_name) = 0
     OR pg_catalog.strpos(v_definition, v_old_assignment) = 0
     OR pg_catalog.strpos(v_definition, 'lab_arena_judgment_cache_source_invalid') = 0 THEN
    RAISE EXCEPTION 'current integrity scoring definition differs';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old_name, v_new_name);
  v_definition := pg_catalog.replace(v_definition, v_old_assignment, v_new_assignment);
  EXECUTE v_definition;
END;
$clone_current_scoring$;
ALTER FUNCTION public.lab_arena_open_scoring_sep16_baseline_only_v1(TEXT, SMALLINT, JSONB)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_open_scoring_sep16_baseline_only_v1(TEXT, SMALLINT, JSONB)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

-- Once prepare has recorded the official rerun, the normal driver must not
-- open the committed plan through the generic integrity scorer.  The driver
-- may race the one-off operator between plan commit and scoring open.  Reject
-- that whole generic INSERT transaction and allow only the exact namespace
-- created by lab_arena_open_sep16_baseline_scoring_v1 below.
CREATE OR REPLACE FUNCTION public.lab_arena_sep16_rerun_score_namespace_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $sep16_rerun_score_namespace_guard$
BEGIN
  IF NEW.round_id = 'arena-2026-09-16'
     AND NEW.kind = 'score'
     AND EXISTS (
       SELECT 1 FROM public.lab_arena_sep16_baseline_rerun_audit
       WHERE round_id = NEW.round_id
     )
     AND (
       NEW.submission_id <> 'baseline-2026-09-16'
       OR NEW.assignment_id IS DISTINCT FROM
            NEW.round_id || ':' || NEW.submission_id || ':' ||
            NEW.stage::TEXT || ':' || NEW.icp_position::TEXT || ':score:rerun265'
     ) THEN
    RAISE EXCEPTION 'sep16 rerun scoring requires exact operator route'
      USING ERRCODE = '55000';
  END IF;
  RETURN NEW;
END;
$sep16_rerun_score_namespace_guard$;
ALTER FUNCTION public.lab_arena_sep16_rerun_score_namespace_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep16_rerun_score_namespace_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep16_rerun_score_namespace_guard
  ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_sep16_rerun_score_namespace_guard
  BEFORE INSERT ON public.lab_arena_runs
  FOR EACH ROW
  EXECUTE FUNCTION public.lab_arena_sep16_rerun_score_namespace_guard_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep16_baseline_rerun_v1(
  p_source_size_bytes BIGINT,
  p_source_sha256 TEXT,
  p_source_commit TEXT,
  p_bank_sha256 TEXT,
  p_forward_schedule JSONB
)
RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $prepare_sep16_baseline_rerun$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-16';
  v_archive_round_id CONSTANT TEXT := 'arena-2026-09-16-archive';
  v_baseline_id CONSTANT TEXT := 'baseline-2026-09-16';
  v_archive_baseline_id CONSTANT TEXT := 'baseline-2026-09-16-archive';
  v_new_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-16/sources/baseline-2026-09-16-native-rerun265.tar.gz';
  v_old_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-16/sources/baseline-2026-09-16.tar.gz';
  v_old_basis_hash CONSTANT TEXT :=
    'sha256:b19a147f1c8cc8365e3cf7dcc96b827ffffdd6c1e8fad65ad9fc25c8895e493f';
  v_bank_sha256 CONSTANT TEXT :=
    '42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390';
  v_round public.lab_arena_rounds%ROWTYPE;
  v_baseline public.lab_arena_submissions%ROWTYPE;
  v_audit public.lab_arena_sep16_baseline_rerun_audit%ROWTYPE;
  v_release public.lab_arena_sep16_rerun_release_authority%ROWTYPE;
  v_archive_configuration JSONB;
  v_new_configuration JSONB;
  v_new_participants JSONB;
  v_round_hash TEXT;
  v_baseline_submission_hash TEXT;
  v_baseline_runs_hash TEXT;
  v_baseline_ledger_hash TEXT;
  v_challenger_runs_hash TEXT;
  v_challenger_submissions_hash TEXT;
  v_challenger_ledger_hash TEXT;
  v_challenger_ledger_max_entry_id BIGINT;
  v_actual_microusd BIGINT;
  v_submission public.lab_arena_submissions%ROWTYPE;
  v_execute_cost JSONB;
  v_score_cost JSONB;
  v_count BIGINT;
  v_position INTEGER;
  v_stage SMALLINT;
  v_assignment TEXT;
BEGIN
  PERFORM pg_catalog.set_config('lock_timeout', '5s', TRUE);
  PERFORM pg_catalog.set_config('statement_timeout', '120s', TRUE);
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('arena-2026-09-16-native-baseline-v1', 0)
  );
  IF p_source_size_bytes NOT BETWEEN 1 AND 10485760
     OR COALESCE(p_source_sha256, '') !~ '^[0-9a-f]{64}$'
     OR COALESCE(p_source_commit, '') !~ '^[0-9a-f]{40}$'
     OR p_bank_sha256 IS DISTINCT FROM v_bank_sha256
     OR pg_catalog.jsonb_typeof(p_forward_schedule) IS DISTINCT FROM 'object' THEN
    RAISE EXCEPTION 'sep16 source or bank proof invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_release
  FROM public.lab_arena_sep16_rerun_release_authority
  WHERE round_id = v_round_id FOR SHARE;
  IF NOT FOUND
     OR v_release.source_ref <> v_new_source_ref
     OR v_release.source_size_bytes IS DISTINCT FROM p_source_size_bytes
     OR v_release.source_sha256 IS DISTINCT FROM p_source_sha256
     OR v_release.source_commit IS DISTINCT FROM p_source_commit
     OR v_release.champion_model_main_commit IS DISTINCT FROM p_source_commit
     OR v_release.champion_model_lab_commit IS DISTINCT FROM p_source_commit
     OR v_release.bank_sha256 IS DISTINCT FROM p_bank_sha256
     OR v_release.forward_schedule IS DISTINCT FROM p_forward_schedule THEN
    RAISE EXCEPTION 'sep16 committed source release authority missing or different'
      USING ERRCODE = '55000';
  END IF;
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_judgment_cache IN SHARE MODE;
  LOCK TABLE public.lab_arena_submission_credentials IN SHARE MODE;

  SELECT * INTO v_audit
  FROM public.lab_arena_sep16_baseline_rerun_audit
  WHERE round_id = v_round_id FOR UPDATE;
  IF FOUND THEN
    IF v_audit.new_source_ref <> v_new_source_ref
       OR v_audit.new_source_sha256 <> p_source_sha256
       OR v_audit.new_source_commit <> p_source_commit
       OR v_audit.bank_sha256 <> p_bank_sha256
       OR v_audit.old_round_hash IS DISTINCT FROM v_release.old_round_hash
       OR v_audit.old_baseline_submission_hash IS DISTINCT FROM
            v_release.old_baseline_submission_hash
       OR v_audit.old_baseline_runs_hash IS DISTINCT FROM
            v_release.old_baseline_runs_hash
       OR v_audit.old_baseline_ledger_hash IS DISTINCT FROM
            v_release.old_baseline_ledger_hash
       OR v_audit.old_challenger_runs_hash IS DISTINCT FROM
            v_release.old_challenger_runs_hash
       OR v_audit.old_challenger_submissions_hash IS DISTINCT FROM
            v_release.old_challenger_submissions_hash
       OR v_audit.old_challenger_ledger_hash IS DISTINCT FROM
            v_release.old_challenger_ledger_hash
       OR v_audit.old_challenger_ledger_max_entry_id IS DISTINCT FROM
            v_release.old_challenger_ledger_max_entry_id
       OR v_audit.old_actual_microusd IS DISTINCT FROM
            v_release.old_baseline_actual_microusd
       OR v_audit.new_scorer_image_digest IS DISTINCT FROM
            v_release.new_scorer_image_digest
       OR NOT EXISTS (
         SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id = v_round_id AND reward_basis_hash = v_old_basis_hash
           AND reward_activated_at IS NOT NULL
           AND configuration_doc ->> 'baseline_source_url' =
             'https://github.com/leadpoet/champion_model/archive/refs/heads/lab.tar.gz'
           AND configuration_doc ->> 'scorer_image_digest' =
             v_release.new_scorer_image_digest
       )
       OR NOT EXISTS (
         SELECT 1 FROM public.lab_arena_submissions
         WHERE submission_id = v_baseline_id AND round_id = v_round_id
           AND source_ref = v_new_source_ref
           AND submission_doc ->> 'source_sha256' = p_source_sha256
           AND submission_doc ->> 'source_commit' = p_source_commit
       )
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id AND submission_id = v_baseline_id
             AND kind = 'execute' AND assignment_id LIKE '%:rerun265') <> 20
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id = v_archive_round_id
             AND submission_id = v_archive_baseline_id) <> 40
       OR NOT public.lab_arena_sep16_challenger_seals_valid_v1() THEN
      RAISE EXCEPTION 'sep16 rerun replay differs' USING ERRCODE = '22023';
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_id', v_round_id);
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = v_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'sep16 round missing' USING ERRCODE = 'P0002';
  END IF;
  SELECT * INTO v_baseline FROM public.lab_arena_submissions
  WHERE submission_id = v_baseline_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'sep16 baseline missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'published'
     OR v_round.status_generation <> 12
     OR v_round.stage_generation <> 8
     OR v_round.evaluation_date <> '2026-09-16'
     OR v_round.icp_set_date <> DATE '2026-09-15'
     OR v_round.benchmark_ref <> 'arena/arena-2026-09-16/benchmark.json'
     OR v_round.king_outcome <> 'no_king'
     OR v_round.publication_doc #>> '{king_decision,outcome}' <> 'no_king'
     OR pg_catalog.jsonb_array_length(
          COALESCE(v_round.publication_doc -> 'final_ranking', '[]'::JSONB)
        ) <> 5
     OR (SELECT pg_catalog.count(*)
         FROM pg_catalog.jsonb_array_elements(
           COALESCE(v_round.publication_doc -> 'final_ranking', '[]'::JSONB)
         ) AS ranked
         WHERE ranked ->> 'submission_id' = v_baseline_id
           AND ranked ->> 'is_baseline' = 'true') <> 1
     OR EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(
         COALESCE(v_round.publication_doc -> 'final_ranking', '[]'::JSONB)
       ) AS ranked
       WHERE ranked ->> 'eligible' IS DISTINCT FROM 'false'
          OR ranked ->> 'eligibility_reason'
               IS DISTINCT FROM 'cost_per_company_exceeded'
     )
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_v1'
     OR EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(
         COALESCE(v_round.publication_doc -> 'final_ranking', '[]'::JSONB)
       ) AS ranked
       WHERE ranked ->> 'is_baseline' = 'false'
         AND (
           ranked #>> '{cost_summary,sourcing_cost_eligibility_policy}'
             IS DISTINCT FROM 'successful_calls_v1'
           OR COALESCE(
             (ranked #>> '{cost_summary,competition_sourcing_microusd}')::BIGINT,
             0
           ) <= COALESCE(
             (ranked #>> '{cost_summary,eligibility_cap_microusd}')::BIGINT,
             0
           )
           OR (public.lab_arena__successful_call_cost_state(
             ranked ->> 'submission_id', 'execute', NULL
           ) ->> 'successful_microusd')::BIGINT <= COALESCE(
             (ranked #>> '{cost_summary,eligibility_cap_microusd}')::BIGINT,
             0
           )
         )
     )
     OR v_round.reward_basis_doc ->> 'king_outcome' <> 'no_king'
     OR v_round.reward_basis_hash <> v_old_basis_hash
     OR v_round.effective_reward_epoch <> 25201
     OR v_round.reward_activated_at IS NULL
     OR v_round.published_at <> TIMESTAMPTZ '2026-09-16 00:44:44+00'
     OR v_round.configuration_doc ->> 'integrity_policy' <> 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'contact_policy' <> 'contacts_v1'
     OR v_round.configuration_doc ->> 'intent_details_policy' <> 'intent_details_v1'
     OR v_round.configuration_doc ? 'quality_policy'
     OR v_round.configuration_doc ->> 'parallel_twenty_icp_execution' <> 'true'
     OR v_round.configuration_doc ->> 'checkpoint_deadline_policy'
          <> 'atomic_checkpoint_45m_v1'
     OR (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER <> 20
     OR (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER <> 2700
     OR (v_round.configuration_doc ->> 'scoring_wall_clock_seconds')::INTEGER <> 900
     OR (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER <> 3600
     OR v_round.configuration_doc ->> 'baseline_source_url'
          <> 'https://github.com/leadpoet/pydantic-harness/archive/refs/heads/lab.tar.gz'
     OR v_release.verified_parallel_runner_slots <> 10
     OR v_round.configuration_doc ->> 'scorer_image_digest'
          IS DISTINCT FROM v_release.old_scorer_image_digest
     OR pg_catalog.jsonb_array_length(v_round.participants) <> 5
     OR pg_catalog.jsonb_array_length(v_round.finalists) <> 4
     OR v_baseline.round_id <> v_round_id
     OR v_baseline.status <> 'frozen'
     OR NOT v_baseline.is_king
     OR v_baseline.source_ref <> v_old_source_ref
     OR v_baseline.source_size_bytes <> 123204
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE round_id = v_round_id AND submission_id <> v_baseline_id
         AND is_king
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_submission_credentials
       WHERE submission_id = v_baseline_id
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE champion_submission_id = v_baseline_id
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id
         AND status IN ('pending', 'leased', 'submitted')
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_ledger
       WHERE round_id = v_round_id
         AND entry_kind IN ('reservation', 'dispatch')
         AND NOT EXISTS (
           SELECT 1 FROM public.lab_arena_ledger AS terminal
           WHERE terminal.call_identity = lab_arena_ledger.call_identity
             AND terminal.entry_kind IN ('settlement', 'uncertain', 'recovery', 'refusal')
         )
     ) THEN
    RAISE EXCEPTION 'sep16 frozen publication preflight differs' USING ERRCODE = '55000';
  END IF;
  FOR v_submission IN
    SELECT * FROM public.lab_arena_submissions
    WHERE round_id = v_round_id AND status = 'frozen'
  LOOP
    v_execute_cost := public.lab_arena__successful_call_cost_state(
      v_submission.submission_id, 'execute', NULL
    );
    v_score_cost := public.lab_arena__successful_call_cost_state(
      v_submission.submission_id, 'score', NULL
    );
    IF (v_execute_cost ->> 'inflight_calls')::BIGINT <> 0
       OR (v_score_cost ->> 'inflight_calls')::BIGINT <> 0
       OR (v_execute_cost ->> 'success_unresolved_calls')::BIGINT <> 0
       OR (v_score_cost ->> 'success_unresolved_calls')::BIGINT <> 0
       OR (v_submission.submission_id = v_baseline_id AND (
         (v_execute_cost ->> 'uncertain_calls')::BIGINT <> 0
         OR (v_score_cost ->> 'uncertain_calls')::BIGINT <> 0
       )) THEN
      RAISE EXCEPTION 'sep16 frozen submission has unresolved provider cost'
        USING ERRCODE = '55000';
    END IF;
  END LOOP;
  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id = v_round_id AND status = 'frozen') <> 5
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'execute' AND status = 'accepted') <> 20
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'score' AND status = 'accepted') <> 20
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id) <> 40 THEN
    RAISE EXCEPTION 'sep16 baseline or challenger proof differs' USING ERRCODE = '55000';
  END IF;
  IF p_forward_schedule #>> '{submission_open}'
       IS DISTINCT FROM v_round.configuration_doc #>> '{schedule,submission_open}'
     OR p_forward_schedule #>> '{submission_cutoff}'
       IS DISTINCT FROM v_round.configuration_doc #>> '{schedule,submission_cutoff}'
     OR (p_forward_schedule ->> 'stage_1_close')::TIMESTAMPTZ
          < pg_catalog.clock_timestamp() +
            pg_catalog.make_interval(secs =>
              pg_catalog.ceil(20.0 / v_release.verified_parallel_runner_slots)::INTEGER
              * 2760)
     OR (p_forward_schedule ->> 'stage_1_scoring_close')::TIMESTAMPTZ
          < (p_forward_schedule ->> 'stage_1_close')::TIMESTAMPTZ
              + pg_catalog.make_interval(secs =>
                  2 * pg_catalog.ceil(
                    10.0 / v_release.verified_parallel_runner_slots
                  )::INTEGER * 960)
     OR (p_forward_schedule ->> 'stage_2_start')::TIMESTAMPTZ
          <= (p_forward_schedule ->> 'stage_1_scoring_close')::TIMESTAMPTZ
     OR (p_forward_schedule ->> 'stage_2_close')::TIMESTAMPTZ
          <= (p_forward_schedule ->> 'stage_2_start')::TIMESTAMPTZ
     OR (p_forward_schedule ->> 'final_scoring_close')::TIMESTAMPTZ
          < (p_forward_schedule ->> 'stage_2_close')::TIMESTAMPTZ
              + pg_catalog.make_interval(secs =>
                  2 * pg_catalog.ceil(
                    10.0 / v_release.verified_parallel_runner_slots
                  )::INTEGER * 960)
     OR (p_forward_schedule ->> 'publication_deadline')::TIMESTAMPTZ
          <= (p_forward_schedule ->> 'final_scoring_close')::TIMESTAMPTZ
     OR (p_forward_schedule ->> 'publication_deadline')::TIMESTAMPTZ
          >= pg_catalog.clock_timestamp() + INTERVAL '24 hours' THEN
    RAISE EXCEPTION 'sep16 forward schedule invalid' USING ERRCODE = '22023';
  END IF;

  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(run) - 'round_id' - 'submission_id')::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY run_id), 'sha256'), 'hex')
  INTO v_baseline_runs_hash
  FROM public.lab_arena_runs AS run
  WHERE run.round_id = v_round_id AND run.submission_id = v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(entry) - 'round_id' - 'submission_id')::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY entry_id), 'sha256'), 'hex')
  INTO v_baseline_ledger_hash
  FROM public.lab_arena_ledger AS entry
  WHERE entry.round_id = v_round_id AND entry.submission_id = v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(run)::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY run_id), 'sha256'), 'hex')
  INTO v_challenger_runs_hash
  FROM public.lab_arena_runs AS run
  WHERE run.round_id = v_round_id AND run.submission_id <> v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(submission)::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY submission_id), 'sha256'), 'hex')
  INTO v_challenger_submissions_hash
  FROM public.lab_arena_submissions AS submission
  WHERE submission.round_id = v_round_id AND submission.submission_id <> v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(entry)::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_challenger_ledger_hash
  FROM public.lab_arena_ledger AS entry
  WHERE entry.round_id = v_round_id AND entry.submission_id <> v_baseline_id;
  SELECT COALESCE(pg_catalog.max(entry_id), 0)
  INTO v_challenger_ledger_max_entry_id
  FROM public.lab_arena_ledger
  WHERE round_id = v_round_id AND submission_id <> v_baseline_id;
  SELECT COALESCE(pg_catalog.sum(amount_microusd), 0)
  INTO v_actual_microusd FROM public.lab_arena_ledger
  WHERE round_id = v_round_id AND submission_id = v_baseline_id
    AND entry_kind IN ('settlement', 'uncertain');
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(v_round)::TEXT, 'sha256'
  ), 'hex') INTO v_round_hash;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.to_jsonb(v_baseline)::TEXT, 'sha256'
  ), 'hex') INTO v_baseline_submission_hash;
  IF v_round_hash IS DISTINCT FROM v_release.old_round_hash
     OR v_baseline_submission_hash IS DISTINCT FROM
          v_release.old_baseline_submission_hash
     OR v_baseline_runs_hash IS DISTINCT FROM v_release.old_baseline_runs_hash
     OR v_baseline_ledger_hash IS DISTINCT FROM v_release.old_baseline_ledger_hash
     OR v_challenger_runs_hash IS DISTINCT FROM v_release.old_challenger_runs_hash
     OR v_challenger_submissions_hash IS DISTINCT FROM
          v_release.old_challenger_submissions_hash
     OR v_challenger_ledger_hash IS DISTINCT FROM
          v_release.old_challenger_ledger_hash
     OR v_challenger_ledger_max_entry_id IS DISTINCT FROM
          v_release.old_challenger_ledger_max_entry_id
     OR v_actual_microusd IS DISTINCT FROM
          v_release.old_baseline_actual_microusd THEN
    RAISE EXCEPTION 'sep16 sealed published state differs'
      USING ERRCODE = '55000';
  END IF;

  INSERT INTO public.lab_arena_sep16_baseline_rerun_audit (
    round_id, archive_round_id, old_round_doc, old_baseline_submission_doc,
    old_baseline_runs_hash, old_baseline_ledger_hash,
    old_challenger_runs_hash, old_challenger_submissions_hash,
    old_challenger_ledger_hash, old_challenger_ledger_max_entry_id,
    old_round_hash, old_baseline_submission_hash, bank_sha256, new_source_ref,
    new_source_sha256, new_source_commit, old_scorer_image_digest,
    new_scorer_image_digest, old_actual_microusd
  ) VALUES (
    v_round_id, v_archive_round_id, pg_catalog.to_jsonb(v_round),
    pg_catalog.to_jsonb(v_baseline), v_baseline_runs_hash,
    v_baseline_ledger_hash, v_challenger_runs_hash,
    v_challenger_submissions_hash, v_challenger_ledger_hash,
    v_challenger_ledger_max_entry_id, v_round_hash,
    v_baseline_submission_hash, p_bank_sha256, v_new_source_ref,
    p_source_sha256, p_source_commit, v_release.old_scorer_image_digest,
    v_release.new_scorer_image_digest, v_actual_microusd
  );

  v_archive_configuration := v_round.configuration_doc ||
    pg_catalog.jsonb_build_object(
      'round_id', v_archive_round_id, 'mode', 'shadow', 'rewards_enabled', FALSE
    );
  INSERT INTO public.lab_arena_rounds (
    round_id, status, configuration_doc, rewards_enabled, cancel_reason
  ) VALUES (
    v_archive_round_id, 'cancelled', v_archive_configuration, FALSE,
    'authorized_sep16_baseline_evidence_archive'
  );
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(v_baseline) ||
      pg_catalog.jsonb_build_object(
        'submission_id', v_archive_baseline_id,
        'round_id', v_archive_round_id
      )
  )).*;

  -- Exclusive table locks keep every normal guard effective for other
  -- transactions. These one-off updates would otherwise be blocked by the
  -- intentional frozen/terminal/append-only protections.
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
  UPDATE public.lab_arena_runs
  SET round_id = v_archive_round_id, submission_id = v_archive_baseline_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 40 THEN
    RAISE EXCEPTION 'sep16 baseline run archive count differs';
  END IF;
  UPDATE public.lab_arena_ledger
  SET round_id = v_archive_round_id, submission_id = v_archive_baseline_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  DELETE FROM public.lab_arena_submissions WHERE submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'sep16 baseline submission archive count differs';
  END IF;
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(v_baseline) || pg_catalog.jsonb_build_object(
      'source_ref', v_new_source_ref,
      'source_size_bytes', p_source_size_bytes,
      'submission_doc', v_baseline.submission_doc ||
        pg_catalog.jsonb_build_object(
          'source_ref', v_new_source_ref,
          'source_size_bytes', p_source_size_bytes,
          'source_sha256', p_source_sha256,
          'source_commit', p_source_commit
        ),
      'accepted_at', pg_catalog.clock_timestamp(),
      'frozen_at', pg_catalog.clock_timestamp(),
      'updated_at', pg_catalog.clock_timestamp()
    )
  )).*;
  SELECT pg_catalog.jsonb_agg(
    CASE WHEN item ->> 'submission_id' = v_baseline_id
      THEN item || pg_catalog.jsonb_build_object(
        'source_ref', v_new_source_ref,
        'source_size_bytes', p_source_size_bytes
      )
      ELSE item END
    ORDER BY ordinal
  ) INTO v_new_participants
  FROM pg_catalog.jsonb_array_elements(v_round.participants)
    WITH ORDINALITY AS entries(item, ordinal);
  v_new_configuration := v_round.configuration_doc ||
    pg_catalog.jsonb_build_object(
      'schedule', p_forward_schedule,
      'scorer_image_reference', v_release.new_scorer_image_reference,
      'scorer_image_digest', v_release.new_scorer_image_digest,
      'runner_slot_ceiling', 10,
      'baseline_source_url',
        'https://github.com/leadpoet/champion_model/archive/refs/heads/lab.tar.gz'
    );
  IF (v_new_configuration - ARRAY[
        'schedule', 'scorer_image_reference', 'scorer_image_digest',
        'runner_slot_ceiling', 'baseline_source_url'
      ]::TEXT[]) IS DISTINCT FROM
      (v_round.configuration_doc - ARRAY[
        'schedule', 'scorer_image_reference', 'scorer_image_digest',
        'runner_slot_ceiling', 'baseline_source_url'
      ]::TEXT[]) THEN
    RAISE EXCEPTION 'sep16 configuration changed outside schedule and image';
  END IF;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1', status_generation = 13, stage_generation = 9,
      configuration_doc = v_new_configuration,
      participants = v_new_participants,
      stage1_scoring_plan_doc = NULL, stage2_scoring_plan_doc = NULL,
      finalists = NULL, publication_doc = NULL,
      king_hotkey = NULL, published_at = NULL,
      cancel_reason = NULL
  WHERE round_id = v_round_id;
  FOR v_position IN 0 .. 19 LOOP
    v_stage := CASE WHEN v_position < 10 THEN 1 ELSE 2 END;
    v_assignment := v_round_id || ':' || v_baseline_id || ':' ||
      v_stage::TEXT || ':' || v_position::TEXT || ':rerun265';
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey,
      stage, icp_position, attempt, status, stage_generation
    ) VALUES (
      v_assignment || ':1', v_assignment, v_round_id, v_baseline_id,
      v_baseline.miner_hotkey, v_stage, v_position, 1, 'pending', 9
    );
  END LOOP;
  ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round_id AND submission_id = v_baseline_id
        AND kind = 'execute' AND status = 'pending') <> 20
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_archive_round_id
           AND submission_id = v_archive_baseline_id) <> 40
     OR (SELECT COALESCE(pg_catalog.sum(amount_microusd), 0)
         FROM public.lab_arena_ledger
         WHERE round_id = v_archive_round_id
           AND submission_id = v_archive_baseline_id
           AND entry_kind IN ('settlement', 'uncertain')) <> v_actual_microusd
     OR (SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
           pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
             (pg_catalog.to_jsonb(run) - 'round_id' - 'submission_id')::TEXT,
             'sha256'
           ), 'hex'), '' ORDER BY run_id), 'sha256'), 'hex')
         FROM public.lab_arena_runs AS run
         WHERE run.round_id = v_archive_round_id
           AND run.submission_id = v_archive_baseline_id)
          <> v_baseline_runs_hash
     OR (SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
           pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
             (pg_catalog.to_jsonb(entry) - 'round_id' - 'submission_id')::TEXT,
             'sha256'
           ), 'hex'), '' ORDER BY entry_id), 'sha256'), 'hex')
         FROM public.lab_arena_ledger AS entry
         WHERE entry.round_id = v_archive_round_id
           AND entry.submission_id = v_archive_baseline_id)
          <> v_baseline_ledger_hash
     OR v_round.reward_basis_hash IS DISTINCT FROM (
       SELECT reward_basis_hash FROM public.lab_arena_rounds
       WHERE round_id = v_round_id
     )
     OR v_round.reward_basis_doc IS DISTINCT FROM (
       SELECT reward_basis_doc FROM public.lab_arena_rounds
       WHERE round_id = v_round_id
     )
     OR v_round.signing_key_doc IS DISTINCT FROM (
       SELECT signing_key_doc FROM public.lab_arena_rounds
       WHERE round_id = v_round_id
     )
     OR v_round.reward_activated_at IS DISTINCT FROM (
       SELECT reward_activated_at FROM public.lab_arena_rounds
       WHERE round_id = v_round_id
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_rounds AS current_round
       WHERE current_round.round_id = v_round_id
         AND (pg_catalog.to_jsonb(current_round) - ARRAY[
           'status', 'status_generation', 'stage_generation', 'configuration_doc',
           'participants', 'stage1_scoring_plan_doc', 'stage2_scoring_plan_doc',
           'finalists', 'publication_doc', 'king_hotkey', 'published_at',
           'cancel_reason', 'updated_at'
         ]::TEXT[]) IS DISTINCT FROM
         (pg_catalog.to_jsonb(v_round) - ARRAY[
           'status', 'status_generation', 'stage_generation', 'configuration_doc',
           'participants', 'stage1_scoring_plan_doc', 'stage2_scoring_plan_doc',
           'finalists', 'publication_doc', 'king_hotkey', 'published_at',
           'cancel_reason', 'updated_at'
         ]::TEXT[])
     ) THEN
    RAISE EXCEPTION 'sep16 post-archive verification differs';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'prepared', 'round_id', v_round_id,
    'baseline_execute_assignments', 20,
    'archived_actual_microusd', v_actual_microusd,
    'old_reward_basis_hash', v_old_basis_hash
  );
END;
$prepare_sep16_baseline_rerun$;
ALTER FUNCTION public.lab_arena_prepare_sep16_baseline_rerun_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep16_baseline_rerun_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_sep16_baseline_rerun_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep16_challenger_seals_valid_v1()
RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep16_challenger_seals_valid$
DECLARE
  v_audit public.lab_arena_sep16_baseline_rerun_audit%ROWTYPE;
  v_runs_hash TEXT;
  v_submissions_hash TEXT;
  v_ledger_hash TEXT;
BEGIN
  SELECT * INTO v_audit FROM public.lab_arena_sep16_baseline_rerun_audit
  WHERE round_id = 'arena-2026-09-16';
  IF NOT FOUND THEN
    RETURN FALSE;
  END IF;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(run)::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY run_id), 'sha256'), 'hex')
  INTO v_runs_hash FROM public.lab_arena_runs AS run
  WHERE run.round_id = 'arena-2026-09-16'
    AND run.submission_id <> 'baseline-2026-09-16';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(submission)::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY submission_id), 'sha256'), 'hex')
  INTO v_submissions_hash FROM public.lab_arena_submissions AS submission
  WHERE submission.round_id = 'arena-2026-09-16'
    AND submission.submission_id <> 'baseline-2026-09-16';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(entry)::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_ledger_hash FROM public.lab_arena_ledger AS entry
  WHERE entry.round_id = 'arena-2026-09-16'
    AND entry.submission_id <> 'baseline-2026-09-16'
    AND entry.entry_id <= v_audit.old_challenger_ledger_max_entry_id;
  RETURN v_runs_hash IS NOT DISTINCT FROM v_audit.old_challenger_runs_hash
     AND v_submissions_hash IS NOT DISTINCT FROM
         v_audit.old_challenger_submissions_hash
     AND v_ledger_hash IS NOT DISTINCT FROM
         v_audit.old_challenger_ledger_hash
     AND NOT EXISTS (
       SELECT 1 FROM public.lab_arena_ledger AS late
       WHERE late.round_id = 'arena-2026-09-16'
         AND late.submission_id <> 'baseline-2026-09-16'
         AND late.entry_id > v_audit.old_challenger_ledger_max_entry_id
         AND (
           late.entry_kind <> 'settlement'
           OR late.entry_doc ->> 'late_reconciliation' IS DISTINCT FROM 'true'
           OR NOT EXISTS (
             SELECT 1 FROM public.lab_arena_ledger AS original
             WHERE original.entry_id <= v_audit.old_challenger_ledger_max_entry_id
               AND original.entry_kind = 'uncertain'
               AND original.round_id = late.round_id
               AND original.submission_id = late.submission_id
               AND original.run_id = late.run_id
               AND original.call_identity = late.call_identity
               AND original.provider = late.provider
               AND original.entry_id =
                   (late.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
           )
         )
     );
END;
$sep16_challenger_seals_valid$;
ALTER FUNCTION public.lab_arena_sep16_challenger_seals_valid_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep16_challenger_seals_valid_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

-- The ordinary service derives every work item from the committed plan and
-- verified output objects. Refuse a partial or different request, then pass
-- only the fresh accepted baseline items to the current integrity-cache scorer.
-- Model-caused terminal execution failures remain the normal zero rows in the
-- committed plan and therefore create no judge assignment.
-- Challenger judgments remain the accepted historical rows of this round.
CREATE OR REPLACE FUNCTION public.lab_arena_open_sep16_baseline_scoring_v1(
  p_round_id TEXT, p_stage SMALLINT, p_work_items JSONB
)
RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $open_sep16_baseline_scoring$
DECLARE
  v_round public.lab_arena_rounds%ROWTYPE;
  v_audit public.lab_arena_sep16_baseline_rerun_audit%ROWTYPE;
  v_plan JSONB;
  v_baseline_items JSONB;
  v_baseline_id CONSTANT TEXT := 'baseline-2026-09-16';
  v_old_basis_hash CONSTANT TEXT :=
    'sha256:b19a147f1c8cc8365e3cf7dcc96b827ffffdd6c1e8fad65ad9fc25c8895e493f';
  v_result JSONB;
  v_count BIGINT;
  v_baseline_count BIGINT;
BEGIN
  IF p_round_id IS DISTINCT FROM 'arena-2026-09-16'
     OR p_stage NOT IN (1, 2)
     OR pg_catalog.jsonb_typeof(p_work_items) IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'sep16 scoring request invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;
  SELECT * INTO v_audit FROM public.lab_arena_sep16_baseline_rerun_audit
  WHERE round_id = p_round_id FOR UPDATE;
  IF v_round.round_id IS NULL OR v_audit.round_id IS NULL
     OR v_round.reward_basis_hash <> v_old_basis_hash
     OR v_round.reward_activated_at IS NULL
     OR v_round.king_outcome <> 'no_king'
     OR v_round.configuration_doc ->> 'parallel_twenty_icp_execution' <> 'true'
     OR v_round.configuration_doc ->> 'checkpoint_deadline_policy'
          <> 'atomic_checkpoint_45m_v1'
     OR v_round.configuration_doc ->> 'integrity_policy' <> 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'contact_policy' <> 'contacts_v1'
     OR v_round.configuration_doc ->> 'intent_details_policy'
          <> 'intent_details_v1'
     OR v_round.configuration_doc ? 'quality_policy'
     OR v_round.configuration_doc #>> '{scorer_policy,scoring_adapter_version}'
          <> 'qualification_contacts_v3'
     OR v_round.configuration_doc #>> '{scorer_policy,intent_details_policy}'
          <> 'intent_details_v1'
     OR NOT public.lab_arena_sep16_challenger_seals_valid_v1()
     OR (p_stage = 2 AND v_audit.stage1_scoring_opened_at IS NULL) THEN
    RAISE EXCEPTION 'sep16 scoring state differs' USING ERRCODE = '55000';
  END IF;
  v_plan := CASE WHEN p_stage = 1 THEN v_round.stage1_scoring_plan_doc
                 ELSE v_round.stage2_scoring_plan_doc END;
  IF v_plan ->> 'round_id' <> p_round_id
     OR (v_plan ->> 'stage')::SMALLINT <> p_stage
     OR pg_catalog.jsonb_typeof(v_plan -> 'work_items') IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(p_work_items)
          <> pg_catalog.jsonb_array_length(v_plan -> 'work_items')
     OR (SELECT pg_catalog.count(DISTINCT item ->> 'scored_run_id')
         FROM pg_catalog.jsonb_array_elements(p_work_items) AS item)
          <> pg_catalog.jsonb_array_length(p_work_items)
     OR EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(p_work_items) AS item
       WHERE NOT EXISTS (
         SELECT 1 FROM pg_catalog.jsonb_array_elements(v_plan -> 'work_items') AS planned
         WHERE planned ->> 'scored_run_id' = item ->> 'scored_run_id'
           AND planned ->> 'submission_id' = item ->> 'submission_id'
           AND planned ->> 'icp_position' = item ->> 'icp_position'
           AND planned ->> 'output_ref' = item ->> 'output_ref'
       )
     ) THEN
    RAISE EXCEPTION 'sep16 committed scoring plan differs' USING ERRCODE = '55000';
  END IF;
  SELECT pg_catalog.jsonb_agg(item ORDER BY item ->> 'scored_run_id')
  INTO v_baseline_items
  FROM pg_catalog.jsonb_array_elements(p_work_items) AS item
  WHERE item ->> 'submission_id' = v_baseline_id;
  v_baseline_items := COALESCE(v_baseline_items, '[]'::JSONB);
  v_baseline_count := pg_catalog.jsonb_array_length(v_baseline_items);
  IF v_baseline_count + (
       SELECT pg_catalog.count(*)
       FROM pg_catalog.jsonb_array_elements(v_plan -> 'zero_rows') AS zero_row
       WHERE zero_row ->> 'submission_id' = v_baseline_id
     ) <> 10
     OR (SELECT pg_catalog.count(DISTINCT position)
         FROM (
           SELECT (item ->> 'icp_position')::INTEGER AS position
           FROM pg_catalog.jsonb_array_elements(v_baseline_items) AS item
           UNION ALL
           SELECT (zero_row ->> 'icp_position')::INTEGER AS position
           FROM pg_catalog.jsonb_array_elements(v_plan -> 'zero_rows') AS zero_row
           WHERE zero_row ->> 'submission_id' = v_baseline_id
         ) AS covered) <> 10
     OR EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(v_plan -> 'work_items') AS planned
       WHERE planned ->> 'submission_id' <> v_baseline_id
         AND NOT EXISTS (
           SELECT 1 FROM public.lab_arena_runs AS scored
           WHERE scored.round_id = p_round_id
             AND scored.submission_id = planned ->> 'submission_id'
             AND scored.scored_run_id = planned ->> 'scored_run_id'
             AND scored.stage = p_stage AND scored.kind = 'score'
             AND scored.status IN ('accepted', 'failed')
         )
     ) THEN
    RAISE EXCEPTION 'sep16 preserved challenger judgment differs'
      USING ERRCODE = '55000';
  END IF;
  IF (p_stage = 1 AND v_audit.stage1_scoring_opened_at IS NOT NULL)
     OR (p_stage = 2 AND v_audit.stage2_scoring_opened_at IS NOT NULL) THEN
    IF v_round.status NOT IN (
         CASE WHEN p_stage = 1 THEN 'stage1_scoring' ELSE 'stage2_scoring' END,
         CASE WHEN p_stage = 1 THEN 'stage1_judged' ELSE 'stage2_judged' END,
         CASE WHEN p_stage = 1 THEN 'stage1_scored' ELSE 'scored' END,
         'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged',
         'scored', 'published'
       )
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = p_round_id AND submission_id = v_baseline_id
             AND stage = p_stage AND kind = 'score'
             AND assignment_id LIKE '%:score:rerun265') <> v_baseline_count
       OR EXISTS (
         SELECT 1 FROM pg_catalog.jsonb_array_elements(v_baseline_items) AS item
         WHERE NOT EXISTS (
           SELECT 1 FROM public.lab_arena_runs AS score
           WHERE score.round_id = p_round_id
             AND score.submission_id = v_baseline_id
             AND score.stage = p_stage AND score.kind = 'score'
             AND score.scored_run_id = item ->> 'scored_run_id'
             AND score.assignment_id LIKE '%:score:rerun265'
         )
       ) THEN
      RAISE EXCEPTION 'sep16 baseline scorer replay differs'
        USING ERRCODE = '55000';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing', 'round_status', v_round.status,
      'assignments', v_baseline_count
    );
  END IF;
  IF v_round.status IS DISTINCT FROM 'stage' || p_stage::TEXT || '_closed' THEN
    RAISE EXCEPTION 'sep16 scoring state differs' USING ERRCODE = '55000';
  END IF;
  v_result := public.lab_arena_open_scoring_sep16_baseline_only_v1(
    p_round_id, p_stage, v_baseline_items
  );
  IF v_result ->> 'status' <> 'ok'
     OR (v_result ->> 'assignments')::INTEGER <> v_baseline_count THEN
    RAISE EXCEPTION 'sep16 baseline scorer open differs' USING ERRCODE = '55000';
  END IF;
  SELECT pg_catalog.count(DISTINCT assignment_id) INTO v_count
  FROM public.lab_arena_runs
  WHERE round_id = p_round_id AND submission_id = v_baseline_id
    AND stage = p_stage AND kind = 'score'
    AND assignment_id LIKE '%:score:rerun265';
  IF v_count <> v_baseline_count THEN
    RAISE EXCEPTION 'sep16 baseline scorer assignments differ';
  END IF;
  UPDATE public.lab_arena_sep16_baseline_rerun_audit
  SET stage1_scoring_opened_at = CASE WHEN p_stage = 1
                                 THEN pg_catalog.clock_timestamp()
                                 ELSE stage1_scoring_opened_at END,
      stage2_scoring_opened_at = CASE WHEN p_stage = 2
                                 THEN pg_catalog.clock_timestamp()
                                 ELSE stage2_scoring_opened_at END
  WHERE round_id = p_round_id;
  RETURN v_result;
END;
$open_sep16_baseline_scoring$;
ALTER FUNCTION public.lab_arena_open_sep16_baseline_scoring_v1(TEXT, SMALLINT, JSONB)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_open_sep16_baseline_scoring_v1(TEXT, SMALLINT, JSONB)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_open_sep16_baseline_scoring_v1(TEXT, SMALLINT, JSONB)
  TO lab_arena_service;

-- The original no-winner basis is already signed and activated for Sep16.
-- A normal publication may refresh the baseline score, including to zero or a
-- positive value, but cannot alter any miner result or activated reward state.
CREATE OR REPLACE FUNCTION public.lab_arena_sep16_rerun_publication_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $sep16_rerun_publication_guard$
DECLARE
  v_audit public.lab_arena_sep16_baseline_rerun_audit%ROWTYPE;
  v_execute_cost JSONB;
  v_score_cost JSONB;
BEGIN
  IF NEW.round_id = 'arena-2026-09-16'
     AND OLD.status = 'scored'
     AND NEW.status = 'published'
     AND EXISTS (
       SELECT 1 FROM public.lab_arena_sep16_baseline_rerun_audit
       WHERE round_id = NEW.round_id
     ) THEN
    SELECT * INTO STRICT v_audit
    FROM public.lab_arena_sep16_baseline_rerun_audit
    WHERE round_id = NEW.round_id;
    v_execute_cost := public.lab_arena__successful_call_cost_state(
      'baseline-2026-09-16', 'execute', NULL
    );
    v_score_cost := public.lab_arena__successful_call_cost_state(
      'baseline-2026-09-16', 'score', NULL
    );
    IF NEW.reward_basis_hash IS DISTINCT FROM
         v_audit.old_round_doc ->> 'reward_basis_hash'
       OR NEW.reward_basis_doc IS DISTINCT FROM
            v_audit.old_round_doc -> 'reward_basis_doc'
       OR NEW.signing_key_doc IS DISTINCT FROM
            v_audit.old_round_doc -> 'signing_key_doc'
       OR NEW.reward_activated_at IS DISTINCT FROM
            (v_audit.old_round_doc ->> 'reward_activated_at')::TIMESTAMPTZ
       OR NEW.effective_reward_epoch IS DISTINCT FROM
            (v_audit.old_round_doc ->> 'effective_reward_epoch')::BIGINT
       OR NEW.king_outcome IS DISTINCT FROM 'no_king'
       OR NEW.publication_doc #>> '{king_decision,outcome}'
            IS DISTINCT FROM 'no_king'
       OR NEW.promotion_required IS DISTINCT FROM
            (v_audit.old_round_doc ->> 'promotion_required')::BOOLEAN
       OR NEW.promotion_doc IS DISTINCT FROM
            NULLIF(v_audit.old_round_doc -> 'promotion_doc', 'null'::JSONB)
       OR NEW.baseline_promoted_at IS DISTINCT FROM
            (v_audit.old_round_doc ->> 'baseline_promoted_at')::TIMESTAMPTZ
       OR NEW.champion_submission_id IS DISTINCT FROM
            v_audit.old_round_doc ->> 'champion_submission_id'
       OR NEW.champion_hotkey IS DISTINCT FROM
            v_audit.old_round_doc ->> 'champion_hotkey'
       OR NEW.champion_funding_frozen IS DISTINCT FROM
            (v_audit.old_round_doc ->> 'champion_funding_frozen')::BOOLEAN
       OR NEW.configuration_doc ->> 'baseline_source_url' IS DISTINCT FROM
            'https://github.com/leadpoet/champion_model/archive/refs/heads/lab.tar.gz'
       OR NEW.configuration_doc ->> 'scorer_image_digest' IS DISTINCT FROM
            v_audit.new_scorer_image_digest
       OR (NEW.configuration_doc ->> 'runner_slot_ceiling')::INTEGER <> 10
       OR (NEW.configuration_doc - ARRAY[
            'schedule', 'scorer_image_reference', 'scorer_image_digest',
            'runner_slot_ceiling', 'baseline_source_url'
          ]::TEXT[]) IS DISTINCT FROM
          ((v_audit.old_round_doc -> 'configuration_doc') - ARRAY[
            'schedule', 'scorer_image_reference', 'scorer_image_digest',
            'runner_slot_ceiling', 'baseline_source_url'
          ]::TEXT[])
       OR NOT public.lab_arena_sep16_challenger_seals_valid_v1()
       OR pg_catalog.jsonb_array_length(NEW.participants) <> 5
       OR EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(
           v_audit.old_round_doc -> 'participants'
         ) AS old_participant
         WHERE old_participant ->> 'submission_id' <> 'baseline-2026-09-16'
           AND NOT EXISTS (
             SELECT 1 FROM pg_catalog.jsonb_array_elements(NEW.participants) AS current_participant
             WHERE current_participant = old_participant
           )
       )
       OR NOT EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(
           v_audit.old_round_doc -> 'participants'
         ) AS old_participant
         JOIN pg_catalog.jsonb_array_elements(NEW.participants) AS current_participant
           ON current_participant ->> 'submission_id' = old_participant ->> 'submission_id'
         WHERE old_participant ->> 'submission_id' = 'baseline-2026-09-16'
           AND (current_participant - ARRAY['source_ref', 'source_size_bytes']::TEXT[])
               = (old_participant - ARRAY['source_ref', 'source_size_bytes']::TEXT[])
       )
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = NEW.round_id
             AND submission_id = 'baseline-2026-09-16'
             AND kind = 'execute'
             AND assignment_id LIKE '%:rerun265') <> 20
       OR (SELECT pg_catalog.count(DISTINCT icp_position)
           FROM public.lab_arena_runs
           WHERE round_id = NEW.round_id
             AND submission_id = 'baseline-2026-09-16'
             AND kind = 'execute'
             AND assignment_id LIKE '%:rerun265') <> 20
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs
         WHERE round_id = NEW.round_id
           AND submission_id = 'baseline-2026-09-16'
           AND kind IN ('execute', 'score')
           AND assignment_id LIKE '%:rerun265'
           AND status IN ('pending', 'leased', 'submitted')
       )
       OR (SELECT pg_catalog.count(DISTINCT icp_position)
           FROM public.lab_arena_runs
           WHERE round_id = NEW.round_id
             AND submission_id = 'baseline-2026-09-16'
             AND kind = 'execute'
             AND assignment_id LIKE '%:rerun265'
             AND per_icp_score IS NOT NULL) <> 20
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs AS executed
         WHERE executed.round_id = NEW.round_id
           AND executed.submission_id = 'baseline-2026-09-16'
           AND executed.kind = 'execute'
           AND executed.assignment_id LIKE '%:rerun265'
           AND executed.status = 'accepted'
           AND NOT EXISTS (
             SELECT 1 FROM public.lab_arena_runs AS judged
             WHERE judged.round_id = executed.round_id
               AND judged.submission_id = executed.submission_id
               AND judged.kind = 'score' AND judged.status = 'accepted'
               AND judged.scored_run_id = executed.run_id
               AND judged.assignment_id LIKE '%:score:rerun265'
           )
       )
       OR (v_execute_cost ->> 'inflight_calls')::BIGINT <> 0
       OR (v_execute_cost ->> 'uncertain_calls')::BIGINT <> 0
       OR (v_execute_cost ->> 'success_unresolved_calls')::BIGINT <> 0
       OR (v_score_cost ->> 'inflight_calls')::BIGINT <> 0
       OR (v_score_cost ->> 'uncertain_calls')::BIGINT <> 0
       OR (v_score_cost ->> 'success_unresolved_calls')::BIGINT <> 0
       OR NOT EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(
           COALESCE(NEW.publication_doc -> 'final_ranking', '[]'::JSONB)
         ) AS ranked
         WHERE ranked ->> 'submission_id' = 'baseline-2026-09-16'
       )
       OR EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(
           v_audit.old_round_doc #> '{publication_doc,final_ranking}'
         ) AS old_ranked
         WHERE old_ranked ->> 'submission_id' <> 'baseline-2026-09-16'
           AND NOT EXISTS (
             SELECT 1 FROM pg_catalog.jsonb_array_elements(
               NEW.publication_doc -> 'final_ranking'
             ) AS current_ranked
             WHERE current_ranked ->> 'submission_id' =
                     old_ranked ->> 'submission_id'
               AND current_ranked - 'rank' = old_ranked - 'rank'
           )
       ) THEN
      RAISE EXCEPTION 'sep16 rerun publication conflicts with sealed reward, miner, cost, or completion state'
        USING ERRCODE = '55000';
    END IF;
  END IF;
  RETURN NEW;
END;
$sep16_rerun_publication_guard$;
ALTER FUNCTION public.lab_arena_sep16_rerun_publication_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep16_rerun_publication_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep16_rerun_publication_guard
  ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_sep16_rerun_publication_guard
  BEFORE UPDATE ON public.lab_arena_rounds
  FOR EACH ROW
  EXECUTE FUNCTION public.lab_arena_sep16_rerun_publication_guard_v1();

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
