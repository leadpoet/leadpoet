-- One-time published-baseline rerun for arena-2026-09-17.
-- Installing this migration changes no competition row. The private prepare
-- RPC performs one atomic archive/reopen after every sealed terminal check.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $requires_sep17_rerun286$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__successful_call_cost_state(text,text,text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep17_recovery284_archive_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_ledger') IS NULL THEN
    RAISE EXCEPTION 'apply current Arena schema before rerun286';
  END IF;
END;
$requires_sep17_rerun286$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Keep the normal driver. Change only the admitted Sep17 baseline assignment
-- suffix and require the exact current integrity scorer definition.
DO $bind_rerun286_score_namespace$
DECLARE
  v_definition TEXT;
  v_old_assignment TEXT := $old$
    v_assignment := p_round_id || ':' || v_scored.submission_id || ':' ||
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score';
$old$;
  v_new_assignment TEXT := $new$
    v_assignment := p_round_id || ':' || v_scored.submission_id || ':' ||
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score' ||
      CASE WHEN p_round_id = 'arena-2026-09-17'
                  AND v_scored.submission_id = 'baseline-2026-09-17'
                  AND EXISTS (
                    SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-17-rerun285archive'
                  )
           THEN ':rerun286' ELSE '' END;
$new$;
BEGIN
  v_definition := pg_catalog.pg_get_functiondef(
    'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure
  );
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_judgment_cache_source_invalid'
     ) = 0 THEN
    RAISE EXCEPTION 'current integrity scoring definition differs';
  END IF;
  IF pg_catalog.strpos(v_definition, v_old_assignment) <> 0
     AND pg_catalog.strpos(
       pg_catalog.substr(
         v_definition, pg_catalog.strpos(v_definition, v_old_assignment)
           + pg_catalog.length(v_old_assignment)
       ), v_old_assignment
     ) = 0
     AND pg_catalog.strpos(v_definition, v_new_assignment) = 0 THEN
    IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex')
         IS DISTINCT FROM 'fe90ee26915aef9e64cb4b6b3187d32e7dc836d169ec0e9ee59f72dd73e61636' THEN
      RAISE EXCEPTION 'current integrity scoring definition differs';
    END IF;
    v_definition := pg_catalog.replace(
      v_definition, v_old_assignment, v_new_assignment
    );
    EXECUTE v_definition;
  ELSIF pg_catalog.strpos(v_definition, v_old_assignment) = 0
     AND pg_catalog.strpos(v_definition, v_new_assignment) <> 0
     AND pg_catalog.strpos(
       pg_catalog.substr(
         v_definition, pg_catalog.strpos(v_definition, v_new_assignment)
           + pg_catalog.length(v_new_assignment)
       ), v_new_assignment
     ) = 0 THEN
    IF pg_catalog.encode(extensions.digest(
         pg_catalog.replace(v_definition, v_new_assignment, v_old_assignment),
         'sha256'), 'hex') IS DISTINCT FROM 'fe90ee26915aef9e64cb4b6b3187d32e7dc836d169ec0e9ee59f72dd73e61636' THEN
      RAISE EXCEPTION 'current integrity scoring definition differs';
    END IF;
  ELSE
    RAISE EXCEPTION 'current integrity scoring namespace binding differs';
  END IF;
END;
$bind_rerun286_score_namespace$;

CREATE OR REPLACE FUNCTION public.lab_arena_sep17_rerun286_score_namespace_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $sep17_rerun286_score_namespace_guard$
BEGIN
  IF NEW.round_id = 'arena-2026-09-17'
     AND NEW.submission_id = 'baseline-2026-09-17'
     AND NEW.kind = 'score'
     AND EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = 'arena-2026-09-17-rerun285archive'
     )
     AND NEW.assignment_id IS DISTINCT FROM
           NEW.round_id || ':' || NEW.submission_id || ':' ||
           NEW.stage::TEXT || ':' || NEW.icp_position::TEXT ||
           ':score:rerun286' THEN
    RAISE EXCEPTION 'Sep17 rerun286 scoring requires exact baseline namespace'
      USING ERRCODE = '55000';
  END IF;
  RETURN NEW;
END;
$sep17_rerun286_score_namespace_guard$;
ALTER FUNCTION public.lab_arena_sep17_rerun286_score_namespace_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep17_rerun286_score_namespace_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep17_rerun286_score_namespace_guard
  ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_sep17_rerun286_score_namespace_guard
  BEFORE INSERT ON public.lab_arena_runs
  FOR EACH ROW
  EXECUTE FUNCTION public.lab_arena_sep17_rerun286_score_namespace_guard_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_sep17_rerun285_archive_valid286_v1()
RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep17_rerun285_archive_valid286$
DECLARE
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
BEGIN
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_runs_hash
  FROM public.lab_arena_runs AS row_value
  WHERE round_id = 'arena-2026-09-17-rerun285archive'
    AND submission_id = 'baseline-2026-09-17-rerun285archive';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_ledger_hash
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = 'arena-2026-09-17-rerun285archive'
    AND submission_id = 'baseline-2026-09-17-rerun285archive';
  RETURN EXISTS (
      SELECT 1 FROM public.lab_arena_rounds
      WHERE round_id = 'arena-2026-09-17-rerun285archive'
        AND status = 'cancelled' AND rewards_enabled IS FALSE
        AND configuration_doc ->> 'mode' = 'shadow'
        AND configuration_doc ->> 'rewards_enabled' = 'false'
        AND cancel_reason = 'authorized_sep17_recovery285_baseline_archive'
        AND reward_basis_hash IS NULL AND reward_basis_doc IS NULL
        AND signing_key_doc IS NULL AND effective_reward_epoch IS NULL
        AND reward_activated_at IS NULL
    )
    AND EXISTS (
      SELECT 1 FROM public.lab_arena_submissions
      WHERE round_id = 'arena-2026-09-17-rerun285archive'
        AND submission_id = 'baseline-2026-09-17-rerun285archive'
        AND source_ref = 'arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery285.tar.gz'
        AND source_size_bytes = 585058
        AND submission_doc ->> 'source_ref' =
          'arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery285.tar.gz'
        AND (submission_doc ->> 'source_size_bytes')::BIGINT =
          585058
        AND submission_doc ->> 'source_sha256' = '7085faa34ddae9994e9f281d5529413611f7e9664457cae6ad5743598982639c'
        AND submission_doc ->> 'source_commit' = '7ccf69c3a0cd3eab4339f57272650c3353a71b9a'
    )
    AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-17-rerun285archive'
           AND submission_id = 'baseline-2026-09-17-rerun285archive') =
        48
    AND (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-17-rerun285archive'
           AND submission_id = 'baseline-2026-09-17-rerun285archive') =
        7626
    AND v_runs_hash = 'sha256:ccb6111ba8b0eecfa6e45458fb79ff1d83d3a97cdc8587bec25e9cf8b220755e'
    AND v_ledger_hash = 'sha256:e2c1193b874b7bf6043d6e38dedbca0a449873ef44e84021853e38d2e2afcc56'
    AND NOT EXISTS (
      SELECT 1 FROM public.lab_arena_runs
      WHERE round_id = 'arena-2026-09-17-rerun285archive'
        AND submission_id = 'baseline-2026-09-17-rerun285archive'
        AND status IN ('pending', 'leased', 'submitted')
    )
    AND public.lab_arena_sep17_recovery284_archive_valid_v1()
    AND public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1();
END;
$sep17_rerun285_archive_valid286$;
ALTER FUNCTION public.lab_arena_sep17_rerun285_archive_valid286_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep17_rerun285_archive_valid286_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep17_rerun286_nonbaseline_valid_v1()
RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep17_rerun286_nonbaseline_valid$
DECLARE
  v_submissions_hash TEXT;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
BEGIN
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY submission_id), ''), 'sha256'), 'hex')
  INTO v_submissions_hash
  FROM public.lab_arena_submissions AS row_value
  WHERE round_id = 'arena-2026-09-17'
    AND submission_id <> 'baseline-2026-09-17';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_runs_hash
  FROM public.lab_arena_runs AS row_value
  WHERE round_id = 'arena-2026-09-17'
    AND submission_id <> 'baseline-2026-09-17';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'
    ), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_ledger_hash
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = 'arena-2026-09-17'
    AND submission_id <> 'baseline-2026-09-17';
  RETURN (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
          WHERE round_id = 'arena-2026-09-17'
            AND submission_id <> 'baseline-2026-09-17') =
           8
     AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
          WHERE round_id = 'arena-2026-09-17'
            AND submission_id <> 'baseline-2026-09-17') =
           0
     AND (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
          WHERE round_id = 'arena-2026-09-17'
            AND submission_id <> 'baseline-2026-09-17') =
           15
     AND v_submissions_hash = 'sha256:7c0d8883f0984000ea0fa2a30726fb489f959b3ce484b8dd5d40d79596684be6'
     AND v_runs_hash = 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
     AND v_ledger_hash = 'sha256:8bd6df08e53b8fe17e846cc5ccb89535a40e42dcc89d87b57825d4f16e67cea2';
END;
$sep17_rerun286_nonbaseline_valid$;
ALTER FUNCTION public.lab_arena_sep17_rerun286_nonbaseline_valid_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep17_rerun286_nonbaseline_valid_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep17_published_rerun286_v1(
  p_source_size_bytes BIGINT,
  p_source_sha256 TEXT,
  p_source_commit TEXT,
  p_bank_sha256 TEXT,
  p_forward_schedule JSONB
)
RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $prepare_sep17_published_rerun286$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-17';
  v_baseline_id CONSTANT TEXT := 'baseline-2026-09-17';
  v_archive_round_id CONSTANT TEXT := 'arena-2026-09-17-rerun285archive';
  v_archive_submission_id CONSTANT TEXT :=
    'baseline-2026-09-17-rerun285archive';
  v_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery286.tar.gz';
  v_terminal_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery285.tar.gz';
  v_bank_sha256 CONSTANT TEXT :=
    '7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871';
  v_terminal_round CONSTANT JSONB :=
    $terminal_round${"arena_netuid":71,"arena_network_name":"finney","baseline_promoted_at":null,"benchmark_ref":"arena/arena-2026-09-17/benchmark.json","cancel_reason":null,"champion_fallback_providers":[],"champion_funding_frozen":true,"champion_hotkey":null,"champion_submission_id":null,"configuration_doc":{"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/pydantic-harness/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-17","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-17T18:30:00Z","final_scoring_close":"2026-09-18T02:20:00Z","publication_deadline":"2026-09-18T02:50:00Z","stage_1_close":"2026-09-17T22:00:00Z","stage_1_scoring_close":"2026-09-18T00:00:00Z","stage_1_start":"2026-09-17T18:30:01Z","stage_2_close":"2026-09-18T00:20:00Z","stage_2_start":"2026-09-18T00:00:01Z","submission_cutoff":"2026-09-17T00:00:00Z","submission_open":"2026-09-16T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_v1","stage_1_icp_count":10,"stage_2_icp_count":10},"confirmation_bank_hash":null,"confirmation_bank_ref":null,"confirmation_cohort":null,"created_at":"2026-09-16T00:02:10.363541+00:00","effective_reward_epoch":25236,"evaluation_date":"2026-09-17","finalists":[],"icp_set_date":"2026-09-16","king_hotkey":null,"king_outcome":"no_king","king_start_epoch":0,"participants":[{"is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","source_ref":"arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery285.tar.gz","source_size_bytes":585058,"submission_id":"baseline-2026-09-17"}],"promotion_doc":null,"promotion_required":true,"publication_doc":{"final_ranking":[{"cost_summary":{"competition_sourcing_microusd":13219891,"cost_per_company_cap_microusd":800000,"eligibility_cap_microusd":800000,"execution":{"call_count":2517,"conservative_microusd":37315934,"inflight_calls":0,"providers":[{"call_count":566,"conservative_microusd":8433000,"inflight_calls":0,"provider":"deepline","refused_calls":0,"reserved_or_uncertain_microusd":35000,"settled_microusd":8398000,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":559,"successful_microusd":8398000,"uncertain_calls":5},{"call_count":1890,"conservative_microusd":28877434,"inflight_calls":0,"provider":"openrouter","refused_calls":0,"reserved_or_uncertain_microusd":24060793,"settled_microusd":4816641,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":1789,"successful_microusd":4816641,"uncertain_calls":101},{"call_count":61,"conservative_microusd":5500,"inflight_calls":0,"provider":"scrapingdog","refused_calls":0,"reserved_or_uncertain_microusd":250,"settled_microusd":5250,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":21,"successful_microusd":5250,"uncertain_calls":1}],"refused_calls":0,"reserved_or_uncertain_microusd":24096043,"settled_microusd":13219891,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":2369,"successful_microusd":13219891,"uncertain_calls":107},"execution_cap_microusd":80000000,"judge":{"call_count":25,"conservative_microusd":157004,"inflight_calls":0,"providers":[{"call_count":5,"conservative_microusd":36000,"inflight_calls":0,"provider":"deepline","refused_calls":0,"reserved_or_uncertain_microusd":0,"settled_microusd":36000,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":5,"successful_microusd":36000,"uncertain_calls":0},{"call_count":13,"conservative_microusd":120504,"inflight_calls":0,"provider":"openrouter","refused_calls":0,"reserved_or_uncertain_microusd":0,"settled_microusd":120504,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":13,"successful_microusd":120504,"uncertain_calls":0},{"call_count":7,"conservative_microusd":500,"inflight_calls":0,"provider":"scrapingdog","refused_calls":0,"reserved_or_uncertain_microusd":0,"settled_microusd":500,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":2,"successful_microusd":500,"uncertain_calls":0}],"refused_calls":0,"reserved_or_uncertain_microusd":0,"settled_microusd":157004,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":20,"successful_microusd":157004,"uncertain_calls":0},"qualified_company_count":1,"returned_company_count":5,"sourcing_cost_eligibility_policy":"successful_calls_v1"},"eligibility_reason":"cost_per_company_exceeded","eligible":false,"final_score":0,"is_baseline":true,"rank":1,"submission_id":"baseline-2026-09-17"}],"finalists":[],"king_decision":{"king_hotkey":"","king_submission_id":null,"outcome":"no_king","winner_submission_id":null},"participants":[{"is_baseline":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","submission_id":"baseline-2026-09-17"}],"published_at":"2026-09-17T18:28:20Z","round_id":"arena-2026-09-17","schema_version":"leadpoet.lab_arena.publication.v1","stage1_ranking":[]},"published_at":"2026-09-17T18:28:20+00:00","reward_activated_at":"2026-09-17T18:28:24.628268+00:00","reward_basis_doc":{"champion_reward_factor_ppm":1000000,"effective_reward_epoch":25236,"king_hotkey":"","king_outcome":"no_king","king_start_epoch":0,"published_at":"2026-09-17T18:28:20Z","reward_basis_hash":"sha256:6b0b7d44bfbc75a5e371aee38545c5496f89fec5beb7ec4b04e7e6ee98cb5234","reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"round_id":"arena-2026-09-17","schema_version":"leadpoet.lab_arena.reward_basis.v1","signature":{"algorithm":"ECDSA_SHA_256","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","signature_b64":"MEUCIQDIrYj4BpBQ3D7aN9JngRzLDsQ7O3heC+H6mA8t/kc6KQIga423Zu2B68rRwPOb5ow9iqnMbuMgJ30aNfrbtwTGU0E="}},"reward_basis_hash":"sha256:6b0b7d44bfbc75a5e371aee38545c5496f89fec5beb7ec4b04e7e6ee98cb5234","rewards_enabled":true,"round_id":"arena-2026-09-17","signing_key_doc":{"algorithm":"ECDSA_SHA_256","key_spec":"ECC_NIST_P256","public_key_der_b64":"MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE9ASH6X2G6rxbVVZuNzvy12Wj0tMPvJ4HCdZTs2me0xUz/llK9Lp00EGdhWYSLhEEXy/YU0weeMMilC+u4jREzg==","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","schema_version":"leadpoet.lab_arena.signing_key.v1"},"stage1_scoring_plan_doc":{"round_id":"arena-2026-09-17","schema_version":"leadpoet.lab_arena.scoring_plan.v1","stage":1,"work_items":[{"icp_position":0,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:0:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:0:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":1,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:1:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:1:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":2,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:2:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:2:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":3,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:3:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:3:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":4,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:4:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:4:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":5,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:5:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:5:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":6,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:6:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:6:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":7,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:7:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:7:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":8,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:8:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:8:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":9,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:9:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:9:rerun285:2","submission_id":"baseline-2026-09-17"}],"zero_rows":[]},"stage2_scoring_plan_doc":{"round_id":"arena-2026-09-17","schema_version":"leadpoet.lab_arena.scoring_plan.v1","stage":2,"work_items":[{"icp_position":10,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:10:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:10:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":11,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:11:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:11:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":12,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:12:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:12:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":13,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:13:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:13:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":14,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:14:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:14:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":15,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:15:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:15:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":16,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:16:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:16:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":17,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:17:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:17:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":18,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:18:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:18:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":19,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:19:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:19:rerun285:1","submission_id":"baseline-2026-09-17"}],"zero_rows":[]},"stage3_scoring_plan_doc":null,"stage_generation":18,"status":"published","status_generation":22,"updated_at":"2026-09-17T18:28:24.629543+00:00"}$terminal_round$::JSONB;
  v_terminal_baseline CONSTANT JSONB :=
    $terminal_baseline${"accepted_at":"2026-09-17T00:01:11.253543+00:00","code_review_attempts":0,"code_review_claim":null,"code_review_doc":null,"code_review_expires_at":null,"code_review_started_at":null,"code_review_status":"pending","consent":{"public_rerun":true},"created_at":"2026-09-17T00:01:11.144651+00:00","frozen_at":"2026-09-17T00:01:11.505222+00:00","is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","owner_block_hash":null,"owner_block_number":null,"owner_coldkey":null,"rejection_rule":null,"replaced_by_submission_id":null,"replaces_submission_id":null,"round_id":"arena-2026-09-17","source_ref":"arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery285.tar.gz","source_size_bytes":585058,"status":"frozen","submission_doc":{"consent":{"public_rerun":true},"is_king":true,"source_commit":"7ccf69c3a0cd3eab4339f57272650c3353a71b9a","source_ref":"arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery285.tar.gz","source_sha256":"7085faa34ddae9994e9f281d5529413611f7e9664457cae6ad5743598982639c","source_size_bytes":585058},"submission_id":"baseline-2026-09-17","updated_at":"2026-09-17T00:01:11.505592+00:00"}$terminal_baseline$::JSONB;
  v_forward_schedule CONSTANT JSONB :=
    $forward_schedule${"benchmark_deadline":"2026-09-17T20:25:00Z","final_scoring_close":"2026-09-18T00:55:00Z","publication_deadline":"2026-09-18T01:05:00Z","stage_1_close":"2026-09-17T23:25:00Z","stage_1_scoring_close":"2026-09-17T23:55:00Z","stage_1_start":"2026-09-17T20:25:01Z","stage_2_close":"2026-09-18T00:25:00Z","stage_2_start":"2026-09-17T23:55:01Z","submission_cutoff":"2026-09-17T00:00:00Z","submission_open":"2026-09-16T00:00:00Z"}$forward_schedule$::JSONB;
  v_round public.lab_arena_rounds%ROWTYPE;
  v_baseline public.lab_arena_submissions%ROWTYPE;
  v_new_configuration JSONB;
  v_new_participants JSONB;
  v_archive_configuration JSONB;
  v_expected_round JSONB;
  v_expected_baseline JSONB;
  v_protected_before JSONB;
  v_protected_after JSONB;
  v_execute_cost JSONB;
  v_score_cost JSONB;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_position INTEGER;
  v_stage SMALLINT;
  v_assignment TEXT;
  v_count BIGINT;
BEGIN
  PERFORM pg_catalog.set_config('lock_timeout', '5s', TRUE);
  PERFORM pg_catalog.set_config('statement_timeout', '120s', TRUE);
  IF p_source_size_bytes IS DISTINCT FROM 599360
     OR p_source_sha256 IS DISTINCT FROM '4851caf8a491ae6ae8a7d1acc5ec5671e34d1ea8357bcc0c4e27f322a2ce02c0'
     OR p_source_commit IS DISTINCT FROM '056117ff321a672793123537156283099814a3cd'
     OR p_bank_sha256 IS DISTINCT FROM v_bank_sha256
     OR p_forward_schedule IS DISTINCT FROM v_forward_schedule THEN
    RAISE EXCEPTION 'Sep17 rerun286 source, bank, or schedule differs'
      USING ERRCODE = '22023';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('arena-2026-09-17-published-rerun286', 0)
  );
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
  SELECT * INTO STRICT v_round FROM public.lab_arena_rounds
  WHERE round_id = v_round_id FOR UPDATE;
  SELECT * INTO STRICT v_baseline FROM public.lab_arena_submissions
  WHERE round_id = v_round_id AND submission_id = v_baseline_id FOR UPDATE;

  v_new_configuration := pg_catalog.jsonb_set(
    v_terminal_round -> 'configuration_doc',
    '{schedule}', v_forward_schedule, FALSE
  );
  SELECT pg_catalog.jsonb_agg(
    CASE WHEN item ->> 'submission_id' = v_baseline_id THEN
      item || pg_catalog.jsonb_build_object(
        'source_ref', v_source_ref,
        'source_size_bytes', p_source_size_bytes
      ) ELSE item END ORDER BY ordinal
  ) INTO v_new_participants
  FROM pg_catalog.jsonb_array_elements(v_terminal_round -> 'participants')
    WITH ORDINALITY AS entries(item, ordinal);

  -- Response-loss replay is read-only and verifies both namespaces.
  IF EXISTS (SELECT 1 FROM public.lab_arena_rounds
             WHERE round_id = v_archive_round_id)
     OR v_baseline.source_ref = v_source_ref
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND submission_id = v_baseline_id
         AND assignment_id LIKE '%:rerun286'
     ) THEN
    IF public.lab_arena_sep17_rerun285_archive_valid286_v1() IS NOT TRUE
       OR public.lab_arena_sep17_rerun286_nonbaseline_valid_v1() IS NOT TRUE
       OR v_round.configuration_doc IS DISTINCT FROM v_new_configuration
       OR v_round.participants IS DISTINCT FROM v_new_participants
       OR v_round.status NOT IN (
         'stage1', 'stage1_closed', 'stage1_scoring', 'stage1_judged',
         'stage1_scored', 'stage2', 'stage2_closed', 'stage2_scoring',
         'stage2_judged', 'scored', 'published'
       )
       OR v_round.reward_basis_hash IS DISTINCT FROM
            v_terminal_round ->> 'reward_basis_hash'
       OR v_round.reward_basis_doc IS DISTINCT FROM
            v_terminal_round -> 'reward_basis_doc'
       OR v_round.signing_key_doc IS DISTINCT FROM
            v_terminal_round -> 'signing_key_doc'
       OR v_round.effective_reward_epoch IS DISTINCT FROM
            (v_terminal_round ->> 'effective_reward_epoch')::BIGINT
       OR v_round.reward_activated_at IS DISTINCT FROM
            (v_terminal_round ->> 'reward_activated_at')::TIMESTAMPTZ
       OR v_baseline.source_ref IS DISTINCT FROM v_source_ref
       OR v_baseline.source_size_bytes IS DISTINCT FROM p_source_size_bytes
       OR v_baseline.submission_doc ->> 'source_ref' IS DISTINCT FROM v_source_ref
       OR (v_baseline.submission_doc ->> 'source_size_bytes')::BIGINT
            IS DISTINCT FROM p_source_size_bytes
       OR v_baseline.submission_doc ->> 'source_sha256'
            IS DISTINCT FROM p_source_sha256
       OR v_baseline.submission_doc ->> 'source_commit'
            IS DISTINCT FROM p_source_commit
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id AND submission_id = v_baseline_id
             AND kind = 'execute' AND assignment_id LIKE '%:rerun286') <> 20
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'score'
           AND assignment_id NOT LIKE '%:score:rerun286'
       ) THEN
      RAISE EXCEPTION 'Sep17 rerun286 replay differs' USING ERRCODE = '55000';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing', 'round_id', v_round_id,
      'baseline_execute_assignments', 20,
      'execute_namespace', 'rerun286',
      'score_namespace', 'score:rerun286',
      'source_size_bytes', p_source_size_bytes,
      'source_sha256', p_source_sha256,
      'source_commit', p_source_commit
    );
  END IF;

  IF (p_forward_schedule ->> 'benchmark_deadline')::TIMESTAMPTZ <=
       pg_catalog.clock_timestamp() THEN
    RAISE EXCEPTION 'Sep17 rerun286 admission window has closed'
      USING ERRCODE = '22023';
  END IF;
  SELECT public.lab_arena__successful_call_cost_state(
    v_baseline_id, 'execute', NULL
  ) INTO v_execute_cost;
  SELECT public.lab_arena__successful_call_cost_state(
    v_baseline_id, 'score', NULL
  ) INTO v_score_cost;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_ledger_hash FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  IF pg_catalog.to_jsonb(v_round) IS DISTINCT FROM v_terminal_round
     OR pg_catalog.to_jsonb(v_baseline) IS DISTINCT FROM v_terminal_baseline
     OR v_round.status IS DISTINCT FROM 'published'
     OR v_round.reward_basis_hash IS NULL
     OR v_round.reward_basis_doc IS NULL
     OR v_round.signing_key_doc IS NULL
     OR v_round.effective_reward_epoch IS NULL
     OR v_round.reward_activated_at IS NULL
     OR v_baseline.source_ref IS DISTINCT FROM v_terminal_source_ref
     OR v_baseline.source_size_bytes IS DISTINCT FROM 585058
     OR v_baseline.submission_doc ->> 'source_ref'
          IS DISTINCT FROM v_terminal_source_ref
     OR (v_baseline.submission_doc ->> 'source_size_bytes')::BIGINT
          IS DISTINCT FROM 585058
     OR v_baseline.submission_doc ->> 'source_sha256'
          IS DISTINCT FROM '7085faa34ddae9994e9f281d5529413611f7e9664457cae6ad5743598982639c'
     OR v_baseline.submission_doc ->> 'source_commit'
          IS DISTINCT FROM '7ccf69c3a0cd3eab4339f57272650c3353a71b9a'
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id) <>
        48
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = v_round_id AND submission_id = v_baseline_id) <>
        7626
     OR v_runs_hash IS DISTINCT FROM 'sha256:ccb6111ba8b0eecfa6e45458fb79ff1d83d3a97cdc8587bec25e9cf8b220755e'
     OR v_ledger_hash IS DISTINCT FROM 'sha256:e2c1193b874b7bf6043d6e38dedbca0a449873ef44e84021853e38d2e2afcc56'
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = v_round_id
                  AND status IN ('pending', 'leased', 'submitted'))
     OR COALESCE((v_execute_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_execute_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_score_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_score_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
     OR EXISTS (SELECT 1 FROM public.lab_arena_rounds
                WHERE round_id = v_archive_round_id)
     OR public.lab_arena_sep17_recovery284_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()
          IS NOT TRUE
     OR public.lab_arena_sep17_rerun286_nonbaseline_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'Sep17 published recovery285 terminal differs'
      USING ERRCODE = '55000';
  END IF;

  -- The ledger comparison covers the active round, next intake, and prior
  -- recovery archive. Older ledger rows cannot match the exact mutation
  -- predicate; do not copy the full billing history into transaction memory.
  SELECT pg_catalog.jsonb_build_object(
    'rounds', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY round_id), '[]'::JSONB)
      FROM public.lab_arena_rounds AS row_value
      WHERE round_id <> v_round_id),
    'submissions', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY submission_id), '[]'::JSONB)
      FROM public.lab_arena_submissions AS row_value
      WHERE NOT (round_id = v_round_id AND submission_id = v_baseline_id)),
    'runs', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY run_id), '[]'::JSONB)
      FROM public.lab_arena_runs AS row_value
      WHERE NOT (round_id = v_round_id AND submission_id = v_baseline_id)),
    'ledger', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY entry_id), '[]'::JSONB)
      FROM public.lab_arena_ledger AS row_value
      WHERE round_id IN (v_round_id, 'arena-2026-09-18',
                         'arena-2026-09-17-rerun284archive')
        AND NOT (round_id = v_round_id AND submission_id = v_baseline_id)),
    'weights', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY network, netuid, epoch), '[]'::JSONB)
      FROM public.lab_arena_accepted_weight_states AS row_value)
  ) INTO v_protected_before;

  v_archive_configuration := (v_terminal_round -> 'configuration_doc') ||
    pg_catalog.jsonb_build_object(
      'round_id', v_archive_round_id, 'mode', 'shadow', 'rewards_enabled', FALSE
    );
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
  INSERT INTO public.lab_arena_rounds (
    round_id, status, configuration_doc, rewards_enabled, cancel_reason
  ) VALUES (
    v_archive_round_id, 'cancelled', v_archive_configuration, FALSE,
    'authorized_sep17_recovery285_baseline_archive'
  );
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    v_terminal_baseline || pg_catalog.jsonb_build_object(
      'submission_id', v_archive_submission_id, 'round_id', v_archive_round_id
    )
  )).*;
  UPDATE public.lab_arena_runs
  SET round_id = v_archive_round_id,
      submission_id = v_archive_submission_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 48 THEN
    RAISE EXCEPTION 'Sep17 rerun286 run archive count differs';
  END IF;
  UPDATE public.lab_arena_ledger
  SET round_id = v_archive_round_id,
      submission_id = v_archive_submission_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 7626 THEN
    RAISE EXCEPTION 'Sep17 rerun286 ledger archive count differs';
  END IF;
  UPDATE public.lab_arena_submissions
  SET source_ref = v_source_ref,
      source_size_bytes = p_source_size_bytes,
      submission_doc = COALESCE(submission_doc, '{}'::JSONB) ||
        pg_catalog.jsonb_build_object(
          'source_ref', v_source_ref,
          'source_size_bytes', p_source_size_bytes,
          'source_sha256', p_source_sha256,
          'source_commit', p_source_commit
        )
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep17 rerun286 baseline update differs';
  END IF;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1',
      status_generation = status_generation + 1,
      stage_generation = stage_generation + 1,
      configuration_doc = v_new_configuration,
      participants = v_new_participants,
      stage1_scoring_plan_doc = NULL,
      stage2_scoring_plan_doc = NULL,
      stage3_scoring_plan_doc = NULL,
      finalists = NULL,
      publication_doc = NULL,
      published_at = NULL,
      cancel_reason = NULL,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round_id;
  FOR v_position IN 0 .. 19 LOOP
    v_stage := CASE WHEN v_position < 10 THEN 1 ELSE 2 END;
    v_assignment := v_round_id || ':' || v_baseline_id || ':' ||
      v_stage::TEXT || ':' || v_position::TEXT || ':rerun286';
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey,
      stage, icp_position, attempt, kind, status, stage_generation
    ) VALUES (
      v_assignment || ':1', v_assignment, v_round_id, v_baseline_id,
      v_baseline.miner_hotkey, v_stage, v_position, 1, 'execute', 'pending',
      v_round.stage_generation + 1
    );
  END LOOP;
  ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

  SELECT pg_catalog.jsonb_build_object(
    'rounds', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY round_id), '[]'::JSONB)
      FROM public.lab_arena_rounds AS row_value
      WHERE round_id <> v_round_id AND round_id <> v_archive_round_id),
    'submissions', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY submission_id), '[]'::JSONB)
      FROM public.lab_arena_submissions AS row_value
      WHERE NOT ((round_id = v_round_id AND submission_id = v_baseline_id)
        OR (round_id = v_archive_round_id
          AND submission_id = v_archive_submission_id))),
    'runs', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY run_id), '[]'::JSONB)
      FROM public.lab_arena_runs AS row_value
      WHERE NOT ((round_id = v_round_id AND submission_id = v_baseline_id)
        OR (round_id = v_archive_round_id
          AND submission_id = v_archive_submission_id))),
    'ledger', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY entry_id), '[]'::JSONB)
      FROM public.lab_arena_ledger AS row_value
      WHERE round_id IN (v_round_id, 'arena-2026-09-18',
                         'arena-2026-09-17-rerun284archive')
        AND NOT ((round_id = v_round_id AND submission_id = v_baseline_id)
        OR (round_id = v_archive_round_id
          AND submission_id = v_archive_submission_id))),
    'weights', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY network, netuid, epoch), '[]'::JSONB)
      FROM public.lab_arena_accepted_weight_states AS row_value)
  ) INTO v_protected_after;
  v_expected_round := v_terminal_round || pg_catalog.jsonb_build_object(
    'status', 'stage1',
    'status_generation', (v_terminal_round ->> 'status_generation')::BIGINT + 1,
    'stage_generation', (v_terminal_round ->> 'stage_generation')::BIGINT + 1,
    'configuration_doc', v_new_configuration,
    'participants', v_new_participants,
    'stage1_scoring_plan_doc', NULL,
    'stage2_scoring_plan_doc', NULL,
    'stage3_scoring_plan_doc', NULL,
    'finalists', NULL, 'publication_doc', NULL,
    'published_at', NULL, 'cancel_reason', NULL
  );
  v_expected_baseline := v_terminal_baseline || pg_catalog.jsonb_build_object(
    'source_ref', v_source_ref,
    'source_size_bytes', p_source_size_bytes,
    'submission_doc', COALESCE(v_terminal_baseline -> 'submission_doc', '{}'::JSONB) ||
      pg_catalog.jsonb_build_object(
        'source_ref', v_source_ref,
        'source_size_bytes', p_source_size_bytes,
        'source_sha256', p_source_sha256,
        'source_commit', p_source_commit
      )
  );
  IF v_protected_after IS DISTINCT FROM v_protected_before
     OR public.lab_arena_sep17_rerun285_archive_valid286_v1() IS NOT TRUE
     OR (SELECT pg_catalog.to_jsonb(row_value) - 'updated_at'
         FROM public.lab_arena_rounds AS row_value
         WHERE round_id = v_round_id) IS DISTINCT FROM
        (v_expected_round - 'updated_at')
     OR (SELECT pg_catalog.to_jsonb(row_value)
         FROM public.lab_arena_submissions AS row_value
         WHERE round_id = v_round_id AND submission_id = v_baseline_id)
        IS DISTINCT FROM v_expected_baseline
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'execute' AND status = 'pending'
           AND output_ref IS NULL AND attempt = 1
           AND assignment_id LIKE '%:rerun286') <> 20
     OR EXISTS (SELECT 1 FROM public.lab_arena_ledger
                WHERE round_id = v_round_id AND submission_id = v_baseline_id)
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = v_round_id AND submission_id = v_baseline_id
                  AND kind = 'score') THEN
    RAISE EXCEPTION 'Sep17 rerun286 atomic preservation differs'
      USING ERRCODE = '55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'prepared', 'round_id', v_round_id,
    'baseline_execute_assignments', 20,
    'archived_runs', 48,
    'archived_ledger_entries', 7626,
    'execute_namespace', 'rerun286',
    'score_namespace', 'score:rerun286',
    'openrouter_calls_per_icp', 200,
    'source_size_bytes', p_source_size_bytes,
    'source_sha256', p_source_sha256,
    'source_commit', p_source_commit
  );
END;
$prepare_sep17_published_rerun286$;
ALTER FUNCTION public.lab_arena_prepare_sep17_published_rerun286_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep17_published_rerun286_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_sep17_published_rerun286_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep17_rerun286_publication_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $sep17_rerun286_publication_guard$
DECLARE
  v_terminal_round CONSTANT JSONB :=
    $terminal_round${"arena_netuid":71,"arena_network_name":"finney","baseline_promoted_at":null,"benchmark_ref":"arena/arena-2026-09-17/benchmark.json","cancel_reason":null,"champion_fallback_providers":[],"champion_funding_frozen":true,"champion_hotkey":null,"champion_submission_id":null,"configuration_doc":{"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/pydantic-harness/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-17","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-17T18:30:00Z","final_scoring_close":"2026-09-18T02:20:00Z","publication_deadline":"2026-09-18T02:50:00Z","stage_1_close":"2026-09-17T22:00:00Z","stage_1_scoring_close":"2026-09-18T00:00:00Z","stage_1_start":"2026-09-17T18:30:01Z","stage_2_close":"2026-09-18T00:20:00Z","stage_2_start":"2026-09-18T00:00:01Z","submission_cutoff":"2026-09-17T00:00:00Z","submission_open":"2026-09-16T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_v1","stage_1_icp_count":10,"stage_2_icp_count":10},"confirmation_bank_hash":null,"confirmation_bank_ref":null,"confirmation_cohort":null,"created_at":"2026-09-16T00:02:10.363541+00:00","effective_reward_epoch":25236,"evaluation_date":"2026-09-17","finalists":[],"icp_set_date":"2026-09-16","king_hotkey":null,"king_outcome":"no_king","king_start_epoch":0,"participants":[{"is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","source_ref":"arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery285.tar.gz","source_size_bytes":585058,"submission_id":"baseline-2026-09-17"}],"promotion_doc":null,"promotion_required":true,"publication_doc":{"final_ranking":[{"cost_summary":{"competition_sourcing_microusd":13219891,"cost_per_company_cap_microusd":800000,"eligibility_cap_microusd":800000,"execution":{"call_count":2517,"conservative_microusd":37315934,"inflight_calls":0,"providers":[{"call_count":566,"conservative_microusd":8433000,"inflight_calls":0,"provider":"deepline","refused_calls":0,"reserved_or_uncertain_microusd":35000,"settled_microusd":8398000,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":559,"successful_microusd":8398000,"uncertain_calls":5},{"call_count":1890,"conservative_microusd":28877434,"inflight_calls":0,"provider":"openrouter","refused_calls":0,"reserved_or_uncertain_microusd":24060793,"settled_microusd":4816641,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":1789,"successful_microusd":4816641,"uncertain_calls":101},{"call_count":61,"conservative_microusd":5500,"inflight_calls":0,"provider":"scrapingdog","refused_calls":0,"reserved_or_uncertain_microusd":250,"settled_microusd":5250,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":21,"successful_microusd":5250,"uncertain_calls":1}],"refused_calls":0,"reserved_or_uncertain_microusd":24096043,"settled_microusd":13219891,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":2369,"successful_microusd":13219891,"uncertain_calls":107},"execution_cap_microusd":80000000,"judge":{"call_count":25,"conservative_microusd":157004,"inflight_calls":0,"providers":[{"call_count":5,"conservative_microusd":36000,"inflight_calls":0,"provider":"deepline","refused_calls":0,"reserved_or_uncertain_microusd":0,"settled_microusd":36000,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":5,"successful_microusd":36000,"uncertain_calls":0},{"call_count":13,"conservative_microusd":120504,"inflight_calls":0,"provider":"openrouter","refused_calls":0,"reserved_or_uncertain_microusd":0,"settled_microusd":120504,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":13,"successful_microusd":120504,"uncertain_calls":0},{"call_count":7,"conservative_microusd":500,"inflight_calls":0,"provider":"scrapingdog","refused_calls":0,"reserved_or_uncertain_microusd":0,"settled_microusd":500,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":2,"successful_microusd":500,"uncertain_calls":0}],"refused_calls":0,"reserved_or_uncertain_microusd":0,"settled_microusd":157004,"success_unresolved_calls":0,"success_unresolved_microusd":0,"successful_calls":20,"successful_microusd":157004,"uncertain_calls":0},"qualified_company_count":1,"returned_company_count":5,"sourcing_cost_eligibility_policy":"successful_calls_v1"},"eligibility_reason":"cost_per_company_exceeded","eligible":false,"final_score":0,"is_baseline":true,"rank":1,"submission_id":"baseline-2026-09-17"}],"finalists":[],"king_decision":{"king_hotkey":"","king_submission_id":null,"outcome":"no_king","winner_submission_id":null},"participants":[{"is_baseline":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","submission_id":"baseline-2026-09-17"}],"published_at":"2026-09-17T18:28:20Z","round_id":"arena-2026-09-17","schema_version":"leadpoet.lab_arena.publication.v1","stage1_ranking":[]},"published_at":"2026-09-17T18:28:20+00:00","reward_activated_at":"2026-09-17T18:28:24.628268+00:00","reward_basis_doc":{"champion_reward_factor_ppm":1000000,"effective_reward_epoch":25236,"king_hotkey":"","king_outcome":"no_king","king_start_epoch":0,"published_at":"2026-09-17T18:28:20Z","reward_basis_hash":"sha256:6b0b7d44bfbc75a5e371aee38545c5496f89fec5beb7ec4b04e7e6ee98cb5234","reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"round_id":"arena-2026-09-17","schema_version":"leadpoet.lab_arena.reward_basis.v1","signature":{"algorithm":"ECDSA_SHA_256","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","signature_b64":"MEUCIQDIrYj4BpBQ3D7aN9JngRzLDsQ7O3heC+H6mA8t/kc6KQIga423Zu2B68rRwPOb5ow9iqnMbuMgJ30aNfrbtwTGU0E="}},"reward_basis_hash":"sha256:6b0b7d44bfbc75a5e371aee38545c5496f89fec5beb7ec4b04e7e6ee98cb5234","rewards_enabled":true,"round_id":"arena-2026-09-17","signing_key_doc":{"algorithm":"ECDSA_SHA_256","key_spec":"ECC_NIST_P256","public_key_der_b64":"MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE9ASH6X2G6rxbVVZuNzvy12Wj0tMPvJ4HCdZTs2me0xUz/llK9Lp00EGdhWYSLhEEXy/YU0weeMMilC+u4jREzg==","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","schema_version":"leadpoet.lab_arena.signing_key.v1"},"stage1_scoring_plan_doc":{"round_id":"arena-2026-09-17","schema_version":"leadpoet.lab_arena.scoring_plan.v1","stage":1,"work_items":[{"icp_position":0,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:0:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:0:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":1,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:1:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:1:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":2,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:2:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:2:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":3,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:3:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:3:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":4,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:4:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:4:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":5,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:5:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:5:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":6,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:6:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:6:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":7,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:7:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:7:rerun285:2","submission_id":"baseline-2026-09-17"},{"icp_position":8,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:8:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:8:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":9,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:1:9:rerun285:2.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:1:9:rerun285:2","submission_id":"baseline-2026-09-17"}],"zero_rows":[]},"stage2_scoring_plan_doc":{"round_id":"arena-2026-09-17","schema_version":"leadpoet.lab_arena.scoring_plan.v1","stage":2,"work_items":[{"icp_position":10,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:10:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:10:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":11,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:11:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:11:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":12,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:12:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:12:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":13,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:13:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:13:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":14,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:14:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:14:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":15,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:15:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:15:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":16,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:16:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:16:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":17,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:17:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:17:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":18,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:18:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:18:rerun285:1","submission_id":"baseline-2026-09-17"},{"icp_position":19,"output_ref":"arena/arena-2026-09-17/outputs/arena-2026-09-17:baseline-2026-09-17:2:19:rerun285:1.json","scored_run_id":"arena-2026-09-17:baseline-2026-09-17:2:19:rerun285:1","submission_id":"baseline-2026-09-17"}],"zero_rows":[]},"stage3_scoring_plan_doc":null,"stage_generation":18,"status":"published","status_generation":22,"updated_at":"2026-09-17T18:28:24.629543+00:00"}$terminal_round$::JSONB;
  v_execute_cost JSONB;
  v_score_cost JSONB;
BEGIN
  IF NEW.round_id = 'arena-2026-09-17'
     AND OLD.status = 'scored' AND NEW.status = 'published'
     AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                 WHERE round_id = 'arena-2026-09-17-rerun285archive') THEN
    v_execute_cost := public.lab_arena__successful_call_cost_state(
      'baseline-2026-09-17', 'execute', NULL
    );
    v_score_cost := public.lab_arena__successful_call_cost_state(
      'baseline-2026-09-17', 'score', NULL
    );
    IF NEW.reward_basis_hash IS DISTINCT FROM
         v_terminal_round ->> 'reward_basis_hash'
       OR NEW.reward_basis_doc IS DISTINCT FROM
            v_terminal_round -> 'reward_basis_doc'
       OR NEW.signing_key_doc IS DISTINCT FROM
            v_terminal_round -> 'signing_key_doc'
       OR NEW.effective_reward_epoch IS DISTINCT FROM
            (v_terminal_round ->> 'effective_reward_epoch')::BIGINT
       OR NEW.reward_activated_at IS DISTINCT FROM
            (v_terminal_round ->> 'reward_activated_at')::TIMESTAMPTZ
       OR NEW.king_outcome IS DISTINCT FROM
            v_terminal_round ->> 'king_outcome'
       OR NEW.promotion_required IS DISTINCT FROM
            (v_terminal_round ->> 'promotion_required')::BOOLEAN
       OR NEW.promotion_doc IS DISTINCT FROM
            NULLIF(v_terminal_round -> 'promotion_doc', 'null'::JSONB)
       OR NEW.baseline_promoted_at IS DISTINCT FROM
            (v_terminal_round ->> 'baseline_promoted_at')::TIMESTAMPTZ
       OR NEW.champion_submission_id IS DISTINCT FROM
            v_terminal_round ->> 'champion_submission_id'
       OR NEW.champion_hotkey IS DISTINCT FROM
            v_terminal_round ->> 'champion_hotkey'
       OR NEW.champion_funding_frozen IS DISTINCT FROM
            (v_terminal_round ->> 'champion_funding_frozen')::BOOLEAN
       OR (NEW.configuration_doc - 'schedule') IS DISTINCT FROM
            ((v_terminal_round -> 'configuration_doc') - 'schedule')
       OR public.lab_arena_sep17_rerun285_archive_valid286_v1() IS NOT TRUE
       OR public.lab_arena_sep17_rerun286_nonbaseline_valid_v1() IS NOT TRUE
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = NEW.round_id
             AND submission_id = 'baseline-2026-09-17'
             AND kind = 'execute' AND assignment_id LIKE '%:rerun286') <> 20
       OR (SELECT pg_catalog.count(DISTINCT icp_position)
           FROM public.lab_arena_runs
           WHERE round_id = NEW.round_id
             AND submission_id = 'baseline-2026-09-17'
             AND kind = 'execute' AND assignment_id LIKE '%:rerun286'
             AND per_icp_score IS NOT NULL) <> 20
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs
         WHERE round_id = NEW.round_id
           AND submission_id = 'baseline-2026-09-17'
           AND kind = 'score'
           AND assignment_id NOT LIKE '%:score:rerun286'
       )
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs AS executed
         WHERE executed.round_id = NEW.round_id
           AND executed.submission_id = 'baseline-2026-09-17'
           AND executed.kind = 'execute'
           AND executed.assignment_id LIKE '%:rerun286'
           AND executed.status = 'accepted'
           AND NOT EXISTS (
             SELECT 1 FROM public.lab_arena_runs AS judged
             WHERE judged.round_id = executed.round_id
               AND judged.submission_id = executed.submission_id
               AND judged.kind = 'score' AND judged.status = 'accepted'
               AND judged.scored_run_id = executed.run_id
               AND judged.assignment_id LIKE '%:score:rerun286'
           )
       )
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                  WHERE round_id = NEW.round_id
                    AND status IN ('pending', 'leased', 'submitted'))
       OR COALESCE((v_execute_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
       OR COALESCE((v_execute_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
       OR COALESCE((v_score_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
       OR COALESCE((v_score_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
       THEN
      RAISE EXCEPTION 'Sep17 rerun286 publication conflicts with sealed reward, source, cost, or completion state'
        USING ERRCODE = '55000';
    END IF;
  END IF;
  RETURN NEW;
END;
$sep17_rerun286_publication_guard$;
ALTER FUNCTION public.lab_arena_sep17_rerun286_publication_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep17_rerun286_publication_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep17_rerun286_publication_guard
  ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_sep17_rerun286_publication_guard
  BEFORE UPDATE ON public.lab_arena_rounds
  FOR EACH ROW
  EXECUTE FUNCTION public.lab_arena_sep17_rerun286_publication_guard_v1();

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
