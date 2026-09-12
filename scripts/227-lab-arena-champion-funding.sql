-- Pin the winning credential owner and retain per-ICP payer decisions.
-- No credential material is copied and no accepted work is changed.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS champion_funding_frozen BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS champion_submission_id TEXT REFERENCES public.lab_arena_submissions(submission_id),
  ADD COLUMN IF NOT EXISTS champion_hotkey TEXT,
  ADD COLUMN IF NOT EXISTS champion_fallback_providers TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE public.lab_arena_runs
  ADD COLUMN IF NOT EXISTS champion_funding_sources JSONB,
  ADD COLUMN IF NOT EXISTS champion_restart_required BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE public.lab_arena_runs DROP CONSTRAINT IF EXISTS lab_arena_runs_attempt_check;
ALTER TABLE public.lab_arena_runs ADD CONSTRAINT lab_arena_runs_attempt_check
  CHECK (attempt BETWEEN 1 AND 5);

CREATE OR REPLACE FUNCTION public.lab_arena_champion_funding_immutable_v1()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog, public
AS $fn$
BEGIN
  IF TG_TABLE_NAME = 'lab_arena_rounds' THEN
    IF OLD.champion_funding_frozen AND (
      NOT NEW.champion_funding_frozen
      OR NEW.champion_submission_id IS DISTINCT FROM OLD.champion_submission_id
      OR NEW.champion_hotkey IS DISTINCT FROM OLD.champion_hotkey
    ) THEN
      RAISE EXCEPTION 'champion funding owner is immutable' USING ERRCODE = '42501';
    END IF;
    IF NOT NEW.champion_fallback_providers @> OLD.champion_fallback_providers
       OR NOT NEW.champion_fallback_providers <@ ARRAY['openrouter','deepline','scrapingdog']::TEXT[]
       OR (NEW.champion_fallback_providers <> OLD.champion_fallback_providers AND
           (OLD.status IN ('published','cancelled') OR NOT NEW.champion_funding_frozen
            OR NEW.champion_submission_id IS NULL)) THEN
      RAISE EXCEPTION 'champion fallback is monotonic within an active round' USING ERRCODE = '42501';
    END IF;
  ELSE
    IF (OLD.champion_funding_sources IS NOT NULL AND
        NEW.champion_funding_sources IS DISTINCT FROM OLD.champion_funding_sources)
       OR (OLD.champion_restart_required AND NOT NEW.champion_restart_required)
       OR (OLD.status IN ('accepted','failed') AND
           (NEW.champion_restart_required <> OLD.champion_restart_required
            OR NEW.champion_funding_sources IS DISTINCT FROM OLD.champion_funding_sources)) THEN
      RAISE EXCEPTION 'champion run funding is immutable' USING ERRCODE = '42501';
    END IF;
  END IF;
  RETURN NEW;
END;
$fn$;
ALTER FUNCTION public.lab_arena_champion_funding_immutable_v1() OWNER TO lab_arena_owner;
DROP TRIGGER IF EXISTS lab_arena_champion_round_funding ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_champion_round_funding BEFORE UPDATE ON public.lab_arena_rounds
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_champion_funding_immutable_v1();
DROP TRIGGER IF EXISTS lab_arena_champion_run_funding ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_champion_run_funding BEFORE UPDATE ON public.lab_arena_runs
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_champion_funding_immutable_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_freeze_champion_funding(p_round_id TEXT)
RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public AS $fn$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002'; END IF;
  IF v_round.champion_funding_frozen OR v_round.status <> 'open' THEN
    -- An already-running pre-migration round keeps its original funding.
    RETURN jsonb_build_object('status','existing');
  END IF;
  IF EXISTS (
    SELECT 1 FROM public.lab_arena_rounds AS prior
    WHERE prior.arena_network_name = v_round.arena_network_name
      AND prior.arena_netuid = v_round.arena_netuid
      AND prior.configuration_doc ->> 'mode' = v_round.configuration_doc ->> 'mode'
      AND (v_round.configuration_doc ->> 'mode' <> 'live' OR prior.rewards_enabled)
      AND prior.status = 'published' AND prior.promotion_required
      AND prior.publication_doc #>> '{king_decision,outcome}' = 'crowned'
      AND prior.baseline_promoted_at IS NULL
  ) THEN RETURN jsonb_build_object('status','promotion_pending'); END IF;
  -- Repository content/commit is deliberately not an ownership input.
  SELECT submission.* INTO v_submission
  FROM public.lab_arena_rounds AS prior
  JOIN public.lab_arena_submissions AS submission
    ON submission.submission_id = prior.publication_doc #>> '{king_decision,winner_submission_id}'
    AND submission.round_id = prior.round_id
    AND submission.miner_hotkey = prior.publication_doc #>> '{king_decision,king_hotkey}'
    AND submission.status = 'frozen' AND NOT submission.is_king
  WHERE prior.arena_network_name = v_round.arena_network_name
    AND prior.arena_netuid = v_round.arena_netuid
    AND prior.configuration_doc ->> 'mode' = v_round.configuration_doc ->> 'mode'
    AND (v_round.configuration_doc ->> 'mode' <> 'live' OR prior.rewards_enabled)
    AND prior.status = 'published' AND prior.baseline_promoted_at IS NOT NULL
    AND prior.publication_doc #>> '{king_decision,outcome}' = 'crowned'
    AND prior.round_id <> p_round_id
  ORDER BY prior.baseline_promoted_at DESC, prior.round_id DESC LIMIT 1;
  UPDATE public.lab_arena_rounds SET champion_funding_frozen = TRUE,
    champion_submission_id = v_submission.submission_id,
    champion_hotkey = v_submission.miner_hotkey
  WHERE round_id = p_round_id;
  RETURN jsonb_build_object('status','frozen');
END;
$fn$;
ALTER FUNCTION public.lab_arena_freeze_champion_funding(TEXT) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_provider_funding(p_run_id TEXT, p_provider TEXT)
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public AS $fn$
DECLARE
  v_run public.lab_arena_runs;
  v_round public.lab_arena_rounds;
  v_baseline BOOLEAN;
  v_champion BOOLEAN;
  v_source TEXT;
BEGIN
  IF p_provider NOT IN ('openrouter','deepline','scrapingdog') THEN
    RAISE EXCEPTION 'lab_arena_provider_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id = p_run_id;
  IF NOT FOUND THEN RETURN jsonb_build_object('status','missing'); END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = v_run.round_id;
  v_baseline := v_run.submission_id = 'baseline-' || regexp_replace(v_run.round_id,'^arena-','')
    AND v_run.miner_hotkey = v_round.configuration_doc ->> 'baseline_hotkey'
    AND EXISTS (SELECT 1 FROM public.lab_arena_submissions
                WHERE submission_id = v_run.submission_id AND is_king);
  v_champion := v_baseline AND v_run.kind = 'execute'
    AND v_round.champion_funding_frozen AND v_round.champion_submission_id IS NOT NULL
    AND v_run.champion_funding_sources IS NOT NULL;
  v_source := CASE WHEN v_champion THEN v_run.champion_funding_sources ->> p_provider
    WHEN v_baseline THEN 'host' ELSE 'miner_key' END;
  IF v_source IS NULL OR v_source NOT IN ('host','miner_key') THEN
    RAISE EXCEPTION 'lab_arena_funding_snapshot_invalid' USING ERRCODE = '23514';
  END IF;
  RETURN jsonb_build_object(
    'status','available','funding_source',v_source,'champion_funding',v_champion,
    'credential_submission_id',CASE WHEN v_source = 'miner_key' THEN
      CASE WHEN v_champion THEN v_round.champion_submission_id ELSE v_run.submission_id END END,
    'credential_miner_hotkey',CASE WHEN v_source = 'miner_key' THEN
      CASE WHEN v_champion THEN v_round.champion_hotkey ELSE v_run.miner_hotkey END END,
    'restart_required',v_champion AND (v_run.champion_restart_required OR
       (v_source = 'miner_key' AND p_provider = ANY(v_round.champion_fallback_providers)))
  );
END;
$fn$;
ALTER FUNCTION public.lab_arena_provider_funding(TEXT,TEXT) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__champion_account_failure_matches(p_call JSONB,p_evidence JSONB)
RETURNS BOOLEAN LANGUAGE sql IMMUTABLE SET search_path = pg_catalog AS $fn$
  SELECT COALESCE(
    jsonb_typeof(p_call) = 'object' AND jsonb_typeof(p_evidence) = 'object'
    AND p_evidence ->> 'error_class' = 'account_credential_failure'
    AND p_evidence ->> 'provider_status' IN ('401','402','403','429')
    AND p_evidence ->> 'base_call_identity' ~ '^sha256:[0-9a-f]{64}$'
    AND p_evidence ->> 'provider_attempt' IN ('1','2','3','4')
    AND p_evidence ->> 'action_sequence' ~ '^[0-9]+$'
    AND p_evidence -> 'base_call_identity' = p_call -> 'base_call_identity'
    AND p_evidence -> 'provider_attempt' = p_call -> 'provider_attempt'
    AND p_evidence -> 'action_sequence' = p_call -> 'action_sequence'
  ,FALSE);
$fn$;
ALTER FUNCTION public.lab_arena__champion_account_failure_matches(JSONB,JSONB) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_mark_champion_provider_fallback(
  p_run_id TEXT, p_lease_token_hash TEXT, p_provider TEXT, p_evidence JSONB
)
RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public AS $fn$
DECLARE
  v_run public.lab_arena_runs;
  v_round public.lab_arena_rounds;
  v_existing BOOLEAN;
  v_failures INTEGER;
  v_rejections INTEGER;
BEGIN
  IF p_provider NOT IN ('openrouter','deepline','scrapingdog')
     OR jsonb_typeof(p_evidence) IS DISTINCT FROM 'object' THEN
    RAISE EXCEPTION 'lab_arena_fallback_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id = p_run_id;
  IF NOT FOUND THEN RETURN jsonb_build_object('status','stale'); END IF;
  -- Match the existing lock order: round before run before ledger/submission.
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = v_run.round_id FOR UPDATE;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id,p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN RETURN jsonb_build_object('status','stale'); END;
  IF v_run.kind <> 'execute' OR NOT v_round.champion_funding_frozen
     OR v_round.champion_submission_id IS NULL
     OR v_run.champion_funding_sources ->> p_provider IS DISTINCT FROM 'miner_key'
     OR v_run.submission_id <> 'baseline-' || regexp_replace(v_run.round_id,'^arena-','')
     OR v_run.miner_hotkey IS DISTINCT FROM v_round.configuration_doc ->> 'baseline_hotkey' THEN
    RAISE EXCEPTION 'lab_arena_fallback_not_champion' USING ERRCODE = '42501';
  END IF;
  IF v_run.champion_restart_required THEN
    -- The model may swallow a failed call and request a different provider.
    -- Stop this ICP without retiring another otherwise usable credential.
    RETURN jsonb_build_object('status','existing','funding_source','host');
  END IF;
  v_existing := p_provider = ANY(v_round.champion_fallback_providers);
  IF NOT v_existing THEN
    IF p_evidence ->> 'error_class' IS DISTINCT FROM 'account_credential_failure'
       OR p_evidence ->> 'provider_attempts' IS DISTINCT FROM '4'
       OR COALESCE(p_evidence ->> 'action_sequence','') !~ '^[0-9]+$'
       OR (p_evidence -> 'provider_status' <> 'null'::JSONB AND
           p_evidence ->> 'provider_status' NOT IN ('401','402','403','429')) THEN
      RAISE EXCEPTION 'lab_arena_fallback_evidence_invalid' USING ERRCODE = '22023';
    END IF;
    IF p_evidence -> 'provider_status' = 'null'::JSONB THEN
      IF EXISTS (SELECT 1 FROM public.lab_arena_submission_credentials
                 WHERE submission_id = v_round.champion_submission_id
                   AND miner_hotkey = v_round.champion_hotkey AND provider = p_provider) THEN
        RAISE EXCEPTION 'lab_arena_fallback_credential_present' USING ERRCODE = '42501';
      END IF;
    ELSE
      SELECT COUNT(DISTINCT reservation.entry_doc ->> 'provider_attempt') INTO v_rejections
      FROM public.lab_arena_ledger AS reservation
      JOIN LATERAL (SELECT entry_kind, entry_doc, terminal_response FROM public.lab_arena_ledger
                    WHERE call_identity = reservation.call_identity
                    ORDER BY entry_id DESC LIMIT 1) AS head ON TRUE
      WHERE reservation.run_id = p_run_id AND reservation.provider = p_provider
        AND reservation.funding_source = 'miner_key' AND reservation.entry_kind = 'reservation'
        AND reservation.entry_doc ->> 'action_sequence' = p_evidence ->> 'action_sequence'
        AND reservation.entry_doc ->> 'provider_attempt' IN ('1','2','3','4')
        AND reservation.entry_doc ->> 'base_call_identity' = p_evidence ->> 'base_call_identity'
        AND (
          (head.entry_kind = 'uncertain' AND public.lab_arena__champion_account_failure_matches(
              reservation.entry_doc,head.entry_doc #> '{call,account_failure_evidence}'))
          OR (head.entry_kind = 'settlement' AND
              public.lab_arena__champion_account_failure_matches(
                reservation.entry_doc,head.terminal_response -> 'account_failure_evidence'))
        );
      -- A dynamic reservation can prevent another dispatch. Those admission
      -- retries remain distinct refusals, bound to the same account rejection.
      SELECT count(DISTINCT evidence.provider_attempt) INTO v_failures FROM (
        SELECT reservation.entry_doc ->> 'provider_attempt' AS provider_attempt
        FROM public.lab_arena_ledger AS reservation
        JOIN LATERAL (SELECT entry_kind,entry_doc,terminal_response FROM public.lab_arena_ledger
                      WHERE call_identity = reservation.call_identity ORDER BY entry_id DESC LIMIT 1) AS head ON TRUE
        WHERE reservation.run_id = p_run_id AND reservation.provider = p_provider
          AND reservation.funding_source = 'miner_key' AND reservation.entry_kind = 'reservation'
          AND reservation.entry_doc ->> 'action_sequence' = p_evidence ->> 'action_sequence'
          AND reservation.entry_doc ->> 'base_call_identity' = p_evidence ->> 'base_call_identity'
          AND ((head.entry_kind = 'uncertain' AND public.lab_arena__champion_account_failure_matches(
                  reservation.entry_doc,head.entry_doc #> '{call,account_failure_evidence}'))
               OR (head.entry_kind = 'settlement' AND public.lab_arena__champion_account_failure_matches(
                  reservation.entry_doc,head.terminal_response -> 'account_failure_evidence')))
        UNION ALL
        SELECT refusal.entry_doc #>> '{call,provider_attempt}'
        FROM public.lab_arena_ledger AS refusal
        WHERE refusal.run_id = p_run_id AND refusal.provider = p_provider
          AND refusal.funding_source = 'miner_key' AND refusal.entry_kind = 'refusal'
          AND refusal.entry_doc ->> 'prior_miner_credential_refusal' = 'true'
          AND refusal.entry_doc #>> '{call,action_sequence}' = p_evidence ->> 'action_sequence'
          AND refusal.entry_doc #>> '{call,base_call_identity}' = p_evidence ->> 'base_call_identity'
      ) AS evidence WHERE evidence.provider_attempt IN ('1','2','3','4');
      IF v_rejections < 1 OR v_failures <> 4 THEN
        RAISE EXCEPTION 'lab_arena_fallback_retries_unproved' USING ERRCODE = '42501';
      END IF;
    END IF;
    UPDATE public.lab_arena_rounds
      SET champion_fallback_providers = array_append(champion_fallback_providers,p_provider)
      WHERE round_id = v_round.round_id;
  END IF;
  UPDATE public.lab_arena_runs SET champion_restart_required = TRUE WHERE run_id = p_run_id;
  RETURN jsonb_build_object('status',CASE WHEN v_existing THEN 'existing' ELSE 'marked' END,
                            'funding_source','host');
END;
$fn$;
ALTER FUNCTION public.lab_arena_mark_champion_provider_fallback(TEXT,TEXT,TEXT,JSONB) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__champion_reward_factor(p_round_id TEXT, p_hotkey TEXT)
RETURNS INTEGER LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public AS $fn$
DECLARE
  v_round public.lab_arena_rounds;
  v_factor INTEGER := 1000000;
  v_positions INTEGER;
  v_completed INTEGER;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  -- The newly winning submission did not need the old champion's fallback.
  IF v_round.publication_doc #>> '{king_decision,outcome}' = 'crowned'
     AND p_hotkey <> ''
     AND p_hotkey IS DISTINCT FROM v_round.configuration_doc ->> 'baseline_hotkey'
     AND v_round.publication_doc #>> '{king_decision,king_hotkey}' = p_hotkey
     AND COALESCE(v_round.publication_doc #>> '{king_decision,winner_submission_id}','') <> ''
     AND v_round.publication_doc #>> '{king_decision,winner_submission_id}'
         <> 'baseline-' || regexp_replace(p_round_id,'^arena-','') THEN
    RETURN 1000000;
  END IF;
  SELECT COALESCE((prior.reward_basis_doc ->> 'champion_reward_factor_ppm')::INTEGER,1000000)
    INTO v_factor FROM public.lab_arena_rounds AS prior
  WHERE prior.reward_activated_at IS NOT NULL
    AND prior.arena_network_name = v_round.arena_network_name
    AND prior.arena_netuid = v_round.arena_netuid
    AND prior.configuration_doc ->> 'mode' = 'live'
    AND prior.reward_basis_doc ->> 'king_outcome' IN ('crowned','defended')
    AND prior.reward_basis_doc ->> 'king_hotkey' = p_hotkey
    AND p_hotkey IS DISTINCT FROM prior.configuration_doc ->> 'baseline_hotkey'
    AND p_hotkey <> ''
  ORDER BY prior.effective_reward_epoch DESC LIMIT 1;
  v_factor := COALESCE(v_factor,1000000);
  IF v_round.champion_funding_frozen AND v_round.champion_hotkey = p_hotkey
     AND v_round.champion_submission_id IS NOT NULL THEN
    IF cardinality(v_round.champion_fallback_providers) > 0 THEN RETURN 500000; END IF;
    v_positions := (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
                 + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER;
    SELECT count(DISTINCT execution.icp_position) INTO v_completed
      FROM public.lab_arena_runs AS execution
      WHERE execution.round_id = p_round_id
        AND execution.submission_id = 'baseline-' || regexp_replace(p_round_id,'^arena-','')
        AND execution.kind = 'execute' AND execution.status = 'accepted'
        AND execution.per_icp_score IS NOT NULL
        AND execution.icp_position >= 0 AND execution.icp_position < v_positions
        AND execution.stage IN (1,2)
        AND EXISTS (SELECT 1 FROM public.lab_arena_runs AS judgment
                    WHERE judgment.round_id = p_round_id AND judgment.kind = 'score'
                      AND judgment.status = 'accepted'
                      AND judgment.scored_run_id = execution.run_id);
    IF v_positions > 0 AND v_completed = v_positions THEN RETURN 1000000; END IF;
  END IF;
  -- Partial success and old host-funded rounds cannot clear an existing penalty.
  RETURN v_factor;
END;
$fn$;
ALTER FUNCTION public.lab_arena__champion_reward_factor(TEXT,TEXT) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__retired_champion_reservations(p_submission_id TEXT)
RETURNS BIGINT LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public AS $fn$
  -- Still present in every cost/eligibility total. Only admission of replacement
  -- work may stop reserving funds for these retired miner credentials.
  SELECT COALESCE(sum(head.amount_microusd),0)::BIGINT
  FROM public.lab_arena_ledger AS reservation
  JOIN public.lab_arena_runs AS run ON run.run_id = reservation.run_id
  JOIN public.lab_arena_rounds AS round ON round.round_id = run.round_id
  JOIN LATERAL (SELECT * FROM public.lab_arena_ledger
                WHERE call_identity = reservation.call_identity ORDER BY entry_id DESC LIMIT 1) AS head ON TRUE
  WHERE reservation.submission_id = p_submission_id AND reservation.entry_kind = 'reservation'
    AND reservation.funding_source = 'miner_key' AND run.kind = 'execute'
    AND run.submission_id = 'baseline-' || regexp_replace(run.round_id,'^arena-','')
    AND run.miner_hotkey = round.configuration_doc ->> 'baseline_hotkey'
    AND round.champion_funding_frozen AND round.champion_submission_id IS NOT NULL
    AND reservation.provider = ANY(round.champion_fallback_providers)
    AND run.champion_funding_sources ->> reservation.provider = 'miner_key'
    AND head.entry_kind = 'uncertain'
    AND public.lab_arena__champion_account_failure_matches(
      reservation.entry_doc,head.entry_doc #> '{call,account_failure_evidence}');
$fn$;
ALTER FUNCTION public.lab_arena__retired_champion_reservations(TEXT) OWNER TO lab_arena_owner;

-- Extend the current definitions only at their reviewed seams. This preserves
-- the company judgment, late settlement, lease and participation fixes.
DO $patch$
DECLARE v_definition TEXT; v_old TEXT; v_new TEXT;
BEGIN
  SELECT pg_get_functiondef('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure) INTO v_definition;
  IF strpos(v_definition,'champion_funding_sources') = 0 THEN
    v_old := 'SET status = ''leased'', runner_hotkey = p_runner_hotkey';
    v_new := $new$SET champion_funding_sources = CASE
        WHEN v_run.kind = 'execute' AND v_submission.is_king
          AND v_run.submission_id = 'baseline-' || regexp_replace(v_run.round_id,'^arena-','')
          AND v_run.miner_hotkey = v_round.configuration_doc ->> 'baseline_hotkey'
          AND v_round.champion_funding_frozen AND v_round.champion_submission_id IS NOT NULL
        THEN (SELECT jsonb_object_agg(provider,CASE WHEN provider = ANY(v_round.champion_fallback_providers)
                THEN 'host' ELSE 'miner_key' END)
              FROM unnest(ARRAY['openrouter','deepline','scrapingdog']) AS provider)
        ELSE NULL END,
      status = 'leased', runner_hotkey = p_runner_hotkey$new$;
    IF strpos(v_definition,v_old) = 0 THEN RAISE EXCEPTION 'champion claim shape unexpected'; END IF;
    EXECUTE replace(v_definition,v_old,v_new);
  END IF;
  SELECT pg_get_functiondef('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure) INTO v_definition;
  IF strpos(v_definition,'lab_arena_provider_funding') = 0 THEN
    v_old := $old$IF (v_is_baseline AND p_funding_source <> 'host')
     OR (NOT v_is_baseline AND p_funding_source <> 'miner_key') THEN$old$;
    v_new := $new$IF p_funding_source IS DISTINCT FROM
       (public.lab_arena_provider_funding(p_run_id,p_provider) ->> 'funding_source') THEN$new$;
    IF strpos(v_definition,v_old) = 0 THEN RAISE EXCEPTION 'champion reserve shape unexpected'; END IF;
    EXECUTE replace(v_definition,v_old,v_new);
  END IF;
  SELECT pg_get_functiondef('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure) INTO v_definition;
  IF strpos(v_definition,'lab_arena__retired_champion_reservations') = 0 THEN
    v_old := $old$  v_quota := CASE v_run.kind$old$;
    v_new := $new$  IF v_run.champion_restart_required OR
       (v_run.champion_funding_sources ->> p_provider = 'miner_key'
        AND p_provider = ANY(v_round.champion_fallback_providers)) THEN
      RETURN jsonb_build_object('status','champion_restart_required');
    END IF;
  v_quota := CASE v_run.kind$new$;
    IF strpos(v_definition,v_old) = 0 THEN RAISE EXCEPTION 'champion reserve restart shape unexpected'; END IF;
    v_definition := replace(v_definition,v_old,v_new);
    v_old := $old$      v_spent := public.lab_arena__submission_kind_spend(
        v_run.submission_id, v_run.kind
      );$old$;
    v_new := v_old || $new$
      IF v_is_baseline AND v_run.kind = 'execute' THEN
        v_spent := GREATEST(0,v_spent - public.lab_arena__retired_champion_reservations(v_run.submission_id));
      END IF;$new$;
    IF strpos(v_definition,v_old) = 0 THEN RAISE EXCEPTION 'champion replacement allowance shape unexpected'; END IF;
    EXECUTE replace(v_definition,v_old,v_new);
  END IF;
  SELECT pg_get_functiondef('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure) INTO v_definition;
  IF strpos(v_definition,'champion_retry_family') = 0 THEN
    v_old := '      ) INTO v_prior_miner_credential_refusal;';
    v_new := v_old || $new$
      IF v_run.champion_funding_sources IS NOT NULL THEN
        -- champion_retry_family: only this operation's authenticated account
        -- rejection can justify a blocked credential retry, including 429.
        SELECT EXISTS (
          SELECT 1 FROM public.lab_arena_ledger AS reservation
          JOIN LATERAL (SELECT entry_kind,entry_doc FROM public.lab_arena_ledger
                        WHERE call_identity = reservation.call_identity ORDER BY entry_id DESC LIMIT 1) AS head ON TRUE
          WHERE reservation.run_id = p_run_id AND reservation.provider = p_provider
            AND reservation.funding_source = 'miner_key' AND reservation.entry_kind = 'reservation'
            AND reservation.entry_doc -> 'base_call_identity' = p_call_doc -> 'base_call_identity'
            AND reservation.entry_doc -> 'action_sequence' = p_call_doc -> 'action_sequence'
            AND head.entry_kind = 'uncertain'
            AND public.lab_arena__champion_account_failure_matches(
              reservation.entry_doc,head.entry_doc #> '{call,account_failure_evidence}')
        ) INTO v_prior_miner_credential_refusal;
      END IF;$new$;
    IF strpos(v_definition,v_old) = 0 THEN RAISE EXCEPTION 'apply 214-lab-arena-prior-credential-refusal.sql before champion funding'; END IF;
    EXECUTE replace(v_definition,v_old,v_new);
  END IF;
  SELECT pg_get_functiondef('public.lab_arena__call_state_view(public.lab_arena_ledger,public.lab_arena_runs)'::regprocedure) INTO v_definition;
  IF strpos(v_definition,'account_failure_evidence') = 0 THEN
    v_old := $old$    'lease_expires_at', p_run.lease_expires_at
  );$old$;
    v_new := $new$    'lease_expires_at', p_run.lease_expires_at
  ) || CASE WHEN p_head.entry_kind = 'uncertain'
                 AND p_head.entry_doc #> '{call,account_failure_evidence}' IS NOT NULL
       THEN jsonb_build_object('account_failure_evidence',p_head.entry_doc #> '{call,account_failure_evidence}')
       ELSE '{}'::JSONB END;$new$;
    IF strpos(v_definition,v_old) = 0 THEN RAISE EXCEPTION 'champion replay shape unexpected'; END IF;
    EXECUTE replace(v_definition,v_old,v_new);
  END IF;
  SELECT pg_get_functiondef('public.lab_arena_complete_attempt(text,text,jsonb,text,text)'::regprocedure) INTO v_definition;
  IF strpos(v_definition,'champion_restart_required') = 0 THEN
    v_old := '  SELECT COUNT(*) INTO v_open FROM (';
    v_new := $new$  IF v_run.champion_restart_required THEN
    -- A swallowed provider error cannot turn partial work into an accepted ICP.
    p_terminal_cause := 'provider_error';
    p_result := jsonb_set(p_result,'{terminal_status}','"provider_error"'::JSONB);
    p_output_ref := '';
  END IF;
  SELECT COUNT(*) INTO v_open FROM ($new$;
    IF strpos(v_definition,v_old) = 0 THEN RAISE EXCEPTION 'champion complete shape unexpected'; END IF;
    v_definition := replace(v_definition,v_old,v_new);
    v_old := 'AND v_run.attempt < 2 THEN';
    v_new := $new$AND v_run.attempt < LEAST(5, 2 +
       (SELECT count(*) FROM public.lab_arena_runs AS failed_run
        WHERE failed_run.assignment_id = v_run.assignment_id AND failed_run.champion_restart_required)) THEN$new$;
    IF strpos(v_definition,v_old) = 0 THEN RAISE EXCEPTION 'champion complete retry shape unexpected'; END IF;
    EXECUTE replace(v_definition,v_old,v_new);
  END IF;
  SELECT pg_get_functiondef('public.lab_arena_expire_leases(text)'::regprocedure) INTO v_definition;
  IF strpos(v_definition,'champion_restart_required') = 0 THEN
    v_old := 'v_run.attempt < 2';
    v_new := $new$v_run.attempt < LEAST(5, 2 +
       (SELECT count(*) FROM public.lab_arena_runs AS failed_run
        WHERE failed_run.assignment_id = v_run.assignment_id AND failed_run.champion_restart_required))$new$;
    IF strpos(v_definition,v_old) = 0 THEN RAISE EXCEPTION 'champion expiry shape unexpected'; END IF;
    EXECUTE replace(v_definition,v_old,v_new);
  END IF;
  SELECT pg_get_functiondef('public.lab_arena_activate_reward(text,jsonb,jsonb)'::regprocedure) INTO v_definition;
  IF strpos(v_definition,'lab_arena__champion_reward_factor') = 0 THEN
    v_old := '  v_effective := (p_reward_basis ->> ''effective_reward_epoch'')::BIGINT;';
    v_new := $new$  IF COALESCE((p_reward_basis ->> 'champion_reward_factor_ppm')::NUMERIC,1000000)
       IS DISTINCT FROM public.lab_arena__champion_reward_factor(p_round_id,v_expected_hotkey) THEN
      RAISE EXCEPTION 'lab_arena_champion_reward_factor_mismatch' USING ERRCODE = '22023';
    END IF;
  v_effective := (p_reward_basis ->> 'effective_reward_epoch')::BIGINT;$new$;
    IF strpos(v_definition,v_old) = 0 THEN RAISE EXCEPTION 'champion reward shape unexpected'; END IF;
    EXECUTE replace(v_definition,v_old,v_new);
  END IF;
END;
$patch$;

CREATE OR REPLACE FUNCTION public.lab_arena_champion_funding_schema_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog AS $fn$
  SELECT jsonb_build_object('version',227,'provider_attempts',4,'reward_factor_ppm',500000);
$fn$;
ALTER FUNCTION public.lab_arena_champion_funding_schema_v1() OWNER TO lab_arena_owner;
DO $acl$
DECLARE v_signature TEXT; v_role TEXT;
BEGIN
  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena_freeze_champion_funding(text)',
    'public.lab_arena_provider_funding(text,text)',
    'public.lab_arena_mark_champion_provider_fallback(text,text,text,jsonb)',
    'public.lab_arena_champion_funding_schema_v1()',
    'public.lab_arena__champion_reward_factor(text,text)',
    'public.lab_arena__retired_champion_reservations(text)',
    'public.lab_arena__champion_account_failure_matches(jsonb,jsonb)',
    'public.lab_arena_champion_funding_immutable_v1()'
  ] LOOP
    EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC',v_signature);
    FOREACH v_role IN ARRAY ARRAY['anon','authenticated','service_role'] LOOP
      IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = v_role) THEN
        EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %I',v_signature,v_role);
      END IF;
    END LOOP;
    EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service',v_signature);
  END LOOP;
END;
$acl$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
