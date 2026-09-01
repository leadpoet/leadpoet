-- Recover independently protected legacy provider terminals whose old
-- dispatch-only request reference collides inside one official run.  New
-- attempt-scoped provider_request_v2 identities remain globally unique.
BEGIN;

CREATE OR REPLACE FUNCTION
    public.research_lab_official_baseline_request_replay_guard_v3()
RETURNS TRIGGER
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $validate_provider_request_replay$
DECLARE
    current_attempt
        public.research_lab_official_baseline_action_attempts_v1%ROWTYPE;
BEGIN
    IF NEW.provider_request_ref IS NULL THEN
        RETURN NEW;
    END IF;

    -- One legacy request hash can occur in multiple protected units.  Keep
    -- its compatibility decision serial without weakening v2 uniqueness.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(NEW.provider_request_ref, 0)
    );

    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_official_baseline_action_terminals_v1 prior
         WHERE prior.provider_request_ref = NEW.provider_request_ref
    ) THEN
        RETURN NEW;
    END IF;

    SELECT * INTO current_attempt
      FROM public.research_lab_official_baseline_action_attempts_v1 attempt
     WHERE attempt.attempt_key = NEW.attempt_key;
    IF NOT FOUND THEN
        RAISE EXCEPTION
            'research_lab_official_baseline_provider_request_attempt_missing'
            USING ERRCODE = '23503';
    END IF;

    -- Only the retired dispatch-hash identity has a compatibility path.
    -- Every new attempt-scoped identity remains globally unique.
    IF NEW.provider_request_ref !~ '^provider_request:[0-9a-f]{64}$'
       OR NEW.terminal_state IS DISTINCT FROM 'terminal_known'
    THEN
        RAISE EXCEPTION
            'research_lab_official_baseline_provider_request_replay_conflict'
            USING ERRCODE = '23505';
    END IF;

    -- A legacy collision is recoverable only when another protected unit in
    -- this exact official run emitted the same immutable action and dispatch.
    -- The provider response itself may legitimately differ: the old request
    -- reference identified a dispatch, not one physical provider execution.
    -- Independent provider and protected-terminal custody must therefore be
    -- distinct while provider identity and bounded accounting stay exact.
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_official_baseline_action_terminals_v1 prior
          JOIN public.research_lab_official_baseline_action_attempts_v1
               prior_attempt
            ON prior_attempt.attempt_key = prior.attempt_key
         WHERE prior.provider_request_ref = NEW.provider_request_ref
           AND prior.attempt_key IS DISTINCT FROM NEW.attempt_key
           AND prior.terminal_state = 'terminal_known'
           AND prior_attempt.run_sha256 = current_attempt.run_sha256
           AND prior_attempt.unit_ref IS DISTINCT FROM current_attempt.unit_ref
           AND prior_attempt.action_idempotency_sha256 =
                current_attempt.action_idempotency_sha256
           AND prior_attempt.action_sha256 = current_attempt.action_sha256
           AND prior_attempt.action_sequence = current_attempt.action_sequence
           AND prior_attempt.action_type = current_attempt.action_type
           AND prior_attempt.tool_id = current_attempt.tool_id
           AND prior_attempt.binding_contract_sha256 =
                current_attempt.binding_contract_sha256
           AND prior_attempt.request_fingerprint_sha256 =
                current_attempt.request_fingerprint_sha256
           AND prior_attempt.request_body_sha256 =
                current_attempt.request_body_sha256
           AND prior_attempt.call_cap = current_attempt.call_cap
           AND prior_attempt.credit_cap_microunits =
                current_attempt.credit_cap_microunits
           AND prior_attempt.timeout_ms = current_attempt.timeout_ms
           AND prior.provider_identity_sha256 = NEW.provider_identity_sha256
           AND prior.outcome = NEW.outcome
           AND prior.call_count = NEW.call_count
           AND prior.cost_microunits = NEW.cost_microunits
           AND prior.provider_receipt_ref IS DISTINCT FROM
                NEW.provider_receipt_ref
           AND prior.provider_receipt_sha256 IS DISTINCT FROM
                NEW.provider_receipt_sha256
           AND prior.protected_result_sha256 IS DISTINCT FROM
                NEW.protected_result_sha256
           AND prior.protected_terminal_receipt_ref IS DISTINCT FROM
                NEW.protected_terminal_receipt_ref
           AND prior.protected_terminal_receipt_sha256 IS DISTINCT FROM
                NEW.protected_terminal_receipt_sha256
    ) THEN
        RAISE EXCEPTION
            'research_lab_official_baseline_provider_request_replay_conflict'
            USING ERRCODE = '23505';
    END IF;

    -- Do not let one compatible row mask a conflicting collision elsewhere
    -- in the same run.  Rows from older official runs are intentionally not
    -- authority for this run's recovery decision.
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_official_baseline_action_terminals_v1 prior
          JOIN public.research_lab_official_baseline_action_attempts_v1
               prior_attempt
            ON prior_attempt.attempt_key = prior.attempt_key
         WHERE prior.provider_request_ref = NEW.provider_request_ref
           AND prior.attempt_key IS DISTINCT FROM NEW.attempt_key
           AND prior_attempt.run_sha256 = current_attempt.run_sha256
           AND (
                prior.terminal_state IS DISTINCT FROM 'terminal_known'
                OR prior_attempt.unit_ref IS NOT DISTINCT FROM
                    current_attempt.unit_ref
                OR prior_attempt.action_idempotency_sha256 IS DISTINCT FROM
                    current_attempt.action_idempotency_sha256
                OR prior_attempt.action_sha256 IS DISTINCT FROM
                    current_attempt.action_sha256
                OR prior_attempt.action_sequence IS DISTINCT FROM
                    current_attempt.action_sequence
                OR prior_attempt.action_type IS DISTINCT FROM
                    current_attempt.action_type
                OR prior_attempt.tool_id IS DISTINCT FROM current_attempt.tool_id
                OR prior_attempt.binding_contract_sha256 IS DISTINCT FROM
                    current_attempt.binding_contract_sha256
                OR prior_attempt.request_fingerprint_sha256 IS DISTINCT FROM
                    current_attempt.request_fingerprint_sha256
                OR prior_attempt.request_body_sha256 IS DISTINCT FROM
                    current_attempt.request_body_sha256
                OR prior_attempt.call_cap IS DISTINCT FROM current_attempt.call_cap
                OR prior_attempt.credit_cap_microunits IS DISTINCT FROM
                    current_attempt.credit_cap_microunits
                OR prior_attempt.timeout_ms IS DISTINCT FROM
                    current_attempt.timeout_ms
                OR prior.provider_identity_sha256 IS DISTINCT FROM
                    NEW.provider_identity_sha256
                OR prior.outcome IS DISTINCT FROM NEW.outcome
                OR prior.call_count IS DISTINCT FROM NEW.call_count
                OR prior.cost_microunits IS DISTINCT FROM NEW.cost_microunits
                OR prior.provider_receipt_ref IS NOT DISTINCT FROM
                    NEW.provider_receipt_ref
                OR prior.provider_receipt_sha256 IS NOT DISTINCT FROM
                    NEW.provider_receipt_sha256
                OR prior.protected_result_sha256 IS NOT DISTINCT FROM
                    NEW.protected_result_sha256
                OR prior.protected_terminal_receipt_ref IS NOT DISTINCT FROM
                    NEW.protected_terminal_receipt_ref
                OR prior.protected_terminal_receipt_sha256 IS NOT DISTINCT FROM
                    NEW.protected_terminal_receipt_sha256
           )
    ) THEN
        RAISE EXCEPTION
            'research_lab_official_baseline_provider_request_replay_conflict'
            USING ERRCODE = '23505';
    END IF;

    RETURN NEW;
END;
$validate_provider_request_replay$;

DROP TRIGGER IF EXISTS
    trg_research_lab_official_baseline_provider_request_replay_v2
    ON public.research_lab_official_baseline_action_terminals_v1;
DROP TRIGGER IF EXISTS
    trg_research_lab_official_baseline_provider_request_replay_v3
    ON public.research_lab_official_baseline_action_terminals_v1;
CREATE TRIGGER trg_research_lab_official_baseline_provider_request_replay_v3
BEFORE INSERT ON public.research_lab_official_baseline_action_terminals_v1
FOR EACH ROW
EXECUTE FUNCTION
    public.research_lab_official_baseline_request_replay_guard_v3();

CREATE OR REPLACE FUNCTION
    public.research_lab_official_baseline_request_scope_v3()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $official_baseline_request_scope$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_provider_request.v3',
        'identity_scope', 'protected_preparation_and_dispatch',
        'legacy_recovery',
            'same_run_exact_action_independent_protected_terminal',
        'new_identity_unique', TRUE
    );
$official_baseline_request_scope$;

REVOKE ALL ON FUNCTION
    public.research_lab_official_baseline_request_replay_guard_v3()
FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION
    public.research_lab_official_baseline_request_scope_v3()
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_official_baseline_request_scope_v3()
TO service_role;

NOTIFY pgrst, 'reload schema';

COMMIT;
