-- Durable exact replay for deterministic SOURCE_ADD catalog snapshots.
--
-- Every measured model invocation consumes the same epoch-bound catalog. A
-- coordinator recycle must reuse that already-attested result rather than
-- execute the same logical job again and collide with immutable receipt
-- uniqueness. This widens only the execution-result replay allowlist.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_attested_execution_results_v2'::REGCLASS
          AND contype = 'c'
          AND (
              pg_catalog.pg_get_constraintdef(oid) ~ '\moperation\M'
              OR pg_catalog.pg_get_constraintdef(oid) ~ '\mpurpose\M'
          )
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_attested_execution_results_v2 '
            'DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$$;

ALTER TABLE public.research_lab_attested_execution_results_v2
    ADD CONSTRAINT
        research_lab_attested_execution_results_v2_operation_check
    CHECK (
        operation IN (
            'research_lab_allocation',
            'allocation_settlement_frontier_bootstrap_v2',
            'attest_weight_input',
            'attest_active_private_model',
            'source_add_catalog_snapshot_v2',
            'observe_chain_realized_weights_v1',
            'attest_chain_realized_settlement_v1'
        )
    ) NOT VALID,
    ADD CONSTRAINT
        research_lab_attested_execution_results_v2_purpose_check
    CHECK (
        purpose IN (
            'research_lab.allocation.v2',
            'research_lab.allocation_settlement_frontier_bootstrap.v2',
            'research_lab.champion_input.v2',
            'research_lab.reimbursement_input.v2',
            'research_lab.source_add_reward_input.v2',
            'research_lab.sourcing_input.v2',
            'research_lab.fulfillment_input.v2',
            'research_lab.leaderboard_input.v2',
            'research_lab.ban_input.v2',
            'research_lab.anomaly_adjustment_input.v2',
            'research_lab.active_private_model.v2',
            'research_lab.source_add_catalog_snapshot.v2',
            'research_lab.chain_weight_observation.v1',
            'research_lab.chain_realized_epoch_settlement.v1'
        )
    ) NOT VALID,
    ADD CONSTRAINT
        research_lab_attested_exec_results_v2_op_purpose_check
    CHECK (
        (
            operation = 'research_lab_allocation'
            AND purpose = 'research_lab.allocation.v2'
        )
        OR (
            operation = 'allocation_settlement_frontier_bootstrap_v2'
            AND purpose =
                'research_lab.allocation_settlement_frontier_bootstrap.v2'
        )
        OR (
            operation = 'attest_weight_input'
            AND purpose IN (
                'research_lab.allocation.v2',
                'research_lab.champion_input.v2',
                'research_lab.reimbursement_input.v2',
                'research_lab.source_add_reward_input.v2',
                'research_lab.sourcing_input.v2',
                'research_lab.fulfillment_input.v2',
                'research_lab.leaderboard_input.v2',
                'research_lab.ban_input.v2',
                'research_lab.anomaly_adjustment_input.v2'
            )
        )
        OR (
            operation = 'attest_active_private_model'
            AND purpose = 'research_lab.active_private_model.v2'
        )
        OR (
            operation = 'source_add_catalog_snapshot_v2'
            AND purpose = 'research_lab.source_add_catalog_snapshot.v2'
        )
        OR (
            operation = 'observe_chain_realized_weights_v1'
            AND purpose = 'research_lab.chain_weight_observation.v1'
        )
        OR (
            operation = 'attest_chain_realized_settlement_v1'
            AND purpose =
                'research_lab.chain_realized_epoch_settlement.v1'
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_attested_execution_results_v2
    VALIDATE CONSTRAINT
        research_lab_attested_execution_results_v2_operation_check;
ALTER TABLE public.research_lab_attested_execution_results_v2
    VALIDATE CONSTRAINT
        research_lab_attested_execution_results_v2_purpose_check;
ALTER TABLE public.research_lab_attested_execution_results_v2
    VALIDATE CONSTRAINT
        research_lab_attested_exec_results_v2_op_purpose_check;

CREATE OR REPLACE FUNCTION
public.research_lab_source_catalog_replay_contract_v2()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT jsonb_build_object(
        'schema_version', 'leadpoet.source_catalog_replay_contract.v2',
        'operation', 'source_add_catalog_snapshot_v2',
        'purpose', 'research_lab.source_add_catalog_snapshot.v2',
        'constraints', COALESCE(
            (
                SELECT jsonb_object_agg(
                    constraint_record.conname,
                    jsonb_build_object(
                        'constraint_definition',
                        pg_get_constraintdef(constraint_record.oid),
                        'constraint_valid', constraint_record.convalidated
                    )
                    ORDER BY constraint_record.conname
                )
                FROM pg_constraint AS constraint_record
                WHERE constraint_record.conrelid =
                      'public.research_lab_attested_execution_results_v2'::REGCLASS
                  AND constraint_record.conname IN (
                      'research_lab_attested_execution_results_v2_operation_check',
                      'research_lab_attested_execution_results_v2_purpose_check',
                      'research_lab_attested_exec_results_v2_op_purpose_check'
                  )
            ),
            '{}'::JSONB
        )
    );
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_catalog_replay_contract_v2()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_catalog_replay_contract_v2()
    TO service_role;

COMMENT ON FUNCTION
    public.research_lab_source_catalog_replay_contract_v2() IS
    'Reports the validated exact-result replay contract for deterministic SOURCE_ADD catalog snapshots.';

NOTIFY pgrst, 'reload schema';

COMMIT;

-- Verify after applying:
-- SELECT public.research_lab_source_catalog_replay_contract_v2();
