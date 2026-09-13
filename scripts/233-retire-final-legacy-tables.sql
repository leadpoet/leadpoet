-- Migration 233: retire the final operator-approved legacy tables and their
-- database-only callers. Preserve the complete SOURCE_ADD audit subset in a
-- private immutable archive before removing public.transparency_log.

BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

CREATE TEMP TABLE _retire_final_legacy_tables (
    table_name TEXT PRIMARY KEY
) ON COMMIT DROP;

INSERT INTO _retire_final_legacy_tables (table_name) VALUES
    ('company_information_table'),
    ('early_access_emails'),
    ('outreach_email_verifications'),
    ('research_lab_official_baseline_action_attempts_v1'),
    ('research_lab_official_baseline_action_terminals_v1'),
    ('research_lab_official_baseline_runs_v1'),
    ('research_lab_official_baseline_unit_closures_v1'),
    ('research_lab_public_loop_card_events'),
    ('research_lab_public_loop_cards'),
    ('suppression_ledger'),
    ('transparency_log'),
    ('validation_evidence_private');

CREATE TEMP TABLE _retire_final_legacy_views (
    view_name TEXT PRIMARY KEY
) ON COMMIT DROP;

INSERT INTO _retire_final_legacy_views (view_name) VALUES
    ('research_lab_public_loop_card_current');

CREATE TEMP TABLE _retire_final_legacy_routines (
    routine_name TEXT NOT NULL,
    identity_arguments TEXT NOT NULL,
    expected_body_md5 TEXT NOT NULL,
    PRIMARY KEY (routine_name, identity_arguments)
) ON COMMIT DROP;

INSERT INTO _retire_final_legacy_routines (
    routine_name,
    identity_arguments,
    expected_body_md5
) VALUES
    ('prevent_research_lab_official_baseline_mutation_v1', '', '3fffea0072be65d57f3888e43ec7a8f2'),
    ('refresh_dashboard_precalc', '', '9d8bbd09aeeddedd247f8e4dc3b851e3'),
    ('refresh_dashboard_precalc_full', '', 'ee21184528511bc0110218ffcf9e25ca'),
    ('research_lab_official_baseline_close_unit_v1', 'p_completion jsonb', '86e8ac27879ba8bdb3cabe6033e2c87d'),
    ('research_lab_official_baseline_exact_keys_v1', 'p_value jsonb, p_keys text[]', 'ea1fa5f47d60923161a1431df51837d9'),
    ('research_lab_official_baseline_hash_v1', 'p_value jsonb', 'e94fad1822bf833dab7f03db1b607c11'),
    ('research_lab_official_baseline_load_frontier_v1', 'p_run_sha256 text, p_unit_ref text', 'b9cf4350aa638d1c4f83e24289c2ded3'),
    ('research_lab_official_baseline_load_replay_v1', 'p_identity jsonb', '2d958c901aededfad2362a2563418d7c'),
    ('research_lab_official_baseline_provider_frontier_doc_v1', 'p_run_sha256 text, p_unit_ref text', '2657c4b6e259255de351946ca8448fd5'),
    ('research_lab_official_baseline_record_terminal_known_v1', 'p_terminal jsonb', 'ad7d7946170c8edd3772ec2ac425d3c1'),
    ('research_lab_official_baseline_record_terminal_uncertain_v1', 'p_terminal jsonb', '949fd24fb4d5380e592348ff3debeba9'),
    ('research_lab_official_baseline_register_run_v1', 'p_registration jsonb', '581e8b1e506ab325215b1d859edaf47f'),
    ('research_lab_official_baseline_reject_secret_doc_v1', 'p_value jsonb, p_label text', 'feb2aa9b4ad2c829e360052f5ef43ff3'),
    ('research_lab_official_baseline_request_replay_guard_v2', '', 'e587ad0fe83448d88a301c9ab2018b0a'),
    ('research_lab_official_baseline_request_replay_guard_v3', '', '7e7d65c3f9851d09364bda212fac317c'),
    ('research_lab_official_baseline_request_scope_v2', '', '2e97641acba7b9bcd4a55773aef36cd0'),
    ('research_lab_official_baseline_request_scope_v3', '', '0c4e6ed3f72d62635462798b7c08e112'),
    ('research_lab_official_baseline_reserve_action_v1', 'p_authorization jsonb', 'ee663a9d056841501ba870f4285b2cb9'),
    ('sync_miner_stats_to_rows', '', 'ef0d84fedb42fa79464e3d33221da0c3'),
    ('update_suppression_timestamp', '', '06bcf30ac3d0a7f279a54cbf228a7bec');

-- Validate the exact reviewed closure before any persistent change. Stored
-- routine bodies and cron command text do not have complete pg_depend edges.
DO $retirement_guard$
DECLARE
    target RECORD;
    expected RECORD;
    preserved_name TEXT;
    unexpected_name TEXT;
    closure_pattern TEXT;
    routine_oid OID;
    actual_md5 TEXT;
BEGIN
    IF (SELECT count(*) FROM _retire_final_legacy_tables) <> 12 THEN
        RAISE EXCEPTION 'final retirement allowlist must contain exactly 12 tables';
    END IF;
    IF (SELECT count(*) FROM _retire_final_legacy_views) <> 1 THEN
        RAISE EXCEPTION 'final retirement allowlist must contain exactly one view';
    END IF;
    IF (SELECT count(*) FROM _retire_final_legacy_routines) <> 20 THEN
        RAISE EXCEPTION 'final retirement allowlist must contain exactly 20 routines';
    END IF;

    FOREACH preserved_name IN ARRAY ARRAY[
        'dashboard_miner_stats',
        'dashboard_precalc',
        'lab_arena_accepted_weight_states',
        'lab_arena_chain_outcomes',
        'lab_arena_company_judgment_reservations',
        'lab_arena_company_judgments',
        'lab_arena_judgment_cache',
        'lab_arena_ledger',
        'lab_arena_restart_claim_control',
        'lab_arena_rounds',
        'lab_arena_runs',
        'lab_arena_submission_credentials',
        'lab_arena_submissions',
        'merkle_checkpoints',
        'published_weight_bundles',
        'qualification_baselines',
        'qualification_private_icp_sets',
        'research_evaluation_score_bundles',
        'research_lab_attested_execution_receipts',
        'research_lab_attested_ancestry_activations_v2',
        'research_lab_attested_ancestry_checkpoints_v2',
        'research_lab_attested_artifact_links_v2',
        'research_lab_attested_boot_identities_v2',
        'research_lab_attested_business_artifact_links_v2',
        'research_lab_attested_execution_receipts_v2',
        'research_lab_attested_execution_results_v2',
        'research_lab_attested_host_operations_v2',
        'research_lab_attested_receipt_edges_v2',
        'research_lab_attested_receipt_transport_v2',
        'research_lab_attested_transport_attempts_v2',
        'research_lab_stateful_subnet_epoch_boundaries_v1',
        'research_lab_stateful_subnet_epoch_candidates_v1',
        'research_lab_stateful_subnet_epoch_cutover_state_v1',
        'research_lab_stateful_subnet_epoch_cutovers_v1',
        'research_lab_stateful_subnet_epoch_snapshots_v1'
    ] LOOP
        IF pg_catalog.to_regclass('public.' || preserved_name) IS NULL THEN
            RAISE EXCEPTION 'required preserved relation public.% is missing',
                preserved_name;
        END IF;
    END LOOP;

    FOR target IN
        SELECT allowlist.table_name, relation.relkind
          FROM _retire_final_legacy_tables AS allowlist
          LEFT JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.nspname = 'public'
          LEFT JOIN pg_catalog.pg_class AS relation
            ON relation.relnamespace = namespace.oid
           AND relation.relname = allowlist.table_name
         ORDER BY allowlist.table_name
    LOOP
        IF target.relkind IS NOT NULL AND target.relkind NOT IN ('r', 'p') THEN
            RAISE EXCEPTION 'refusing to retire public.% because it is not a table',
                target.table_name;
        END IF;
    END LOOP;

    FOR expected IN SELECT * FROM _retire_final_legacy_routines LOOP
        SELECT routine.oid, pg_catalog.md5(routine.prosrc)
          INTO routine_oid, actual_md5
          FROM pg_catalog.pg_proc AS routine
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = routine.pronamespace
         WHERE namespace.nspname = 'public'
           AND routine.proname = expected.routine_name
           AND pg_catalog.pg_get_function_identity_arguments(routine.oid) =
               expected.identity_arguments;
        IF routine_oid IS NULL THEN
            CONTINUE;
        END IF;
        IF actual_md5 IS DISTINCT FROM expected.expected_body_md5 THEN
            RAISE EXCEPTION 'retirement routine public.%(%) changed after review',
                expected.routine_name, expected.identity_arguments;
        END IF;
        routine_oid := NULL;
        actual_md5 := NULL;
    END LOOP;

    routine_oid := pg_catalog.to_regprocedure(
        'public.enforce_research_lab_stateful_epoch_fence_v1()'
    );
    IF routine_oid IS NULL THEN
        RAISE EXCEPTION 'required shared stateful epoch fence is missing';
    END IF;
    SELECT pg_catalog.md5(routine.prosrc)
      INTO actual_md5
      FROM pg_catalog.pg_proc AS routine
     WHERE routine.oid = routine_oid;
    IF actual_md5 NOT IN (
        '59fe7131073dabd508d7195975c77c5a',
        '5b48baac95474f877b66f84deb78d8f4'
    ) THEN
        RAISE EXCEPTION 'shared stateful epoch fence changed after review: %',
            actual_md5;
    END IF;

    SELECT '(^|[^a-zA-Z0-9_])(' || pg_catalog.string_agg(name, '|') ||
           ')([^a-zA-Z0-9_]|$)'
      INTO closure_pattern
      FROM (
          SELECT table_name AS name FROM _retire_final_legacy_tables
          UNION
          SELECT view_name FROM _retire_final_legacy_views
          UNION
          SELECT routine_name FROM _retire_final_legacy_routines
      ) AS closure_names;

    SELECT pg_catalog.format('%I.%I', candidate.schemaname, candidate.viewname)
      INTO unexpected_name
      FROM (
          SELECT schemaname, viewname, definition FROM pg_catalog.pg_views
          UNION ALL
          SELECT schemaname, matviewname, definition FROM pg_catalog.pg_matviews
      ) AS candidate
     WHERE candidate.schemaname <> 'information_schema'
       AND candidate.schemaname !~ '^pg_'
       AND candidate.definition ~* closure_pattern
       AND NOT (
           candidate.schemaname = 'public'
           AND EXISTS (
               SELECT 1 FROM _retire_final_legacy_views AS allowed
                WHERE allowed.view_name = candidate.viewname
           )
       )
     LIMIT 1;
    IF unexpected_name IS NOT NULL THEN
        RAISE EXCEPTION 'unexpected view depends on final retirement closure: %',
            unexpected_name;
    END IF;

    SELECT pg_catalog.format(
               '%I.%I(%s)',
               namespace.nspname,
               routine.proname,
               pg_catalog.pg_get_function_identity_arguments(routine.oid)
           )
      INTO unexpected_name
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
     WHERE routine.prokind IN ('f', 'p')
       AND namespace.nspname <> 'information_schema'
       AND namespace.nspname !~ '^pg_'
       AND pg_catalog.pg_get_functiondef(routine.oid) ~* closure_pattern
       AND NOT (
           namespace.nspname = 'public'
           AND (
               routine.proname = 'enforce_research_lab_stateful_epoch_fence_v1'
               OR EXISTS (
                   SELECT 1 FROM _retire_final_legacy_routines AS allowed
                    WHERE allowed.routine_name = routine.proname
                      AND allowed.identity_arguments =
                          pg_catalog.pg_get_function_identity_arguments(routine.oid)
               )
           )
       )
     LIMIT 1;
    IF unexpected_name IS NOT NULL THEN
        RAISE EXCEPTION 'unexpected routine depends on final retirement closure: %',
            unexpected_name;
    END IF;

    SELECT pg_catalog.format('%I on %I.%I', trigger.tgname,
                             namespace.nspname, relation.relname)
      INTO unexpected_name
      FROM pg_catalog.pg_trigger AS trigger
      JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger.tgrelid
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = relation.relnamespace
      JOIN pg_catalog.pg_proc AS routine ON routine.oid = trigger.tgfoid
      JOIN _retire_final_legacy_routines AS retired
        ON retired.routine_name = routine.proname
       AND retired.identity_arguments =
           pg_catalog.pg_get_function_identity_arguments(routine.oid)
     WHERE NOT trigger.tgisinternal
       AND NOT (
           namespace.nspname = 'public'
           AND EXISTS (
               SELECT 1 FROM _retire_final_legacy_tables AS target_table
                WHERE target_table.table_name = relation.relname
           )
       )
     LIMIT 1;
    IF unexpected_name IS NOT NULL THEN
        RAISE EXCEPTION 'retired routine is used by a retained trigger: %',
            unexpected_name;
    END IF;

    IF pg_catalog.to_regclass('cron.job') IS NOT NULL THEN
        EXECUTE $cron_guard$
            SELECT pg_catalog.format('%s:%s', jobid, jobname)
              FROM cron.job
             WHERE command ~* $1
               AND jobname <> 'refresh-dashboard-precalc'
             ORDER BY jobid
             LIMIT 1
        $cron_guard$
        INTO unexpected_name
        USING closure_pattern;
        IF unexpected_name IS NOT NULL THEN
            RAISE EXCEPTION 'unexpected cron job depends on retirement closure: %',
                unexpected_name;
        END IF;

        EXECUTE $cron_hash$
            SELECT pg_catalog.md5(command)
              FROM cron.job
             WHERE jobname = 'refresh-dashboard-precalc'
             ORDER BY jobid
             LIMIT 1
        $cron_hash$
        INTO actual_md5;
        IF actual_md5 IS NOT NULL
           AND actual_md5 <> '3ff9ca8d232d8ceed4fd6019a6d22cc5' THEN
            RAISE EXCEPTION 'refresh-dashboard-precalc cron changed after review: %',
                actual_md5;
        END IF;
    END IF;
END;
$retirement_guard$;

-- The archive stores only reviewed SOURCE_ADD history, keeps the complete
-- original row as JSONB, and is not exposed through the public API schema.
CREATE SCHEMA IF NOT EXISTS source_add_history;
REVOKE ALL ON SCHEMA source_add_history FROM PUBLIC, anon, authenticated,
    service_role;

CREATE TABLE IF NOT EXISTS source_add_history.legacy_audit_rows (
    source_relation TEXT NOT NULL,
    source_pk TEXT NOT NULL,
    row_doc JSONB NOT NULL,
    row_md5 TEXT NOT NULL CHECK (row_md5 ~ '^[0-9a-f]{32}$'),
    archived_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (source_relation, source_pk),
    CHECK (row_md5 = pg_catalog.md5(row_doc::TEXT))
);

REVOKE ALL ON TABLE source_add_history.legacy_audit_rows
    FROM PUBLIC, anon, authenticated, service_role;

CREATE TABLE IF NOT EXISTS source_add_history.archive_manifests (
    source_relation TEXT PRIMARY KEY,
    archive_floor BIGINT NOT NULL,
    historical_ceiling BIGINT NOT NULL,
    historical_row_count BIGINT NOT NULL,
    archived_row_count BIGINT NOT NULL,
    min_source_pk BIGINT NOT NULL,
    max_source_pk BIGINT NOT NULL,
    source_row_fingerprint TEXT NOT NULL
        CHECK (source_row_fingerprint ~ '^[0-9a-f]{32}$'),
    audit_row_count BIGINT NOT NULL,
    source_add_marker_row_count BIGINT NOT NULL,
    missing_nonzero_parent_count BIGINT NOT NULL,
    missing_parent_fingerprint TEXT
        CHECK (
            missing_parent_fingerprint IS NULL
            OR missing_parent_fingerprint ~ '^[0-9a-f]{32}$'
        ),
    source_columns JSONB NOT NULL,
    archived_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

REVOKE ALL ON TABLE source_add_history.archive_manifests
    FROM PUBLIC, anon, authenticated, service_role;

CREATE OR REPLACE FUNCTION source_add_history.reject_legacy_audit_mutation_v1()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION 'source_add_history.legacy_audit_rows is append-only';
END;
$$;

REVOKE ALL ON FUNCTION source_add_history.reject_legacy_audit_mutation_v1()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS reject_legacy_audit_mutation_v1
    ON source_add_history.legacy_audit_rows;
CREATE TRIGGER reject_legacy_audit_mutation_v1
    BEFORE UPDATE OR DELETE ON source_add_history.legacy_audit_rows
    FOR EACH ROW
    EXECUTE FUNCTION source_add_history.reject_legacy_audit_mutation_v1();

DROP TRIGGER IF EXISTS reject_legacy_audit_mutation_v1
    ON source_add_history.archive_manifests;
CREATE TRIGGER reject_legacy_audit_mutation_v1
    BEFORE UPDATE OR DELETE ON source_add_history.archive_manifests
    FOR EACH ROW
    EXECUTE FUNCTION source_add_history.reject_legacy_audit_mutation_v1();

-- The schema and archive tables are security boundaries. Refuse a same-name
-- object with a different owner, kind, column contract, or primary key.
DO $archive_shape_guard$
DECLARE
    actual_columns TEXT[];
    actual_primary_key TEXT[];
BEGIN
    IF (
        SELECT pg_catalog.pg_get_userbyid(namespace.nspowner)
        FROM pg_catalog.pg_namespace AS namespace
        WHERE namespace.nspname = 'source_add_history'
    ) IS DISTINCT FROM CURRENT_USER THEN
        RAISE EXCEPTION 'source_add_history must be owned by the migration role';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class AS relation
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'source_add_history'
          AND relation.relname = 'legacy_audit_rows'
          AND relation.relkind = 'r'
          AND pg_catalog.pg_get_userbyid(relation.relowner) = CURRENT_USER
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD row archive has an unsafe owner or kind';
    END IF;

    SELECT pg_catalog.array_agg(
               attribute.attname || ':' ||
               pg_catalog.format_type(attribute.atttypid, attribute.atttypmod) ||
               ':' || attribute.attnotnull::TEXT
               ORDER BY attribute.attnum
           )
      INTO actual_columns
      FROM pg_catalog.pg_attribute AS attribute
     WHERE attribute.attrelid =
           'source_add_history.legacy_audit_rows'::REGCLASS
       AND attribute.attnum > 0
       AND NOT attribute.attisdropped;
    IF actual_columns IS DISTINCT FROM ARRAY[
        'source_relation:text:true',
        'source_pk:text:true',
        'row_doc:jsonb:true',
        'row_md5:text:true',
        'archived_at:timestamp with time zone:true'
    ]::TEXT[] THEN
        RAISE EXCEPTION 'SOURCE_ADD row archive shape differs: %', actual_columns;
    END IF;

    SELECT pg_catalog.array_agg(attribute.attname ORDER BY key.ordinality)
      INTO actual_primary_key
      FROM pg_catalog.pg_constraint AS constraint_row
      CROSS JOIN LATERAL pg_catalog.unnest(constraint_row.conkey)
          WITH ORDINALITY AS key(attnum, ordinality)
      JOIN pg_catalog.pg_attribute AS attribute
        ON attribute.attrelid = constraint_row.conrelid
       AND attribute.attnum = key.attnum
     WHERE constraint_row.conrelid =
           'source_add_history.legacy_audit_rows'::REGCLASS
       AND constraint_row.contype = 'p';
    IF actual_primary_key IS DISTINCT FROM
       ARRAY['source_relation', 'source_pk']::TEXT[] THEN
        RAISE EXCEPTION 'SOURCE_ADD row archive primary key differs: %',
            actual_primary_key;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class AS relation
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'source_add_history'
          AND relation.relname = 'archive_manifests'
          AND relation.relkind = 'r'
          AND pg_catalog.pg_get_userbyid(relation.relowner) = CURRENT_USER
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD archive manifest has an unsafe owner or kind';
    END IF;
END;
$archive_shape_guard$;

-- The reviewed archive is the complete transparency-log suffix beginning at
-- the recursively closed SOURCE_ADD boundary. The historical part ends at
-- 41982623; rows appended before this transaction are also archived.
LOCK TABLE source_add_history.legacy_audit_rows IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE source_add_history.archive_manifests IN SHARE ROW EXCLUSIVE MODE;

DO $archive_source_add_history$
DECLARE
    archive_floor CONSTANT BIGINT := 41960931;
    historical_ceiling CONSTANT BIGINT := 41982623;
    expected_historical_rows CONSTANT BIGINT := 21660;
    source_exists BOOLEAN;
    historical_rows BIGINT;
    historical_min_id BIGINT;
    historical_max_id BIGINT;
    historical_audit_rows BIGINT;
    historical_marker_rows BIGINT;
    source_rows BIGINT;
    source_min_id BIGINT;
    source_max_id BIGINT;
    source_fingerprint TEXT;
    archive_rows BIGINT;
    archive_min_id BIGINT;
    archive_max_id BIGINT;
    archive_fingerprint TEXT;
    outside_existing_parents BIGINT;
    missing_nonzero_parents BIGINT;
    missing_parent_fingerprint TEXT;
    weight_audit_rows BIGINT;
    weight_reference_rows BIGINT;
    weight_companion_rows BIGINT;
    source_columns JSONB;
    stored_manifest source_add_history.archive_manifests%ROWTYPE;
BEGIN
    source_exists := pg_catalog.to_regclass('public.transparency_log') IS NOT NULL;

    IF source_exists THEN
        IF EXISTS (
            SELECT 1
            FROM source_add_history.archive_manifests
            WHERE source_relation = 'public.transparency_log'
        ) OR EXISTS (
            SELECT 1
            FROM source_add_history.legacy_audit_rows
            WHERE source_relation = 'public.transparency_log'
        ) THEN
            RAISE EXCEPTION
                'refusing to mix a live transparency_log with a prior archive';
        END IF;

        EXECUTE 'LOCK TABLE public.transparency_log IN ACCESS EXCLUSIVE MODE';

        SELECT count(*), min(id), max(id),
               count(*) FILTER (
                   WHERE event_type = 'RESEARCH_LAB_EPOCH_AUDIT'
               ),
               count(*) FILTER (
                   WHERE event_type = 'RESEARCH_LAB_EPOCH_AUDIT'
                     AND (
                         payload::TEXT ~* 'source[_. -]?add'
                         OR signed_log_entry::TEXT ~* 'source[_. -]?add'
                     )
               )
          INTO historical_rows, historical_min_id, historical_max_id,
               historical_audit_rows, historical_marker_rows
          FROM public.transparency_log
         WHERE id BETWEEN archive_floor AND historical_ceiling;

        IF historical_rows <> expected_historical_rows
           OR historical_min_id <> archive_floor
           OR historical_max_id <> historical_ceiling
           OR historical_audit_rows <> 324
           OR historical_marker_rows <> 152 THEN
            RAISE EXCEPTION
                'historical SOURCE_ADD boundary changed: rows %, ids %..%, audits %, markers %',
                historical_rows, historical_min_id, historical_max_id,
                historical_audit_rows, historical_marker_rows;
        END IF;

        SELECT count(*), min(id), max(id),
               pg_catalog.md5(pg_catalog.string_agg(
                   pg_catalog.md5(pg_catalog.to_jsonb(source_row)::TEXT),
                   '' ORDER BY id
               ))
          INTO source_rows, source_min_id, source_max_id, source_fingerprint
          FROM public.transparency_log AS source_row
         WHERE id >= archive_floor;

        IF source_rows < expected_historical_rows OR source_rows > 100000
           OR source_min_id <> archive_floor
           OR source_max_id < historical_ceiling THEN
            RAISE EXCEPTION
                'SOURCE_ADD archive suffix is outside reviewed bounds: rows %, ids %..%',
                source_rows, source_min_id, source_max_id;
        END IF;

        IF NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_index AS index_row
            JOIN pg_catalog.pg_attribute AS attribute
              ON attribute.attrelid = index_row.indrelid
             AND attribute.attnum = index_row.indkey[0]
            WHERE index_row.indrelid = 'public.transparency_log'::REGCLASS
              AND index_row.indisunique
              AND index_row.indisvalid
              AND index_row.indisready
              AND index_row.indnatts = 1
              AND index_row.indpred IS NULL
              AND attribute.attname = 'event_hash'
        ) THEN
            RAISE EXCEPTION
                'transparency_log event_hash index required for bounded closure check';
        END IF;

        PERFORM pg_catalog.set_config('enable_hashjoin', 'off', TRUE);
        PERFORM pg_catalog.set_config('enable_mergejoin', 'off', TRUE);
        SELECT
            count(*) FILTER (WHERE parent_row.id < archive_floor),
            count(*) FILTER (
                WHERE parent_row.id IS NULL
                  AND child_row.prev_event_hash !~ '^(sha256:)?0+$'
            ),
            pg_catalog.md5(pg_catalog.string_agg(
                child_row.prev_event_hash, ',' ORDER BY child_row.id
            ) FILTER (
                WHERE parent_row.id IS NULL
                  AND child_row.prev_event_hash !~ '^(sha256:)?0+$'
            ))
          INTO outside_existing_parents, missing_nonzero_parents,
               missing_parent_fingerprint
          FROM public.transparency_log AS child_row
          LEFT JOIN public.transparency_log AS parent_row
            ON parent_row.event_hash = child_row.prev_event_hash
         WHERE child_row.id >= archive_floor
           AND child_row.prev_event_hash IS NOT NULL;
        PERFORM pg_catalog.set_config('enable_hashjoin', 'on', TRUE);
        PERFORM pg_catalog.set_config('enable_mergejoin', 'on', TRUE);

        IF outside_existing_parents <> 0 OR missing_nonzero_parents <> 4 THEN
            RAISE EXCEPTION
                'SOURCE_ADD signed-chain boundary changed: outside %, missing %',
                outside_existing_parents, missing_nonzero_parents;
        END IF;

        SELECT count(*),
               count(weight_event_hash),
               count(companion.id)
          INTO weight_audit_rows, weight_reference_rows, weight_companion_rows
          FROM (
              SELECT COALESCE(
                         audit.payload #>>
                             '{weights,weight_submission_event_hash}',
                         audit.signed_log_entry #>>
                             '{signed_event,payload,weights,weight_submission_event_hash}'
                     ) AS weight_event_hash
              FROM public.transparency_log AS audit
              WHERE audit.id >= archive_floor
                AND audit.event_type = 'RESEARCH_LAB_EPOCH_AUDIT'
          ) AS source_audit
          LEFT JOIN public.transparency_log AS companion
            ON companion.event_hash = source_audit.weight_event_hash
           AND companion.id >= archive_floor;
        IF weight_audit_rows < 324
           OR weight_reference_rows <> weight_audit_rows
           OR weight_companion_rows <> weight_audit_rows THEN
            RAISE EXCEPTION
                'SOURCE_ADD weight closure changed: audits %, references %, companions %',
                weight_audit_rows, weight_reference_rows, weight_companion_rows;
        END IF;

        IF EXISTS (
            SELECT 1
            FROM public.merkle_checkpoints
            WHERE seq_end >= archive_floor
        ) THEN
            RAISE EXCEPTION
                'a generic Merkle checkpoint intersects the SOURCE_ADD suffix';
        END IF;

        SELECT pg_catalog.jsonb_agg(
                   pg_catalog.jsonb_build_object(
                       'ordinal_position', attribute.attnum,
                       'column_name', attribute.attname,
                       'data_type', pg_catalog.format_type(
                           attribute.atttypid, attribute.atttypmod
                       ),
                       'not_null', attribute.attnotnull,
                       'default_expression', pg_catalog.pg_get_expr(
                           default_row.adbin, default_row.adrelid
                       ),
                       'identity', attribute.attidentity,
                       'generated', attribute.attgenerated
                   ) ORDER BY attribute.attnum
               )
          INTO source_columns
          FROM pg_catalog.pg_attribute AS attribute
          LEFT JOIN pg_catalog.pg_attrdef AS default_row
            ON default_row.adrelid = attribute.attrelid
           AND default_row.adnum = attribute.attnum
         WHERE attribute.attrelid = 'public.transparency_log'::REGCLASS
           AND attribute.attnum > 0
           AND NOT attribute.attisdropped;

        INSERT INTO source_add_history.legacy_audit_rows (
            source_relation, source_pk, row_doc, row_md5
        )
        SELECT 'public.transparency_log', source_row.id::TEXT,
               pg_catalog.to_jsonb(source_row),
               pg_catalog.md5(pg_catalog.to_jsonb(source_row)::TEXT)
          FROM public.transparency_log AS source_row
         WHERE source_row.id >= archive_floor
         ORDER BY source_row.id;

        SELECT count(*), min(source_pk::BIGINT), max(source_pk::BIGINT),
               pg_catalog.md5(pg_catalog.string_agg(
                   pg_catalog.md5(row_doc::TEXT),
                   '' ORDER BY source_pk::BIGINT
               ))
          INTO archive_rows, archive_min_id, archive_max_id,
               archive_fingerprint
          FROM source_add_history.legacy_audit_rows
         WHERE source_relation = 'public.transparency_log';

        IF (archive_rows, archive_min_id, archive_max_id, archive_fingerprint)
           IS DISTINCT FROM
           (source_rows, source_min_id, source_max_id, source_fingerprint) THEN
            RAISE EXCEPTION
                'SOURCE_ADD full-row archive does not equal its locked source';
        END IF;

        INSERT INTO source_add_history.archive_manifests (
            source_relation, archive_floor, historical_ceiling,
            historical_row_count, archived_row_count, min_source_pk,
            max_source_pk, source_row_fingerprint, audit_row_count,
            source_add_marker_row_count, missing_nonzero_parent_count,
            missing_parent_fingerprint, source_columns
        ) VALUES (
            'public.transparency_log', archive_floor, historical_ceiling,
            historical_rows, source_rows, source_min_id, source_max_id,
            source_fingerprint, historical_audit_rows, historical_marker_rows,
            missing_nonzero_parents, missing_parent_fingerprint, source_columns
        );
    ELSE
        SELECT * INTO stored_manifest
        FROM source_add_history.archive_manifests
        WHERE source_relation = 'public.transparency_log';
        IF NOT FOUND THEN
            RAISE EXCEPTION
                'transparency_log is absent without a SOURCE_ADD archive manifest';
        END IF;

        IF EXISTS (
            SELECT 1
            FROM source_add_history.legacy_audit_rows
            WHERE source_relation = 'public.transparency_log'
              AND source_pk !~ '^[0-9]+$'
        ) THEN
            RAISE EXCEPTION 'stored SOURCE_ADD archive has a nonnumeric key';
        END IF;

        SELECT count(*), min(source_pk::BIGINT), max(source_pk::BIGINT),
               pg_catalog.md5(pg_catalog.string_agg(
                   pg_catalog.md5(row_doc::TEXT),
                   '' ORDER BY source_pk::BIGINT
               ))
          INTO archive_rows, archive_min_id, archive_max_id,
               archive_fingerprint
          FROM source_add_history.legacy_audit_rows
         WHERE source_relation = 'public.transparency_log';
        IF stored_manifest.archive_floor <> archive_floor
           OR stored_manifest.historical_ceiling <> historical_ceiling
           OR stored_manifest.historical_row_count <> expected_historical_rows
           OR stored_manifest.audit_row_count <> 324
           OR stored_manifest.source_add_marker_row_count <> 152
           OR stored_manifest.missing_nonzero_parent_count <> 4
           OR stored_manifest.source_columns IS NULL
           OR (archive_rows, archive_min_id, archive_max_id,
               archive_fingerprint) IS DISTINCT FROM
              (stored_manifest.archived_row_count,
               stored_manifest.min_source_pk,
               stored_manifest.max_source_pk,
               stored_manifest.source_row_fingerprint) THEN
            RAISE EXCEPTION 'stored SOURCE_ADD archive failed idempotence checks';
        END IF;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM source_add_history.legacy_audit_rows
        WHERE source_relation <> 'public.transparency_log'
           OR source_pk !~ '^[0-9]+$'
           OR row_md5 <> pg_catalog.md5(row_doc::TEXT)
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD archive contains an unexpected row';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM source_add_history.legacy_audit_rows
        WHERE source_pk::BIGINT < archive_floor
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD archive contains a row below its floor';
    END IF;
END;
$archive_source_add_history$;

-- Retire the reviewed scheduled cache refresh. Existing cache rows stay.
DO $unschedule_dashboard_refresh$
DECLARE
    legacy_job RECORD;
BEGIN
    IF pg_catalog.to_regclass('cron.job') IS NULL THEN
        RETURN;
    END IF;
    FOR legacy_job IN EXECUTE
        'SELECT jobid, jobname FROM cron.job '
        || 'WHERE jobname = ''refresh-dashboard-precalc'' ORDER BY jobid'
    LOOP
        IF NOT cron.unschedule(legacy_job.jobid) THEN
            RAISE EXCEPTION 'could not unschedule legacy cron job % (%)',
                legacy_job.jobname, legacy_job.jobid;
        END IF;
    END LOOP;
END;
$unschedule_dashboard_refresh$;

-- Keep the shared stateful epoch fence for retained Arena and weight tables,
-- but remove both branches that existed only for transparency_log.
CREATE OR REPLACE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    row_doc JSONB := pg_catalog.to_jsonb(NEW);
    identity_key TEXT;
    identity_text TEXT;
    identity_value BIGINT;
    payload JSONB;
    authority_value JSONB;
    has_stateful_receipt BOOLEAN;
    linked_bundle RECORD;
BEGIN
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'stateful epoch cutover singleton state is missing';
    END IF;
    IF state_row.lifecycle_state IN ('cutover_fenced', 'stateful_staged') THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
        SELECT * INTO state_row
        FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
        WHERE singleton;
    END IF;

    IF TG_TABLE_NAME = 'research_lab_stateful_subnet_epoch_candidates_v1' THEN
        IF state_row.lifecycle_state = 'legacy_open' THEN
            RAISE EXCEPTION
                'stateful epoch candidate requires the durable pre-boundary fence';
        END IF;
        IF state_row.lifecycle_state = 'cutover_fenced'
           AND state_row.mapping_hash IS NULL
           AND row_doc->>'network_genesis_hash' = state_row.network_genesis_hash
           AND (row_doc->>'netuid')::INTEGER = state_row.netuid
           AND (row_doc->>'proposed_settlement_epoch_id')::INTEGER =
               state_row.first_settlement_epoch_id THEN
            RETURN NEW;
        END IF;
        IF row_doc->>'mapping_hash' IS NOT DISTINCT FROM state_row.mapping_hash
           AND row_doc->>'snapshot_hash' IS NOT DISTINCT FROM
              state_row.candidate_snapshot_hash
           AND row_doc->>'chain_state_receipt_hash' IS NOT DISTINCT FROM
              state_row.candidate_receipt_hash THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence rejects an unbound candidate';
    ELSIF TG_TABLE_NAME = 'research_lab_stateful_subnet_epoch_cutovers_v1' THEN
        IF state_row.cutover_receipt_hash IS NOT NULL
           AND state_row.initialization_nonce IS NOT NULL
           AND state_row.initialization_payload_hash IS NOT NULL
           AND row_doc->>'mapping_hash' IS NOT DISTINCT FROM state_row.mapping_hash
           AND row_doc->>'cutover_authority_hash' IS NOT DISTINCT FROM
              state_row.cutover_authority_hash
           AND row_doc->>'cutover_receipt_hash' IS NOT DISTINCT FROM
              state_row.cutover_receipt_hash
           AND (row_doc->>'first_settlement_epoch_id')::BIGINT IS NOT DISTINCT FROM
              state_row.first_settlement_epoch_id::BIGINT THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence requires the atomic staged cutover RPC';
    ELSIF TG_TABLE_NAME IN (
        'research_lab_stateful_subnet_epoch_boundaries_v1',
        'research_lab_stateful_subnet_epoch_snapshots_v1'
    ) THEN
        IF state_row.lifecycle_state = 'stateful_active' THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence rejects boundary/snapshot writes before activation';
    END IF;

    IF state_row.lifecycle_state = 'legacy_open' THEN
        RETURN NEW;
    END IF;

    IF state_row.lifecycle_state = 'stateful_active' THEN
        FOR authority_value IN
            SELECT value
            FROM pg_catalog.jsonb_path_query(
                row_doc,
                'strict $.**.cutover_mapping_hash'
            ) AS authority_item(value)
        LOOP
            IF pg_catalog.jsonb_typeof(authority_value) <> 'string'
               OR authority_value #>> '{}' <> state_row.mapping_hash THEN
                RAISE EXCEPTION
                    'stateful epoch active mapping authority differs on %',
                    TG_TABLE_NAME;
            END IF;
        END LOOP;

        IF TG_TABLE_NAME IN (
            'published_weight_bundles',
            'research_lab_attested_execution_receipts',
            'research_lab_attested_weight_bundles',
            'research_lab_legacy_finalized_allocation_migrations_v2'
        ) THEN
            FOREACH identity_key IN ARRAY ARRAY[
                'epoch', 'epoch_id', 'evaluation_epoch'
            ]
            LOOP
                IF row_doc ? identity_key
                   AND pg_catalog.jsonb_typeof(row_doc->identity_key) <> 'null'
                   AND row_doc->>identity_key ~ '^[0-9]+$'
                   AND (row_doc->>identity_key)::BIGINT >=
                       state_row.first_settlement_epoch_id::BIGINT THEN
                    RAISE EXCEPTION
                        'stateful epoch active namespace rejects legacy %.% identity %',
                        TG_TABLE_NAME,
                        identity_key,
                        row_doc->>identity_key;
                END IF;
            END LOOP;
            RETURN NEW;
        END IF;

        IF TG_TABLE_NAME = 'research_lab_attested_execution_receipts_v2'
           AND row_doc->>'epoch_id' ~ '^[0-9]+$'
           AND (row_doc->>'epoch_id')::BIGINT >=
               state_row.first_settlement_epoch_id::BIGINT
           AND row_doc->>'purpose' IN (
               'validator.weight_snapshot.v2',
               'validator.weights.computed.v2',
               'validator.weights.finalized.v2',
               'gateway.weights.publication.v2'
           )
           AND NOT EXISTS (
               SELECT 1
               FROM public.research_lab_attested_execution_receipts_v2 AS receipt
               WHERE receipt.epoch_id = (row_doc->>'epoch_id')::INTEGER
                 AND receipt.role = 'validator_weights'
                 AND receipt.purpose = 'validator.subnet_epoch_snapshot.v2'
                 AND receipt.receipt_status = 'succeeded'
           ) THEN
            RAISE EXCEPTION
                'stateful epoch active V2 receipt lacks subnet authority %',
                row_doc->>'epoch_id';
        END IF;

        IF TG_TABLE_NAME = 'research_lab_attested_weight_bundles_v2'
           AND row_doc->>'epoch_id' ~ '^[0-9]+$'
           AND (row_doc->>'epoch_id')::BIGINT >=
               state_row.first_settlement_epoch_id::BIGINT THEN
            payload := row_doc->'bundle_doc'->'receipt_graph'->'receipts';
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.jsonb_array_elements(payload) AS graph_receipt(doc)
                JOIN public.research_lab_attested_execution_receipts_v2 AS receipt
                  ON receipt.receipt_hash = graph_receipt.doc->>'receipt_hash'
                WHERE graph_receipt.doc->>'role' = 'validator_weights'
                  AND graph_receipt.doc->>'purpose' =
                      'validator.subnet_epoch_snapshot.v2'
                  AND graph_receipt.doc->>'epoch_id' ~ '^[0-9]+$'
                  AND (graph_receipt.doc->>'epoch_id')::BIGINT =
                      (row_doc->>'epoch_id')::BIGINT
                  AND receipt.epoch_id = (row_doc->>'epoch_id')::INTEGER
                  AND receipt.role = 'validator_weights'
                  AND receipt.purpose = 'validator.subnet_epoch_snapshot.v2'
                  AND receipt.receipt_status = 'succeeded'
            ) INTO has_stateful_receipt
            WHERE pg_catalog.jsonb_typeof(payload) = 'array';
            IF NOT COALESCE(has_stateful_receipt, FALSE) THEN
                RAISE EXCEPTION
                    'stateful epoch active V2 bundle lacks subnet authority %',
                    row_doc->>'epoch_id';
            END IF;
        END IF;

        IF TG_TABLE_NAME = 'research_lab_attested_weight_finalizations_v2' THEN
            SELECT bundle.epoch_id, bundle.bundle_doc
            INTO linked_bundle
            FROM public.research_lab_attested_weight_bundles_v2 AS bundle
            WHERE bundle.bundle_hash = row_doc->>'bundle_hash';
            IF FOUND
               AND linked_bundle.epoch_id >=
                   state_row.first_settlement_epoch_id THEN
                payload := linked_bundle.bundle_doc->'receipt_graph'->'receipts';
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.jsonb_array_elements(payload)
                         AS graph_receipt(doc)
                    JOIN public.research_lab_attested_execution_receipts_v2 AS receipt
                      ON receipt.receipt_hash = graph_receipt.doc->>'receipt_hash'
                    WHERE graph_receipt.doc->>'purpose' =
                          'validator.subnet_epoch_snapshot.v2'
                      AND graph_receipt.doc->>'epoch_id' ~ '^[0-9]+$'
                      AND (graph_receipt.doc->>'epoch_id')::INTEGER =
                          linked_bundle.epoch_id
                      AND receipt.epoch_id = linked_bundle.epoch_id
                      AND receipt.purpose = 'validator.subnet_epoch_snapshot.v2'
                      AND receipt.receipt_status = 'succeeded'
                ) INTO has_stateful_receipt
                WHERE pg_catalog.jsonb_typeof(payload) = 'array';
                IF NOT COALESCE(has_stateful_receipt, FALSE) THEN
                    RAISE EXCEPTION
                        'stateful epoch active V2 finalization lacks subnet authority %',
                        linked_bundle.epoch_id;
                END IF;
            END IF;
        END IF;

        RETURN NEW;
    END IF;

    IF TG_TABLE_NAME = 'research_lab_attested_execution_receipts_v2'
       AND row_doc ? 'epoch_id'
       AND row_doc->>'epoch_id' ~ '^[0-9]+$'
       AND (row_doc->>'epoch_id')::BIGINT >=
           state_row.first_settlement_epoch_id::BIGINT THEN
        IF row_doc->>'role' = 'validator_weights'
           AND row_doc->>'purpose' = 'validator.subnet_epoch_snapshot.v2'
           AND row_doc->>'receipt_status' = 'succeeded'
           AND row_doc->>'output_root' ~ '^sha256:[0-9a-f]{64}$'
           AND (row_doc->>'epoch_id')::BIGINT =
               state_row.first_settlement_epoch_id::BIGINT
           AND (
               state_row.candidate_receipt_hash IS NULL
               OR (
                   row_doc->>'receipt_hash' = state_row.candidate_receipt_hash
                   AND row_doc->>'output_root' = state_row.candidate_snapshot_hash
               )
           ) THEN
            RETURN NEW;
        END IF;
        IF row_doc->>'role' = 'gateway_coordinator'
           AND row_doc->>'purpose' = 'research_lab.subnet_epoch_cutover.v2'
           AND row_doc->>'receipt_status' = 'succeeded'
           AND row_doc->>'output_root' = state_row.cutover_authority_hash
           AND (row_doc->>'epoch_id')::BIGINT =
               state_row.first_settlement_epoch_id::BIGINT
           AND pg_catalog.jsonb_typeof(
                  row_doc->'receipt_doc'->'parent_receipt_hashes'
               ) = 'array'
           AND pg_catalog.jsonb_array_length(
                  row_doc->'receipt_doc'->'parent_receipt_hashes'
               ) = 2
           AND row_doc->'receipt_doc'->'parent_receipt_hashes'
               @> pg_catalog.jsonb_build_array(
                   state_row.candidate_receipt_hash,
                   state_row.last_legacy_finalization_receipt_hash
               ) THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence rejects receipt epoch identity %',
            row_doc->>'epoch_id';
    END IF;

    FOREACH identity_key IN ARRAY ARRAY['epoch', 'epoch_id', 'evaluation_epoch']
    LOOP
        IF NOT (row_doc ? identity_key)
           OR pg_catalog.jsonb_typeof(row_doc->identity_key) = 'null' THEN
            CONTINUE;
        END IF;
        identity_text := row_doc->>identity_key;
        IF identity_text !~ '^-?[0-9]+$' THEN
            RAISE EXCEPTION 'stateful epoch fence rejects malformed %.% identity',
                TG_TABLE_NAME, identity_key;
        END IF;
        identity_value := identity_text::BIGINT;
        IF identity_value >= state_row.first_settlement_epoch_id::BIGINT THEN
            RAISE EXCEPTION 'stateful epoch fence rejects %.% identity %',
                TG_TABLE_NAME, identity_key, identity_value;
        END IF;
    END LOOP;
    RETURN NEW;
END;
$$;

-- Drop the only reviewed view dependency before its base tables.
DROP VIEW IF EXISTS public.research_lab_public_loop_card_current;

-- Drop callers before helpers. No CASCADE is used; an unreviewed dependency
-- aborts the transaction.
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_close_unit_v1(JSONB);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_load_frontier_v1(TEXT, TEXT);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_load_replay_v1(JSONB);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_record_terminal_known_v1(JSONB);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_record_terminal_uncertain_v1(JSONB);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_register_run_v1(JSONB);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_reserve_action_v1(JSONB);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_provider_frontier_doc_v1(TEXT, TEXT);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_exact_keys_v1(JSONB, TEXT[]);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_reject_secret_doc_v1(JSONB, TEXT);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_request_replay_guard_v2();
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_request_scope_v2();
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_request_scope_v3();

DROP FUNCTION IF EXISTS public.refresh_dashboard_precalc();
DROP FUNCTION IF EXISTS public.refresh_dashboard_precalc_full();
DROP FUNCTION IF EXISTS public.sync_miner_stats_to_rows();

-- Children precede parents. Owned constraints, indexes, policies, triggers,
-- sequences, and publication membership disappear with their exact table.
DROP TABLE IF EXISTS public.research_lab_official_baseline_action_terminals_v1;
DROP TABLE IF EXISTS public.research_lab_official_baseline_action_attempts_v1;
DROP TABLE IF EXISTS public.research_lab_official_baseline_unit_closures_v1;
DROP TABLE IF EXISTS public.research_lab_official_baseline_runs_v1;
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_hash_v1(JSONB);
DROP FUNCTION IF EXISTS public.research_lab_official_baseline_request_replay_guard_v3();
DROP FUNCTION IF EXISTS public.prevent_research_lab_official_baseline_mutation_v1();

DROP TABLE IF EXISTS public.research_lab_public_loop_card_events;
DROP TABLE IF EXISTS public.research_lab_public_loop_cards;

DROP TABLE IF EXISTS public.company_information_table;
DROP TABLE IF EXISTS public.early_access_emails;
DROP TABLE IF EXISTS public.outreach_email_verifications;
DROP TABLE IF EXISTS public.suppression_ledger;
DROP TABLE IF EXISTS public.transparency_log;
DROP TABLE IF EXISTS public.validation_evidence_private;

DROP SEQUENCE IF EXISTS public.company_information_table_id_seq;
DROP SEQUENCE IF EXISTS public.transparency_log_id_seq;
DROP FUNCTION IF EXISTS public.update_suppression_timestamp();

-- Re-scan every non-system routine and cron command. This catches dynamic SQL
-- and private-schema callers that pg_depend cannot represent.
DO $assert_no_hidden_callers$
DECLARE
    closure_pattern TEXT;
    unexpected_name TEXT;
BEGIN
    SELECT '(^|[^a-zA-Z0-9_])(' || pg_catalog.string_agg(name, '|') ||
           ')([^a-zA-Z0-9_]|$)'
      INTO closure_pattern
      FROM (
          SELECT table_name AS name FROM _retire_final_legacy_tables
          UNION
          SELECT view_name FROM _retire_final_legacy_views
          UNION
          SELECT routine_name FROM _retire_final_legacy_routines
      ) AS closure_names;

    SELECT pg_catalog.format(
               '%I.%I(%s)',
               namespace.nspname,
               routine.proname,
               pg_catalog.pg_get_function_identity_arguments(routine.oid)
           )
      INTO unexpected_name
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
     WHERE routine.prokind IN ('f', 'p')
       AND namespace.nspname <> 'information_schema'
       AND namespace.nspname !~ '^pg_'
       AND pg_catalog.pg_get_functiondef(routine.oid) ~* closure_pattern
     LIMIT 1;
    IF unexpected_name IS NOT NULL THEN
        RAISE EXCEPTION 'remaining routine references final retirement closure: %',
            unexpected_name;
    END IF;

    IF pg_catalog.to_regclass('cron.job') IS NOT NULL THEN
        EXECUTE $cron_scan$
            SELECT pg_catalog.format('%s:%s', jobid, jobname)
              FROM cron.job
             WHERE command ~* $1
             ORDER BY jobid
             LIMIT 1
        $cron_scan$
        INTO unexpected_name
        USING closure_pattern;
        IF unexpected_name IS NOT NULL THEN
            RAISE EXCEPTION 'remaining cron job references final retirement closure: %',
                unexpected_name;
        END IF;
    END IF;
END;
$assert_no_hidden_callers$;

DO $assert_final_retirement_complete$
DECLARE
    target_name TEXT;
    archive_columns TEXT[];
    manifest_columns TEXT[];
BEGIN
    FOREACH target_name IN ARRAY ARRAY[
        'company_information_table',
        'company_information_table_id_seq',
        'early_access_emails',
        'outreach_email_verifications',
        'research_lab_official_baseline_action_attempts_v1',
        'research_lab_official_baseline_action_terminals_v1',
        'research_lab_official_baseline_runs_v1',
        'research_lab_official_baseline_unit_closures_v1',
        'research_lab_public_loop_card_current',
        'research_lab_public_loop_card_events',
        'research_lab_public_loop_cards',
        'suppression_ledger',
        'transparency_log',
        'transparency_log_id_seq',
        'validation_evidence_private'
    ] LOOP
        IF pg_catalog.to_regclass('public.' || target_name) IS NOT NULL THEN
            RAISE EXCEPTION 'legacy relation public.% remains', target_name;
        END IF;
    END LOOP;

    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_proc AS routine
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = routine.pronamespace
          JOIN _retire_final_legacy_routines AS retired
            ON retired.routine_name = routine.proname
           AND retired.identity_arguments =
               pg_catalog.pg_get_function_identity_arguments(routine.oid)
         WHERE namespace.nspname = 'public'
    ) THEN
        RAISE EXCEPTION 'a reviewed legacy routine remains';
    END IF;

    IF pg_catalog.pg_get_functiondef(
        'public.enforce_research_lab_stateful_epoch_fence_v1()'::REGPROCEDURE
    ) ~* '(^|[^a-zA-Z0-9_])transparency_log([^a-zA-Z0-9_]|$)' THEN
        RAISE EXCEPTION 'shared stateful epoch fence still names transparency_log';
    END IF;

    SELECT pg_catalog.array_agg(
               attribute.attname ORDER BY attribute.attnum
           )
      INTO archive_columns
      FROM pg_catalog.pg_attribute AS attribute
     WHERE attribute.attrelid =
           'source_add_history.legacy_audit_rows'::REGCLASS
       AND attribute.attnum > 0
       AND NOT attribute.attisdropped;
    IF archive_columns IS DISTINCT FROM ARRAY[
        'source_relation', 'source_pk', 'row_doc', 'row_md5', 'archived_at'
    ]::TEXT[] THEN
        RAISE EXCEPTION 'SOURCE_ADD archive shape differs: %', archive_columns;
    END IF;

    SELECT pg_catalog.array_agg(
               attribute.attname ORDER BY attribute.attnum
           )
      INTO manifest_columns
      FROM pg_catalog.pg_attribute AS attribute
     WHERE attribute.attrelid =
           'source_add_history.archive_manifests'::REGCLASS
       AND attribute.attnum > 0
       AND NOT attribute.attisdropped;
    IF manifest_columns IS DISTINCT FROM ARRAY[
        'source_relation', 'archive_floor', 'historical_ceiling',
        'historical_row_count', 'archived_row_count', 'min_source_pk',
        'max_source_pk', 'source_row_fingerprint', 'audit_row_count',
        'source_add_marker_row_count', 'missing_nonzero_parent_count',
        'missing_parent_fingerprint', 'source_columns', 'archived_at'
    ]::TEXT[] THEN
        RAISE EXCEPTION 'SOURCE_ADD archive manifest shape differs: %',
            manifest_columns;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_namespace AS namespace
        CROSS JOIN LATERAL pg_catalog.aclexplode(
            COALESCE(
                namespace.nspacl,
                pg_catalog.acldefault('n', namespace.nspowner)
            )
        ) AS privilege
        WHERE namespace.nspname = 'source_add_history'
          AND privilege.grantee <> namespace.nspowner
    ) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class AS relation
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        CROSS JOIN LATERAL pg_catalog.aclexplode(
            COALESCE(
                relation.relacl,
                pg_catalog.acldefault('r', relation.relowner)
            )
        ) AS privilege
        WHERE namespace.nspname = 'source_add_history'
          AND relation.relname IN ('legacy_audit_rows', 'archive_manifests')
          AND privilege.grantee <> relation.relowner
    ) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc AS routine
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = routine.pronamespace
        CROSS JOIN LATERAL pg_catalog.aclexplode(
            COALESCE(
                routine.proacl,
                pg_catalog.acldefault('f', routine.proowner)
            )
        ) AS privilege
        WHERE namespace.nspname = 'source_add_history'
          AND routine.proname = 'reject_legacy_audit_mutation_v1'
          AND privilege.grantee <> routine.proowner
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD archive grants privileges outside its owner';
    END IF;

    IF (
        SELECT count(*)
        FROM pg_catalog.pg_trigger AS trigger
        JOIN pg_catalog.pg_class AS relation
          ON relation.oid = trigger.tgrelid
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        JOIN pg_catalog.pg_proc AS routine
          ON routine.oid = trigger.tgfoid
        WHERE namespace.nspname = 'source_add_history'
          AND relation.relname IN ('legacy_audit_rows', 'archive_manifests')
          AND trigger.tgname = 'reject_legacy_audit_mutation_v1'
          AND NOT trigger.tgisinternal
          AND trigger.tgenabled <> 'D'
          AND routine.proname = 'reject_legacy_audit_mutation_v1'
    ) <> 2 THEN
        RAISE EXCEPTION 'SOURCE_ADD append-only triggers are incomplete';
    END IF;
END;
$assert_final_retirement_complete$;

COMMIT;
