-- Retire the exact 15-table historical SOURCE_ADD graph after its full logical
-- archive has been restored and count-verified outside production.
--
-- The constants below bind this migration to the versioned private archive
-- manifest verified by full S3 readback and a successful PG15 data restore.
-- The transaction refuses an invalid archive URI or SHA-256.

BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

CREATE TEMP TABLE _retire_242_tables (
    table_name TEXT PRIMARY KEY
) ON COMMIT DROP;
INSERT INTO _retire_242_tables (table_name) VALUES
    ('published_weight_bundles'),
    ('research_lab_attested_ancestry_activations_v2'),
    ('research_lab_attested_ancestry_checkpoints_v2'),
    ('research_lab_attested_artifact_links_v2'),
    ('research_lab_attested_boot_identities_v2'),
    ('research_lab_attested_business_artifact_links_v2'),
    ('research_lab_attested_execution_receipts_v2'),
    ('research_lab_attested_execution_results_v2'),
    ('research_lab_attested_host_operations_v2'),
    ('research_lab_attested_receipt_edges_v2'),
    ('research_lab_attested_receipt_transport_v2'),
    ('research_lab_attested_transport_attempts_v2'),
    ('research_lab_stateful_subnet_epoch_boundaries_v1'),
    ('research_lab_stateful_subnet_epoch_candidates_v1'),
    ('research_lab_stateful_subnet_epoch_snapshots_v1');

-- Bound to the reviewed export baseline in evidence/snapshot-before.json.
-- These counters match the pre-export observation at 2026-09-13 16:59:55 UTC.
-- A changed OID or write counter means the archive snapshot is stale.
CREATE TEMP TABLE _retire_242_source_stats (
    table_name TEXT PRIMARY KEY,
    expected_oid OID NOT NULL,
    n_tup_ins BIGINT NOT NULL,
    n_tup_upd BIGINT NOT NULL,
    n_tup_del BIGINT NOT NULL
) ON COMMIT DROP;
-- TEST_FIXTURE_STATS_BEGIN
INSERT INTO _retire_242_source_stats VALUES
    ('published_weight_bundles', 828843, 1523, 0, 0),
    ('research_lab_attested_ancestry_activations_v2', 67542149, 167737, 0, 0),
    ('research_lab_attested_ancestry_checkpoints_v2', 67542091, 167737, 0, 0),
    ('research_lab_attested_artifact_links_v2', 67044467, 3211110, 0, 0),
    ('research_lab_attested_boot_identities_v2', 67044314, 9755, 0, 0),
    ('research_lab_attested_business_artifact_links_v2', 67044486, 6652, 0, 0),
    ('research_lab_attested_execution_receipts_v2', 67044368, 226866, 0, 0),
    ('research_lab_attested_execution_results_v2', 67094039, 14778, 0, 0),
    ('research_lab_attested_host_operations_v2', 67044442, 0, 0, 0),
    ('research_lab_attested_receipt_edges_v2', 67044405, 178027, 0, 0),
    ('research_lab_attested_receipt_transport_v2', 67044424, 8238927, 0, 0),
    ('research_lab_attested_transport_attempts_v2', 67044340, 8326255, 0, 0),
    ('research_lab_stateful_subnet_epoch_boundaries_v1', 67060831, 745, 0, 0),
    ('research_lab_stateful_subnet_epoch_candidates_v1', 67060634, 5, 0, 0),
    ('research_lab_stateful_subnet_epoch_snapshots_v1', 67060904, 745, 0, 0);
-- TEST_FIXTURE_STATS_END

CREATE TEMP TABLE _retire_242_routines (
    routine_name TEXT NOT NULL,
    identity_arguments TEXT NOT NULL,
    expected_body_md5 TEXT NOT NULL,
    PRIMARY KEY (routine_name, identity_arguments)
) ON COMMIT DROP;
INSERT INTO _retire_242_routines VALUES
    ('persist_research_lab_ancestry_checkpoint_v2', 'checkpoint jsonb', '266871ca5d1f1ce4e0ae22e7da76a3e6'),
    ('research_lab_active_model_replay_contract_v2', '', 'b9950de24ed91e7d3718570c6554c220'),
    ('research_lab_ancestry_disclosure_lookup_contract_v1', '', '881ae11e2e3e02659d820b07ac80bd2d'),
    ('research_lab_ancestry_checkpoint_bootstrap_contract_v2', '', '67b952a7d04124501d68554ca4392373'),
    ('research_lab_attested_transport_purpose_contract_v2', '', 'b940563a3e0f9b15df459f6ab3b47a4a'),
    ('research_lab_attested_transport_terminal_contract_v2', '', '1b5f105d75a5c21c726ee70a6bfef31e'),
    ('research_lab_candidate_hybrid_purpose_contract_v1', '', '246c0992c7a64f715ec4c5875bffa2d1'),
    ('research_lab_compact_checkpoint_graph_contract_v1', '', 'c982d13b2eddf870fd19bbde9b2b0264'),
    ('validate_research_lab_compact_checkpoint_sidecars_v1', '', '0f42e80c1c38aaf72b1f70b0923d4047'),
    ('validate_research_lab_stateful_subnet_epoch_v1', '', '7259893de35bbc17e0483af31a3ca654');

DO $archive_precondition$
DECLARE
    archive_uri CONSTANT TEXT := 's3://leadpoet-attested-v2-artifacts-493765492819/supabase-history/qplwoislplkcegvdmbim/2026-09-13/ixfa_wgp/archive-manifest.json?versionId=2vryaziHpEnYl80qRhnq1.KkyjYCQhMo';
    archive_sha256 CONSTANT TEXT := '942bd27d4f5d56b94e4200fafd60e706b2d2cbac3ce59f7a84714f5eb59eb738';
BEGIN
    IF pg_catalog.strpos(archive_uri, '__') > 0
       OR archive_uri !~ '^s3://[^[:space:]]+$'
       OR pg_catalog.strpos(archive_sha256, '__') > 0
       OR archive_sha256 !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION
            'verified historical archive URI and SHA-256 are required before migration 242';
    END IF;
END;
$archive_precondition$;

-- Acquire every destructive target in one stable name order. The retained
-- cutover table is included because five reviewed FKs are altered below.
DO $lock_targets$
DECLARE
    target RECORD;
BEGIN
    FOR target IN
        SELECT schema_name, relation_name
        FROM (
            SELECT 'public'::TEXT AS schema_name, table_name AS relation_name
            FROM _retire_242_tables
            UNION ALL
            SELECT 'public', 'research_lab_stateful_subnet_epoch_cutovers_v1'
        ) AS lock_set
        ORDER BY schema_name, relation_name
    LOOP
        IF pg_catalog.to_regclass(
               pg_catalog.format('%I.%I', target.schema_name, target.relation_name)
           ) IS NOT NULL THEN
            EXECUTE pg_catalog.format(
                'LOCK TABLE %I.%I IN ACCESS EXCLUSIVE MODE',
                target.schema_name,
                target.relation_name
            );
        END IF;
    END LOOP;
END;
$lock_targets$;

DO $source_snapshot_guard$
DECLARE
    changed TEXT;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM _retire_242_tables target
        WHERE pg_catalog.to_regclass(
            pg_catalog.format('public.%I', target.table_name)
        ) IS NOT NULL
    ) THEN
        RETURN;
    END IF;

    IF (SELECT stats_reset FROM pg_catalog.pg_stat_database
        WHERE datname = pg_catalog.current_database()) IS NOT NULL THEN
        RAISE EXCEPTION 'database statistics were reset after the reviewed baseline';
    END IF;

    SELECT expected.table_name INTO changed
    FROM _retire_242_source_stats expected
    LEFT JOIN pg_catalog.pg_stat_user_tables actual
      ON actual.relid = expected.expected_oid
     AND actual.schemaname = 'public'
     AND actual.relname = expected.table_name
    WHERE actual.relid IS NULL
       OR actual.n_tup_ins IS DISTINCT FROM expected.n_tup_ins
       OR actual.n_tup_upd IS DISTINCT FROM expected.n_tup_upd
       OR actual.n_tup_del IS DISTINCT FROM expected.n_tup_del
    LIMIT 1;
    IF changed IS NOT NULL THEN
        RAISE EXCEPTION
            'historical source changed after archive baseline: %', changed;
    END IF;
END;
$source_snapshot_guard$;

DO $dependency_preflight$
DECLARE
    candidate_count INTEGER;
    unexpected TEXT;
    routine_row RECORD;
    actual_md5 TEXT;
BEGIN
    SELECT count(*) INTO candidate_count
    FROM _retire_242_tables target
    WHERE pg_catalog.to_regclass(
        pg_catalog.format('public.%I', target.table_name)
    ) IS NOT NULL;

    IF candidate_count NOT IN (0, 15) THEN
        RAISE EXCEPTION
            'historical retirement set is partial: expected 0 or 15 tables, found %',
            candidate_count;
    END IF;

    IF candidate_count = 15 THEN
        SELECT pg_catalog.format(
                   '%I on %I.%I -> %I',
                   constraint_row.conname,
                   source_namespace.nspname,
                   source_relation.relname,
                   target_relation.relname
               )
          INTO unexpected
          FROM pg_catalog.pg_constraint AS constraint_row
          JOIN pg_catalog.pg_class AS source_relation
            ON source_relation.oid = constraint_row.conrelid
          JOIN pg_catalog.pg_namespace AS source_namespace
            ON source_namespace.oid = source_relation.relnamespace
          JOIN pg_catalog.pg_class AS target_relation
            ON target_relation.oid = constraint_row.confrelid
          JOIN pg_catalog.pg_namespace AS target_namespace
            ON target_namespace.oid = target_relation.relnamespace
         WHERE constraint_row.contype = 'f'
           AND target_namespace.nspname = 'public'
           AND EXISTS (
               SELECT 1 FROM _retire_242_tables target
               WHERE target.table_name = target_relation.relname
           )
           AND NOT EXISTS (
               SELECT 1 FROM _retire_242_tables source
               WHERE source_namespace.nspname = 'public'
                 AND source.table_name = source_relation.relname
           )
           AND NOT (
               source_namespace.nspname = 'public'
               AND source_relation.relname =
                   'research_lab_stateful_subnet_epoch_cutovers_v1'
               AND constraint_row.conname IN (
                   'research_lab_stateful_subnet__last_legacy_finalization_rec_fkey',
                   'research_lab_stateful_subnet_e_first_snapshot_receipt_hash_fkey',
                   'research_lab_stateful_subnet_epoc_predecessor_receipt_hash_fkey',
                   'research_lab_stateful_subnet_epoch_cu_cutover_receipt_hash_fkey',
                   'research_lab_stateful_subnet_epoch_cut_first_snapshot_hash_fkey'
               )
           )
         LIMIT 1;
        IF unexpected IS NOT NULL THEN
            RAISE EXCEPTION 'unexpected retained foreign key into historical set: %',
                unexpected;
        END IF;

        IF (
            SELECT count(*)
            FROM pg_catalog.pg_constraint AS constraint_row
            JOIN (VALUES
                ('research_lab_stateful_subnet__last_legacy_finalization_rec_fkey',
                 'FOREIGN KEY (last_legacy_finalization_receipt_hash) REFERENCES research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT'),
                ('research_lab_stateful_subnet_e_first_snapshot_receipt_hash_fkey',
                 'FOREIGN KEY (first_snapshot_receipt_hash) REFERENCES research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT'),
                ('research_lab_stateful_subnet_epoc_predecessor_receipt_hash_fkey',
                 'FOREIGN KEY (predecessor_receipt_hash) REFERENCES research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT'),
                ('research_lab_stateful_subnet_epoch_cu_cutover_receipt_hash_fkey',
                 'FOREIGN KEY (cutover_receipt_hash) REFERENCES research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT'),
                ('research_lab_stateful_subnet_epoch_cut_first_snapshot_hash_fkey',
                 'FOREIGN KEY (first_snapshot_hash) REFERENCES research_lab_stateful_subnet_epoch_candidates_v1(snapshot_hash) ON DELETE RESTRICT')
            ) AS expected(constraint_name, definition)
              ON expected.constraint_name = constraint_row.conname
             AND expected.definition =
                 pg_catalog.pg_get_constraintdef(constraint_row.oid, true)
            WHERE constraint_row.conrelid =
                  'public.research_lab_stateful_subnet_epoch_cutovers_v1'::REGCLASS
        ) <> 5 THEN
            RAISE EXCEPTION 'reviewed retained cutover foreign keys changed';
        END IF;

        SELECT pg_catalog.format('%I.%I', namespace.nspname, relation.relname)
          INTO unexpected
          FROM pg_catalog.pg_depend AS dependency
          JOIN pg_catalog.pg_rewrite AS rewrite
            ON rewrite.oid = dependency.objid
          JOIN pg_catalog.pg_class AS relation
            ON relation.oid = rewrite.ev_class
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
          JOIN pg_catalog.pg_class AS referenced
            ON referenced.oid = dependency.refobjid
          JOIN pg_catalog.pg_namespace AS referenced_namespace
            ON referenced_namespace.oid = referenced.relnamespace
         WHERE dependency.classid = 'pg_catalog.pg_rewrite'::REGCLASS
           AND dependency.refclassid = 'pg_catalog.pg_class'::REGCLASS
           AND referenced_namespace.nspname = 'public'
           AND EXISTS (
               SELECT 1 FROM _retire_242_tables target
               WHERE target.table_name = referenced.relname
           )
           AND NOT (
               namespace.nspname = 'public'
               AND EXISTS (
                   SELECT 1 FROM _retire_242_tables target
                   WHERE target.table_name = relation.relname
               )
           )
         LIMIT 1;
        IF unexpected IS NOT NULL THEN
            RAISE EXCEPTION 'unexpected view depends on historical set: %',
                unexpected;
        END IF;

        SELECT pg_catalog.format(
                   '%I.%I(%s)', namespace.nspname, routine.proname,
                   pg_catalog.pg_get_function_identity_arguments(routine.oid)
               )
          INTO unexpected
          FROM pg_catalog.pg_proc AS routine
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = routine.pronamespace
         WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
           AND namespace.nspname NOT LIKE 'pg_toast%'
           AND namespace.nspname NOT LIKE 'pg_temp_%'
           AND EXISTS (
               SELECT 1 FROM _retire_242_tables target
               WHERE pg_catalog.strpos(
                   pg_catalog.lower(routine.prosrc),
                   pg_catalog.lower(target.table_name)
               ) > 0
           )
           AND NOT (
               namespace.nspname = 'public'
               AND (
                   EXISTS (
                       SELECT 1 FROM _retire_242_routines allowed
                       WHERE allowed.routine_name = routine.proname
                         AND allowed.identity_arguments =
                             pg_catalog.pg_get_function_identity_arguments(routine.oid)
                   )
                   OR (
                       routine.proname = 'enforce_research_lab_stateful_epoch_fence_v1'
                       AND pg_catalog.pg_get_function_identity_arguments(routine.oid) = ''
                   )
                   OR (
                       routine.proname = 'validate_research_lab_stateful_epoch_cutover_v2'
                       AND pg_catalog.pg_get_function_identity_arguments(routine.oid) = ''
                   )
               )
           )
         LIMIT 1;
        IF unexpected IS NOT NULL THEN
            RAISE EXCEPTION 'unexpected routine text references historical set: %',
                unexpected;
        END IF;

        SELECT pg_catalog.format(
                   '%I.%I(%s)', routine_namespace.nspname, routine.proname,
                   pg_catalog.pg_get_function_identity_arguments(routine.oid)
               )
          INTO unexpected
          FROM pg_catalog.pg_depend AS dependency
          JOIN pg_catalog.pg_proc AS routine
            ON routine.oid = dependency.objid
          JOIN pg_catalog.pg_namespace AS routine_namespace
            ON routine_namespace.oid = routine.pronamespace
          JOIN pg_catalog.pg_class AS referenced_relation
            ON referenced_relation.oid = dependency.refobjid
          JOIN pg_catalog.pg_namespace AS referenced_namespace
            ON referenced_namespace.oid = referenced_relation.relnamespace
         WHERE dependency.classid = 'pg_catalog.pg_proc'::REGCLASS
           AND dependency.refclassid = 'pg_catalog.pg_class'::REGCLASS
           AND referenced_namespace.nspname = 'public'
           AND EXISTS (
               SELECT 1 FROM _retire_242_tables target
               WHERE target.table_name = referenced_relation.relname
           )
           AND NOT (
               routine_namespace.nspname = 'public'
               AND (
                   EXISTS (
                       SELECT 1 FROM _retire_242_routines allowed
                       WHERE allowed.routine_name = routine.proname
                         AND allowed.identity_arguments =
                             pg_catalog.pg_get_function_identity_arguments(routine.oid)
                   )
                   OR (
                       routine.proname = 'enforce_research_lab_stateful_epoch_fence_v1'
                       AND pg_catalog.pg_get_function_identity_arguments(routine.oid) = ''
                   )
                   OR (
                       routine.proname = 'validate_research_lab_stateful_epoch_cutover_v2'
                       AND pg_catalog.pg_get_function_identity_arguments(routine.oid) = ''
                   )
               )
           )
         LIMIT 1;
        IF unexpected IS NOT NULL THEN
            RAISE EXCEPTION 'unexpected parsed routine depends on historical set: %',
                unexpected;
        END IF;

        SELECT pg_catalog.format(
                   '%I.%I(%s) -> %I', namespace.nspname, routine.proname,
                   pg_catalog.pg_get_function_identity_arguments(routine.oid),
                   retired.routine_name
               )
          INTO unexpected
          FROM pg_catalog.pg_proc AS routine
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = routine.pronamespace
          CROSS JOIN _retire_242_routines AS retired
         WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
           AND namespace.nspname NOT LIKE 'pg_toast%'
           AND namespace.nspname NOT LIKE 'pg_temp_%'
           AND pg_catalog.strpos(
                   pg_catalog.lower(routine.prosrc),
                   pg_catalog.lower(retired.routine_name)
               ) > 0
           AND NOT (
               namespace.nspname = 'public'
               AND EXISTS (
                   SELECT 1 FROM _retire_242_routines caller
                   WHERE caller.routine_name = routine.proname
                     AND caller.identity_arguments =
                         pg_catalog.pg_get_function_identity_arguments(routine.oid)
               )
           )
         LIMIT 1;
        IF unexpected IS NOT NULL THEN
            RAISE EXCEPTION 'unexpected caller of historical routine: %',
                unexpected;
        END IF;

        FOR routine_row IN SELECT * FROM _retire_242_routines LOOP
            SELECT pg_catalog.md5(routine.prosrc)
              INTO actual_md5
              FROM pg_catalog.pg_proc AS routine
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = routine.pronamespace
             WHERE namespace.nspname = 'public'
               AND routine.proname = routine_row.routine_name
               AND pg_catalog.pg_get_function_identity_arguments(routine.oid) =
                   routine_row.identity_arguments;
            IF actual_md5 IS DISTINCT FROM routine_row.expected_body_md5 THEN
                RAISE EXCEPTION 'reviewed historical routine changed: %(%)',
                    routine_row.routine_name, routine_row.identity_arguments;
            END IF;
        END LOOP;

        SELECT pg_catalog.md5(routine.prosrc) INTO actual_md5
        FROM pg_catalog.pg_proc AS routine
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = routine.pronamespace
        WHERE namespace.nspname = 'public'
          AND routine.proname = 'enforce_research_lab_stateful_epoch_fence_v1'
          AND pg_catalog.pg_get_function_identity_arguments(routine.oid) = '';
        IF actual_md5 IS DISTINCT FROM '5b97ae04b866110b43b6cf8ec159463b' THEN
            RAISE EXCEPTION 'shared epoch fence changed after audit';
        END IF;

        SELECT pg_catalog.md5(routine.prosrc) INTO actual_md5
        FROM pg_catalog.pg_proc AS routine
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = routine.pronamespace
        WHERE namespace.nspname = 'public'
          AND routine.proname = 'validate_research_lab_stateful_epoch_cutover_v2'
          AND pg_catalog.pg_get_function_identity_arguments(routine.oid) = '';
        IF actual_md5 IS DISTINCT FROM 'bc22df15eda65827f8aade10a2e0f5b4' THEN
            RAISE EXCEPTION 'cutover validator changed after audit';
        END IF;

        IF pg_catalog.to_regclass('cron.job') IS NOT NULL THEN
            EXECUTE $cron$
                SELECT pg_catalog.format('%s:%s', jobid, jobname)
                FROM cron.job AS job
                WHERE EXISTS (
                    SELECT 1 FROM _retire_242_tables target
                    WHERE pg_catalog.strpos(
                        pg_catalog.lower(job.command),
                        pg_catalog.lower(target.table_name)
                    ) > 0
                )
                LIMIT 1
            $cron$ INTO unexpected;
            IF unexpected IS NOT NULL THEN
                RAISE EXCEPTION 'unexpected cron references historical set: %',
                    unexpected;
            END IF;
        END IF;

        SELECT publication.pubname || ':' || publication.schemaname || '.' ||
               publication.tablename
          INTO unexpected
          FROM pg_catalog.pg_publication_tables AS publication
         WHERE publication.schemaname = 'public'
           AND EXISTS (
               SELECT 1 FROM _retire_242_tables target
               WHERE target.table_name = publication.tablename
           )
         LIMIT 1;
        IF unexpected IS NOT NULL THEN
            RAISE EXCEPTION 'unexpected publication contains historical table: %',
                unexpected;
        END IF;
    END IF;
END;
$dependency_preflight$;

CREATE TEMP TABLE _retain_242_relations_before ON COMMIT DROP AS
SELECT namespace.nspname AS schema_name,
       relation.relname AS relation_name,
       relation.oid,
       relation.relkind,
       relation.relowner,
       relation.relacl,
       relation.relrowsecurity,
       relation.relforcerowsecurity,
       relation.relreplident,
       (
           SELECT pg_catalog.md5(pg_catalog.string_agg(
               pg_catalog.concat_ws(':', attribute.attnum, attribute.attname,
                   attribute.atttypid, attribute.atttypmod, attribute.attnotnull,
                   attribute.attidentity, attribute.attgenerated),
               '|' ORDER BY attribute.attnum
           ))
           FROM pg_catalog.pg_attribute AS attribute
           WHERE attribute.attrelid = relation.oid
             AND attribute.attnum > 0
             AND NOT attribute.attisdropped
       ) AS column_signature
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
  AND namespace.nspname NOT LIKE 'pg_toast%'
  AND namespace.nspname NOT LIKE 'pg_temp_%'
  AND relation.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
  AND NOT (
      namespace.nspname = 'public'
      AND EXISTS (
          SELECT 1 FROM _retire_242_tables target
          WHERE target.table_name = relation.relname
      )
  )
  AND NOT EXISTS (
      SELECT 1
      FROM pg_catalog.pg_depend dependency
      JOIN pg_catalog.pg_class owner_relation
        ON owner_relation.oid = dependency.refobjid
      JOIN pg_catalog.pg_namespace owner_namespace
        ON owner_namespace.oid = owner_relation.relnamespace
      WHERE dependency.classid = 'pg_catalog.pg_class'::REGCLASS
        AND dependency.objid = relation.oid
        AND dependency.refclassid = 'pg_catalog.pg_class'::REGCLASS
        AND dependency.deptype IN ('a', 'i')
        AND owner_namespace.nspname = 'public'
        AND EXISTS (SELECT 1 FROM _retire_242_tables target
                    WHERE target.table_name = owner_relation.relname)
  );

CREATE TEMP TABLE _retain_242_cutover_rows_before ON COMMIT DROP AS
SELECT 'state'::TEXT AS relation_name, count(*) AS row_count,
       pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(row_value)::TEXT, '|' ORDER BY
           pg_catalog.to_jsonb(row_value)::TEXT
       ), '')) AS row_fingerprint
FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1 AS row_value
UNION ALL
SELECT 'cutovers', count(*),
       pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(row_value)::TEXT, '|' ORDER BY
           pg_catalog.to_jsonb(row_value)::TEXT
       ), ''))
FROM public.research_lab_stateful_subnet_epoch_cutovers_v1 AS row_value;

CREATE TEMP TABLE _retain_242_function_metadata_before ON COMMIT DROP AS
SELECT namespace.nspname, routine.proname,
       pg_catalog.pg_get_function_identity_arguments(routine.oid) AS arguments,
       routine.oid, routine.proowner, routine.proacl, routine.prosecdef,
       routine.proleakproof, routine.provolatile, routine.proparallel,
       routine.proconfig
FROM pg_catalog.pg_proc AS routine
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = routine.pronamespace
WHERE namespace.nspname = 'public'
  AND routine.proname IN (
      'enforce_research_lab_stateful_epoch_fence_v1',
      'validate_research_lab_stateful_epoch_cutover_v2',
      'prevent_research_lab_attested_v2_mutation',
      'put_research_lab_provider_evidence_cache_v2',
      'research_lab_stateful_subnet_epoch_cutover_public_state_v1'
  );

CREATE OR REPLACE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO ''
AS $function$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    row_doc JSONB := pg_catalog.to_jsonb(NEW);
BEGIN
    IF TG_TABLE_SCHEMA IS DISTINCT FROM 'public'
       OR TG_TABLE_NAME IS DISTINCT FROM
          'research_lab_stateful_subnet_epoch_cutovers_v1' THEN
        RAISE EXCEPTION 'retired epoch fence is attached to unexpected table %.%',
            TG_TABLE_SCHEMA, TG_TABLE_NAME;
    END IF;

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
    RAISE EXCEPTION 'stateful epoch fence requires the bound cutover authority';
END;
$function$;

CREATE OR REPLACE FUNCTION public.validate_research_lab_stateful_epoch_cutover_v2()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO ''
AS $function$
BEGIN
    RAISE EXCEPTION
        'historical cutover registration is retired; existing mappings are read-only';
END;
$function$;

ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    DROP CONSTRAINT IF EXISTS research_lab_stateful_subnet__last_legacy_finalization_rec_fkey,
    DROP CONSTRAINT IF EXISTS research_lab_stateful_subnet_e_first_snapshot_receipt_hash_fkey,
    DROP CONSTRAINT IF EXISTS research_lab_stateful_subnet_epoc_predecessor_receipt_hash_fkey,
    DROP CONSTRAINT IF EXISTS research_lab_stateful_subnet_epoch_cu_cutover_receipt_hash_fkey,
    DROP CONSTRAINT IF EXISTS research_lab_stateful_subnet_epoch_cut_first_snapshot_hash_fkey;

DROP TABLE IF EXISTS public.research_lab_attested_ancestry_activations_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_ancestry_checkpoints_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_artifact_links_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_business_artifact_links_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_execution_results_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_host_operations_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_receipt_edges_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_receipt_transport_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_stateful_subnet_epoch_boundaries_v1 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_stateful_subnet_epoch_snapshots_v1 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_stateful_subnet_epoch_candidates_v1 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_transport_attempts_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_execution_receipts_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_boot_identities_v2 RESTRICT;
DROP TABLE IF EXISTS public.published_weight_bundles RESTRICT;

DROP FUNCTION IF EXISTS public.persist_research_lab_ancestry_checkpoint_v2(JSONB) RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_active_model_replay_contract_v2() RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_ancestry_disclosure_lookup_contract_v1() RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_ancestry_checkpoint_bootstrap_contract_v2() RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_attested_transport_purpose_contract_v2() RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_attested_transport_terminal_contract_v2() RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_candidate_hybrid_purpose_contract_v1() RESTRICT;
DROP FUNCTION IF EXISTS public.research_lab_compact_checkpoint_graph_contract_v1() RESTRICT;
DROP FUNCTION IF EXISTS public.validate_research_lab_compact_checkpoint_sidecars_v1() RESTRICT;
DROP FUNCTION IF EXISTS public.validate_research_lab_stateful_subnet_epoch_v1() RESTRICT;

CREATE TEMP TABLE _retain_242_cutover_rows_after ON COMMIT DROP AS
SELECT 'state'::TEXT AS relation_name, count(*) AS row_count,
       pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(row_value)::TEXT, '|' ORDER BY
           pg_catalog.to_jsonb(row_value)::TEXT
       ), '')) AS row_fingerprint
FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1 AS row_value
UNION ALL
SELECT 'cutovers', count(*),
       pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(row_value)::TEXT, '|' ORDER BY
           pg_catalog.to_jsonb(row_value)::TEXT
       ), ''))
FROM public.research_lab_stateful_subnet_epoch_cutovers_v1 AS row_value;

DO $postflight$
DECLARE
    unexpected TEXT;
BEGIN
    SELECT target.table_name INTO unexpected
    FROM _retire_242_tables target
    WHERE pg_catalog.to_regclass(
        pg_catalog.format('public.%I', target.table_name)
    ) IS NOT NULL
    LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'historical table survived migration 242: %', unexpected;
    END IF;

    IF EXISTS (
        (SELECT * FROM _retain_242_relations_before
         EXCEPT
         SELECT namespace.nspname, relation.relname, relation.oid,
                relation.relkind, relation.relowner, relation.relacl,
                relation.relrowsecurity, relation.relforcerowsecurity,
                relation.relreplident,
                (SELECT pg_catalog.md5(pg_catalog.string_agg(
                    pg_catalog.concat_ws(':', attribute.attnum,
                        attribute.attname, attribute.atttypid,
                        attribute.atttypmod, attribute.attnotnull,
                        attribute.attidentity, attribute.attgenerated),
                    '|' ORDER BY attribute.attnum))
                 FROM pg_catalog.pg_attribute AS attribute
                 WHERE attribute.attrelid = relation.oid
                   AND attribute.attnum > 0 AND NOT attribute.attisdropped)
         FROM pg_catalog.pg_class AS relation
         JOIN pg_catalog.pg_namespace AS namespace
           ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
           AND namespace.nspname NOT LIKE 'pg_toast%'
           AND namespace.nspname NOT LIKE 'pg_temp_%'
           AND relation.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
           AND NOT (namespace.nspname = 'public' AND EXISTS (
               SELECT 1 FROM _retire_242_tables target
               WHERE target.table_name = relation.relname))
           AND NOT EXISTS (
               SELECT 1 FROM pg_catalog.pg_depend dependency
               JOIN pg_catalog.pg_class owner_relation
                 ON owner_relation.oid = dependency.refobjid
               JOIN pg_catalog.pg_namespace owner_namespace
                 ON owner_namespace.oid = owner_relation.relnamespace
               WHERE dependency.classid = 'pg_catalog.pg_class'::REGCLASS
                 AND dependency.objid = relation.oid
                 AND dependency.refclassid = 'pg_catalog.pg_class'::REGCLASS
                 AND dependency.deptype IN ('a', 'i')
                 AND owner_namespace.nspname = 'public'
                 AND EXISTS (SELECT 1 FROM _retire_242_tables target
                             WHERE target.table_name = owner_relation.relname)))
        UNION ALL
        (SELECT namespace.nspname, relation.relname, relation.oid,
                relation.relkind, relation.relowner, relation.relacl,
                relation.relrowsecurity, relation.relforcerowsecurity,
                relation.relreplident,
                (SELECT pg_catalog.md5(pg_catalog.string_agg(
                    pg_catalog.concat_ws(':', attribute.attnum,
                        attribute.attname, attribute.atttypid,
                        attribute.atttypmod, attribute.attnotnull,
                        attribute.attidentity, attribute.attgenerated),
                    '|' ORDER BY attribute.attnum))
                 FROM pg_catalog.pg_attribute AS attribute
                 WHERE attribute.attrelid = relation.oid
                   AND attribute.attnum > 0 AND NOT attribute.attisdropped)
         FROM pg_catalog.pg_class AS relation
         JOIN pg_catalog.pg_namespace AS namespace
           ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
           AND namespace.nspname NOT LIKE 'pg_toast%'
           AND namespace.nspname NOT LIKE 'pg_temp_%'
           AND relation.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
           AND NOT (namespace.nspname = 'public' AND EXISTS (
               SELECT 1 FROM _retire_242_tables target
               WHERE target.table_name = relation.relname))
           AND NOT EXISTS (
               SELECT 1 FROM pg_catalog.pg_depend dependency
               JOIN pg_catalog.pg_class owner_relation
                 ON owner_relation.oid = dependency.refobjid
               JOIN pg_catalog.pg_namespace owner_namespace
                 ON owner_namespace.oid = owner_relation.relnamespace
               WHERE dependency.classid = 'pg_catalog.pg_class'::REGCLASS
                 AND dependency.objid = relation.oid
                 AND dependency.refclassid = 'pg_catalog.pg_class'::REGCLASS
                 AND dependency.deptype IN ('a', 'i')
                 AND owner_namespace.nspname = 'public'
                 AND EXISTS (SELECT 1 FROM _retire_242_tables target
                             WHERE target.table_name = owner_relation.relname))
         EXCEPT SELECT * FROM _retain_242_relations_before)
    ) THEN
        SELECT pg_catalog.concat_ws('.', changed.schema_name, changed.relation_name)
          INTO unexpected
          FROM (
              SELECT * FROM _retain_242_relations_before
              EXCEPT
              SELECT namespace.nspname, relation.relname, relation.oid,
                     relation.relkind, relation.relowner, relation.relacl,
                     relation.relrowsecurity, relation.relforcerowsecurity,
                     relation.relreplident,
                     (SELECT pg_catalog.md5(pg_catalog.string_agg(
                         pg_catalog.concat_ws(':', attribute.attnum,
                             attribute.attname, attribute.atttypid,
                             attribute.atttypmod, attribute.attnotnull,
                             attribute.attidentity, attribute.attgenerated),
                         '|' ORDER BY attribute.attnum))
                      FROM pg_catalog.pg_attribute AS attribute
                      WHERE attribute.attrelid = relation.oid
                        AND attribute.attnum > 0 AND NOT attribute.attisdropped)
              FROM pg_catalog.pg_class AS relation
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = relation.relnamespace
              WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
                AND namespace.nspname NOT LIKE 'pg_toast%'
                AND namespace.nspname NOT LIKE 'pg_temp_%'
                AND relation.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
                AND NOT (namespace.nspname = 'public' AND EXISTS (
                    SELECT 1 FROM _retire_242_tables target
                    WHERE target.table_name = relation.relname))
                AND NOT EXISTS (
                    SELECT 1 FROM pg_catalog.pg_depend dependency
                    JOIN pg_catalog.pg_class owner_relation
                      ON owner_relation.oid = dependency.refobjid
                    JOIN pg_catalog.pg_namespace owner_namespace
                      ON owner_namespace.oid = owner_relation.relnamespace
                    WHERE dependency.classid = 'pg_catalog.pg_class'::REGCLASS
                      AND dependency.objid = relation.oid
                      AND dependency.refclassid = 'pg_catalog.pg_class'::REGCLASS
                      AND dependency.deptype IN ('a', 'i')
                      AND owner_namespace.nspname = 'public'
                      AND EXISTS (SELECT 1 FROM _retire_242_tables target
                                  WHERE target.table_name = owner_relation.relname))
          ) changed
          LIMIT 1;
        RAISE EXCEPTION 'non-candidate relation metadata changed during migration 242: %',
            unexpected;
    END IF;

    IF EXISTS (
        (SELECT * FROM _retain_242_cutover_rows_before
         EXCEPT SELECT * FROM _retain_242_cutover_rows_after)
        UNION ALL
        (SELECT * FROM _retain_242_cutover_rows_after
         EXCEPT SELECT * FROM _retain_242_cutover_rows_before)
    ) THEN
        RAISE EXCEPTION 'retained cutover rows changed during migration 242';
    END IF;

    IF EXISTS (
        SELECT 1 FROM _retain_242_function_metadata_before before
        LEFT JOIN pg_catalog.pg_proc AS routine ON routine.oid = before.oid
        WHERE routine.oid IS NULL
           OR routine.proowner IS DISTINCT FROM before.proowner
           OR routine.proacl IS DISTINCT FROM before.proacl
           OR routine.prosecdef IS DISTINCT FROM before.prosecdef
           OR routine.proleakproof IS DISTINCT FROM before.proleakproof
           OR routine.provolatile IS DISTINCT FROM before.provolatile
           OR routine.proparallel IS DISTINCT FROM before.proparallel
           OR routine.proconfig IS DISTINCT FROM before.proconfig
    ) THEN
        RAISE EXCEPTION 'retained function metadata changed during migration 242';
    END IF;

    SELECT pg_catalog.format('%I.%I(%s)', namespace.nspname, routine.proname,
               pg_catalog.pg_get_function_identity_arguments(routine.oid))
      INTO unexpected
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
     WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
       AND namespace.nspname NOT LIKE 'pg_toast%'
       AND namespace.nspname NOT LIKE 'pg_temp_%'
       AND EXISTS (
           SELECT 1 FROM _retire_242_tables target
           WHERE pg_catalog.strpos(pg_catalog.lower(routine.prosrc),
                 pg_catalog.lower(target.table_name)) > 0
       )
     LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'routine still references retired historical table: %',
            unexpected;
    END IF;

    SELECT pg_catalog.format('%I.%I(%s) -> %I', namespace.nspname,
               routine.proname,
               pg_catalog.pg_get_function_identity_arguments(routine.oid),
               retired.routine_name)
      INTO unexpected
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
      CROSS JOIN _retire_242_routines AS retired
     WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
       AND namespace.nspname NOT LIKE 'pg_toast%'
       AND namespace.nspname NOT LIKE 'pg_temp_%'
       AND pg_catalog.strpos(pg_catalog.lower(routine.prosrc),
               pg_catalog.lower(retired.routine_name)) > 0
     LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'routine still calls retired historical routine: %',
            unexpected;
    END IF;
END;
$postflight$;

NOTIFY pgrst, 'reload schema';

COMMIT;
