-- Crash-safe Git-tree replacement after an active private-model root change.
--
-- Additive lineage hardening only. Cancelled trees, nodes, operations, costs,
-- and events remain immutable. A run may have sequential tree generations,
-- but PostgreSQL permits only one successor for each cancelled generation.
-- No scoring, promotion, reward, payment, allocation, emission, validator
-- weight, or chain-submission behavior is changed.
--
-- Apply after scripts/95-research-lab-git-tree-autoresearch.sql.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_autoresearch_trees
    DROP CONSTRAINT IF EXISTS research_lab_autoresearch_trees_run_id_key;

ALTER TABLE public.research_lab_autoresearch_trees
    ADD COLUMN IF NOT EXISTS tree_generation INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS replaces_tree_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS cancellation_event_hash TEXT NULL,
    ADD COLUMN IF NOT EXISTS replacement_hash TEXT NULL;

ALTER TABLE public.research_lab_autoresearch_trees
    DROP CONSTRAINT IF EXISTS research_lab_autoresearch_trees_generation_check,
    ADD CONSTRAINT research_lab_autoresearch_trees_generation_check
        CHECK (tree_generation BETWEEN 0 AND 1000000) NOT VALID,
    DROP CONSTRAINT IF EXISTS research_lab_autoresearch_trees_replacement_check,
    ADD CONSTRAINT research_lab_autoresearch_trees_replacement_check CHECK (
        (
            tree_generation = 0
            AND replaces_tree_id IS NULL
            AND cancellation_event_hash IS NULL
            AND replacement_hash IS NULL
        )
        OR
        (
            tree_generation > 0
            AND replaces_tree_id ~ '^sha256:[0-9a-f]{64}$'
            AND cancellation_event_hash ~ '^sha256:[0-9a-f]{64}$'
            AND replacement_hash ~ '^sha256:[0-9a-f]{64}$'
            AND replaces_tree_id <> tree_id
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_autoresearch_trees
    DROP CONSTRAINT IF EXISTS research_lab_autoresearch_trees_replaces_tree_fkey,
    ADD CONSTRAINT research_lab_autoresearch_trees_replaces_tree_fkey
        FOREIGN KEY (replaces_tree_id)
        REFERENCES public.research_lab_autoresearch_trees(tree_id)
        ON DELETE RESTRICT
        NOT VALID;

ALTER TABLE public.research_lab_autoresearch_trees
    VALIDATE CONSTRAINT research_lab_autoresearch_trees_generation_check;
ALTER TABLE public.research_lab_autoresearch_trees
    VALIDATE CONSTRAINT research_lab_autoresearch_trees_replacement_check;
ALTER TABLE public.research_lab_autoresearch_trees
    VALIDATE CONSTRAINT research_lab_autoresearch_trees_replaces_tree_fkey;

CREATE UNIQUE INDEX IF NOT EXISTS uq_research_lab_tree_run_generation
    ON public.research_lab_autoresearch_trees(run_id, tree_generation);

CREATE UNIQUE INDEX IF NOT EXISTS uq_research_lab_tree_one_successor
    ON public.research_lab_autoresearch_trees(replaces_tree_id)
    WHERE replaces_tree_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_research_lab_tree_run_latest
    ON public.research_lab_autoresearch_trees(
        run_id, tree_generation DESC, created_at DESC
    );

CREATE OR REPLACE VIEW public.research_lab_autoresearch_run_tree_current
WITH (security_invoker = true) AS
SELECT
    latest.*,
    event_row.seq AS current_event_seq,
    event_row.event_type AS current_event_type,
    event_row.event_doc AS current_event_doc,
    event_row.event_hash AS current_event_hash,
    event_row.created_at AS current_event_at,
    frontier_row.round_index AS current_round_index,
    frontier_row.frontier_hash AS current_frontier_hash,
    frontier_row.frontier_doc AS current_frontier_doc
FROM (
    SELECT DISTINCT ON (tree.run_id)
        tree.*
    FROM public.research_lab_autoresearch_trees tree
    ORDER BY
        tree.run_id,
        tree.tree_generation DESC,
        tree.created_at DESC,
        tree.tree_id DESC
) latest
LEFT JOIN LATERAL (
    SELECT event.*
    FROM public.research_lab_autoresearch_tree_events event
    WHERE event.tree_id = latest.tree_id
    ORDER BY event.seq DESC
    LIMIT 1
) event_row ON TRUE
LEFT JOIN LATERAL (
    SELECT frontier.*
    FROM public.research_lab_autoresearch_frontier_commitments frontier
    WHERE frontier.tree_id = latest.tree_id
    ORDER BY frontier.round_index DESC
    LIMIT 1
) frontier_row ON TRUE;

-- Serialize every private-model status transition, not only the new active
-- event. This closes the superseded -> active gap against a final tree handoff.
CREATE OR REPLACE FUNCTION public.guard_research_lab_one_active_private_model_version()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    conflicting_version TEXT;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_private_model_version'),
        pg_catalog.hashtext('one_active_version')
    );

    IF NEW.version_status <> 'active' THEN
        RETURN NEW;
    END IF;

    SELECT latest.private_model_version_id
      INTO conflicting_version
      FROM (
        SELECT DISTINCT ON (event.private_model_version_id)
               event.private_model_version_id,
               event.version_status
          FROM public.research_lab_private_model_version_events event
         WHERE event.private_model_version_id <> NEW.private_model_version_id
         ORDER BY
               event.private_model_version_id,
               event.seq DESC,
               event.created_at DESC
      ) latest
     WHERE latest.version_status = 'active'
     LIMIT 1;

    IF conflicting_version IS NOT NULL THEN
        RAISE EXCEPTION
            'research_lab_one_active_version_conflict: version % cannot become active while version % is active; supersede it first',
            NEW.private_model_version_id,
            conflicting_version
            USING ERRCODE = '23505';
    END IF;

    RETURN NEW;
END;
$$;

-- The handoff trigger shares the private-model transition lock and the run
-- generation lock. Whichever transaction commits first is authoritative:
-- either this tree hands off while its root is still active, or the handoff
-- fails and the worker creates a replacement generation.
CREATE OR REPLACE FUNCTION public.guard_research_lab_git_tree_handoff_active_root()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    tree_row public.research_lab_autoresearch_trees;
    latest_tree public.research_lab_autoresearch_trees;
    active_count BIGINT;
    active_artifact_hash TEXT;
    active_manifest_hash TEXT;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree'),
        pg_catalog.hashtext(NEW.run_id::TEXT)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_private_model_version'),
        pg_catalog.hashtext('one_active_version')
    );

    SELECT *
      INTO tree_row
      FROM public.research_lab_autoresearch_trees
     WHERE tree_id = NEW.tree_id;
    SELECT *
      INTO latest_tree
      FROM public.research_lab_autoresearch_trees
     WHERE run_id = NEW.run_id
     ORDER BY tree_generation DESC, created_at DESC, tree_id DESC
     LIMIT 1;
    SELECT
        COUNT(*),
        MIN(model_artifact_hash),
        MIN(private_model_manifest_hash)
      INTO active_count, active_artifact_hash, active_manifest_hash
      FROM public.research_lab_private_model_version_current
     WHERE current_version_status = 'active';

    IF tree_row.tree_id IS NULL
       OR latest_tree.tree_id IS DISTINCT FROM NEW.tree_id
       OR active_count IS DISTINCT FROM 1::BIGINT
       OR active_artifact_hash IS DISTINCT FROM tree_row.root_artifact_hash
       OR active_manifest_hash IS DISTINCT FROM tree_row.root_manifest_hash THEN
        RAISE EXCEPTION 'research_lab_git_tree_handoff_stale_active_root'
            USING ERRCODE = '40001';
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS guard_research_lab_git_tree_handoff_active_root
    ON public.research_lab_autoresearch_tree_handoffs;
CREATE TRIGGER guard_research_lab_git_tree_handoff_active_root
    BEFORE INSERT ON public.research_lab_autoresearch_tree_handoffs
    FOR EACH ROW
    EXECUTE FUNCTION public.guard_research_lab_git_tree_handoff_active_root();

CREATE OR REPLACE FUNCTION public.create_research_lab_autoresearch_tree(
    requested_tree_id TEXT,
    requested_run_id UUID,
    requested_root_artifact_hash TEXT,
    requested_root_manifest_hash TEXT,
    requested_root_source_tree_hash TEXT,
    requested_root_git_commit TEXT,
    requested_root_image_digest TEXT,
    requested_policy_hash TEXT,
    requested_evaluator_commitment_hash TEXT,
    requested_tree_doc JSONB,
    requested_identity_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    inserted public.research_lab_autoresearch_trees;
    latest public.research_lab_autoresearch_trees;
    predecessor_event public.research_lab_autoresearch_tree_events;
    replacement_doc JSONB;
    requested_generation INTEGER := 0;
    requested_replaces_tree_id TEXT := NULL;
    requested_cancellation_event_hash TEXT := NULL;
    requested_replacement_hash TEXT := NULL;
    active_count BIGINT;
    active_artifact_hash TEXT;
    active_manifest_hash TEXT;
    created BOOLEAN := FALSE;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree'),
        pg_catalog.hashtext(requested_run_id::TEXT)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_private_model_version'),
        pg_catalog.hashtext('one_active_version')
    );

    SELECT
        COUNT(*),
        MIN(model_artifact_hash),
        MIN(private_model_manifest_hash)
      INTO active_count, active_artifact_hash, active_manifest_hash
      FROM public.research_lab_private_model_version_current
     WHERE current_version_status = 'active';
    IF active_count IS DISTINCT FROM 1::BIGINT
       OR active_artifact_hash IS DISTINCT FROM requested_root_artifact_hash
       OR active_manifest_hash IS DISTINCT FROM requested_root_manifest_hash THEN
        RAISE EXCEPTION 'research_lab_git_tree_create_stale_active_root'
            USING ERRCODE = '40001';
    END IF;

    SELECT *
      INTO inserted
      FROM public.research_lab_autoresearch_trees
     WHERE tree_id = requested_tree_id;
    IF inserted.tree_id IS NOT NULL THEN
        IF inserted.run_id IS DISTINCT FROM requested_run_id
           OR inserted.identity_hash IS DISTINCT FROM requested_identity_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_identity_conflict'
                USING ERRCODE = '40001';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'created', FALSE,
            'tree', pg_catalog.to_jsonb(inserted)
        );
    END IF;

    replacement_doc := requested_tree_doc->'replacement';
    IF replacement_doc IS NOT NULL THEN
        IF pg_catalog.jsonb_typeof(replacement_doc) <> 'object'
           OR replacement_doc->>'schema_version'
              IS DISTINCT FROM 'research_lab.git_tree_replacement.v1' THEN
            RAISE EXCEPTION 'research_lab_git_tree_replacement_invalid'
                USING ERRCODE = '23514';
        END IF;
        requested_generation := (replacement_doc->>'generation')::INTEGER;
        requested_replaces_tree_id := replacement_doc->>'replaces_tree_id';
        requested_cancellation_event_hash :=
            replacement_doc->>'cancellation_event_hash';
        requested_replacement_hash := replacement_doc->>'replacement_hash';
        IF replacement_doc->>'root_artifact_hash'
              IS DISTINCT FROM requested_root_artifact_hash
           OR replacement_doc->>'root_manifest_hash'
              IS DISTINCT FROM requested_root_manifest_hash
           OR replacement_doc->>'policy_hash'
              IS DISTINCT FROM requested_policy_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_replacement_root_conflict'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    SELECT *
      INTO latest
      FROM public.research_lab_autoresearch_trees
     WHERE run_id = requested_run_id
     ORDER BY tree_generation DESC, created_at DESC, tree_id DESC
     LIMIT 1;

    IF latest.tree_id IS NULL THEN
        IF replacement_doc IS NOT NULL OR requested_generation <> 0 THEN
            RAISE EXCEPTION 'research_lab_git_tree_replacement_predecessor_missing'
                USING ERRCODE = '40001';
        END IF;
    ELSE
        IF replacement_doc IS NULL THEN
            RAISE EXCEPTION 'research_lab_git_tree_replacement_authority_required'
                USING ERRCODE = '40001';
        END IF;
        IF requested_replaces_tree_id IS DISTINCT FROM latest.tree_id
           OR requested_generation IS DISTINCT FROM latest.tree_generation + 1 THEN
            RAISE EXCEPTION 'research_lab_git_tree_replacement_generation_conflict'
                USING ERRCODE = '40001';
        END IF;
        SELECT *
          INTO predecessor_event
          FROM public.research_lab_autoresearch_tree_events
         WHERE tree_id = latest.tree_id
         ORDER BY seq DESC
         LIMIT 1;
        IF predecessor_event.event_type
              IS DISTINCT FROM 'tree_cancelled_root_changed'
           OR predecessor_event.event_hash
              IS DISTINCT FROM requested_cancellation_event_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_replacement_predecessor_active'
                USING ERRCODE = '40001';
        END IF;
        IF replacement_doc->>'prior_root_artifact_hash'
              IS DISTINCT FROM latest.root_artifact_hash
           OR replacement_doc->>'prior_root_manifest_hash'
              IS DISTINCT FROM latest.root_manifest_hash
           OR replacement_doc->>'prior_policy_hash'
              IS DISTINCT FROM latest.policy_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_replacement_lineage_conflict'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    INSERT INTO public.research_lab_autoresearch_trees (
        tree_id, schema_version, run_id, tree_generation, replaces_tree_id,
        cancellation_event_hash, replacement_hash, root_artifact_hash,
        root_manifest_hash, root_source_tree_hash, root_git_commit,
        root_image_digest, policy_hash, evaluator_commitment_hash,
        tree_doc, identity_hash
    ) VALUES (
        requested_tree_id, 'research_lab.git_tree.v1', requested_run_id,
        requested_generation, requested_replaces_tree_id,
        requested_cancellation_event_hash, requested_replacement_hash,
        requested_root_artifact_hash, requested_root_manifest_hash,
        requested_root_source_tree_hash, requested_root_git_commit,
        requested_root_image_digest, requested_policy_hash,
        requested_evaluator_commitment_hash, requested_tree_doc,
        requested_identity_hash
    )
    RETURNING * INTO inserted;
    created := TRUE;

    RETURN pg_catalog.jsonb_build_object(
        'created', created,
        'tree', pg_catalog.to_jsonb(inserted)
    );
END;
$$;

-- Candidate creation and its official tree handoff must commit together.
-- A stale-root handoff failure rolls the candidate insert back, so restart
-- recovery cannot mistake an orphaned row for an official finalist.
CREATE OR REPLACE FUNCTION public.create_research_lab_git_tree_candidate_handoff(
    requested_candidate_doc JSONB,
    requested_tree_id TEXT,
    requested_run_id UUID,
    requested_candidate_id TEXT,
    requested_node_id TEXT,
    requested_root_git_commit TEXT,
    requested_node_git_commit TEXT,
    requested_lineage_hash TEXT,
    requested_handoff_doc JSONB,
    requested_handoff_hash TEXT,
    requested_previous_event_hash TEXT,
    requested_completed_event_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    candidate_row public.research_lab_candidate_artifacts;
    handoff_result JSONB;
    created BOOLEAN := FALSE;
BEGIN
    IF pg_catalog.jsonb_typeof(requested_candidate_doc) <> 'object'
       OR requested_candidate_doc->>'candidate_id'
          IS DISTINCT FROM requested_candidate_id
       OR requested_candidate_doc->>'run_id'
          IS DISTINCT FROM requested_run_id::TEXT
       OR requested_candidate_doc->>'git_tree_id'
          IS DISTINCT FROM requested_tree_id
       OR requested_candidate_doc->>'git_tree_node_id'
          IS DISTINCT FROM requested_node_id
       OR requested_candidate_doc->>'git_tree_root_commit'
          IS DISTINCT FROM requested_root_git_commit
       OR requested_candidate_doc->>'git_tree_node_commit'
          IS DISTINCT FROM requested_node_git_commit
       OR requested_candidate_doc->>'git_tree_lineage_hash'
          IS DISTINCT FROM requested_lineage_hash THEN
        RAISE EXCEPTION 'research_lab_git_tree_candidate_document_conflict'
            USING ERRCODE = '23514';
    END IF;

    SELECT *
      INTO candidate_row
      FROM public.research_lab_candidate_artifacts
     WHERE candidate_id = requested_candidate_id;
    IF candidate_row.candidate_id IS NULL THEN
        INSERT INTO public.research_lab_candidate_artifacts (
            candidate_id,
            schema_version,
            run_id,
            ticket_id,
            receipt_id,
            miner_hotkey,
            island,
            parent_artifact_hash,
            candidate_artifact_hash,
            private_model_manifest_hash,
            private_model_manifest_doc,
            candidate_patch_hash,
            candidate_patch_manifest,
            hypothesis_doc,
            redacted_public_summary,
            anchored_hash,
            candidate_kind,
            candidate_model_manifest_hash,
            candidate_model_manifest_doc,
            candidate_source_diff_hash,
            candidate_build_doc,
            git_tree_id,
            git_tree_node_id,
            git_tree_root_commit,
            git_tree_node_commit,
            git_tree_lineage_hash
        ) VALUES (
            requested_candidate_doc->>'candidate_id',
            requested_candidate_doc->>'schema_version',
            (requested_candidate_doc->>'run_id')::UUID,
            (requested_candidate_doc->>'ticket_id')::UUID,
            NULLIF(requested_candidate_doc->>'receipt_id', '')::UUID,
            requested_candidate_doc->>'miner_hotkey',
            requested_candidate_doc->>'island',
            requested_candidate_doc->>'parent_artifact_hash',
            requested_candidate_doc->>'candidate_artifact_hash',
            requested_candidate_doc->>'private_model_manifest_hash',
            requested_candidate_doc->'private_model_manifest_doc',
            requested_candidate_doc->>'candidate_patch_hash',
            requested_candidate_doc->'candidate_patch_manifest',
            requested_candidate_doc->'hypothesis_doc',
            requested_candidate_doc->>'redacted_public_summary',
            requested_candidate_doc->>'anchored_hash',
            requested_candidate_doc->>'candidate_kind',
            requested_candidate_doc->>'candidate_model_manifest_hash',
            requested_candidate_doc->'candidate_model_manifest_doc',
            requested_candidate_doc->>'candidate_source_diff_hash',
            requested_candidate_doc->'candidate_build_doc',
            requested_candidate_doc->>'git_tree_id',
            requested_candidate_doc->>'git_tree_node_id',
            requested_candidate_doc->>'git_tree_root_commit',
            requested_candidate_doc->>'git_tree_node_commit',
            requested_candidate_doc->>'git_tree_lineage_hash'
        )
        RETURNING * INTO candidate_row;
        created := TRUE;
    ELSIF candidate_row.candidate_artifact_hash
              IS DISTINCT FROM requested_candidate_doc->>'candidate_artifact_hash'
       OR candidate_row.candidate_model_manifest_hash
              IS DISTINCT FROM requested_candidate_doc->>'candidate_model_manifest_hash'
       OR candidate_row.candidate_model_manifest_doc
              IS DISTINCT FROM requested_candidate_doc->'candidate_model_manifest_doc'
       OR candidate_row.candidate_source_diff_hash
              IS DISTINCT FROM requested_candidate_doc->>'candidate_source_diff_hash'
       OR candidate_row.candidate_patch_hash
              IS DISTINCT FROM requested_candidate_doc->>'candidate_patch_hash'
       OR candidate_row.private_model_manifest_hash
              IS DISTINCT FROM requested_candidate_doc->>'private_model_manifest_hash' THEN
        RAISE EXCEPTION 'research_lab_git_tree_candidate_content_conflict'
            USING ERRCODE = '23514';
    END IF;

    handoff_result :=
        public.record_research_lab_autoresearch_tree_handoff(
            requested_tree_id,
            requested_run_id,
            requested_candidate_id,
            requested_node_id,
            requested_root_git_commit,
            requested_node_git_commit,
            requested_lineage_hash,
            requested_handoff_doc,
            requested_handoff_hash,
            requested_previous_event_hash,
            requested_completed_event_hash
        );

    RETURN pg_catalog.jsonb_build_object(
        'created', created,
        'candidate', pg_catalog.to_jsonb(candidate_row),
        'handoff', handoff_result
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_autoresearch_run_evaluation_usage(
    requested_run_id UUID
)
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
    WITH current_operations AS (
        SELECT DISTINCT ON (settlement.logical_operation_id)
            settlement.logical_operation_id,
            settlement.operation_status,
            settlement.settled_cost_microusd,
            settlement.provider_call_count
        FROM public.research_lab_autoresearch_operation_settlements settlement
        JOIN public.research_lab_autoresearch_trees tree
          ON tree.tree_id = settlement.tree_id
        WHERE tree.run_id = requested_run_id
          AND settlement.operation_kind = 'evaluation'
        ORDER BY settlement.logical_operation_id, settlement.seq DESC
    )
    SELECT pg_catalog.jsonb_build_object(
        'settled_cost_microusd',
        COALESCE(
            SUM(settled_cost_microusd) FILTER (
                WHERE operation_status IN (
                    'succeeded', 'failed', 'indeterminate'
                )
            ),
            0
        ),
        'provider_call_count',
        COALESCE(
            SUM(provider_call_count) FILTER (
                WHERE operation_status IN (
                    'succeeded', 'failed', 'indeterminate'
                )
            ),
            0
        ),
        'terminal_operation_count',
        COUNT(*) FILTER (
            WHERE operation_status IN (
                'succeeded', 'failed', 'indeterminate'
            )
        ),
        'unsettled_operation_ids',
        COALESCE(
            pg_catalog.jsonb_agg(
                logical_operation_id ORDER BY logical_operation_id
            ) FILTER (
                WHERE operation_status = 'reserved'
            ),
            '[]'::JSONB
        ),
        'indeterminate_operation_ids',
        COALESCE(
            pg_catalog.jsonb_agg(
                logical_operation_id ORDER BY logical_operation_id
            ) FILTER (
                WHERE operation_status = 'indeterminate'
            ),
            '[]'::JSONB
        )
    )
    FROM current_operations;
$$;

REVOKE ALL ON TABLE public.research_lab_autoresearch_run_tree_current
    FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_autoresearch_run_tree_current
    TO service_role;

REVOKE ALL ON FUNCTION public.guard_research_lab_git_tree_handoff_active_root()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE
    ON FUNCTION public.guard_research_lab_git_tree_handoff_active_root()
    TO service_role;

REVOKE ALL ON FUNCTION public.create_research_lab_git_tree_candidate_handoff(
    JSONB, TEXT, UUID, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, TEXT, TEXT
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE
    ON FUNCTION public.create_research_lab_git_tree_candidate_handoff(
        JSONB, TEXT, UUID, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, TEXT, TEXT
    )
    TO service_role;

REVOKE ALL ON FUNCTION public.research_lab_autoresearch_run_evaluation_usage(UUID)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE
    ON FUNCTION public.research_lab_autoresearch_run_evaluation_usage(UUID)
    TO service_role;

COMMENT ON VIEW public.research_lab_autoresearch_run_tree_current IS
    'Latest immutable Git-tree generation per run; older cancelled generations remain queryable in the base tables.';
COMMENT ON FUNCTION public.create_research_lab_git_tree_candidate_handoff(
    JSONB, TEXT, UUID, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, TEXT, TEXT
) IS
    'Atomically creates one selected candidate and hands it off only while its latest Git-tree root remains the active private model.';
COMMENT ON FUNCTION public.research_lab_autoresearch_run_evaluation_usage(UUID) IS
    'Run-wide settled evaluation usage plus interrupted-operation recovery evidence across every immutable tree generation.';

NOTIFY pgrst, 'reload schema';

COMMIT;
