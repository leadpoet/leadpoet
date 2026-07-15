-- Authoritative Git-tree autoresearch state (apply after migration 94).
--
-- This migration is additive. It does not change candidate scoring,
-- promotion, rewards, reimbursements, allocations, emissions, validator
-- weights, or chain submission. Historical flat/sequential rows remain
-- readable, but new autoresearch trees use the append-only records below.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE TABLE IF NOT EXISTS public.research_lab_autoresearch_trees (
    tree_id                     TEXT        PRIMARY KEY
                                           CHECK (tree_id ~ '^sha256:[0-9a-f]{64}$'),
    schema_version              TEXT        NOT NULL
                                           CHECK (schema_version = 'research_lab.git_tree.v1'),
    run_id                      UUID        NOT NULL UNIQUE,
    root_artifact_hash          TEXT        NOT NULL
                                           CHECK (root_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    root_manifest_hash          TEXT        NOT NULL
                                           CHECK (root_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    root_source_tree_hash       TEXT        NOT NULL
                                           CHECK (root_source_tree_hash ~ '^sha256:[0-9a-f]{64}$'),
    root_git_commit             TEXT        NOT NULL CHECK (root_git_commit ~ '^[0-9a-f]{64}$'),
    root_image_digest           TEXT        NOT NULL
                                           CHECK (root_image_digest ~ '^sha256:[0-9a-f]{64}$'),
    policy_hash                 TEXT        NOT NULL
                                           CHECK (policy_hash ~ '^sha256:[0-9a-f]{64}$'),
    evaluator_commitment_hash   TEXT        NOT NULL
                                           CHECK (evaluator_commitment_hash ~ '^sha256:[0-9a-f]{64}$'),
    tree_doc                    JSONB       NOT NULL CHECK (jsonb_typeof(tree_doc) = 'object'),
    identity_hash               TEXT        NOT NULL UNIQUE
                                           CHECK (identity_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_lab_autoresearch_tree_nodes (
    tree_id                     TEXT        NOT NULL
                                           REFERENCES public.research_lab_autoresearch_trees(tree_id)
                                           ON DELETE RESTRICT,
    node_id                     TEXT        NOT NULL
                                           CHECK (node_id ~ '^tree-node:[0-9a-f]{64}$'),
    schema_version              TEXT        NOT NULL
                                           CHECK (schema_version = 'research_lab.git_tree_node_identity.v1'),
    parent_node_id              TEXT        NOT NULL
                                           CHECK (parent_node_id = 'root'
                                                  OR parent_node_id ~ '^tree-node:[0-9a-f]{64}$'),
    root_branch_id              TEXT        NOT NULL
                                           CHECK (root_branch_id ~ '^tree-node:[0-9a-f]{64}$'),
    depth                       INTEGER     NOT NULL CHECK (depth BETWEEN 1 AND 128),
    child_ordinal               INTEGER     NOT NULL CHECK (child_ordinal BETWEEN 0 AND 128),
    generation_operation_id     TEXT        NOT NULL UNIQUE
                                           CHECK (generation_operation_id ~ '^sha256:[0-9a-f]{64}$'),
    node_doc                    JSONB       NOT NULL CHECK (jsonb_typeof(node_doc) = 'object'),
    identity_hash               TEXT        NOT NULL UNIQUE
                                           CHECK (identity_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tree_id, node_id),
    UNIQUE (tree_id, parent_node_id, depth, child_ordinal),
    CHECK (
        (parent_node_id = 'root' AND depth = 1 AND root_branch_id = node_id)
        OR (parent_node_id <> 'root' AND depth > 1)
    ),
    CHECK (parent_node_id <> node_id)
);

CREATE TABLE IF NOT EXISTS public.research_lab_autoresearch_tree_events (
    tree_id                     TEXT        NOT NULL
                                           REFERENCES public.research_lab_autoresearch_trees(tree_id)
                                           ON DELETE RESTRICT,
    seq                         BIGINT      NOT NULL CHECK (seq >= 0),
    schema_version              TEXT        NOT NULL
                                           CHECK (schema_version = 'research_lab.git_tree_event.v1'),
    event_type                  TEXT        NOT NULL CHECK (
                                           event_type IN (
                                               'tree_created',
                                               'node_planned',
                                               'node_generated',
                                               'node_build_started',
                                               'node_built',
                                               'node_evaluated',
                                               'node_failed',
                                               'frontier_committed',
                                               'checkpoint_committed',
                                               'tree_paused',
                                               'tree_cancelled_root_changed',
                                               'final_selected',
                                               'tree_completed',
                                               'tree_failed'
                                           )),
    node_id                     TEXT        NULL
                                           CHECK (node_id IS NULL
                                                  OR node_id ~ '^tree-node:[0-9a-f]{64}$'),
    previous_event_hash         TEXT        NOT NULL
                                           CHECK (previous_event_hash ~ '^sha256:[0-9a-f]{64}$'),
    event_doc                   JSONB       NOT NULL CHECK (jsonb_typeof(event_doc) = 'object'),
    event_hash                  TEXT        NOT NULL UNIQUE
                                           CHECK (event_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tree_id, seq),
    UNIQUE (tree_id, event_hash)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_research_lab_tree_final_selected
    ON public.research_lab_autoresearch_tree_events(tree_id)
    WHERE event_type = 'final_selected';

CREATE UNIQUE INDEX IF NOT EXISTS uq_research_lab_tree_node_generated
    ON public.research_lab_autoresearch_tree_events(tree_id, node_id)
    WHERE event_type = 'node_generated';

CREATE UNIQUE INDEX IF NOT EXISTS uq_research_lab_tree_terminal
    ON public.research_lab_autoresearch_tree_events(tree_id)
    WHERE event_type IN (
        'tree_completed', 'tree_failed', 'tree_cancelled_root_changed'
    );

CREATE TABLE IF NOT EXISTS public.research_lab_autoresearch_operation_settlements (
    logical_operation_id        TEXT        NOT NULL
                                           CHECK (logical_operation_id ~ '^sha256:[0-9a-f]{64}$'),
    seq                         INTEGER     NOT NULL CHECK (seq >= 0),
    schema_version              TEXT        NOT NULL
                                           CHECK (schema_version = 'research_lab.git_tree_operation.v1'),
    tree_id                     TEXT        NOT NULL
                                           REFERENCES public.research_lab_autoresearch_trees(tree_id)
                                           ON DELETE RESTRICT,
    node_id                     TEXT        NULL
                                           CHECK (node_id IS NULL
                                                  OR node_id ~ '^tree-node:[0-9a-f]{64}$'),
    operation_kind              TEXT        NOT NULL CHECK (
                                           operation_kind IN (
                                               'generation', 'build', 'provider',
                                               'evaluation', 'artifact', 'checkpoint'
                                           )),
    operation_status            TEXT        NOT NULL CHECK (
                                           operation_status IN (
                                               'reserved', 'succeeded', 'failed', 'indeterminate'
                                           )),
    request_hash                TEXT        NOT NULL
                                           CHECK (request_hash ~ '^sha256:[0-9a-f]{64}$'),
    result_hash                 TEXT        NULL
                                           CHECK (result_hash IS NULL
                                                  OR result_hash ~ '^sha256:[0-9a-f]{64}$'),
    settled_cost_microusd       BIGINT      NOT NULL DEFAULT 0
                                           CHECK (settled_cost_microusd >= 0),
    provider_call_count         INTEGER     NOT NULL DEFAULT 0
                                           CHECK (provider_call_count >= 0),
    settlement_doc              JSONB       NOT NULL CHECK (jsonb_typeof(settlement_doc) = 'object'),
    transition_hash             TEXT        NOT NULL UNIQUE
                                           CHECK (transition_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (logical_operation_id, seq)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_research_lab_tree_operation_terminal
    ON public.research_lab_autoresearch_operation_settlements(logical_operation_id)
    WHERE operation_status IN ('succeeded', 'failed', 'indeterminate');

CREATE TABLE IF NOT EXISTS public.research_lab_autoresearch_frontier_commitments (
    tree_id                     TEXT        NOT NULL
                                           REFERENCES public.research_lab_autoresearch_trees(tree_id)
                                           ON DELETE RESTRICT,
    round_index                 INTEGER     NOT NULL CHECK (round_index >= 0),
    schema_version              TEXT        NOT NULL
                                           CHECK (schema_version = 'research_lab.git_tree_frontier.v1'),
    expected_previous_hash      TEXT        NOT NULL
                                           CHECK (expected_previous_hash ~ '^sha256:[0-9a-f]{64}$'),
    frontier_hash               TEXT        NOT NULL
                                           CHECK (frontier_hash ~ '^sha256:[0-9a-f]{64}$'),
    frontier_doc                JSONB       NOT NULL CHECK (jsonb_typeof(frontier_doc) = 'object'),
    commitment_hash             TEXT        NOT NULL UNIQUE
                                           CHECK (commitment_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tree_id, round_index),
    UNIQUE (tree_id, frontier_hash)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_tree_nodes_parent
    ON public.research_lab_autoresearch_tree_nodes(tree_id, parent_node_id, depth, child_ordinal);
CREATE INDEX IF NOT EXISTS idx_research_lab_tree_events_created
    ON public.research_lab_autoresearch_tree_events(tree_id, created_at DESC, seq DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_tree_operations_tree
    ON public.research_lab_autoresearch_operation_settlements(tree_id, node_id, created_at DESC);

ALTER TABLE public.research_lab_candidate_artifacts
    ADD COLUMN IF NOT EXISTS git_tree_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS git_tree_node_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS git_tree_root_commit TEXT NULL,
    ADD COLUMN IF NOT EXISTS git_tree_node_commit TEXT NULL,
    ADD COLUMN IF NOT EXISTS git_tree_lineage_hash TEXT NULL;

ALTER TABLE public.research_lab_candidate_artifacts
    DROP CONSTRAINT IF EXISTS research_lab_candidate_artifacts_git_tree_id_check,
    ADD CONSTRAINT research_lab_candidate_artifacts_git_tree_id_check
        CHECK (git_tree_id IS NULL OR git_tree_id ~ '^sha256:[0-9a-f]{64}$') NOT VALID,
    DROP CONSTRAINT IF EXISTS research_lab_candidate_artifacts_git_tree_node_id_check,
    ADD CONSTRAINT research_lab_candidate_artifacts_git_tree_node_id_check
        CHECK (git_tree_node_id IS NULL OR git_tree_node_id ~ '^tree-node:[0-9a-f]{64}$') NOT VALID,
    DROP CONSTRAINT IF EXISTS research_lab_candidate_artifacts_git_tree_commits_check,
    ADD CONSTRAINT research_lab_candidate_artifacts_git_tree_commits_check CHECK (
        (git_tree_id IS NULL AND git_tree_node_id IS NULL
         AND git_tree_root_commit IS NULL AND git_tree_node_commit IS NULL
         AND git_tree_lineage_hash IS NULL)
        OR
        (git_tree_id IS NOT NULL AND git_tree_node_id IS NOT NULL
         AND git_tree_root_commit ~ '^[0-9a-f]{64}$'
         AND git_tree_node_commit ~ '^[0-9a-f]{64}$'
         AND git_tree_lineage_hash ~ '^sha256:[0-9a-f]{64}$')
    ) NOT VALID;

CREATE UNIQUE INDEX IF NOT EXISTS uq_research_lab_candidate_one_per_git_tree
    ON public.research_lab_candidate_artifacts(git_tree_id)
    WHERE git_tree_id IS NOT NULL;

ALTER TABLE public.research_lab_candidate_artifacts
    VALIDATE CONSTRAINT research_lab_candidate_artifacts_git_tree_id_check;
ALTER TABLE public.research_lab_candidate_artifacts
    VALIDATE CONSTRAINT research_lab_candidate_artifacts_git_tree_node_id_check;
ALTER TABLE public.research_lab_candidate_artifacts
    VALIDATE CONSTRAINT research_lab_candidate_artifacts_git_tree_commits_check;

-- A content-addressed candidate can already exist from an earlier run. Keep
-- the immutable candidate row reusable while recording exactly one official
-- scoring handoff for every tree.
CREATE TABLE IF NOT EXISTS public.research_lab_autoresearch_tree_handoffs (
    tree_id                 TEXT        PRIMARY KEY
                                       REFERENCES public.research_lab_autoresearch_trees(tree_id)
                                       ON DELETE RESTRICT,
    schema_version          TEXT        NOT NULL CHECK (
                                       schema_version = 'research_lab.git_tree_candidate_handoff.v1'
                                   ),
    run_id                  UUID        NOT NULL UNIQUE,
    candidate_id            TEXT        NOT NULL
                                       REFERENCES public.research_lab_candidate_artifacts(candidate_id)
                                       ON DELETE RESTRICT,
    node_id                 TEXT        NOT NULL
                                       CHECK (node_id ~ '^tree-node:[0-9a-f]{64}$'),
    root_git_commit         TEXT        NOT NULL CHECK (root_git_commit ~ '^[0-9a-f]{64}$'),
    node_git_commit         TEXT        NOT NULL CHECK (node_git_commit ~ '^[0-9a-f]{64}$'),
    lineage_hash            TEXT        NOT NULL CHECK (lineage_hash ~ '^sha256:[0-9a-f]{64}$'),
    handoff_doc             JSONB       NOT NULL CHECK (jsonb_typeof(handoff_doc) = 'object'),
    handoff_hash            TEXT        NOT NULL UNIQUE
                                       CHECK (handoff_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE VIEW public.research_lab_autoresearch_tree_node_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (n.tree_id, n.node_id)
    n.*,
    e.event_type AS current_event_type,
    e.event_doc AS current_event_doc,
    e.event_hash AS current_event_hash,
    e.created_at AS current_event_at
FROM public.research_lab_autoresearch_tree_nodes n
LEFT JOIN public.research_lab_autoresearch_tree_events e
  ON e.tree_id = n.tree_id AND e.node_id = n.node_id
ORDER BY n.tree_id, n.node_id, e.seq DESC NULLS LAST;

CREATE OR REPLACE VIEW public.research_lab_autoresearch_operation_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (logical_operation_id)
    *
FROM public.research_lab_autoresearch_operation_settlements
ORDER BY logical_operation_id, seq DESC;

CREATE OR REPLACE VIEW public.research_lab_autoresearch_tree_current
WITH (security_invoker = true) AS
SELECT
    t.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.event_doc AS current_event_doc,
    e.event_hash AS current_event_hash,
    e.created_at AS current_event_at,
    f.round_index AS current_round_index,
    f.frontier_hash AS current_frontier_hash,
    f.frontier_doc AS current_frontier_doc
FROM public.research_lab_autoresearch_trees t
LEFT JOIN LATERAL (
    SELECT * FROM public.research_lab_autoresearch_tree_events te
    WHERE te.tree_id = t.tree_id ORDER BY te.seq DESC LIMIT 1
) e ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM public.research_lab_autoresearch_frontier_commitments fc
    WHERE fc.tree_id = t.tree_id ORDER BY fc.round_index DESC LIMIT 1
) f ON TRUE;

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
    created BOOLEAN := FALSE;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree'),
        pg_catalog.hashtext(requested_run_id::TEXT)
    );
    INSERT INTO public.research_lab_autoresearch_trees (
        tree_id, schema_version, run_id, root_artifact_hash,
        root_manifest_hash, root_source_tree_hash, root_git_commit,
        root_image_digest, policy_hash, evaluator_commitment_hash,
        tree_doc, identity_hash
    ) VALUES (
        requested_tree_id, 'research_lab.git_tree.v1', requested_run_id,
        requested_root_artifact_hash, requested_root_manifest_hash,
        requested_root_source_tree_hash, requested_root_git_commit,
        requested_root_image_digest, requested_policy_hash,
        requested_evaluator_commitment_hash, requested_tree_doc,
        requested_identity_hash
    ) ON CONFLICT (tree_id) DO NOTHING RETURNING * INTO inserted;
    created := inserted.tree_id IS NOT NULL;
    IF inserted.tree_id IS NULL THEN
        SELECT * INTO inserted FROM public.research_lab_autoresearch_trees
        WHERE tree_id = requested_tree_id;
        IF inserted.run_id IS DISTINCT FROM requested_run_id
           OR inserted.identity_hash IS DISTINCT FROM requested_identity_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_identity_conflict'
                USING ERRCODE = '40001';
        END IF;
    END IF;
    RETURN pg_catalog.jsonb_build_object('created', created, 'tree', pg_catalog.to_jsonb(inserted));
END;
$$;

CREATE OR REPLACE FUNCTION public.plan_research_lab_autoresearch_tree_node(
    requested_tree_id TEXT,
    requested_node_id TEXT,
    requested_parent_node_id TEXT,
    requested_root_branch_id TEXT,
    requested_depth INTEGER,
    requested_child_ordinal INTEGER,
    requested_generation_operation_id TEXT,
    requested_generation_request_hash TEXT,
    requested_generation_transition_hash TEXT,
    requested_node_doc JSONB,
    requested_identity_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    parent_depth INTEGER;
    parent_branch TEXT;
    inserted public.research_lab_autoresearch_tree_nodes;
    reserved public.research_lab_autoresearch_operation_settlements;
    created BOOLEAN := FALSE;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree_node'),
        pg_catalog.hashtext(requested_tree_id || ':' || requested_node_id)
    );
    IF requested_parent_node_id <> 'root' THEN
        SELECT depth, root_branch_id INTO parent_depth, parent_branch
        FROM public.research_lab_autoresearch_tree_nodes
        WHERE tree_id = requested_tree_id AND node_id = requested_parent_node_id;
        IF parent_depth IS NULL
           OR requested_depth <> parent_depth + 1
           OR requested_root_branch_id IS DISTINCT FROM parent_branch THEN
            RAISE EXCEPTION 'research_lab_git_tree_parent_conflict'
                USING ERRCODE = '40001';
        END IF;
    END IF;
    INSERT INTO public.research_lab_autoresearch_tree_nodes (
        tree_id, node_id, schema_version, parent_node_id, root_branch_id,
        depth, child_ordinal, generation_operation_id, node_doc, identity_hash
    ) VALUES (
        requested_tree_id, requested_node_id,
        'research_lab.git_tree_node_identity.v1', requested_parent_node_id,
        requested_root_branch_id, requested_depth, requested_child_ordinal,
        requested_generation_operation_id, requested_node_doc,
        requested_identity_hash
    ) ON CONFLICT (tree_id, node_id) DO NOTHING RETURNING * INTO inserted;
    created := inserted.node_id IS NOT NULL;
    IF inserted.node_id IS NULL THEN
        SELECT * INTO inserted FROM public.research_lab_autoresearch_tree_nodes
        WHERE tree_id = requested_tree_id AND node_id = requested_node_id;
        IF inserted.identity_hash IS DISTINCT FROM requested_identity_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_node_identity_conflict'
                USING ERRCODE = '40001';
        END IF;
    END IF;
    INSERT INTO public.research_lab_autoresearch_operation_settlements (
        logical_operation_id, seq, schema_version, tree_id, node_id,
        operation_kind, operation_status, request_hash, result_hash,
        settled_cost_microusd, provider_call_count, settlement_doc,
        transition_hash
    ) VALUES (
        requested_generation_operation_id, 0,
        'research_lab.git_tree_operation.v1', requested_tree_id,
        requested_node_id, 'generation', 'reserved',
        requested_generation_request_hash, NULL, 0, 0, '{}'::JSONB,
        requested_generation_transition_hash
    ) ON CONFLICT (logical_operation_id, seq) DO NOTHING
      RETURNING * INTO reserved;
    IF reserved.logical_operation_id IS NULL THEN
        SELECT * INTO reserved
        FROM public.research_lab_autoresearch_operation_settlements
        WHERE logical_operation_id = requested_generation_operation_id
        ORDER BY seq DESC LIMIT 1;
        IF reserved.tree_id IS DISTINCT FROM requested_tree_id
           OR reserved.node_id IS DISTINCT FROM requested_node_id
           OR reserved.operation_kind IS DISTINCT FROM 'generation'
           OR reserved.request_hash IS DISTINCT FROM requested_generation_request_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_generation_reservation_conflict'
                USING ERRCODE = '40001';
        END IF;
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'created', created,
        'node', pg_catalog.to_jsonb(inserted),
        'operation', pg_catalog.to_jsonb(reserved)
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.append_research_lab_autoresearch_tree_event(
    requested_tree_id TEXT,
    requested_event_type TEXT,
    requested_node_id TEXT,
    requested_previous_event_hash TEXT,
    requested_event_doc JSONB,
    requested_event_hash TEXT
)
RETURNS public.research_lab_autoresearch_tree_events
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    latest_seq BIGINT;
    latest_hash TEXT;
    latest_type TEXT;
    existing public.research_lab_autoresearch_tree_events;
    inserted public.research_lab_autoresearch_tree_events;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree_event'),
        pg_catalog.hashtext(requested_tree_id)
    );
    SELECT * INTO existing
    FROM public.research_lab_autoresearch_tree_events
    WHERE tree_id = requested_tree_id
      AND event_type = requested_event_type
      AND node_id IS NOT DISTINCT FROM NULLIF(requested_node_id, '')
      AND event_doc = requested_event_doc
    ORDER BY seq ASC
    LIMIT 1;
    IF existing.event_hash IS NOT NULL THEN
        IF existing.previous_event_hash IS DISTINCT FROM requested_previous_event_hash
           OR existing.event_hash IS DISTINCT FROM requested_event_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_event_identity_conflict'
                USING ERRCODE = '40001';
        END IF;
        RETURN existing;
    END IF;
    SELECT * INTO existing
    FROM public.research_lab_autoresearch_tree_events
    WHERE event_hash = requested_event_hash;
    IF existing.event_hash IS NOT NULL THEN
        IF existing.tree_id IS DISTINCT FROM requested_tree_id
           OR existing.event_type IS DISTINCT FROM requested_event_type
           OR existing.node_id IS DISTINCT FROM NULLIF(requested_node_id, '')
           OR existing.previous_event_hash IS DISTINCT FROM requested_previous_event_hash
           OR existing.event_doc IS DISTINCT FROM requested_event_doc THEN
            RAISE EXCEPTION 'research_lab_git_tree_event_hash_conflict'
                USING ERRCODE = '40001';
        END IF;
        RETURN existing;
    END IF;
    SELECT seq, event_hash, event_type INTO latest_seq, latest_hash, latest_type
    FROM public.research_lab_autoresearch_tree_events
    WHERE tree_id = requested_tree_id ORDER BY seq DESC LIMIT 1;
    IF latest_type IN (
        'tree_completed', 'tree_failed', 'tree_cancelled_root_changed'
    ) THEN
        RAISE EXCEPTION 'research_lab_git_tree_already_terminal'
            USING ERRCODE = '40001';
    END IF;
    IF COALESCE(latest_hash, 'sha256:' || repeat('0', 64))
       IS DISTINCT FROM requested_previous_event_hash THEN
        RAISE EXCEPTION 'research_lab_git_tree_event_conflict'
            USING ERRCODE = '40001';
    END IF;
    INSERT INTO public.research_lab_autoresearch_tree_events (
        tree_id, seq, schema_version, event_type, node_id,
        previous_event_hash, event_doc, event_hash
    ) VALUES (
        requested_tree_id, COALESCE(latest_seq + 1, 0),
        'research_lab.git_tree_event.v1', requested_event_type,
        NULLIF(requested_node_id, ''), requested_previous_event_hash,
        requested_event_doc, requested_event_hash
    ) RETURNING * INTO inserted;
    RETURN inserted;
END;
$$;

CREATE OR REPLACE FUNCTION public.transition_research_lab_autoresearch_operation(
    requested_logical_operation_id TEXT,
    requested_tree_id TEXT,
    requested_node_id TEXT,
    requested_operation_kind TEXT,
    requested_operation_status TEXT,
    requested_request_hash TEXT,
    requested_result_hash TEXT,
    requested_settled_cost_microusd BIGINT,
    requested_provider_call_count INTEGER,
    requested_settlement_doc JSONB,
    requested_transition_hash TEXT,
    expected_current_status TEXT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    latest public.research_lab_autoresearch_operation_settlements;
    inserted public.research_lab_autoresearch_operation_settlements;
    created BOOLEAN := FALSE;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree_operation'),
        pg_catalog.hashtext(requested_logical_operation_id)
    );
    SELECT * INTO latest
    FROM public.research_lab_autoresearch_operation_settlements
    WHERE logical_operation_id = requested_logical_operation_id
    ORDER BY seq DESC LIMIT 1;
    IF latest.logical_operation_id IS NOT NULL THEN
        IF latest.tree_id IS DISTINCT FROM requested_tree_id
           OR latest.node_id IS DISTINCT FROM NULLIF(requested_node_id, '')
           OR latest.operation_kind IS DISTINCT FROM requested_operation_kind
           OR latest.request_hash IS DISTINCT FROM requested_request_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_operation_identity_conflict'
                USING ERRCODE = '40001';
        END IF;
        IF latest.operation_status IN ('succeeded', 'failed', 'indeterminate') THEN
            IF latest.transition_hash = requested_transition_hash THEN
                RETURN pg_catalog.jsonb_build_object(
                    'created', FALSE, 'operation', pg_catalog.to_jsonb(latest)
                );
            END IF;
            RAISE EXCEPTION 'research_lab_git_tree_operation_already_terminal'
                USING ERRCODE = '40001';
        END IF;
    END IF;
    IF expected_current_status IS NOT NULL
       AND COALESCE(latest.operation_status, '') IS DISTINCT FROM expected_current_status THEN
        RAISE EXCEPTION 'research_lab_git_tree_operation_state_conflict'
            USING ERRCODE = '40001';
    END IF;
    IF latest.logical_operation_id IS NULL AND requested_operation_status <> 'reserved' THEN
        RAISE EXCEPTION 'research_lab_git_tree_operation_not_reserved'
            USING ERRCODE = '40001';
    END IF;
    IF latest.operation_status = 'reserved' AND requested_operation_status = 'reserved' THEN
        RETURN pg_catalog.jsonb_build_object(
            'created', FALSE, 'operation', pg_catalog.to_jsonb(latest)
        );
    END IF;
    INSERT INTO public.research_lab_autoresearch_operation_settlements (
        logical_operation_id, seq, schema_version, tree_id, node_id,
        operation_kind, operation_status, request_hash, result_hash,
        settled_cost_microusd, provider_call_count, settlement_doc,
        transition_hash
    ) VALUES (
        requested_logical_operation_id, COALESCE(latest.seq + 1, 0),
        'research_lab.git_tree_operation.v1', requested_tree_id,
        NULLIF(requested_node_id, ''), requested_operation_kind,
        requested_operation_status, requested_request_hash,
        NULLIF(requested_result_hash, ''), requested_settled_cost_microusd,
        requested_provider_call_count, requested_settlement_doc,
        requested_transition_hash
    ) RETURNING * INTO inserted;
    created := TRUE;
    RETURN pg_catalog.jsonb_build_object(
        'created', created, 'operation', pg_catalog.to_jsonb(inserted)
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.commit_research_lab_autoresearch_frontier(
    requested_tree_id TEXT,
    requested_round_index INTEGER,
    requested_expected_previous_hash TEXT,
    requested_frontier_hash TEXT,
    requested_frontier_doc JSONB,
    requested_commitment_hash TEXT
)
RETURNS public.research_lab_autoresearch_frontier_commitments
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    latest public.research_lab_autoresearch_frontier_commitments;
    inserted public.research_lab_autoresearch_frontier_commitments;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree_frontier'),
        pg_catalog.hashtext(requested_tree_id)
    );
    SELECT * INTO inserted
    FROM public.research_lab_autoresearch_frontier_commitments
    WHERE tree_id = requested_tree_id
      AND frontier_hash = requested_frontier_hash;
    IF inserted.commitment_hash IS NOT NULL THEN
        IF inserted.round_index IS DISTINCT FROM requested_round_index
           OR inserted.expected_previous_hash IS DISTINCT FROM requested_expected_previous_hash
           OR inserted.frontier_doc IS DISTINCT FROM requested_frontier_doc
           OR inserted.commitment_hash IS DISTINCT FROM requested_commitment_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_frontier_identity_conflict'
                USING ERRCODE = '40001';
        END IF;
        RETURN inserted;
    END IF;
    SELECT * INTO latest
    FROM public.research_lab_autoresearch_frontier_commitments
    WHERE tree_id = requested_tree_id ORDER BY round_index DESC LIMIT 1;
    IF COALESCE(latest.frontier_hash, 'sha256:' || repeat('0', 64))
       IS DISTINCT FROM requested_expected_previous_hash
       OR requested_round_index <> COALESCE(latest.round_index + 1, 0) THEN
        RAISE EXCEPTION 'research_lab_git_tree_frontier_conflict'
            USING ERRCODE = '40001';
    END IF;
    INSERT INTO public.research_lab_autoresearch_frontier_commitments (
        tree_id, round_index, schema_version, expected_previous_hash,
        frontier_hash, frontier_doc, commitment_hash
    ) VALUES (
        requested_tree_id, requested_round_index,
        'research_lab.git_tree_frontier.v1', requested_expected_previous_hash,
        requested_frontier_hash, requested_frontier_doc,
        requested_commitment_hash
    ) ON CONFLICT (commitment_hash) DO NOTHING RETURNING * INTO inserted;
    IF inserted.commitment_hash IS NULL THEN
        SELECT * INTO inserted
        FROM public.research_lab_autoresearch_frontier_commitments
        WHERE commitment_hash = requested_commitment_hash;
        IF inserted.tree_id IS DISTINCT FROM requested_tree_id
           OR inserted.round_index IS DISTINCT FROM requested_round_index
           OR inserted.expected_previous_hash IS DISTINCT FROM requested_expected_previous_hash
           OR inserted.frontier_hash IS DISTINCT FROM requested_frontier_hash
           OR inserted.frontier_doc IS DISTINCT FROM requested_frontier_doc THEN
            RAISE EXCEPTION 'research_lab_git_tree_frontier_commitment_conflict'
                USING ERRCODE = '40001';
        END IF;
    END IF;
    RETURN inserted;
END;
$$;

CREATE OR REPLACE FUNCTION public.select_research_lab_autoresearch_tree_final(
    requested_tree_id TEXT,
    requested_node_id TEXT,
    requested_selection_hash TEXT,
    requested_selection_doc JSONB,
    requested_previous_event_hash TEXT,
    requested_event_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    latest_seq BIGINT;
    latest_hash TEXT;
    latest_type TEXT;
    selected_node_event public.research_lab_autoresearch_tree_events;
    existing public.research_lab_autoresearch_tree_events;
    inserted public.research_lab_autoresearch_tree_events;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree_final'),
        pg_catalog.hashtext(requested_tree_id)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree_event'),
        pg_catalog.hashtext(requested_tree_id)
    );
    SELECT * INTO selected_node_event
    FROM public.research_lab_autoresearch_tree_events
    WHERE tree_id = requested_tree_id
      AND node_id = requested_node_id
      AND event_type = 'node_evaluated'
    ORDER BY seq DESC
    LIMIT 1;
    IF selected_node_event.event_hash IS NULL
       OR selected_node_event.event_doc->>'status' IS DISTINCT FROM 'eligible'
       OR requested_selection_doc->>'tree_id' IS DISTINCT FROM requested_tree_id
       OR requested_selection_doc->>'selected_node_id' IS DISTINCT FROM requested_node_id
       OR requested_selection_doc->>'paid_finalist_count' IS DISTINCT FROM '1'
       OR requested_selection_doc->>'selected_candidate_artifact_hash'
          IS DISTINCT FROM selected_node_event.event_doc->>'candidate_artifact_hash'
       OR requested_selection_doc->>'selected_node_git_commit'
          IS DISTINCT FROM selected_node_event.event_doc->>'git_commit'
       OR requested_selection_doc->>'selected_lineage_hash'
          IS DISTINCT FROM selected_node_event.event_doc->>'lineage_hash'
    THEN
        RAISE EXCEPTION 'research_lab_git_tree_final_selection_authority_conflict'
            USING ERRCODE = '40001';
    END IF;
    SELECT * INTO existing
    FROM public.research_lab_autoresearch_tree_events
    WHERE tree_id = requested_tree_id AND event_type = 'final_selected'
    LIMIT 1;
    IF existing.event_hash IS NOT NULL THEN
        IF existing.node_id IS DISTINCT FROM requested_node_id
           OR existing.event_doc->>'selection_hash'
              IS DISTINCT FROM requested_selection_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_final_selection_conflict'
                USING ERRCODE = '40001';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'created', FALSE, 'event', pg_catalog.to_jsonb(existing)
        );
    END IF;
    SELECT seq, event_hash, event_type INTO latest_seq, latest_hash, latest_type
    FROM public.research_lab_autoresearch_tree_events
    WHERE tree_id = requested_tree_id ORDER BY seq DESC LIMIT 1;
    IF latest_type IN (
        'tree_completed', 'tree_failed', 'tree_cancelled_root_changed'
    ) THEN
        RAISE EXCEPTION 'research_lab_git_tree_already_terminal'
            USING ERRCODE = '40001';
    END IF;
    IF COALESCE(latest_hash, 'sha256:' || repeat('0', 64))
       IS DISTINCT FROM requested_previous_event_hash THEN
        RAISE EXCEPTION 'research_lab_git_tree_event_conflict'
            USING ERRCODE = '40001';
    END IF;
    INSERT INTO public.research_lab_autoresearch_tree_events (
        tree_id, seq, schema_version, event_type, node_id,
        previous_event_hash, event_doc, event_hash
    ) VALUES (
        requested_tree_id, COALESCE(latest_seq + 1, 0),
        'research_lab.git_tree_event.v1', 'final_selected',
        requested_node_id, requested_previous_event_hash,
        pg_catalog.jsonb_build_object(
            'selection_hash', requested_selection_hash,
            'selection', requested_selection_doc
        ),
        requested_event_hash
    ) RETURNING * INTO inserted;
    RETURN pg_catalog.jsonb_build_object(
        'created', TRUE, 'event', pg_catalog.to_jsonb(inserted)
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.record_research_lab_autoresearch_tree_handoff(
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
    tree_row public.research_lab_autoresearch_trees;
    candidate_row public.research_lab_candidate_artifacts;
    final_event public.research_lab_autoresearch_tree_events;
    selected_node_event public.research_lab_autoresearch_tree_events;
    latest_event public.research_lab_autoresearch_tree_events;
    completed_event public.research_lab_autoresearch_tree_events;
    inserted public.research_lab_autoresearch_tree_handoffs;
    completed_doc JSONB;
    created BOOLEAN := FALSE;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree_handoff'),
        pg_catalog.hashtext(requested_tree_id)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_git_tree_event'),
        pg_catalog.hashtext(requested_tree_id)
    );
    SELECT * INTO tree_row
    FROM public.research_lab_autoresearch_trees
    WHERE tree_id = requested_tree_id;
    SELECT * INTO final_event
    FROM public.research_lab_autoresearch_tree_events
    WHERE tree_id = requested_tree_id AND event_type = 'final_selected'
    LIMIT 1;
    SELECT * INTO selected_node_event
    FROM public.research_lab_autoresearch_tree_events
    WHERE tree_id = requested_tree_id
      AND node_id = requested_node_id
      AND event_type = 'node_evaluated'
    ORDER BY seq DESC
    LIMIT 1;
    SELECT * INTO candidate_row
    FROM public.research_lab_candidate_artifacts
    WHERE candidate_id = requested_candidate_id;
    IF candidate_row.candidate_id IS NULL THEN
        RAISE EXCEPTION 'research_lab_git_tree_handoff_candidate_missing'
            USING ERRCODE = '23503';
    END IF;
    IF tree_row.tree_id IS NULL
       OR tree_row.run_id IS DISTINCT FROM requested_run_id
       OR tree_row.root_git_commit IS DISTINCT FROM requested_root_git_commit
       OR final_event.node_id IS DISTINCT FROM requested_node_id
       OR final_event.event_doc->'selection'->>'paid_finalist_count' IS DISTINCT FROM '1'
       OR final_event.event_doc->'selection'->>'selected_candidate_artifact_hash'
          IS DISTINCT FROM candidate_row.candidate_artifact_hash
       OR final_event.event_doc->'selection'->>'selected_node_git_commit'
          IS DISTINCT FROM requested_node_git_commit
       OR final_event.event_doc->'selection'->>'selected_lineage_hash'
          IS DISTINCT FROM requested_lineage_hash
       OR selected_node_event.event_hash IS NULL
       OR selected_node_event.event_doc->>'status' IS DISTINCT FROM 'eligible'
       OR selected_node_event.event_doc->>'candidate_artifact_hash'
          IS DISTINCT FROM candidate_row.candidate_artifact_hash
       OR selected_node_event.event_doc->>'git_commit'
          IS DISTINCT FROM requested_node_git_commit
       OR selected_node_event.event_doc->>'lineage_hash'
          IS DISTINCT FROM requested_lineage_hash
       OR requested_candidate_id IS DISTINCT FROM
          'candidate:' || pg_catalog.substr(candidate_row.candidate_artifact_hash, 8)
       OR requested_handoff_doc->>'schema_version'
          IS DISTINCT FROM 'research_lab.git_tree_candidate_handoff.v1'
       OR requested_handoff_doc->>'tree_id' IS DISTINCT FROM requested_tree_id
       OR requested_handoff_doc->>'run_id' IS DISTINCT FROM requested_run_id::TEXT
       OR requested_handoff_doc->>'candidate_id' IS DISTINCT FROM requested_candidate_id
       OR requested_handoff_doc->>'node_id' IS DISTINCT FROM requested_node_id
       OR requested_handoff_doc->>'root_git_commit'
          IS DISTINCT FROM requested_root_git_commit
       OR requested_handoff_doc->>'node_git_commit'
          IS DISTINCT FROM requested_node_git_commit
       OR requested_handoff_doc->>'lineage_hash'
          IS DISTINCT FROM requested_lineage_hash
    THEN
        RAISE EXCEPTION 'research_lab_git_tree_handoff_authority_conflict'
            USING ERRCODE = '40001';
    END IF;
    completed_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'research_lab.git_tree_completed.v1',
        'tree_id', requested_tree_id,
        'run_id', requested_run_id::TEXT,
        'candidate_id', requested_candidate_id,
        'node_id', requested_node_id,
        'handoff_hash', requested_handoff_hash,
        'paid_finalist_count', 1
    );
    SELECT * INTO completed_event
    FROM public.research_lab_autoresearch_tree_events
    WHERE tree_id = requested_tree_id AND event_type = 'tree_completed'
    LIMIT 1;
    IF completed_event.event_hash IS NOT NULL THEN
        IF completed_event.node_id IS DISTINCT FROM requested_node_id
           OR completed_event.event_doc IS DISTINCT FROM completed_doc
           OR completed_event.event_hash IS DISTINCT FROM requested_completed_event_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_completion_conflict'
                USING ERRCODE = '40001';
        END IF;
    END IF;
    INSERT INTO public.research_lab_autoresearch_tree_handoffs (
        tree_id, schema_version, run_id, candidate_id, node_id,
        root_git_commit, node_git_commit, lineage_hash,
        handoff_doc, handoff_hash
    ) VALUES (
        requested_tree_id, 'research_lab.git_tree_candidate_handoff.v1',
        requested_run_id, requested_candidate_id, requested_node_id,
        requested_root_git_commit, requested_node_git_commit,
        requested_lineage_hash, requested_handoff_doc,
        requested_handoff_hash
    ) ON CONFLICT (tree_id) DO NOTHING RETURNING * INTO inserted;
    created := inserted.tree_id IS NOT NULL;
    IF inserted.tree_id IS NULL THEN
        SELECT * INTO inserted
        FROM public.research_lab_autoresearch_tree_handoffs
        WHERE tree_id = requested_tree_id;
        IF inserted.handoff_hash IS DISTINCT FROM requested_handoff_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_handoff_conflict'
                USING ERRCODE = '40001';
        END IF;
    END IF;
    IF completed_event.event_hash IS NULL THEN
        SELECT * INTO latest_event
        FROM public.research_lab_autoresearch_tree_events
        WHERE tree_id = requested_tree_id ORDER BY seq DESC LIMIT 1;
        IF latest_event.event_type IN (
            'tree_completed', 'tree_failed', 'tree_cancelled_root_changed'
        ) THEN
            RAISE EXCEPTION 'research_lab_git_tree_already_terminal'
                USING ERRCODE = '40001';
        END IF;
        IF COALESCE(latest_event.event_hash, 'sha256:' || repeat('0', 64))
           IS DISTINCT FROM requested_previous_event_hash THEN
            RAISE EXCEPTION 'research_lab_git_tree_event_conflict'
                USING ERRCODE = '40001';
        END IF;
        INSERT INTO public.research_lab_autoresearch_tree_events (
            tree_id, seq, schema_version, event_type, node_id,
            previous_event_hash, event_doc, event_hash
        ) VALUES (
            requested_tree_id, COALESCE(latest_event.seq + 1, 0),
            'research_lab.git_tree_event.v1', 'tree_completed',
            requested_node_id, requested_previous_event_hash,
            completed_doc, requested_completed_event_hash
        ) RETURNING * INTO completed_event;
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'created', created,
        'handoff', pg_catalog.to_jsonb(inserted),
        'completion_event', pg_catalog.to_jsonb(completed_event)
    );
END;
$$;

-- Repair migration 88's reduced event allowlist with the complete historical
-- compatibility set. Tree state itself is authoritative in the new tables.
ALTER TABLE public.research_lab_auto_research_loop_events
    DROP CONSTRAINT IF EXISTS research_lab_auto_research_loop_events_event_type_check;
ALTER TABLE public.research_lab_auto_research_loop_events
    ADD CONSTRAINT research_lab_auto_research_loop_events_event_type_check CHECK (
        event_type IN (
            'loop_started', 'loop_resumed', 'hypothesis_drafted', 'patch_drafted',
            'patch_validation_passed', 'patch_validation_failed', 'dev_check_passed',
            'dev_check_failed', 'reflection_recorded', 'checkpoint_saved', 'loop_paused',
            'candidate_selected', 'loop_completed', 'loop_failed', 'code_edit_drafted',
            'code_edit_validation_passed', 'code_edit_validation_failed',
            'candidate_build_started', 'candidate_build_passed', 'candidate_build_failed',
            'source_inspection_requested', 'source_inspection_seeded',
            'source_inspection_resolved',
            'source_inspection_failed', 'code_edit_repair_requested',
            'code_edit_repair_drafted', 'code_edit_repair_failed',
            'candidate_patch_apply_failed', 'candidate_patch_parse_failed',
            'candidate_patch_empty_or_noop', 'candidate_test_failed',
            'candidate_patch_test_failed', 'candidate_image_build_failed',
            'candidate_artifact_missing', 'candidate_repair_exhausted',
            'candidate_generation_fallback_requested',
            'candidate_generation_fallback_drafted',
            'candidate_generation_fallback_failed', 'loop_direction_planned',
            'plan_alignment_judged', 'code_edit_alignment_rejected',
            'duplicate_candidate_reused', 'no_viable_patch', 'allocator_decision',
            'probe_requested', 'probe_resolved', 'probe_blocked'
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_auto_research_loop_events
    VALIDATE CONSTRAINT research_lab_auto_research_loop_events_event_type_check;

DO $grants$
DECLARE
    relation_name TEXT;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'research_lab_autoresearch_trees',
        'research_lab_autoresearch_tree_nodes',
        'research_lab_autoresearch_tree_events',
        'research_lab_autoresearch_operation_settlements',
        'research_lab_autoresearch_frontier_commitments',
        'research_lab_autoresearch_tree_handoffs'
    ] LOOP
        EXECUTE pg_catalog.format(
            'REVOKE ALL ON TABLE public.%I FROM anon, authenticated, service_role', relation_name
        );
        EXECUTE pg_catalog.format(
            'GRANT SELECT ON TABLE public.%I TO service_role', relation_name
        );
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', relation_name
        );
        EXECUTE pg_catalog.format(
            'DROP POLICY IF EXISTS service_role_read ON public.%I', relation_name
        );
        EXECUTE pg_catalog.format(
            'CREATE POLICY service_role_read ON public.%I FOR SELECT TO service_role USING (true)',
            relation_name
        );
        EXECUTE pg_catalog.format(
            'DROP POLICY IF EXISTS service_role_insert ON public.%I', relation_name
        );
        EXECUTE pg_catalog.format(
            'CREATE POLICY service_role_insert ON public.%I FOR INSERT TO service_role WITH CHECK (true)',
            relation_name
        );
        EXECUTE pg_catalog.format(
            'DROP TRIGGER IF EXISTS %I ON public.%I',
            'prevent_' || relation_name || '_mutation', relation_name
        );
        EXECUTE pg_catalog.format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON public.%I '
            || 'FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation()',
            'prevent_' || relation_name || '_mutation', relation_name
        );
    END LOOP;
END;
$grants$;

REVOKE ALL ON TABLE public.research_lab_autoresearch_tree_current,
    public.research_lab_autoresearch_tree_node_current,
    public.research_lab_autoresearch_operation_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_autoresearch_tree_current,
    public.research_lab_autoresearch_tree_node_current,
    public.research_lab_autoresearch_operation_current TO service_role;

REVOKE ALL ON FUNCTION public.create_research_lab_autoresearch_tree(
    TEXT, UUID, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.create_research_lab_autoresearch_tree(
    TEXT, UUID, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT
) TO service_role;
REVOKE ALL ON FUNCTION public.plan_research_lab_autoresearch_tree_node(
    TEXT, TEXT, TEXT, TEXT, INTEGER, INTEGER, TEXT, TEXT, TEXT, JSONB, TEXT
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.plan_research_lab_autoresearch_tree_node(
    TEXT, TEXT, TEXT, TEXT, INTEGER, INTEGER, TEXT, TEXT, TEXT, JSONB, TEXT
) TO service_role;
REVOKE ALL ON FUNCTION public.append_research_lab_autoresearch_tree_event(
    TEXT, TEXT, TEXT, TEXT, JSONB, TEXT
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.append_research_lab_autoresearch_tree_event(
    TEXT, TEXT, TEXT, TEXT, JSONB, TEXT
) TO service_role;
REVOKE ALL ON FUNCTION public.transition_research_lab_autoresearch_operation(
    TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, INTEGER, JSONB, TEXT, TEXT
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.transition_research_lab_autoresearch_operation(
    TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, INTEGER, JSONB, TEXT, TEXT
) TO service_role;
REVOKE ALL ON FUNCTION public.commit_research_lab_autoresearch_frontier(
    TEXT, INTEGER, TEXT, TEXT, JSONB, TEXT
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.commit_research_lab_autoresearch_frontier(
    TEXT, INTEGER, TEXT, TEXT, JSONB, TEXT
) TO service_role;
REVOKE ALL ON FUNCTION public.select_research_lab_autoresearch_tree_final(
    TEXT, TEXT, TEXT, JSONB, TEXT, TEXT
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.select_research_lab_autoresearch_tree_final(
    TEXT, TEXT, TEXT, JSONB, TEXT, TEXT
) TO service_role;
REVOKE ALL ON FUNCTION public.record_research_lab_autoresearch_tree_handoff(
    TEXT, UUID, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, TEXT, TEXT
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.record_research_lab_autoresearch_tree_handoff(
    TEXT, UUID, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, TEXT, TEXT
) TO service_role;

COMMENT ON TABLE public.research_lab_autoresearch_trees IS
    'Immutable identity and measured root commitments for V2 Git-tree autoresearch.';
COMMENT ON TABLE public.research_lab_autoresearch_operation_settlements IS
    'Append-only logical-operation transitions; one terminal settlement prevents duplicate execution and spend.';

NOTIFY pgrst, 'reload schema';

COMMIT;
