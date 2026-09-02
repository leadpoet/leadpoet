-- Make automatic provenance Leg 1 unique at the normalized provider host.
--
-- Migration 175 correctly moved the reward boundary to attested provenance,
-- but historical terminal submissions had already released their host
-- reservation. A later path alias could therefore pass provenance too. Repair
-- those histories deterministically: the earliest qualified submission owns
-- the host, later submissions are cancelled and terminalized append-only, and
-- terminal qualified winners regain the reservations needed by the authority.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $quiet_pause$
BEGIN
    LOCK TABLE public.research_lab_source_add_control
        IN ACCESS EXCLUSIVE MODE NOWAIT;
    IF NOT COALESCE((
        SELECT control.paused
        FROM public.research_lab_source_add_control control
        WHERE control.singleton
    ), FALSE) THEN
        RAISE EXCEPTION
            'SOURCE_ADD must be paused before provenance-origin repair';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_control control
        WHERE control.singleton
          AND control.restart_guard_commitment <> ''
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD restart guard is active during provenance-origin repair';
    END IF;
    LOCK TABLE
        public.research_lab_source_add_work_items,
        public.research_lab_source_add_submissions,
        public.research_lab_source_add_identity_events,
        public.research_lab_source_add_provider_origin_events,
        public.research_lab_source_add_reward_intents,
        public.research_lab_source_add_reward_slots,
        public.research_lab_source_add_reward_obligations,
        public.research_lab_source_add_reward_events,
        public.research_lab_source_catalog,
        public.research_lab_source_add_provisioning_events,
        public.research_lab_attested_execution_receipts_v2,
        public.research_lab_attested_receipt_edges_v2,
        public.research_lab_attested_business_artifact_links_v2
        IN SHARE ROW EXCLUSIVE MODE NOWAIT;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_work_items work
        WHERE work.work_status = 'leased'
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD work is leased during provenance-origin repair';
    END IF;
    IF pg_catalog.to_regclass(
        'public.research_lab_source_add_provenance_leg1_authority_v1'
    ) IS NULL OR pg_catalog.to_regprocedure(
        'public.research_lab_source_add_reconcile_provenance_leg1_v1()'
    ) IS NULL THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance Leg 1 migration 175 is unavailable';
    END IF;
END;
$quiet_pause$;

CREATE TEMP TABLE source_add_provenance_candidates_176
ON COMMIT DROP
AS
WITH exact_provenance AS (
    SELECT DISTINCT ON (history.submission_id)
        history.submission_id,
        history.adapter_id,
        history.miner_hotkey,
        receipt.receipt_hash AS provenance_receipt_hash,
        receipt.output_root AS provenance_artifact_hash,
        history.created_at AS provenance_created_at,
        public.research_lab_source_add_provider_origin_hash_v1(
            history.submission_doc #>> '{source_metadata,api_base_url}'
        ) AS provider_origin_hash
    FROM public.research_lab_source_add_submissions history
    JOIN public.research_lab_attested_execution_receipts_v2 receipt
      ON receipt.receipt_hash =
         history.submission_doc->>'provenance_receipt_hash'
    JOIN public.research_lab_attested_business_artifact_links_v2 link
      ON link.receipt_hash = receipt.receipt_hash
     AND link.artifact_kind = 'source_add_provenance'
     AND link.artifact_ref = history.submission_id
     AND link.artifact_hash = receipt.output_root
    WHERE history.precheck_status = 'provenance_precheck_passed'
      AND history.precheck_doc->>'precheck_status' =
          'provenance_precheck_passed'
      AND history.submission_doc->>'provenance_receipt_hash'
          ~ '^sha256:[0-9a-f]{64}$'
      AND receipt.role = 'gateway_coordinator'
      AND receipt.purpose = 'research_lab.source_add_provenance.v2'
      AND receipt.receipt_status = 'succeeded'
      AND receipt.output_root ~ '^sha256:[0-9a-f]{64}$'
      AND receipt.receipt_doc->'parent_receipt_hashes' = '[]'::JSONB
      AND NOT EXISTS (
          SELECT 1
          FROM public.research_lab_attested_receipt_edges_v2 edge
          WHERE edge.child_receipt_hash = receipt.receipt_hash
      )
    ORDER BY history.submission_id, history.seq ASC, history.created_at ASC
), ranked AS (
    SELECT
        exact_provenance.*,
        ROW_NUMBER() OVER (
            PARTITION BY exact_provenance.provider_origin_hash
            ORDER BY
                exact_provenance.provenance_created_at ASC,
                exact_provenance.submission_id ASC
        ) AS origin_rank,
        COUNT(*) OVER (
            PARTITION BY exact_provenance.provider_origin_hash
        ) AS origin_count
    FROM exact_provenance
)
SELECT * FROM ranked;

DO $candidate_preflight$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM source_add_provenance_candidates_176 candidate
        WHERE candidate.provider_origin_hash
              !~ '^sha256:[0-9a-f]{64}$'
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance provider-origin candidate is malformed';
    END IF;
    -- Never rewrite a collision after either side became economic/catalog
    -- authority. Such a state needs a separately reviewed compensation plan.
    IF EXISTS (
        SELECT 1
        FROM source_add_provenance_candidates_176 candidate
        WHERE candidate.origin_count > 1
          AND candidate.origin_rank > 1
          AND (
              EXISTS (
                  SELECT 1
                  FROM public.research_lab_source_add_reward_obligations reward
                  WHERE reward.adapter_id = candidate.adapter_id
                    AND reward.leg = 1
              )
              OR EXISTS (
                  SELECT 1
                  FROM public.research_lab_source_catalog catalog
                  WHERE catalog.adapter_id = candidate.adapter_id
              )
          )
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance-origin collision has finalized authority';
    END IF;
END;
$candidate_preflight$;

CREATE TEMP TABLE source_add_provenance_winners_176
ON COMMIT DROP
AS
SELECT *
FROM source_add_provenance_candidates_176 candidate
WHERE candidate.origin_rank = 1;

CREATE TEMP TABLE source_add_provenance_losers_176
ON COMMIT DROP
AS
SELECT DISTINCT
    loser.submission_id,
    loser.adapter_id,
    loser.miner_hotkey,
    winner.provider_origin_hash
FROM source_add_provenance_winners_176 winner
JOIN (
    SELECT
        candidate.submission_id,
        candidate.adapter_id,
        candidate.miner_hotkey,
        candidate.provider_origin_hash
    FROM source_add_provenance_candidates_176 candidate
    WHERE candidate.origin_rank > 1
    UNION ALL
    SELECT
        current.submission_id,
        current.adapter_id,
        current.miner_hotkey,
        current.provider_origin_hash
    FROM public.research_lab_source_add_provider_origin_current current
    WHERE current.reservation_status = 'reserved'
) loser
  ON loser.provider_origin_hash = winner.provider_origin_hash
 AND (
     loser.submission_id <> winner.submission_id
     OR loser.adapter_id <> winner.adapter_id
     OR loser.miner_hotkey <> winner.miner_hotkey
 );

DO $loser_preflight$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM source_add_provenance_losers_176 loser
        LEFT JOIN public.research_lab_source_add_submission_current current
          ON current.submission_id = loser.submission_id
         AND current.adapter_id = loser.adapter_id
         AND current.miner_hotkey = loser.miner_hotkey
        WHERE current.submission_id IS NULL
    ) OR EXISTS (
        SELECT 1
        FROM source_add_provenance_losers_176 loser
        WHERE EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_reward_obligations reward
            WHERE reward.adapter_id = loser.adapter_id
              AND reward.leg = 1
        ) OR EXISTS (
            SELECT 1
            FROM public.research_lab_source_catalog catalog
            WHERE catalog.adapter_id = loser.adapter_id
        )
    ) OR EXISTS (
        SELECT 1
        FROM source_add_provenance_losers_176 loser
        JOIN public.research_lab_source_add_reward_intents intent
          ON intent.adapter_id = loser.adapter_id
         AND intent.leg = 1
        WHERE intent.intent_status NOT IN (
            'queued', 'retry_wait', 'cancelled'
        )
    ) OR EXISTS (
        SELECT 1
        FROM source_add_provenance_losers_176 loser
        JOIN public.research_lab_source_add_reward_slots slot
          ON slot.intent_id IN (
              SELECT intent.intent_id
              FROM public.research_lab_source_add_reward_intents intent
              WHERE intent.adapter_id = loser.adapter_id
                AND intent.leg = 1
          )
        WHERE slot.slot_status = 'finalized'
           OR slot.reward_ref IS NOT NULL
    ) OR EXISTS (
        SELECT 1
        FROM source_add_provenance_losers_176 loser
        JOIN public.research_lab_source_add_work_items work
          ON work.submission_id = loser.submission_id
         AND work.adapter_id = loser.adapter_id
         AND work.work_kind = 'leg1_reward'
        WHERE work.work_status NOT IN (
            'queued', 'retry_wait', 'cancelled'
        )
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance-origin loser has conflicting authority';
    END IF;
END;
$loser_preflight$;

-- Cancel economic scheduling before appending the terminal state. This makes
-- the revised release trigger see the loser as non-permanent.
UPDATE public.research_lab_source_add_reward_slots slot
SET slot_status = 'released',
    lease_expires_at = LEAST(slot.lease_expires_at, NOW()),
    updated_at = NOW()
WHERE slot.slot_status = 'reserved'
  AND slot.intent_id IN (
      SELECT intent.intent_id
      FROM public.research_lab_source_add_reward_intents intent
      JOIN source_add_provenance_losers_176 loser
        ON loser.adapter_id = intent.adapter_id
       AND intent.leg = 1
  );

UPDATE public.research_lab_source_add_work_items work
SET work_status = 'cancelled',
    lease_token = NULL,
    leased_by = '',
    lease_expires_at = NULL,
    job_doc = work.job_doc
        - 'provider_execution_state'
        - 'provider_execution_attempt'
        - 'provider_execution_started_at'
        - 'provider_execution_recovery',
    result_doc = pg_catalog.jsonb_build_object(
        'status', 'submission_failed'
    ),
    completed_at = COALESCE(work.completed_at, NOW()),
    updated_at = NOW()
FROM source_add_provenance_losers_176 loser
WHERE work.submission_id = loser.submission_id
  AND work.adapter_id = loser.adapter_id
  AND work.work_status IN ('queued', 'retry_wait');

UPDATE public.research_lab_source_add_reward_intents intent
SET intent_status = 'cancelled',
    updated_at = NOW()
FROM source_add_provenance_losers_176 loser
WHERE intent.adapter_id = loser.adapter_id
  AND intent.leg = 1
  AND intent.intent_status IN ('queued', 'retry_wait');

CREATE OR REPLACE FUNCTION
    public.release_research_lab_source_add_provider_origin_terminal()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_origin RECORD;
BEGIN
    IF NEW.stage NOT IN (
        'rejected', 'rejected_precheck', 'functional_probe_failed'
    ) THEN
        RETURN NEW;
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_intents intent
        WHERE (
            intent.submission_id = NEW.submission_id
            OR intent.adapter_id = NEW.adapter_id
        )
          AND intent.intent_status <> 'cancelled'
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_obligations obligation
        WHERE obligation.adapter_id = NEW.adapter_id
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_catalog catalog
        WHERE catalog.adapter_id = NEW.adapter_id
    ) THEN
        RETURN NEW;
    END IF;
    FOR v_origin IN
        SELECT *
        FROM public.research_lab_source_add_provider_origin_current current
        WHERE current.submission_id = NEW.submission_id
          AND current.adapter_id = NEW.adapter_id
          AND current.miner_hotkey = NEW.miner_hotkey
          AND current.reservation_status = 'reserved'
    LOOP
        INSERT INTO public.research_lab_source_add_provider_origin_events (
            origin_version, provider_origin_hash, submission_id, adapter_id,
            miner_hotkey, reservation_status, seq, reason
        ) VALUES (
            'v1', v_origin.provider_origin_hash, v_origin.submission_id,
            v_origin.adapter_id, v_origin.miner_hotkey, 'released',
            v_origin.seq + 1, 'terminal_pre_reward'
        );
    END LOOP;
    RETURN NEW;
END;
$function$;

REVOKE ALL ON FUNCTION
    public.release_research_lab_source_add_provider_origin_terminal()
    FROM PUBLIC, anon, authenticated;

-- Append the generic terminal result while the current loser still owns the
-- host, satisfying the existing BEFORE INSERT ownership guard. The AFTER
-- INSERT terminal trigger above releases that reservation in the same tx.
INSERT INTO public.research_lab_source_add_submissions (
    submission_id,
    schema_version,
    adapter_id,
    miner_hotkey,
    stage,
    seq,
    measured_trial_yield,
    submission_doc,
    precheck_status,
    precheck_doc,
    source_identity_hash,
    source_identity_version
)
SELECT
    current.submission_id,
    current.schema_version,
    current.adapter_id,
    current.miner_hotkey,
    'rejected_precheck',
    current.seq + 1,
    current.measured_trial_yield,
    current.submission_doc || pg_catalog.jsonb_build_object(
        'stage', 'rejected_precheck',
        'status', 'submission_failed'
    ),
    'rejected_precheck',
    pg_catalog.jsonb_build_object(
        'status', 'rejected_precheck',
        'reason_codes', pg_catalog.jsonb_build_array(
            'submission_not_eligible'
        )
    ),
    current.source_identity_hash,
    current.source_identity_version
FROM public.research_lab_source_add_submission_current current
JOIN source_add_provenance_losers_176 loser
  ON loser.submission_id = current.submission_id
 AND loser.adapter_id = current.adapter_id
 AND loser.miner_hotkey = current.miner_hotkey
JOIN public.research_lab_source_add_provider_origin_current origin
  ON origin.provider_origin_hash = loser.provider_origin_hash
 AND origin.submission_id = loser.submission_id
 AND origin.adapter_id = loser.adapter_id
 AND origin.miner_hotkey = loser.miner_hotkey
 AND origin.reservation_status = 'reserved'
WHERE current.stage NOT IN (
    'rejected', 'rejected_precheck', 'functional_probe_failed'
);

-- A loser may already be terminal while retaining a stale reservation (for
-- example after an interrupted older migration).  The insert above lets the
-- normal trigger release every nonterminal loser.  Append a release for any
-- remaining current loser so the winner transfer below never relies on that
-- historical stage shape.
INSERT INTO public.research_lab_source_add_provider_origin_events (
    origin_version,
    provider_origin_hash,
    submission_id,
    adapter_id,
    miner_hotkey,
    reservation_status,
    seq,
    reason
)
SELECT
    current.origin_version,
    current.provider_origin_hash,
    current.submission_id,
    current.adapter_id,
    current.miner_hotkey,
    'released',
    current.seq + 1,
    'provenance_origin_duplicate_migration_176'
FROM public.research_lab_source_add_provider_origin_current current
JOIN source_add_provenance_losers_176 loser
  ON loser.provider_origin_hash = current.provider_origin_hash
 AND loser.submission_id = current.submission_id
 AND loser.adapter_id = current.adapter_id
 AND loser.miner_hotkey = current.miner_hotkey
WHERE current.reservation_status = 'reserved';

-- Identity aliases are independent append-only reservations. Release only
-- aliases still owned by a loser; never mutate their prior history.
INSERT INTO public.research_lab_source_add_identity_events (
    identity_version,
    source_identity_hash,
    submission_id,
    adapter_id,
    miner_hotkey,
    reservation_status,
    seq,
    reason
)
SELECT
    identity.identity_version,
    identity.source_identity_hash,
    identity.submission_id,
    identity.adapter_id,
    identity.miner_hotkey,
    'released',
    identity.seq + 1,
    'provenance_origin_duplicate_migration_176'
FROM public.research_lab_source_add_identity_current identity
JOIN source_add_provenance_losers_176 loser
  ON loser.submission_id = identity.submission_id
 AND loser.adapter_id = identity.adapter_id
 AND loser.miner_hotkey = identity.miner_hotkey
WHERE identity.reservation_status = 'reserved';

-- Transfer/recreate the exact-host reservation for every qualified winner,
-- including historical terminal submissions whose provenance already earned
-- Leg 1 before a later functional result released the host.
INSERT INTO public.research_lab_source_add_provider_origin_events (
    origin_version,
    provider_origin_hash,
    submission_id,
    adapter_id,
    miner_hotkey,
    reservation_status,
    seq,
    reason
)
SELECT
    'v1',
    winner.provider_origin_hash,
    winner.submission_id,
    winner.adapter_id,
    winner.miner_hotkey,
    'reserved',
    COALESCE((
        SELECT MAX(history.seq) + 1
        FROM public.research_lab_source_add_provider_origin_events history
        WHERE history.provider_origin_hash = winner.provider_origin_hash
    ), 0),
    'provenance_leg1_owner_migration_176'
FROM source_add_provenance_winners_176 winner
WHERE NOT EXISTS (
    SELECT 1
    FROM public.research_lab_source_add_provider_origin_current current
    WHERE current.provider_origin_hash = winner.provider_origin_hash
      AND current.submission_id = winner.submission_id
      AND current.adapter_id = winner.adapter_id
      AND current.miner_hotkey = winner.miner_hotkey
      AND current.reservation_status = 'reserved'
);

-- The authority is now one earliest exact provenance result per normalized
-- host, and it exists only while that exact winner owns the host reservation.
CREATE OR REPLACE VIEW
    public.research_lab_source_add_provenance_leg1_authority_v1
WITH (security_invoker = true) AS
WITH exact_provenance AS (
    SELECT DISTINCT ON (history.submission_id)
        history.submission_id,
        history.adapter_id,
        history.miner_hotkey,
        history.precheck_status,
        receipt.receipt_hash AS provenance_receipt_hash,
        receipt.output_root AS provenance_artifact_hash,
        history.created_at AS provenance_created_at,
        public.research_lab_source_add_provider_origin_hash_v1(
            history.submission_doc #>> '{source_metadata,api_base_url}'
        ) AS provider_origin_hash
    FROM public.research_lab_source_add_submissions history
    JOIN public.research_lab_attested_execution_receipts_v2 receipt
      ON receipt.receipt_hash =
         history.submission_doc->>'provenance_receipt_hash'
    JOIN public.research_lab_attested_business_artifact_links_v2 link
      ON link.receipt_hash = receipt.receipt_hash
     AND link.artifact_kind = 'source_add_provenance'
     AND link.artifact_ref = history.submission_id
     AND link.artifact_hash = receipt.output_root
    WHERE history.precheck_status = 'provenance_precheck_passed'
      AND history.precheck_doc->>'precheck_status' =
          'provenance_precheck_passed'
      AND history.submission_doc->>'provenance_receipt_hash'
          ~ '^sha256:[0-9a-f]{64}$'
      AND receipt.role = 'gateway_coordinator'
      AND receipt.purpose = 'research_lab.source_add_provenance.v2'
      AND receipt.receipt_status = 'succeeded'
      AND receipt.output_root ~ '^sha256:[0-9a-f]{64}$'
      AND receipt.receipt_doc->'parent_receipt_hashes' = '[]'::JSONB
      AND NOT EXISTS (
          SELECT 1
          FROM public.research_lab_attested_receipt_edges_v2 edge
          WHERE edge.child_receipt_hash = receipt.receipt_hash
      )
    ORDER BY history.submission_id, history.seq ASC, history.created_at ASC
), ranked AS (
    SELECT
        exact_provenance.*,
        ROW_NUMBER() OVER (
            PARTITION BY exact_provenance.provider_origin_hash
            ORDER BY
                exact_provenance.provenance_created_at ASC,
                exact_provenance.submission_id ASC
        ) AS origin_rank
    FROM exact_provenance
)
SELECT
    ranked.submission_id,
    ranked.adapter_id,
    ranked.miner_hotkey,
    ranked.precheck_status,
    ranked.provenance_receipt_hash,
    ranked.provenance_artifact_hash,
    ranked.provenance_created_at
FROM ranked
JOIN public.research_lab_source_add_provider_origin_current origin
  ON origin.provider_origin_hash = ranked.provider_origin_hash
 AND origin.submission_id = ranked.submission_id
 AND origin.adapter_id = ranked.adapter_id
 AND origin.miner_hotkey = ranked.miner_hotkey
 AND origin.reservation_status = 'reserved'
WHERE ranked.origin_rank = 1;

REVOKE ALL ON TABLE
    public.research_lab_source_add_provenance_leg1_authority_v1
    FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE
    public.research_lab_source_add_provenance_leg1_authority_v1
    TO service_role;

-- Cancelled historical intents are audit records, not permanent owners.
CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_provider_origin_contract_v1()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
    WITH permanent_adapters AS (
        SELECT adapter_id FROM public.research_lab_source_catalog
        UNION
        SELECT adapter_id
        FROM public.research_lab_source_add_reward_intents
        WHERE intent_status <> 'cancelled'
        UNION
        SELECT adapter_id
        FROM public.research_lab_source_add_reward_obligations
    ), owners AS (
        SELECT
            current.submission_id,
            current.adapter_id,
            current.miner_hotkey,
            public.research_lab_source_add_provider_origin_hash_v1(
                current.submission_doc #>> '{source_metadata,api_base_url}'
            ) AS provider_origin_hash
        FROM public.research_lab_source_add_submission_current current
        WHERE current.stage NOT IN (
            'rejected', 'rejected_precheck', 'functional_probe_failed'
        ) OR current.adapter_id IN (SELECT adapter_id FROM permanent_adapters)
    )
    SELECT pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_provider_origin_contract.v1',
        'identity_version', 'v1',
        'identity_scope', 'normalized_exact_host',
        'admission_rpc', 'research_lab_source_add_admit_v2',
        'recheck_rpc', 'research_lab_source_add_requeue_provenance_v2',
        'owner_count', (SELECT COUNT(*) FROM owners),
        'reserved_count', (
            SELECT COUNT(*)
            FROM public.research_lab_source_add_provider_origin_current current
            WHERE current.reservation_status = 'reserved'
        ),
        'coverage_complete', (
            NOT EXISTS (
                SELECT 1 FROM owners owner
                WHERE COALESCE(owner.provider_origin_hash, '') = ''
                   OR NOT EXISTS (
                       SELECT 1
                       FROM public.research_lab_source_add_provider_origin_current current
                       WHERE current.provider_origin_hash = owner.provider_origin_hash
                         AND current.submission_id = owner.submission_id
                         AND current.adapter_id = owner.adapter_id
                         AND current.miner_hotkey = owner.miner_hotkey
                         AND current.reservation_status = 'reserved'
                   )
            ) AND NOT EXISTS (
                SELECT 1
                FROM public.research_lab_source_add_provider_origin_current current
                WHERE current.reservation_status = 'reserved'
                  AND NOT EXISTS (
                      SELECT 1 FROM owners owner
                      WHERE owner.provider_origin_hash = current.provider_origin_hash
                        AND owner.submission_id = current.submission_id
                        AND owner.adapter_id = current.adapter_id
                        AND owner.miner_hotkey = current.miner_hotkey
                  )
            )
        ),
        'collision_free', NOT EXISTS (
            SELECT 1 FROM owners
            GROUP BY provider_origin_hash
            HAVING COUNT(DISTINCT submission_id) > 1
        ),
        'submission_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            JOIN pg_catalog.pg_class relation ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'research_lab_source_add_submissions'
              AND trigger.tgname = 'trg_source_add_provider_origin_submission'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'catalog_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            JOIN pg_catalog.pg_class relation ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'research_lab_source_catalog'
              AND trigger.tgname = 'trg_source_catalog_provider_origin'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'provision_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            JOIN pg_catalog.pg_class relation ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname =
                  'research_lab_source_add_provisioning_events'
              AND trigger.tgname = 'trg_source_add_provision_provider_origin'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'terminal_release_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            JOIN pg_catalog.pg_class relation ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'research_lab_source_add_submissions'
              AND trigger.tgname = 'trg_source_add_provider_origin_terminal'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'append_only_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            JOIN pg_catalog.pg_class relation ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname =
                  'research_lab_source_add_provider_origin_events'
              AND trigger.tgname =
                  'trg_source_add_provider_origin_no_mutation'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'row_level_security_enabled', COALESCE((
            SELECT relation.relrowsecurity
            FROM pg_catalog.pg_class relation
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname =
                  'research_lab_source_add_provider_origin_events'
        ), FALSE),
        'service_role_policy_enabled', EXISTS (
            SELECT 1
            FROM pg_catalog.pg_policies policy
            WHERE policy.schemaname = 'public'
              AND policy.tablename =
                  'research_lab_source_add_provider_origin_events'
              AND policy.policyname = 'source_add_provider_origin_service_all'
              AND policy.cmd = 'ALL'
              AND 'service_role' = ANY(policy.roles)
        )
    )
$function$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_provider_origin_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_provider_origin_contract_v1()
    TO service_role;

CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_post_accept_leg1_contract_v4()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
    WITH base AS (
        SELECT public.research_lab_source_add_post_accept_leg1_contract_v3()
            AS contract
    ), repair_authority AS (
        SELECT 'sha256:' || pg_catalog.encode(
            extensions.digest(
                pg_catalog.convert_to(
                    COALESCE(
                        pg_catalog.jsonb_object_agg(
                            expected.name,
                            pg_catalog.jsonb_build_object(
                                'body', function_row.prosrc,
                                'security_definer', function_row.prosecdef,
                                'configuration', pg_catalog.to_jsonb(
                                    function_row.proconfig
                                ),
                                'identity_arguments',
                                    pg_catalog.pg_get_function_identity_arguments(
                                        function_row.oid
                                    ),
                                'argument_names', pg_catalog.to_jsonb(
                                    function_row.proargnames
                                ),
                                'language', language.lanname,
                                'volatility', function_row.provolatile,
                                'parallel', function_row.proparallel,
                                'kind', function_row.prokind,
                                'return_type',
                                    function_row.prorettype::REGTYPE::TEXT
                            )
                        ),
                        '{}'::JSONB
                    )::TEXT,
                    'UTF8'
                ),
                'sha256'
            ),
            'hex'
        ) AS authority_hash
        FROM (
            VALUES
                (
                    'provider_origin_contract_v1',
                    'public.research_lab_source_add_provider_origin_contract_v1()'
                ),
                (
                    'provider_origin_terminal_release',
                    'public.release_research_lab_source_add_provider_origin_terminal()'
                )
        ) expected(name, signature)
        LEFT JOIN pg_catalog.pg_proc function_row
          ON function_row.oid =
             pg_catalog.to_regprocedure(expected.signature)
        LEFT JOIN pg_catalog.pg_language language
          ON language.oid = function_row.prolang
    )
    SELECT
        (base.contract - 'schema_version' - 'backfill_policy')
        || pg_catalog.jsonb_build_object(
            'schema_version',
                'leadpoet.source_add_post_accept_leg1_contract.v4',
            'required_migration',
                'scripts/176-research-lab-source-add-provenance-origin-repair.sql',
            'backfill_policy',
                'earliest_exact_attested_provenance_per_provider_origin',
            'provider_origin_scope', 'normalized_exact_host',
            'provider_origin_winner_order', pg_catalog.jsonb_build_array(
                'provenance_created_at', 'submission_id'
            ),
            'cancelled_intents_are_authority', FALSE,
            'repair_function_authority_sha256',
                repair_authority.authority_hash
        )
    FROM base CROSS JOIN repair_authority
$function$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_post_accept_leg1_contract_v4()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_post_accept_leg1_contract_v4()
    TO service_role;

DO $reconcile_and_assert$
DECLARE
    v_result JSONB;
    v_contract JSONB;
BEGIN
    v_result :=
        public.research_lab_source_add_reconcile_provenance_leg1_v1();
    v_contract :=
        public.research_lab_source_add_provider_origin_contract_v1();
    IF COALESCE(v_result->>'status', '') <> 'reconciled'
       OR COALESCE((v_contract->>'coverage_complete')::BOOLEAN, FALSE)
          IS NOT TRUE
       OR COALESCE((v_contract->>'collision_free')::BOOLEAN, FALSE)
          IS NOT TRUE
       OR COALESCE((v_contract->>'owner_count')::INTEGER, -1)
          <> COALESCE((v_contract->>'reserved_count')::INTEGER, -2)
       OR EXISTS (
           SELECT 1
           FROM source_add_provenance_winners_176 winner
           WHERE NOT EXISTS (
               SELECT 1
               FROM public.research_lab_source_add_provenance_leg1_authority_v1 authority
               WHERE authority.submission_id = winner.submission_id
                 AND authority.adapter_id = winner.adapter_id
                 AND authority.miner_hotkey = winner.miner_hotkey
                 AND authority.provenance_receipt_hash =
                     winner.provenance_receipt_hash
                 AND authority.provenance_artifact_hash =
                     winner.provenance_artifact_hash
           )
       ) OR EXISTS (
           SELECT 1
           FROM source_add_provenance_losers_176 loser
           WHERE EXISTS (
               SELECT 1
               FROM public.research_lab_source_add_reward_intents intent
               WHERE intent.adapter_id = loser.adapter_id
                 AND intent.leg = 1
                 AND intent.intent_status <> 'cancelled'
           ) OR EXISTS (
               SELECT 1
               FROM public.research_lab_source_add_work_items work
               WHERE work.submission_id = loser.submission_id
                 AND work.adapter_id = loser.adapter_id
                 AND work.work_status IN ('queued', 'leased', 'retry_wait')
           ) OR EXISTS (
               SELECT 1
               FROM public.research_lab_source_add_reward_obligations reward
               WHERE reward.adapter_id = loser.adapter_id
                 AND reward.leg = 1
           )
       )
    THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance-origin repair readback differs';
    END IF;
END;
$reconcile_and_assert$;

NOTIFY pgrst, 'reload schema';

COMMIT;
