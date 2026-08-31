-- Reserve one path-independent exact provider host for every SOURCE_ADD owner.
--
-- The existing v1/v2 source identity hashes remain byte-for-byte unchanged.
-- This independent reservation closes the path-alias gap (for example /v1
-- versus /v2 on the same provider host) without merging distinct subdomains.

BEGIN;

SET LOCAL lock_timeout = '5s';

-- Installing the submission ownership trigger intentionally makes N-1 new
-- admissions fail closed. Require the same quiet handoff used by migration
-- 167 so no N-1 request or worker crosses that behavior boundary in flight.
DO $quiet_pause$
BEGIN
    -- Exclude a claim which already read the old unpaused state. Its ACCESS
    -- SHARE lock survives to transaction end, so NOWAIT makes the migration
    -- fail for a clean retry instead of allowing the claim to lease afterward.
    LOCK TABLE public.research_lab_source_add_control
        IN ACCESS EXCLUSIVE MODE NOWAIT;
    IF NOT COALESCE((
        SELECT paused
        FROM public.research_lab_source_add_control
        WHERE singleton
    ), FALSE) THEN
        RAISE EXCEPTION 'SOURCE_ADD must be paused before provider-origin migration';
    END IF;
    -- Writer orders differ across legacy admission, worker finalization, and
    -- operator provisioning. Lock every existing origin/preflight target up
    -- front without waiting; a conflict aborts this transaction for retry.
    LOCK TABLE
        public.research_lab_source_add_work_items,
        public.research_lab_source_add_submissions,
        public.research_lab_source_add_identity_events,
        public.research_lab_source_add_functional_probe_attempts,
        public.research_lab_source_catalog,
        public.research_lab_source_add_provisioning_events,
        public.research_lab_source_add_reward_intents,
        public.research_lab_source_add_reward_obligations
        IN SHARE ROW EXCLUSIVE MODE NOWAIT;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_work_items
        WHERE work_status = 'leased'
          AND work_kind IN (
              'provenance', 'functional_probe',
              'provisioning_smoke', 'leg1_reward'
          )
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD work is leased during provider-origin migration';
    END IF;
END;
$quiet_pause$;

DO $preflight$
BEGIN
    IF pg_catalog.to_regprocedure('extensions.digest(bytea,text)') IS NULL THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin hashing requires extensions.digest';
    END IF;
END;
$preflight$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_provider_origin_host_v1(
    p_api_base_url TEXT
)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
STRICT
SET search_path = pg_catalog
AS $function$
DECLARE
    v_url TEXT := btrim(p_api_base_url);
    v_remainder TEXT;
    v_authority TEXT;
    v_host TEXT;
    v_port TEXT := '';
    v_inet INET;
    v_label TEXT;
BEGIN
    IF v_url !~* '^https://' OR v_url ~ '[?#[:space:][:cntrl:]]' THEN
        RETURN '';
    END IF;
    v_remainder := regexp_replace(v_url, '^https://', '', 'i');
    v_authority := split_part(v_remainder, '/', 1);
    IF v_authority = '' OR v_authority ~ '@' THEN
        RETURN '';
    END IF;

    IF left(v_authority, 1) = '[' THEN
        IF v_authority !~ '^\[[0-9A-Fa-f:.]+\](?::443)?$' THEN
            RETURN '';
        END IF;
        v_host := substring(v_authority FROM '^\[([^]]+)\]');
        IF v_host !~ ':' THEN
            RETURN '';
        END IF;
    ELSE
        IF v_authority ~ ':' THEN
            IF v_authority !~ '^[^:]+:443$' THEN
                RETURN '';
            END IF;
            v_host := split_part(v_authority, ':', 1);
            v_port := split_part(v_authority, ':', 2);
        ELSE
            v_host := v_authority;
        END IF;
        IF v_port NOT IN ('', '443') THEN
            RETURN '';
        END IF;
    END IF;

    v_host := lower(btrim(v_host, '.'));
    IF left(v_host, 4) = 'www.' THEN
        v_host := substring(v_host FROM 5);
    END IF;
    IF v_host = '' OR length(v_host) > 253 THEN
        RETURN '';
    END IF;

    IF v_host ~ ':' THEN
        BEGIN
            v_inet := v_host::INET;
        EXCEPTION WHEN invalid_text_representation THEN
            RETURN '';
        END;
        IF family(v_inet) <> 6 OR masklen(v_inet) <> 128
           OR v_inet <<= '::ffff:0:0/96'::INET THEN
            RETURN '';
        END IF;
        RETURN host(v_inet);
    END IF;
    IF v_host ~ '^[0-9.]+$' THEN
        IF v_host !~ '^[0-9]{1,3}(\.[0-9]{1,3}){3}$' THEN
            RETURN '';
        END IF;
        BEGIN
            v_inet := v_host::INET;
        EXCEPTION WHEN invalid_text_representation THEN
            RETURN '';
        END;
        IF family(v_inet) <> 4 OR masklen(v_inet) <> 32
           OR host(v_inet) <> v_host THEN
            RETURN '';
        END IF;
        RETURN host(v_inet);
    END IF;

    IF v_host !~ '^[a-z0-9.-]+$' OR v_host !~ '[.]' THEN
        RETURN '';
    END IF;
    FOREACH v_label IN ARRAY string_to_array(v_host, '.')
    LOOP
        IF v_label = '' OR length(v_label) > 63
           OR v_label !~ '^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$' THEN
            RETURN '';
        END IF;
    END LOOP;
    RETURN v_host;
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_provider_origin_hash_v1(
    p_api_base_url TEXT
)
RETURNS TEXT
LANGUAGE SQL
IMMUTABLE
STRICT
SET search_path = ''
AS $function$
    SELECT CASE WHEN normalized.provider_host = '' THEN '' ELSE
        'sha256:' || pg_catalog.encode(
            extensions.digest(
                pg_catalog.convert_to(
                    '{"source_identity":{"identity_kind":"provider_origin",'
                    || '"identity_version":"v1","provider_host":"'
                    || normalized.provider_host || '"}}',
                    'UTF8'
                ),
                'sha256'
            ),
            'hex'
        )
    END
    FROM (
        SELECT public.research_lab_source_add_provider_origin_host_v1(
            p_api_base_url
        ) AS provider_host
    ) normalized
$function$;

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_provider_origin_events (
    origin_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    origin_version TEXT NOT NULL CHECK (origin_version = 'v1'),
    provider_origin_hash TEXT NOT NULL CHECK (
        provider_origin_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    submission_id TEXT NOT NULL CHECK (
        submission_id ~ '^source_add_submission:[0-9a-f]{16}$'
    ),
    adapter_id TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    reservation_status TEXT NOT NULL CHECK (
        reservation_status IN ('reserved', 'released')
    ),
    seq INTEGER NOT NULL CHECK (seq >= 0),
    reason TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider_origin_hash, seq)
);

-- On first application this table is transaction-local. On reapplication it
-- can be touched by an already-upgraded caller, so fence it fail-fast before
-- replacing its append-only trigger or deriving the reservation snapshot.
LOCK TABLE public.research_lab_source_add_provider_origin_events
    IN SHARE ROW EXCLUSIVE MODE NOWAIT;

CREATE INDEX IF NOT EXISTS idx_source_add_provider_origin_submission
    ON public.research_lab_source_add_provider_origin_events (
        submission_id, created_at DESC
    );

CREATE OR REPLACE VIEW public.research_lab_source_add_provider_origin_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (provider_origin_hash)
    origin_event_id,
    origin_version,
    provider_origin_hash,
    submission_id,
    adapter_id,
    miner_hotkey,
    reservation_status,
    seq,
    reason,
    created_at
FROM public.research_lab_source_add_provider_origin_events
ORDER BY provider_origin_hash, seq DESC, created_at DESC;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_source_add_provider_origin_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $function$
BEGIN
    RAISE EXCEPTION 'research_lab_source_add_provider_origin_events is append-only';
END;
$function$;

DROP TRIGGER IF EXISTS trg_source_add_provider_origin_no_mutation
    ON public.research_lab_source_add_provider_origin_events;
CREATE TRIGGER trg_source_add_provider_origin_no_mutation
    BEFORE UPDATE OR DELETE
    ON public.research_lab_source_add_provider_origin_events
    FOR EACH ROW EXECUTE FUNCTION
        public.prevent_research_lab_source_add_provider_origin_mutation();

CREATE TEMP TABLE source_add_provider_origin_backfill
ON COMMIT DROP
AS
WITH permanent_adapters AS (
    SELECT adapter_id FROM public.research_lab_source_catalog
    UNION
    SELECT adapter_id FROM public.research_lab_source_add_reward_intents
    UNION
    SELECT adapter_id FROM public.research_lab_source_add_reward_obligations
), owner_rows AS (
    SELECT
        current.submission_id,
        current.adapter_id,
        current.miner_hotkey,
        current.adapter_id IN (SELECT adapter_id FROM permanent_adapters)
            AS permanent_owner,
        (
            SELECT MIN(history.created_at)
            FROM public.research_lab_source_add_submissions history
            WHERE history.submission_id = current.submission_id
        ) AS admitted_at,
        current.submission_doc #>> '{source_metadata,api_base_url}'
            AS api_base_url
    FROM public.research_lab_source_add_submission_current current
    WHERE current.stage NOT IN (
        'rejected', 'rejected_precheck', 'functional_probe_failed'
    ) OR current.adapter_id IN (SELECT adapter_id FROM permanent_adapters)
)
SELECT
    owner_rows.*,
    public.research_lab_source_add_provider_origin_host_v1(api_base_url)
        AS provider_origin_host,
    public.research_lab_source_add_provider_origin_hash_v1(api_base_url)
        AS provider_origin_hash
FROM owner_rows;

-- Legacy admission keyed uniqueness to the full source identity, so path
-- aliases on one host can already have multiple owners. Preserve an existing
-- rewarded/cataloged owner. Otherwise preserve the earliest admission, which
-- is the owner atomic host-level admission would have selected at the time.
-- Every non-permanent loser is terminalized append-only before the unique
-- provider-origin reservation is installed.
CREATE TEMP TABLE source_add_provider_origin_losers
ON COMMIT DROP
AS
SELECT ranked.*
FROM (
    SELECT
        backfill.*,
        ROW_NUMBER() OVER (
            PARTITION BY backfill.provider_origin_hash
            ORDER BY
                backfill.permanent_owner DESC,
                backfill.admitted_at ASC,
                backfill.submission_id ASC
        ) AS owner_rank,
        COUNT(*) FILTER (WHERE backfill.permanent_owner) OVER (
            PARTITION BY backfill.provider_origin_hash
        ) AS permanent_owner_count
    FROM source_add_provider_origin_backfill backfill
) ranked
WHERE ranked.owner_rank > 1;

DO $backfill_checks$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_catalog catalog
        WHERE NOT EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_submission_current current
            WHERE current.adapter_id = catalog.adapter_id
              AND current.miner_hotkey = catalog.miner_ref
        )
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_obligations obligation
        WHERE NOT EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_submission_current current
            WHERE current.adapter_id = obligation.adapter_id
              AND current.miner_hotkey = obligation.miner_hotkey
        )
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_intents intent
        WHERE NOT EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_submission_current current
            WHERE current.submission_id = intent.submission_id
              AND current.adapter_id = intent.adapter_id
              AND current.miner_hotkey = intent.miner_hotkey
        )
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin permanent owner is orphaned';
    END IF;
    IF EXISTS (
        SELECT 1 FROM source_add_provider_origin_backfill
        WHERE COALESCE(api_base_url, '') = ''
           OR provider_origin_host = ''
           OR provider_origin_hash !~ '^sha256:[0-9a-f]{64}$'
           OR admitted_at IS NULL
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin backfill input is malformed';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM source_add_provider_origin_losers
        WHERE permanent_owner_count > 1
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin has multiple permanent owners';
    END IF;
END;
$backfill_checks$;

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
    result_doc = work.result_doc || jsonb_build_object(
        'status', 'provider_origin_duplicate_migration',
        'reason_code', 'duplicate_provider_origin_existing_owner'
    ),
    completed_at = COALESCE(work.completed_at, NOW()),
    updated_at = NOW()
WHERE work.submission_id IN (
    SELECT loser.submission_id
    FROM source_add_provider_origin_losers loser
)
  AND work.work_status IN ('queued', 'leased', 'retry_wait');

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
    'provider_origin_duplicate_migration_169'
FROM public.research_lab_source_add_identity_current identity
JOIN source_add_provider_origin_losers loser
  ON loser.submission_id = identity.submission_id
 AND loser.adapter_id = identity.adapter_id
 AND loser.miner_hotkey = identity.miner_hotkey
WHERE identity.reservation_status = 'reserved';

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
    current.submission_doc || jsonb_build_object(
        'stage', 'rejected_precheck',
        'provider_origin_migration', jsonb_build_object(
            'migration', '169',
            'reason_code', 'duplicate_provider_origin_existing_owner',
            'status', 'rejected_duplicate'
        )
    ),
    'rejected_precheck',
    current.precheck_doc || jsonb_build_object(
        'status', 'rejected_precheck',
        'reason_codes', jsonb_build_array(
            'duplicate_provider_origin_existing_owner'
        ),
        'migration', '169'
    ),
    current.source_identity_hash,
    current.source_identity_version
FROM public.research_lab_source_add_submission_current current
JOIN source_add_provider_origin_losers loser
  ON loser.submission_id = current.submission_id
 AND loser.adapter_id = current.adapter_id
 AND loser.miner_hotkey = current.miner_hotkey
WHERE current.stage NOT IN (
    'rejected', 'rejected_precheck', 'functional_probe_failed'
);

DO $reconciliation_checks$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM source_add_provider_origin_losers loser
        JOIN public.research_lab_source_add_submission_current current
          ON current.submission_id = loser.submission_id
         AND current.adapter_id = loser.adapter_id
         AND current.miner_hotkey = loser.miner_hotkey
        WHERE current.stage <> 'rejected_precheck'
           OR current.precheck_status <> 'rejected_precheck'
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_work_items work
        JOIN source_add_provider_origin_losers loser
          ON loser.submission_id = work.submission_id
        WHERE work.work_status IN ('queued', 'leased', 'retry_wait')
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_identity_current identity
        JOIN source_add_provider_origin_losers loser
          ON loser.submission_id = identity.submission_id
         AND loser.adapter_id = identity.adapter_id
         AND loser.miner_hotkey = identity.miner_hotkey
        WHERE identity.reservation_status = 'reserved'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin reconciliation differs';
    END IF;
END;
$reconciliation_checks$;

DELETE FROM source_add_provider_origin_backfill backfill
USING source_add_provider_origin_losers loser
WHERE loser.submission_id = backfill.submission_id
  AND loser.adapter_id = backfill.adapter_id
  AND loser.miner_hotkey = backfill.miner_hotkey;

DO $post_reconciliation_collision_check$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM source_add_provider_origin_backfill
        GROUP BY provider_origin_hash
        HAVING COUNT(DISTINCT submission_id) > 1
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin reconciliation is incomplete';
    END IF;
END;
$post_reconciliation_collision_check$;

INSERT INTO public.research_lab_source_add_provider_origin_events (
    origin_version, provider_origin_hash, submission_id, adapter_id,
    miner_hotkey, reservation_status, seq, reason
)
SELECT
    'v1', backfill.provider_origin_hash, backfill.submission_id,
    backfill.adapter_id, backfill.miner_hotkey, 'reserved', 0,
    'migration_169_live_or_permanent_owner'
FROM source_add_provider_origin_backfill backfill
WHERE NOT EXISTS (
    SELECT 1
    FROM public.research_lab_source_add_provider_origin_current current
    WHERE current.provider_origin_hash = backfill.provider_origin_hash
      AND current.submission_id = backfill.submission_id
      AND current.adapter_id = backfill.adapter_id
      AND current.miner_hotkey = backfill.miner_hotkey
      AND current.reservation_status = 'reserved'
)
ON CONFLICT (provider_origin_hash, seq) DO NOTHING;

DO $coverage_checks$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM source_add_provider_origin_backfill backfill
        WHERE NOT EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_provider_origin_current current
            WHERE current.provider_origin_hash = backfill.provider_origin_hash
              AND current.submission_id = backfill.submission_id
              AND current.adapter_id = backfill.adapter_id
              AND current.miner_hotkey = backfill.miner_hotkey
              AND current.reservation_status = 'reserved'
        )
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_provider_origin_current current
        WHERE current.reservation_status = 'reserved'
          AND NOT EXISTS (
              SELECT 1 FROM source_add_provider_origin_backfill backfill
              WHERE backfill.provider_origin_hash = current.provider_origin_hash
                AND backfill.submission_id = current.submission_id
                AND backfill.adapter_id = current.adapter_id
                AND backfill.miner_hotkey = current.miner_hotkey
          )
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin backfill coverage differs';
    END IF;
END;
$coverage_checks$;

CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_provider_origin_submission()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_api_base_url TEXT := NEW.submission_doc #>> '{source_metadata,api_base_url}';
    v_expected_host TEXT;
    v_expected_hash TEXT;
    v_doc_host TEXT := COALESCE(NEW.submission_doc->>'provider_origin_host', '');
    v_doc_hash TEXT := COALESCE(NEW.submission_doc->>'provider_origin_hash', '');
    v_owner RECORD;
    v_owner_count INTEGER;
BEGIN
    v_expected_host := public.research_lab_source_add_provider_origin_host_v1(
        v_api_base_url
    );
    v_expected_hash := public.research_lab_source_add_provider_origin_hash_v1(
        v_api_base_url
    );
    SELECT COUNT(*), MIN(current.provider_origin_hash)
    INTO v_owner_count, v_doc_hash
    FROM public.research_lab_source_add_provider_origin_current current
    WHERE current.submission_id = NEW.submission_id
      AND current.adapter_id = NEW.adapter_id
      AND current.miner_hotkey = NEW.miner_hotkey
      AND current.reservation_status = 'reserved';
    IF v_expected_host = '' OR v_expected_hash = '' OR v_owner_count <> 1
       OR v_doc_hash <> v_expected_hash THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin submission ownership differs';
    END IF;
    v_doc_host := COALESCE(NEW.submission_doc->>'provider_origin_host', '');
    IF v_doc_host NOT IN ('', v_expected_host)
       OR COALESCE(NEW.submission_doc->>'provider_origin_hash', '')
          NOT IN ('', v_expected_hash) THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin submission document differs';
    END IF;
    NEW.submission_doc := NEW.submission_doc || jsonb_build_object(
        'provider_origin_host', v_expected_host,
        'provider_origin_hash', v_expected_hash
    );
    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS trg_source_add_provider_origin_submission
    ON public.research_lab_source_add_submissions;
CREATE TRIGGER trg_source_add_provider_origin_submission
    BEFORE INSERT ON public.research_lab_source_add_submissions
    FOR EACH ROW EXECUTE FUNCTION
        public.enforce_research_lab_source_add_provider_origin_submission();

CREATE OR REPLACE FUNCTION public.release_research_lab_source_add_provider_origin_terminal()
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
        SELECT 1 FROM public.research_lab_source_add_reward_intents intent
        WHERE intent.submission_id = NEW.submission_id
           OR intent.adapter_id = NEW.adapter_id
    ) OR EXISTS (
        SELECT 1 FROM public.research_lab_source_add_reward_obligations obligation
        WHERE obligation.adapter_id = NEW.adapter_id
    ) OR EXISTS (
        SELECT 1 FROM public.research_lab_source_catalog catalog
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

DROP TRIGGER IF EXISTS trg_source_add_provider_origin_terminal
    ON public.research_lab_source_add_submissions;
CREATE TRIGGER trg_source_add_provider_origin_terminal
    AFTER INSERT ON public.research_lab_source_add_submissions
    FOR EACH ROW EXECUTE FUNCTION
        public.release_research_lab_source_add_provider_origin_terminal();

CREATE OR REPLACE FUNCTION public.assert_research_lab_source_add_provider_origin_owner(
    p_submission_id TEXT,
    p_adapter_id TEXT,
    p_miner_hotkey TEXT
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_count
    FROM public.research_lab_source_add_provider_origin_current current
    JOIN public.research_lab_source_add_submission_current submission
      ON submission.submission_id = current.submission_id
     AND submission.adapter_id = current.adapter_id
     AND submission.miner_hotkey = current.miner_hotkey
    WHERE current.submission_id = p_submission_id
      AND current.adapter_id = p_adapter_id
      AND current.miner_hotkey = p_miner_hotkey
      AND current.reservation_status = 'reserved'
      AND public.research_lab_source_add_provider_origin_hash_v1(
          submission.submission_doc #>> '{source_metadata,api_base_url}'
      ) = current.provider_origin_hash
      AND COALESCE(submission.submission_doc->>'provider_origin_hash', '')
          IN ('', current.provider_origin_hash)
      AND COALESCE(submission.submission_doc->>'provider_origin_host', '')
          IN (
              '',
              public.research_lab_source_add_provider_origin_host_v1(
                  submission.submission_doc #>> '{source_metadata,api_base_url}'
              )
          );
    IF v_count <> 1 THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin owner is unavailable';
    END IF;
END;
$function$;

CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_catalog_provider_origin()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_submission_id TEXT;
    v_count INTEGER;
BEGIN
    SELECT COUNT(*), MIN(current.submission_id)
    INTO v_count, v_submission_id
    FROM public.research_lab_source_add_submission_current current
    WHERE current.adapter_id = NEW.adapter_id
      AND current.miner_hotkey = NEW.miner_ref
      AND current.source_identity_hash = NEW.source_identity_hash;
    IF v_count <> 1 THEN
        RAISE EXCEPTION 'SOURCE_ADD catalog provider-origin owner differs';
    END IF;
    PERFORM public.assert_research_lab_source_add_provider_origin_owner(
        v_submission_id, NEW.adapter_id, NEW.miner_ref
    );
    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS trg_source_catalog_provider_origin
    ON public.research_lab_source_catalog;
CREATE TRIGGER trg_source_catalog_provider_origin
    BEFORE INSERT ON public.research_lab_source_catalog
    FOR EACH ROW EXECUTE FUNCTION
        public.enforce_research_lab_source_catalog_provider_origin();

CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_provision_provider_origin()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
BEGIN
    PERFORM public.assert_research_lab_source_add_provider_origin_owner(
        NEW.submission_id, NEW.adapter_id, NEW.miner_hotkey
    );
    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS trg_source_add_provision_provider_origin
    ON public.research_lab_source_add_provisioning_events;
CREATE TRIGGER trg_source_add_provision_provider_origin
    BEFORE INSERT ON public.research_lab_source_add_provisioning_events
    FOR EACH ROW EXECUTE FUNCTION
        public.enforce_research_lab_source_add_provision_provider_origin();

CREATE OR REPLACE FUNCTION public.research_lab_source_add_admit_v2(
    p_record_doc JSONB,
    p_identity_hash TEXT,
    p_documentation_identity_hash TEXT,
    p_legacy_identity_hash TEXT,
    p_provider_origin_hash TEXT,
    p_work_id TEXT,
    p_max_open INTEGER,
    p_max_day INTEGER,
    p_max_30d INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_submission_id TEXT := p_record_doc->>'submission_id';
    v_adapter_id TEXT := p_record_doc->>'adapter_id';
    v_miner_hotkey TEXT := p_record_doc->>'miner_hotkey';
    v_api_base_url TEXT := p_record_doc #>> '{source_metadata,api_base_url}';
    v_origin_host TEXT;
    v_record_doc JSONB;
    v_result JSONB;
    v_seq INTEGER;
    v_lock_key TEXT;
BEGIN
    v_origin_host := public.research_lab_source_add_provider_origin_host_v1(
        v_api_base_url
    );
    IF p_provider_origin_hash !~ '^sha256:[0-9a-f]{64}$'
       OR v_origin_host = ''
       OR p_provider_origin_hash
          <> public.research_lab_source_add_provider_origin_hash_v1(v_api_base_url)
       OR COALESCE(p_record_doc->>'provider_origin_host', '')
          NOT IN ('', v_origin_host)
       OR COALESCE(p_record_doc->>'provider_origin_hash', '')
          NOT IN ('', p_provider_origin_hash) THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin admission input is invalid';
    END IF;
    FOR v_lock_key IN
        SELECT DISTINCT lock_key
        FROM unnest(ARRAY[
            'source-add-provider-origin:' || p_provider_origin_hash,
            'source-add-identity:' || p_identity_hash,
            CASE WHEN COALESCE(p_documentation_identity_hash, '') = ''
                 THEN NULL ELSE 'source-add-identity:' || p_documentation_identity_hash END,
            CASE WHEN COALESCE(p_legacy_identity_hash, '') = ''
                 THEN NULL ELSE 'source-add-identity:' || p_legacy_identity_hash END
        ]) lock_key
        WHERE lock_key IS NOT NULL
        ORDER BY lock_key
    LOOP
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended(v_lock_key, 0)
        );
    END LOOP;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_provider_origin_current current
        WHERE current.provider_origin_hash = p_provider_origin_hash
          AND current.reservation_status = 'reserved'
    ) THEN
        RETURN jsonb_build_object('status', 'duplicate');
    END IF;
    SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
    FROM public.research_lab_source_add_provider_origin_events
    WHERE provider_origin_hash = p_provider_origin_hash;
    INSERT INTO public.research_lab_source_add_provider_origin_events (
        origin_version, provider_origin_hash, submission_id, adapter_id,
        miner_hotkey, reservation_status, seq, reason
    ) VALUES (
        'v1', p_provider_origin_hash, v_submission_id, v_adapter_id,
        v_miner_hotkey, 'reserved', v_seq, 'atomic_admission_v2'
    );
    v_record_doc := p_record_doc || jsonb_build_object(
        'provider_origin_host', v_origin_host,
        'provider_origin_hash', p_provider_origin_hash
    );
    v_result := public.research_lab_source_add_admit(
        v_record_doc,
        p_identity_hash,
        p_documentation_identity_hash,
        p_legacy_identity_hash,
        p_work_id,
        p_max_open,
        p_max_day,
        p_max_30d
    );
    IF COALESCE(v_result->>'status', '') <> 'admitted' THEN
        INSERT INTO public.research_lab_source_add_provider_origin_events (
            origin_version, provider_origin_hash, submission_id, adapter_id,
            miner_hotkey, reservation_status, seq, reason
        ) VALUES (
            'v1', p_provider_origin_hash, v_submission_id, v_adapter_id,
            v_miner_hotkey, 'released', v_seq + 1,
            'admission_v2_not_admitted'
        );
    END IF;
    RETURN v_result;
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_requeue_provenance_v2(
    p_submission_id TEXT,
    p_identity_hash TEXT,
    p_documentation_identity_hash TEXT,
    p_legacy_identity_hash TEXT,
    p_provider_origin_hash TEXT,
    p_work_id TEXT,
    p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_current RECORD;
    v_api_base_url TEXT;
    v_result JSONB;
    v_seq INTEGER;
    v_inserted BOOLEAN := FALSE;
    v_lock_key TEXT;
BEGIN
    SELECT * INTO v_current
    FROM public.research_lab_source_add_submission_current current
    WHERE current.submission_id = p_submission_id;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('status', 'missing');
    END IF;
    IF v_current.stage IN (
        'accepted', 'rejected', 'rejected_precheck', 'functional_probe_failed'
    ) THEN
        RETURN jsonb_build_object('status', 'terminal');
    END IF;
    v_api_base_url := v_current.submission_doc #>> '{source_metadata,api_base_url}';
    IF p_provider_origin_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_provider_origin_hash
          <> public.research_lab_source_add_provider_origin_hash_v1(v_api_base_url) THEN
        RAISE EXCEPTION 'SOURCE_ADD provider-origin recheck input is invalid';
    END IF;
    FOR v_lock_key IN
        SELECT DISTINCT lock_key
        FROM unnest(ARRAY[
            'source-add-provider-origin:' || p_provider_origin_hash,
            'source-add-identity:' || p_identity_hash,
            CASE WHEN COALESCE(p_documentation_identity_hash, '') = ''
                 THEN NULL ELSE 'source-add-identity:' || p_documentation_identity_hash END,
            CASE WHEN COALESCE(p_legacy_identity_hash, '') = ''
                 THEN NULL ELSE 'source-add-identity:' || p_legacy_identity_hash END
        ]) lock_key
        WHERE lock_key IS NOT NULL
        ORDER BY lock_key
    LOOP
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended(v_lock_key, 0)
        );
    END LOOP;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_provider_origin_current current
        WHERE current.provider_origin_hash = p_provider_origin_hash
          AND current.reservation_status = 'reserved'
          AND (
              current.submission_id <> p_submission_id
              OR current.adapter_id <> v_current.adapter_id
              OR current.miner_hotkey <> v_current.miner_hotkey
          )
    ) THEN
        RETURN jsonb_build_object('status', 'duplicate');
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_provider_origin_current current
        WHERE current.provider_origin_hash = p_provider_origin_hash
          AND current.submission_id = p_submission_id
          AND current.adapter_id = v_current.adapter_id
          AND current.miner_hotkey = v_current.miner_hotkey
          AND current.reservation_status = 'reserved'
    ) THEN
        SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
        FROM public.research_lab_source_add_provider_origin_events
        WHERE provider_origin_hash = p_provider_origin_hash;
        INSERT INTO public.research_lab_source_add_provider_origin_events (
            origin_version, provider_origin_hash, submission_id, adapter_id,
            miner_hotkey, reservation_status, seq, reason
        ) VALUES (
            'v1', p_provider_origin_hash, p_submission_id,
            v_current.adapter_id, v_current.miner_hotkey, 'reserved', v_seq,
            'operator_provenance_recheck_v2'
        );
        v_inserted := TRUE;
    END IF;
    v_result := public.research_lab_source_add_requeue_provenance(
        p_submission_id,
        p_identity_hash,
        p_documentation_identity_hash,
        p_legacy_identity_hash,
        p_work_id,
        p_actor_ref
    );
    IF v_inserted AND COALESCE(v_result->>'status', '') <> 'queued' THEN
        INSERT INTO public.research_lab_source_add_provider_origin_events (
            origin_version, provider_origin_hash, submission_id, adapter_id,
            miner_hotkey, reservation_status, seq, reason
        ) VALUES (
            'v1', p_provider_origin_hash, p_submission_id,
            v_current.adapter_id, v_current.miner_hotkey, 'released', v_seq + 1,
            'provenance_recheck_v2_not_queued'
        );
    END IF;
    RETURN v_result;
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_provider_origin_contract_v1()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
    WITH permanent_adapters AS (
        SELECT adapter_id FROM public.research_lab_source_catalog
        UNION
        SELECT adapter_id FROM public.research_lab_source_add_reward_intents
        UNION
        SELECT adapter_id FROM public.research_lab_source_add_reward_obligations
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
    SELECT jsonb_build_object(
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
                WHERE COALESCE(owner.provider_origin_hash, '') = '' OR NOT EXISTS (
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
            JOIN pg_catalog.pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'research_lab_source_add_submissions'
              AND trigger.tgname = 'trg_source_add_provider_origin_submission'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'catalog_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            JOIN pg_catalog.pg_class relation ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'research_lab_source_catalog'
              AND trigger.tgname = 'trg_source_catalog_provider_origin'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'provision_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            JOIN pg_catalog.pg_class relation ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'research_lab_source_add_provisioning_events'
              AND trigger.tgname = 'trg_source_add_provision_provider_origin'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'terminal_release_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            JOIN pg_catalog.pg_class relation ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'research_lab_source_add_submissions'
              AND trigger.tgname = 'trg_source_add_provider_origin_terminal'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'append_only_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            JOIN pg_catalog.pg_class relation ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace namespace ON namespace.oid = relation.relnamespace
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

REVOKE ALL ON TABLE public.research_lab_source_add_provider_origin_events
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_provider_origin_current
    FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT
    ON TABLE public.research_lab_source_add_provider_origin_events
    TO service_role;
GRANT SELECT
    ON TABLE public.research_lab_source_add_provider_origin_current
    TO service_role;

ALTER TABLE public.research_lab_source_add_provider_origin_events
    ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS source_add_provider_origin_service_all
    ON public.research_lab_source_add_provider_origin_events;
CREATE POLICY source_add_provider_origin_service_all
    ON public.research_lab_source_add_provider_origin_events
    FOR ALL TO service_role USING (true) WITH CHECK (true);

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_provider_origin_host_v1(TEXT)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.research_lab_source_add_provider_origin_hash_v1(TEXT)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.research_lab_source_add_admit_v2(
        JSONB, TEXT, TEXT, TEXT, TEXT, TEXT, INTEGER, INTEGER, INTEGER
    ) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.research_lab_source_add_requeue_provenance_v2(
        TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT
    ) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.research_lab_source_add_provider_origin_contract_v1()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.assert_research_lab_source_add_provider_origin_owner(TEXT, TEXT, TEXT)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.prevent_research_lab_source_add_provider_origin_mutation()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.enforce_research_lab_source_add_provider_origin_submission()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.release_research_lab_source_add_provider_origin_terminal()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.enforce_research_lab_source_catalog_provider_origin()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.enforce_research_lab_source_add_provision_provider_origin()
    FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_admit_v2(
        JSONB, TEXT, TEXT, TEXT, TEXT, TEXT, INTEGER, INTEGER, INTEGER
    ) TO service_role;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_requeue_provenance_v2(
        TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT
    ) TO service_role;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_provider_origin_contract_v1()
    TO service_role;

COMMENT ON TABLE public.research_lab_source_add_provider_origin_events IS
    'Append-only exact-host SOURCE_ADD provider reservations; API paths are excluded.';
COMMENT ON FUNCTION public.research_lab_source_add_provider_origin_contract_v1() IS
    'Read-only release preflight for exact-host SOURCE_ADD uniqueness and backfill coverage.';

NOTIFY pgrst, 'reload schema';

COMMIT;
