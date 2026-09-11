-- Commit the private Arena benchmark before scoring and reveal it no earlier
-- than twenty-four hours after the stored submission cutoff.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_210_requires_209$
DECLARE
  v_publication_guard TEXT;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_validator_scoring_authority_schema_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply Arena migrations through 208 first';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
           pg_catalog.to_regprocedure(
             'public.lab_arena_publication_baseline_guard_v1()'
           )
         )
  INTO v_publication_guard;
  IF COALESCE(pg_catalog.strpos(v_publication_guard, 'provider_cost_uncertain'), 0) = 0 THEN
    RAISE EXCEPTION 'apply 209-lab-arena-uncertain-cost-eligibility.sql first';
  END IF;
END;
$lab_arena_210_requires_209$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS benchmark_reveal_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS benchmark_commitment_doc JSONB,
  ADD COLUMN IF NOT EXISTS benchmark_committed_at TIMESTAMPTZ;

-- Validate only the public commitment. The plaintext ICPs and their nonces
-- remain in the content-addressed private artifact referenced by benchmark_ref.
CREATE OR REPLACE FUNCTION public.lab_arena__benchmark_commitment_valid_v1(
  p_round_id TEXT,
  p_network_name TEXT,
  p_netuid BIGINT,
  p_icp_set_date DATE,
  p_evaluation_date TEXT,
  p_reveal_at TIMESTAMPTZ,
  p_commitment_doc JSONB
)
RETURNS BOOLEAN
LANGUAGE plpgsql
IMMUTABLE
SECURITY INVOKER
SET search_path = ''
AS $lab_arena_benchmark_commitment_valid$
DECLARE
  v_manifest JSONB;
  v_canonical_manifest TEXT;
  v_keys TEXT[];
BEGIN
  IF pg_catalog.jsonb_typeof(p_commitment_doc) IS DISTINCT FROM 'object' THEN
    RETURN FALSE;
  END IF;
  SELECT pg_catalog.array_agg(key ORDER BY key)
  INTO v_keys
  FROM pg_catalog.jsonb_object_keys(p_commitment_doc) AS key;
  IF v_keys IS DISTINCT FROM
       ARRAY['canonical_manifest', 'manifest', 'manifest_hash']::TEXT[] THEN
    RETURN FALSE;
  END IF;

  v_manifest := p_commitment_doc -> 'manifest';
  v_canonical_manifest := p_commitment_doc ->> 'canonical_manifest';
  IF pg_catalog.jsonb_typeof(v_manifest) IS DISTINCT FROM 'object'
     OR pg_catalog.jsonb_typeof(p_commitment_doc -> 'canonical_manifest')
          IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(p_commitment_doc -> 'manifest_hash')
          IS DISTINCT FROM 'string'
     OR COALESCE(p_commitment_doc ->> 'manifest_hash', '')
          !~ '^sha256:[0-9a-f]{64}$'
     OR p_commitment_doc ->> 'manifest_hash' IS DISTINCT FROM
          'sha256:' || pg_catalog.encode(
            extensions.digest(
              pg_catalog.convert_to(v_canonical_manifest, 'UTF8'), 'sha256'
            ),
            'hex'
          )
     OR v_canonical_manifest::JSONB IS DISTINCT FROM v_manifest THEN
    RETURN FALSE;
  END IF;

  SELECT pg_catalog.array_agg(key ORDER BY key)
  INTO v_keys
  FROM pg_catalog.jsonb_object_keys(v_manifest) AS key;
  IF v_keys IS DISTINCT FROM ARRAY[
       'disclosure_policy', 'entries', 'evaluation_date', 'icp_count',
       'icp_set_date', 'netuid', 'network_name', 'public_at', 'round_id',
       'schema_version'
     ]::TEXT[]
     OR v_manifest ->> 'schema_version' IS DISTINCT FROM
          'leadpoet.lab_arena.benchmark_commitment.v1'
     OR v_manifest ->> 'disclosure_policy' IS DISTINCT FROM
          'commit_reveal_day2_v1'
     OR v_manifest ->> 'network_name' IS DISTINCT FROM p_network_name
     OR pg_catalog.jsonb_typeof(v_manifest -> 'netuid') IS DISTINCT FROM 'number'
     OR (v_manifest ->> 'netuid')::BIGINT IS DISTINCT FROM p_netuid
     OR v_manifest ->> 'round_id' IS DISTINCT FROM p_round_id
     OR v_manifest ->> 'icp_set_date' IS DISTINCT FROM p_icp_set_date::TEXT
     OR v_manifest ->> 'evaluation_date' IS DISTINCT FROM p_evaluation_date
     OR pg_catalog.jsonb_typeof(v_manifest -> 'public_at') IS DISTINCT FROM 'string'
     OR COALESCE(v_manifest ->> 'public_at', '') !~
          '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?(Z|[+]00:00)$'
     OR (v_manifest ->> 'public_at')::TIMESTAMPTZ IS DISTINCT FROM p_reveal_at
     OR pg_catalog.jsonb_typeof(v_manifest -> 'icp_count') IS DISTINCT FROM 'number'
     OR (v_manifest ->> 'icp_count')::INTEGER <> 20
     OR pg_catalog.jsonb_typeof(v_manifest -> 'entries') IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_manifest -> 'entries') <> 20 THEN
    RETURN FALSE;
  END IF;

  IF EXISTS (
    SELECT 1
    FROM pg_catalog.jsonb_array_elements(v_manifest -> 'entries')
         WITH ORDINALITY AS entry(value, position)
    WHERE pg_catalog.jsonb_typeof(entry.value) IS DISTINCT FROM 'object'
       OR (
         SELECT pg_catalog.array_agg(key ORDER BY key)
         FROM pg_catalog.jsonb_object_keys(entry.value) AS key
       ) IS DISTINCT FROM ARRAY['icp_hash', 'icp_position']::TEXT[]
       OR pg_catalog.jsonb_typeof(entry.value -> 'icp_position')
            IS DISTINCT FROM 'number'
       OR COALESCE(entry.value ->> 'icp_position', '') !~ '^(0|[1-9][0-9]?)$'
       OR (entry.value ->> 'icp_position')::INTEGER <> entry.position - 1
       OR pg_catalog.jsonb_typeof(entry.value -> 'icp_hash')
            IS DISTINCT FROM 'string'
       OR COALESCE(entry.value ->> 'icp_hash', '')
            !~ '^sha256:[0-9a-f]{64}$'
  ) THEN
    RETURN FALSE;
  END IF;
  RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
  RETURN FALSE;
END;
$lab_arena_benchmark_commitment_valid$;
ALTER FUNCTION public.lab_arena__benchmark_commitment_valid_v1(
  TEXT, TEXT, BIGINT, DATE, TEXT, TIMESTAMPTZ, JSONB
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__benchmark_commitment_valid_v1(
  TEXT, TEXT, BIGINT, DATE, TEXT, TIMESTAMPTZ, JSONB
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

-- This trigger is independent of the broad round write-once trigger. That
-- keeps its promotion, reward, billing, and publication exceptions unchanged.
CREATE OR REPLACE FUNCTION public.lab_arena_benchmark_commitment_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $lab_arena_benchmark_commitment_guard$
DECLARE
  v_policy TEXT;
  v_submission_cutoff TIMESTAMPTZ;
  v_expected_reveal TIMESTAMPTZ;
  v_has_commitment BOOLEAN;
BEGIN
  IF NEW.configuration_doc ? 'benchmark_disclosure_policy' THEN
    v_policy := NEW.configuration_doc ->> 'benchmark_disclosure_policy';
    IF pg_catalog.jsonb_typeof(
         NEW.configuration_doc -> 'benchmark_disclosure_policy'
       ) IS DISTINCT FROM 'string'
       OR v_policy IS DISTINCT FROM 'commit_reveal_day2_v1' THEN
      RAISE EXCEPTION 'lab_arena_benchmark_disclosure_policy_invalid'
        USING ERRCODE = '22023';
    END IF;
  END IF;

  IF TG_OP = 'INSERT' THEN
    IF v_policy IS NULL THEN
      IF NEW.benchmark_reveal_at IS NOT NULL
         OR NEW.benchmark_commitment_doc IS NOT NULL
         OR NEW.benchmark_committed_at IS NOT NULL THEN
        RAISE EXCEPTION 'lab_arena_legacy_benchmark_metadata_forbidden'
          USING ERRCODE = '23514';
      END IF;
      RETURN NEW;
    END IF;
    BEGIN
      v_submission_cutoff :=
        (NEW.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ;
      v_expected_reveal := v_submission_cutoff
        + pg_catalog.make_interval(hours => 24);
    EXCEPTION WHEN OTHERS THEN
      RAISE EXCEPTION 'lab_arena_benchmark_reveal_time_invalid'
        USING ERRCODE = '22023';
    END;
    IF NEW.status <> 'open'
       OR NEW.benchmark_commitment_doc IS NOT NULL
       OR NEW.benchmark_committed_at IS NOT NULL
       OR (
         NEW.benchmark_reveal_at IS NOT NULL
         AND NEW.benchmark_reveal_at IS DISTINCT FROM v_expected_reveal
       ) THEN
      RAISE EXCEPTION 'lab_arena_benchmark_commitment_state_invalid'
        USING ERRCODE = '23514';
    END IF;
    NEW.benchmark_reveal_at := v_expected_reveal;
    RETURN NEW;
  END IF;

  IF (OLD.configuration_doc ? 'benchmark_disclosure_policy') IS DISTINCT FROM
       (NEW.configuration_doc ? 'benchmark_disclosure_policy')
     OR OLD.configuration_doc -> 'benchmark_disclosure_policy' IS DISTINCT FROM
       NEW.configuration_doc -> 'benchmark_disclosure_policy' THEN
    RAISE EXCEPTION 'lab_arena_benchmark_disclosure_policy_immutable'
      USING ERRCODE = '42501';
  END IF;
  IF v_policy IS NULL THEN
    IF OLD.benchmark_reveal_at IS NOT NULL
       OR OLD.benchmark_commitment_doc IS NOT NULL
       OR OLD.benchmark_committed_at IS NOT NULL
       OR NEW.benchmark_reveal_at IS NOT NULL
       OR NEW.benchmark_commitment_doc IS NOT NULL
       OR NEW.benchmark_committed_at IS NOT NULL THEN
      RAISE EXCEPTION 'lab_arena_legacy_benchmark_metadata_forbidden'
        USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
  END IF;

  BEGIN
    v_submission_cutoff :=
      (NEW.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ;
    v_expected_reveal := v_submission_cutoff
      + pg_catalog.make_interval(hours => 24);
  EXCEPTION WHEN OTHERS THEN
    RAISE EXCEPTION 'lab_arena_benchmark_reveal_time_invalid'
      USING ERRCODE = '22023';
  END;
  IF NEW.benchmark_reveal_at IS DISTINCT FROM OLD.benchmark_reveal_at
     OR NEW.benchmark_reveal_at IS DISTINCT FROM v_expected_reveal THEN
    RAISE EXCEPTION 'lab_arena_benchmark_reveal_time_immutable'
      USING ERRCODE = '42501';
  END IF;

  v_has_commitment := NEW.benchmark_commitment_doc IS NOT NULL
    AND NEW.benchmark_committed_at IS NOT NULL;
  IF (NEW.benchmark_commitment_doc IS NULL) <>
       (NEW.benchmark_committed_at IS NULL) THEN
    RAISE EXCEPTION 'lab_arena_benchmark_commitment_state_invalid'
      USING ERRCODE = '23514';
  END IF;
  IF OLD.benchmark_commitment_doc IS NOT NULL
     AND (
       NEW.benchmark_commitment_doc IS DISTINCT FROM OLD.benchmark_commitment_doc
       OR NEW.benchmark_committed_at IS DISTINCT FROM OLD.benchmark_committed_at
       OR NEW.benchmark_ref IS DISTINCT FROM OLD.benchmark_ref
       OR NEW.icp_set_date IS DISTINCT FROM OLD.icp_set_date
       OR NEW.evaluation_date IS DISTINCT FROM OLD.evaluation_date
     ) THEN
    RAISE EXCEPTION 'lab_arena_benchmark_commitment_immutable'
      USING ERRCODE = '42501';
  END IF;
  IF OLD.benchmark_commitment_doc IS NULL AND v_has_commitment
     AND (
       OLD.status <> 'open'
       OR NEW.status <> 'committed'
       OR NEW.benchmark_committed_at < v_submission_cutoff
       OR NEW.benchmark_committed_at > pg_catalog.clock_timestamp()
     ) THEN
    RAISE EXCEPTION 'lab_arena_benchmark_commitment_transition_invalid'
      USING ERRCODE = '42501';
  END IF;
  IF v_has_commitment AND (
       NEW.benchmark_ref IS NULL
       OR NEW.benchmark_ref !~
            '^arena/[A-Za-z0-9._:-]+/benchmarks/[0-9a-f]{64}[.]json$'
       OR pg_catalog.split_part(NEW.benchmark_ref, '/', 2)
            IS DISTINCT FROM NEW.round_id
       OR NEW.icp_set_date IS NULL
       OR NEW.evaluation_date IS NULL
       OR NOT public.lab_arena__benchmark_commitment_valid_v1(
         NEW.round_id,
         COALESCE(NEW.configuration_doc ->> 'network_name', 'finney'),
         COALESCE((NEW.configuration_doc ->> 'netuid')::BIGINT, 71),
         NEW.icp_set_date,
         NEW.evaluation_date,
         NEW.benchmark_reveal_at,
         NEW.benchmark_commitment_doc
       )
     ) THEN
    RAISE EXCEPTION 'lab_arena_benchmark_commitment_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF NEW.status = 'open' AND v_has_commitment THEN
    RAISE EXCEPTION 'lab_arena_benchmark_commitment_state_invalid'
      USING ERRCODE = '23514';
  END IF;
  IF NEW.status NOT IN ('open', 'cancelled') AND NOT v_has_commitment THEN
    RAISE EXCEPTION 'lab_arena_benchmark_commitment_required'
      USING ERRCODE = '23514';
  END IF;
  IF NEW.status = 'cancelled'
     AND OLD.status <> 'open'
     AND NOT v_has_commitment THEN
    RAISE EXCEPTION 'lab_arena_benchmark_commitment_required'
      USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$lab_arena_benchmark_commitment_guard$;
ALTER FUNCTION public.lab_arena_benchmark_commitment_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_benchmark_commitment_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DROP TRIGGER IF EXISTS lab_arena_benchmark_commitment_guard
  ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_benchmark_commitment_guard
  BEFORE INSERT OR UPDATE ON public.lab_arena_rounds
  FOR EACH ROW EXECUTE FUNCTION
    public.lab_arena_benchmark_commitment_guard_v1();

-- Build this from the effective migration-200 RPC plus migration-206's live
-- budget adoption. Do not delegate to v2: its separate transaction-visible
-- update cannot include the commitment tuple atomically.
CREATE OR REPLACE FUNCTION public.lab_arena_commit_round_v3(
  p_round_id TEXT,
  p_participants JSONB,
  p_benchmark_ref TEXT,
  p_evaluation_date TEXT,
  p_icp_set_date DATE,
  p_scorer_image_digest TEXT,
  p_scorer_image_reference TEXT,
  p_benchmark_commitment_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_commit_round_v3$
DECLARE
  v_round public.lab_arena_rounds;
  v_expected_bank_date DATE;
  v_expected_evaluation_date DATE;
  v_submission_cutoff TIMESTAMPTZ;
  v_pending_admissions INTEGER;
  v_participant_count INTEGER;
  v_frozen_count INTEGER;
  v_invalid_participants INTEGER;
  v_baseline_count INTEGER;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale',
      'round_status', v_round.status,
      'status_generation', v_round.status_generation
    );
  END IF;
  IF v_round.configuration_doc ->> 'benchmark_disclosure_policy'
       IS DISTINCT FROM 'commit_reveal_day2_v1'
     OR pg_catalog.jsonb_typeof(
       v_round.configuration_doc -> 'benchmark_disclosure_policy'
     ) IS DISTINCT FROM 'string' THEN
    RAISE EXCEPTION 'lab_arena_benchmark_disclosure_policy_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT COUNT(*) INTO v_pending_admissions
  FROM public.lab_arena_submissions
  WHERE round_id = p_round_id AND status = 'accepted';
  IF v_pending_admissions <> 0 THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'retry',
      'round_status', v_round.status,
      'remaining_admissions', v_pending_admissions
    );
  END IF;
  BEGIN
    v_expected_bank_date := pg_catalog.timezone(
      'UTC',
      (v_round.configuration_doc #>> '{schedule,submission_open}')::TIMESTAMPTZ
    )::DATE;
    v_submission_cutoff :=
      (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ;
    v_expected_evaluation_date := pg_catalog.timezone(
      'UTC', v_submission_cutoff
    )::DATE;
  EXCEPTION WHEN OTHERS THEN
    RAISE EXCEPTION 'lab_arena_round_commit_invalid' USING ERRCODE = '22023';
  END;
  IF pg_catalog.clock_timestamp() < v_submission_cutoff THEN
    RAISE EXCEPTION 'lab_arena_round_commit_too_early'
      USING ERRCODE = '55000';
  END IF;
  IF pg_catalog.jsonb_typeof(p_participants) IS DISTINCT FROM 'array'
     OR pg_catalog.char_length(COALESCE(p_benchmark_ref, '')) NOT BETWEEN 1 AND 1024
     OR COALESCE(p_benchmark_ref, '') !~
          '^arena/[A-Za-z0-9._:-]+/benchmarks/[0-9a-f]{64}[.]json$'
     OR pg_catalog.split_part(p_benchmark_ref, '/', 2) IS DISTINCT FROM p_round_id
     OR COALESCE(p_evaluation_date, '') !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
     OR p_evaluation_date::DATE <> v_expected_evaluation_date
     OR p_icp_set_date IS NULL
     OR p_icp_set_date <> v_expected_bank_date
     OR v_expected_evaluation_date <> v_expected_bank_date + 1 THEN
    RAISE EXCEPTION 'lab_arena_round_commit_invalid' USING ERRCODE = '22023';
  END IF;
  IF NOT public.lab_arena__benchmark_commitment_valid_v1(
       p_round_id,
       v_round.arena_network_name,
       v_round.arena_netuid,
       p_icp_set_date,
       p_evaluation_date,
       v_round.benchmark_reveal_at,
       p_benchmark_commitment_doc
     ) THEN
    RAISE EXCEPTION 'lab_arena_benchmark_commitment_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_participant_count := pg_catalog.jsonb_array_length(p_participants);

  SELECT COUNT(*)
  INTO v_frozen_count
  FROM public.lab_arena_submissions
  WHERE round_id = p_round_id AND status = 'frozen';
  SELECT COUNT(*) INTO v_invalid_participants
  FROM pg_catalog.jsonb_array_elements(p_participants) AS participant
  WHERE pg_catalog.jsonb_typeof(participant) IS DISTINCT FROM 'object'
     OR NOT EXISTS (
       SELECT 1
       FROM public.lab_arena_submissions AS submission
       WHERE submission.round_id = p_round_id
         AND submission.status = 'frozen'
         AND submission.submission_id = participant ->> 'submission_id'
         AND submission.miner_hotkey = participant ->> 'miner_hotkey'
         AND submission.is_king = COALESCE(
           (participant ->> 'is_king')::BOOLEAN, FALSE
         )
     );
  SELECT COUNT(*) INTO v_baseline_count
  FROM pg_catalog.jsonb_array_elements(p_participants) AS participant
  WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
  IF v_participant_count <> v_frozen_count
     OR v_invalid_participants <> 0
     OR (
       SELECT COUNT(DISTINCT participant ->> 'submission_id')
       FROM pg_catalog.jsonb_array_elements(p_participants) AS participant
     ) <> v_participant_count
     OR v_baseline_count <> 1
     OR v_participant_count >
       COALESCE((v_round.configuration_doc ->> 'max_challengers')::INTEGER, 100) + 1 THEN
    RAISE EXCEPTION 'lab_arena_round_participants_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF COALESCE(p_scorer_image_digest, '') !~ '^sha256:[0-9a-f]{64}$'
     OR pg_catalog.char_length(COALESCE(p_scorer_image_reference, ''))
          NOT BETWEEN 1 AND 512
     OR pg_catalog.right(
       p_scorer_image_reference,
       pg_catalog.char_length(p_scorer_image_digest) + 1
     ) <> '@' || p_scorer_image_digest THEN
    RAISE EXCEPTION 'lab_arena_scorer_image_invalid' USING ERRCODE = '22023';
  END IF;
  UPDATE public.lab_arena_rounds
  SET status = 'committed',
      status_generation = status_generation + 1,
      participants = p_participants,
      benchmark_ref = p_benchmark_ref,
      evaluation_date = p_evaluation_date,
      icp_set_date = p_icp_set_date,
      benchmark_commitment_doc = p_benchmark_commitment_doc,
      benchmark_committed_at = pg_catalog.clock_timestamp(),
      configuration_doc = (
        CASE
          WHEN v_round.configuration_doc ->> 'mode' = 'live'
               AND NOT v_round.configuration_doc ? 'cost_per_company_microusd'
          THEN v_round.configuration_doc || pg_catalog.jsonb_build_object(
            'execution_cap_microusd', 50000000,
            'cost_per_company_microusd', 500000
          )
          ELSE v_round.configuration_doc
        END
      ) || pg_catalog.jsonb_build_object(
        'scorer_image_digest', p_scorer_image_digest,
        'scorer_image_reference', p_scorer_image_reference
      )
  WHERE round_id = p_round_id;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok',
    'round_status', v_round.status,
    'status_generation', v_round.status_generation,
    'benchmark_committed_at', v_round.benchmark_committed_at
  );
END;
$lab_arena_commit_round_v3$;
ALTER FUNCTION public.lab_arena_commit_round_v3(
  TEXT, JSONB, TEXT, TEXT, DATE, TEXT, TEXT, JSONB
) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_benchmark_disclosure_schema_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = pg_catalog
AS $lab_arena_benchmark_disclosure_schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.benchmark_disclosure.v1',
    'version', 210,
    'policy', 'commit_reveal_day2_v1'
  )
$lab_arena_benchmark_disclosure_schema$;
ALTER FUNCTION public.lab_arena_benchmark_disclosure_schema_v1()
  OWNER TO lab_arena_owner;

DO $lab_arena_benchmark_disclosure_acl$
DECLARE
  signature TEXT;
  role_name TEXT;
BEGIN
  FOREACH signature IN ARRAY ARRAY[
    'public.lab_arena_commit_round_v3(TEXT, JSONB, TEXT, TEXT, DATE, TEXT, TEXT, JSONB)',
    'public.lab_arena_benchmark_disclosure_schema_v1()'
  ] LOOP
    EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', signature);
    FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
      IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
        EXECUTE pg_catalog.format(
          'REVOKE ALL ON FUNCTION %s FROM %I', signature, role_name
        );
      END IF;
    END LOOP;
    EXECUTE pg_catalog.format(
      'GRANT EXECUTE ON FUNCTION %s TO lab_arena_service', signature
    );
  END LOOP;
END;
$lab_arena_benchmark_disclosure_acl$;

-- A first application must not silently adopt a prematurely marked row. A
-- reapplication validates every already-created new-policy row in place.
DO $lab_arena_benchmark_existing_rows$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM public.lab_arena_rounds AS legacy_round
    WHERE NOT legacy_round.configuration_doc ? 'benchmark_disclosure_policy'
      AND (
        legacy_round.benchmark_reveal_at IS NOT NULL
        OR legacy_round.benchmark_commitment_doc IS NOT NULL
        OR legacy_round.benchmark_committed_at IS NOT NULL
      )
  ) OR EXISTS (
    SELECT 1
    FROM public.lab_arena_rounds AS round_row
    WHERE round_row.configuration_doc ? 'benchmark_disclosure_policy'
      AND (
        pg_catalog.jsonb_typeof(
          round_row.configuration_doc -> 'benchmark_disclosure_policy'
        ) IS DISTINCT FROM 'string'
        OR round_row.configuration_doc ->> 'benchmark_disclosure_policy'
             IS DISTINCT FROM 'commit_reveal_day2_v1'
        OR round_row.benchmark_reveal_at IS NULL
        OR round_row.benchmark_reveal_at IS DISTINCT FROM
             (round_row.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
             + pg_catalog.make_interval(hours => 24)
        OR (round_row.benchmark_commitment_doc IS NULL) <>
             (round_row.benchmark_committed_at IS NULL)
        OR (
          round_row.status = 'open'
          AND round_row.benchmark_commitment_doc IS NOT NULL
        )
        OR (
          round_row.status NOT IN ('open', 'cancelled')
          AND (
            round_row.benchmark_commitment_doc IS NULL
            OR round_row.benchmark_committed_at IS NULL
          )
        )
        OR (
          round_row.benchmark_commitment_doc IS NOT NULL
          AND (
            round_row.benchmark_ref IS NULL
            OR round_row.benchmark_ref !~
                 '^arena/[A-Za-z0-9._:-]+/benchmarks/[0-9a-f]{64}[.]json$'
            OR pg_catalog.split_part(round_row.benchmark_ref, '/', 2)
                 IS DISTINCT FROM round_row.round_id
            OR round_row.icp_set_date IS NULL
            OR round_row.evaluation_date IS NULL
            OR round_row.benchmark_committed_at <
                 (round_row.configuration_doc #>>
                   '{schedule,submission_cutoff}')::TIMESTAMPTZ
            OR round_row.benchmark_committed_at > pg_catalog.clock_timestamp()
            OR NOT public.lab_arena__benchmark_commitment_valid_v1(
              round_row.round_id,
              round_row.arena_network_name,
              round_row.arena_netuid,
              round_row.icp_set_date,
              round_row.evaluation_date,
              round_row.benchmark_reveal_at,
              round_row.benchmark_commitment_doc
            )
          )
        )
      )
  ) THEN
    RAISE EXCEPTION 'lab_arena_existing_benchmark_disclosure_state_invalid';
  END IF;
END;
$lab_arena_benchmark_existing_rows$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
