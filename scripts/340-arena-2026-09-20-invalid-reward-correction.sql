-- Inactive, one-use recovery capability. Render only from the protected
-- production preparation artifact. The signed replacement basis is supplied
-- at call time and is never committed to this repository.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

-- The migration runner transfers this temporary function to the locked Arena
-- owner. That target role needs CREATE only for the ownership transfer.
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_sep20_revoke_invalid_reward340(
  p_reward_basis JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $correction340$
DECLARE
  active public.lab_arena_rounds;
  archived public.lab_arena_rounds;
  active_sha256 TEXT;
  payload_sha256 TEXT;
  archive_found BOOLEAN;
  revoked_basis JSONB;
  revoked_signing_key JSONB;
  revoked_activated_at TIMESTAMPTZ;
  archive_round_id CONSTANT TEXT := 'arena-2026-09-20-rewardhistory340';
  replacement_epoch CONSTANT BIGINT := 25294;
  last_immutable_epoch CONSTANT BIGINT := 25293;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab_arena.reward_epoch', 0)
  );
  SELECT * INTO STRICT active
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-20'
  FOR UPDATE;
  SELECT * INTO archived
  FROM public.lab_arena_rounds
  WHERE round_id = archive_round_id
  FOR UPDATE;
  archive_found := FOUND;

  payload_sha256 := pg_catalog.encode(
    extensions.digest(p_reward_basis::TEXT, 'sha256'), 'hex'
  );
  IF payload_sha256 <> 'd8c260a141cca310451a26125746cd52a1dd9b87be7b506ff2ee92cd91bf1c4f'
     OR p_reward_basis ->> 'reward_basis_hash'
          <> 'sha256:4ca55a8c4cd2cbe13270eb78852b87be4d9bd4ec45a80f262120e97379ac4124'
     OR p_reward_basis ->> 'schema_version'
          <> 'leadpoet.lab_arena.reward_basis.v1'
     OR p_reward_basis ->> 'round_id' <> active.round_id
     OR (p_reward_basis ->> 'effective_reward_epoch')::BIGINT
          <> replacement_epoch
     OR p_reward_basis ->> 'king_outcome' <> 'no_king'
     OR COALESCE(p_reward_basis ->> 'king_hotkey', '') <> ''
     OR (p_reward_basis ->> 'king_start_epoch')::BIGINT <> 0
     OR (p_reward_basis ->> 'champion_reward_factor_ppm')::INTEGER <> 1000000
     OR (p_reward_basis ->> 'published_at')::TIMESTAMPTZ
          IS DISTINCT FROM active.published_at
     OR p_reward_basis -> 'reward_constants'
          IS DISTINCT FROM active.configuration_doc -> 'reward_constants'
     OR pg_catalog.jsonb_typeof(p_reward_basis -> 'signature')
          IS DISTINCT FROM 'object'
     OR p_reward_basis -> 'signature' ->> 'algorithm' <> 'ECDSA_SHA_256'
     OR p_reward_basis -> 'signature' ->> 'public_key_hash'
          IS DISTINCT FROM active.signing_key_doc ->> 'public_key_hash'
     OR pg_catalog.jsonb_typeof(active.signing_key_doc) IS DISTINCT FROM 'object'
     OR active.signing_key_doc ->> 'algorithm' <> 'ECDSA_SHA_256'
     OR active.signing_key_doc ->> 'key_spec' <> 'ECC_NIST_P256'
     OR active.signing_key_doc ->> 'public_key_hash'
          <> 'sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a'
     OR (SELECT pg_catalog.array_agg(key ORDER BY key)
         FROM pg_catalog.jsonb_object_keys(p_reward_basis) key)
          IS DISTINCT FROM ARRAY[
            'champion_reward_factor_ppm','effective_reward_epoch','king_hotkey',
            'king_outcome','king_start_epoch','published_at','reward_basis_hash',
            'reward_constants','round_id','schema_version','signature'
          ]::TEXT[]
     OR (SELECT pg_catalog.array_agg(key ORDER BY key)
         FROM pg_catalog.jsonb_object_keys(p_reward_basis -> 'signature') key)
          IS DISTINCT FROM ARRAY[
            'algorithm','public_key_hash','signature_b64'
          ]::TEXT[] THEN
    RAISE EXCEPTION 'Sep20 replacement reward payload differs'
      USING ERRCODE = '22023';
  END IF;

  -- Exact retries are read-only, including after the replacement epoch has
  -- acquired its own immutable accepted state.
  IF active.reward_basis_hash = 'sha256:4ca55a8c4cd2cbe13270eb78852b87be4d9bd4ec45a80f262120e97379ac4124'
     AND active.reward_basis_doc IS NOT DISTINCT FROM p_reward_basis
     AND active.effective_reward_epoch = replacement_epoch
     AND active.king_start_epoch = 0
     AND active.reward_activated_at IS NOT NULL
     AND archived.round_id = archive_round_id
     AND archived.status = 'cancelled'
     AND archived.cancel_reason =
          'authorized_sep20_invalid_champion_reward_archive340'
     AND archived.reward_basis_hash = 'sha256:c3193add5ccb652c0e94a5e0aeb67cc26f11f579e3fd07f1ce2eb56ee51652fd'
     AND archived.effective_reward_epoch = 25288
     AND archived.reward_basis_doc ->> 'reward_basis_hash'
          = 'sha256:c3193add5ccb652c0e94a5e0aeb67cc26f11f579e3fd07f1ce2eb56ee51652fd'
     AND archived.signing_key_doc IS NOT NULL
     AND archived.reward_activated_at IS NOT NULL THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing',
      'archived_reward_basis_hash', archived.reward_basis_hash,
      'active_reward_basis_hash', active.reward_basis_hash
    );
  END IF;

  active_sha256 := pg_catalog.encode(
    extensions.digest(pg_catalog.to_jsonb(active)::TEXT, 'sha256'), 'hex'
  );
  IF archive_found
     OR active_sha256 <> 'cc420aa2ca8e0eed3196bdc136d814db4e081ba880fbf852693f9825f6a07271'
     OR pg_catalog.encode(extensions.digest(
          active.publication_doc::TEXT, 'sha256'), 'hex')
          <> 'f87acfaa1e209413d2f35524abfe07dd3523045c5cd52055e78ad3b5fd00cc1c'
     OR pg_catalog.encode(extensions.digest(
          active.signing_key_doc::TEXT, 'sha256'), 'hex')
          <> 'ccbad44c8b39edb9f508a96fa65be4a9fe0923d0e3230c950701d57b6611cdb1'
     OR active.status <> 'published'
     OR active.configuration_doc ->> 'mode' <> 'live'
     OR NOT active.rewards_enabled
     OR active.publication_doc #>> '{king_decision,outcome}' <> 'no_king'
     OR COALESCE(active.publication_doc #>> '{king_decision,king_hotkey}', '') <> ''
     OR active.king_outcome <> 'no_king'
     OR active.king_hotkey IS NOT NULL
     OR active.reward_basis_hash <> 'sha256:c3193add5ccb652c0e94a5e0aeb67cc26f11f579e3fd07f1ce2eb56ee51652fd'
     OR active.reward_basis_doc ->> 'reward_basis_hash'
          <> 'sha256:c3193add5ccb652c0e94a5e0aeb67cc26f11f579e3fd07f1ce2eb56ee51652fd'
     OR active.reward_basis_doc ->> 'king_outcome' NOT IN ('crowned', 'defended')
     OR COALESCE(active.reward_basis_doc ->> 'king_hotkey', '') = ''
     OR active.effective_reward_epoch <> 25288
     OR active.reward_activated_at IS NULL
     OR replacement_epoch <= active.effective_reward_epoch
     OR replacement_epoch <> last_immutable_epoch + 1
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_accepted_weight_states accepted
       WHERE accepted.network = active.arena_network_name
         AND accepted.netuid = active.arena_netuid
         AND accepted.epoch = last_immutable_epoch
         AND accepted.state_doc #>> '{reward_basis,reward_basis_hash}'
              = active.reward_basis_hash
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_accepted_weight_states accepted
       WHERE accepted.network = active.arena_network_name
         AND accepted.netuid = active.arena_netuid
         AND accepted.epoch >= replacement_epoch
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_chain_outcomes outcome
       WHERE outcome.network = active.arena_network_name
         AND outcome.netuid = active.arena_netuid
         AND outcome.epoch >= replacement_epoch
     )
     OR EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger trigger_row
       WHERE trigger_row.tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND NOT trigger_row.tgisinternal
         AND trigger_row.tgenabled <> 'O'
     ) THEN
    RAISE EXCEPTION 'Sep20 invalid reward correction preimage differs'
      USING ERRCODE = '55000';
  END IF;

  revoked_basis := active.reward_basis_doc;
  revoked_signing_key := active.signing_key_doc;
  revoked_activated_at := active.reward_activated_at;

  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  UPDATE public.lab_arena_rounds
  SET effective_reward_epoch = NULL,
      reward_basis_hash = NULL,
      reward_basis_doc = NULL,
      signing_key_doc = NULL,
      reward_activated_at = NULL,
      king_start_epoch = NULL
  WHERE round_id = active.round_id;

  INSERT INTO public.lab_arena_rounds(
    round_id,status,status_generation,stage_generation,configuration_doc,
    rewards_enabled,participants,benchmark_ref,evaluation_date,
    stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
    king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
    reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
    cancel_reason,published_at,created_at,updated_at,promotion_required,
    promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
    confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
    champion_funding_frozen,champion_submission_id,champion_hotkey,
    champion_fallback_providers
  )
  SELECT round_id,status,status_generation,stage_generation,configuration_doc,
    rewards_enabled,participants,benchmark_ref,evaluation_date,
    stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
    king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
    reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
    cancel_reason,published_at,created_at,updated_at,promotion_required,
    promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
    confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
    champion_funding_frozen,champion_submission_id,champion_hotkey,
    champion_fallback_providers
  FROM pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_rounds,
    pg_catalog.to_jsonb(active) || pg_catalog.jsonb_build_object(
      'round_id', archive_round_id,
      'status', 'cancelled',
      'configuration_doc', active.configuration_doc ||
        pg_catalog.jsonb_build_object('round_id', archive_round_id),
      'publication_doc', NULL,
      'promotion_required', FALSE,
      'cancel_reason',
        'authorized_sep20_invalid_champion_reward_archive340'
    )
  );

  UPDATE public.lab_arena_rounds
  SET effective_reward_epoch = replacement_epoch,
      reward_basis_hash = p_reward_basis ->> 'reward_basis_hash',
      reward_basis_doc = p_reward_basis,
      signing_key_doc = active.signing_key_doc,
      reward_activated_at = pg_catalog.clock_timestamp(),
      king_start_epoch = 0
  WHERE round_id = active.round_id;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

  SELECT * INTO STRICT active
  FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-09-20';
  SELECT * INTO STRICT archived
  FROM public.lab_arena_rounds WHERE round_id = archive_round_id;
  IF active.reward_basis_doc IS DISTINCT FROM p_reward_basis
     OR active.reward_basis_hash <> 'sha256:4ca55a8c4cd2cbe13270eb78852b87be4d9bd4ec45a80f262120e97379ac4124'
     OR active.effective_reward_epoch <> replacement_epoch
     OR active.king_start_epoch <> 0
     OR active.status <> 'published'
     OR active.publication_doc #>> '{king_decision,outcome}' <> 'no_king'
     OR archived.reward_basis_doc IS DISTINCT FROM revoked_basis
     OR archived.signing_key_doc IS DISTINCT FROM revoked_signing_key
     OR archived.reward_activated_at IS DISTINCT FROM revoked_activated_at
     OR archived.reward_basis_hash <> 'sha256:c3193add5ccb652c0e94a5e0aeb67cc26f11f579e3fd07f1ce2eb56ee51652fd'
     OR archived.effective_reward_epoch <> 25288
     OR archived.status <> 'cancelled'
     OR archived.publication_doc IS NOT NULL
     OR archived.promotion_required
     OR archived.configuration_doc ->> 'mode' <> 'live'
     OR NOT archived.rewards_enabled
     OR EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger trigger_row
       WHERE trigger_row.tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND NOT trigger_row.tgisinternal
         AND trigger_row.tgenabled <> 'O'
     ) THEN
    RAISE EXCEPTION 'Sep20 invalid reward correction postimage differs'
      USING ERRCODE = '55000';
  END IF;

  RETURN pg_catalog.jsonb_build_object(
    'status', 'corrected',
    'archived_reward_basis_hash', archived.reward_basis_hash,
    'active_reward_basis_hash', active.reward_basis_hash
  );
END;
$correction340$;
ALTER FUNCTION public.lab_arena_sep20_revoke_invalid_reward340(JSONB)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep20_revoke_invalid_reward340(JSONB)
  FROM PUBLIC, lab_arena_service;
DO $correction340_acl$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'service_role'
  ) THEN
    RAISE EXCEPTION 'service_role is required for Sep20 reward correction';
  END IF;
  EXECUTE 'GRANT EXECUTE ON FUNCTION public.lab_arena_sep20_revoke_invalid_reward340(JSONB) TO service_role';
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'anon') THEN
    EXECUTE 'REVOKE ALL ON FUNCTION public.lab_arena_sep20_revoke_invalid_reward340(JSONB) FROM anon';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'authenticated') THEN
    EXECUTE 'REVOKE ALL ON FUNCTION public.lab_arena_sep20_revoke_invalid_reward340(JSONB) FROM authenticated';
  END IF;
END;
$correction340_acl$;

NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
