-- Remove the one-use Sep20 correction capability only after the exact durable
-- replacement and archive can be read back. This migration is replay-safe.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $cleanup341$
DECLARE
  active public.lab_arena_rounds;
  archived public.lab_arena_rounds;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab_arena.reward_epoch', 0)
  );
  SELECT * INTO STRICT active
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-20'
  FOR UPDATE;
  SELECT * INTO STRICT archived
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-20-rewardhistory340'
  FOR UPDATE;

  IF active.status <> 'published'
     OR active.publication_doc #>> '{king_decision,outcome}' <> 'no_king'
     OR active.king_outcome <> 'no_king'
     OR active.king_hotkey IS NOT NULL
     OR active.reward_basis_hash <> 'sha256:4ca55a8c4cd2cbe13270eb78852b87be4d9bd4ec45a80f262120e97379ac4124'
     OR active.reward_basis_doc ->> 'reward_basis_hash'
          <> 'sha256:4ca55a8c4cd2cbe13270eb78852b87be4d9bd4ec45a80f262120e97379ac4124'
     OR pg_catalog.encode(extensions.digest(
          active.reward_basis_doc::TEXT, 'sha256'), 'hex')
          <> 'd8c260a141cca310451a26125746cd52a1dd9b87be7b506ff2ee92cd91bf1c4f'
     OR active.reward_basis_doc ->> 'king_outcome' <> 'no_king'
     OR COALESCE(active.reward_basis_doc ->> 'king_hotkey', '') <> ''
     OR active.effective_reward_epoch <> 25294
     OR active.reward_activated_at IS NULL
     OR archived.status <> 'cancelled'
     OR archived.cancel_reason <>
          'authorized_sep20_invalid_champion_reward_archive340'
     OR archived.publication_doc IS NOT NULL
     OR archived.promotion_required
     OR archived.configuration_doc ->> 'mode' <> 'live'
     OR NOT archived.rewards_enabled
     OR archived.reward_basis_hash <> 'sha256:c3193add5ccb652c0e94a5e0aeb67cc26f11f579e3fd07f1ce2eb56ee51652fd'
     OR archived.reward_basis_doc ->> 'reward_basis_hash'
          <> 'sha256:c3193add5ccb652c0e94a5e0aeb67cc26f11f579e3fd07f1ce2eb56ee51652fd'
     OR archived.effective_reward_epoch <> 25288
     OR archived.reward_activated_at IS NULL THEN
    RAISE EXCEPTION 'Sep20 reward correction cleanup readback differs'
      USING ERRCODE = '55000';
  END IF;
END;
$cleanup341$;

DROP FUNCTION IF EXISTS
  public.lab_arena_sep20_revoke_invalid_reward340(JSONB);

DO $cleanup341_assert$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_sep20_revoke_invalid_reward340(jsonb)'
     ) IS NOT NULL THEN
    RAISE EXCEPTION 'Sep20 reward correction capability remains installed'
      USING ERRCODE = '55000';
  END IF;
END;
$cleanup341_assert$;

NOTIFY pgrst, 'reload schema';
COMMIT;
