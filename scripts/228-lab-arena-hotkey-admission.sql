-- Admit one challenger per hotkey rather than one challenger per coldkey.
-- Finalized ownership remains immutable audit metadata. Only untouched open
-- daily Finney 71 rounds using the former standard caps move to twenty.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Patch the exact migration-211 owner-collision seam. The surrounding input,
-- hotkey, source, baseline, and owner-tampering checks remain byte-for-byte in
-- the installed function. Replays accept only the already-patched shape.
DO $lab_arena_hotkey_admission$
DECLARE
  v_definition TEXT;
  v_owner_declaration TEXT := $old$
  v_owner_existing public.lab_arena_submissions;
$old$;
  v_owner_collision TEXT := $old$
  IF v_integrity AND NOT v_is_baseline THEN
    SELECT * INTO v_owner_existing
    FROM public.lab_arena_submissions
    WHERE round_id = p_round_id
      AND owner_coldkey = v_fixed_owner_coldkey
      AND status IN ('uploading', 'accepted', 'frozen')
    ORDER BY created_at LIMIT 1 FOR UPDATE;
    IF FOUND THEN
      RAISE EXCEPTION 'lab_arena_owner_active_submission' USING ERRCODE = '23505';
    END IF;
  END IF;

$old$;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena__register_submission_v2(text,text,text,jsonb,text,bigint,text)'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_submissions_one_active_per_miner_uq'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply the current Arena admission schema first';
  END IF;

  SELECT pg_catalog.pg_get_functiondef(procedure.oid)
  INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena__register_submission_v2'
    AND procedure.pronargs = 7;

  IF pg_catalog.strpos(
       v_definition,
       'v_existing.owner_coldkey IS DISTINCT FROM p_owner_coldkey'
     ) = 0
     OR pg_catalog.strpos(
       v_definition,
       'WHERE round_id = p_round_id AND miner_hotkey = p_miner_hotkey'
     ) = 0 THEN
    RAISE EXCEPTION 'lab_arena_hotkey_admission_shape_unexpected';
  END IF;

  IF pg_catalog.strpos(
       v_definition, 'lab_arena_owner_active_submission'
     ) > 0 THEN
    IF pg_catalog.strpos(v_definition, v_owner_declaration) = 0
       OR pg_catalog.strpos(v_definition, v_owner_collision) = 0 THEN
      RAISE EXCEPTION 'lab_arena_hotkey_admission_shape_unexpected';
    END IF;
    v_definition := pg_catalog.replace(
      v_definition, v_owner_declaration, E'\n'
    );
    v_definition := pg_catalog.replace(
      v_definition, v_owner_collision, ''
    );
    EXECUTE v_definition;
  ELSIF pg_catalog.strpos(v_definition, 'v_owner_existing') > 0 THEN
    RAISE EXCEPTION 'lab_arena_hotkey_admission_shape_unexpected';
  END IF;
END;
$lab_arena_hotkey_admission$;

DROP INDEX IF EXISTS public.lab_arena_submissions_one_active_owner_uq;

-- Serialize the one-time cap repair with cutoff, participant freeze, and run
-- creation. The write-once trigger is restored in this same transaction.
LOCK TABLE public.lab_arena_rounds IN SHARE ROW EXCLUSIVE MODE;
ALTER TABLE public.lab_arena_rounds
  DISABLE TRIGGER lab_arena_rounds_write_once;

UPDATE public.lab_arena_rounds AS round
SET configuration_doc = pg_catalog.jsonb_set(
  round.configuration_doc, '{max_challengers}', '20'::JSONB, FALSE
)
WHERE round.round_id ~ '^arena-[0-9]{4}-[0-9]{2}-[0-9]{2}$'
  AND round.status = 'open'
  AND round.arena_network_name = 'finney'
  AND round.arena_netuid = 71
  AND round.configuration_doc ->> 'mode' = 'live'
  AND round.configuration_doc ->> 'max_challengers' IN ('8', '16')
  AND COALESCE(
    (round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
      > pg_catalog.clock_timestamp(),
    FALSE
  )
  AND round.benchmark_ref IS NULL
  AND (
    round.participants IS NULL
    OR round.participants = '[]'::JSONB
  )
  AND NOT EXISTS (
    SELECT 1
    FROM public.lab_arena_runs AS run
    WHERE run.round_id = round.round_id
  );

ALTER TABLE public.lab_arena_rounds
  ENABLE TRIGGER lab_arena_rounds_write_once;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
