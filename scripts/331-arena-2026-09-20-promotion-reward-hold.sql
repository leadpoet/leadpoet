-- Hold only Sep20 rerun328 promotion and reward activation while its published
-- scores are reviewed. Judging, publication, execution data and billing stay live.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $sep20_rerun328_authority_hold$
DECLARE
 active public.lab_arena_rounds%ROWTYPE;
 baseline public.lab_arena_submissions%ROWTYPE;
 before_row JSONB;
 after_row JSONB;
 original_configuration_sha256 TEXT;
 expected_hold JSONB;
 baseline_zero_positions BIGINT;
 baseline_publication_count BIGINT;
 zero_baseline_publication_count BIGINT;
 changed BIGINT;
BEGIN
 PERFORM pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('arena-2026-09-20',0));

 SELECT * INTO active FROM public.lab_arena_rounds
 WHERE round_id='arena-2026-09-20' FOR UPDATE;
 SELECT * INTO baseline FROM public.lab_arena_submissions
 WHERE round_id='arena-2026-09-20'
  AND submission_id='baseline-2026-09-20' FOR UPDATE;
 IF active.round_id IS NULL OR baseline.submission_id IS NULL THEN
  RAISE EXCEPTION 'Sep20 rerun328 authority hold target missing'
   USING ERRCODE='55000';
 END IF;

 -- A repeat after the hold may observe later judging or publication. It is
 -- idempotent only while every authority side effect remains absent.
 IF active.configuration_doc ? 'sep20_rerun328_promotion_reward_hold' THEN
  expected_hold:=active.configuration_doc->
   'sep20_rerun328_promotion_reward_hold';
  IF active.rewards_enabled IS DISTINCT FROM FALSE
   OR active.promotion_required IS DISTINCT FROM FALSE
   OR active.configuration_doc->'rewards_enabled' IS DISTINCT FROM 'false'::JSONB
   OR pg_catalog.jsonb_typeof(expected_hold) IS DISTINCT FROM 'object'
   OR (SELECT pg_catalog.count(*) FROM
       pg_catalog.jsonb_object_keys(expected_hold))<>8
   OR expected_hold->>'schema_version' IS DISTINCT FROM
      'leadpoet.sep20.rerun328.authority_hold.v1'
   OR expected_hold->>'migration' IS DISTINCT FROM
      '331-arena-2026-09-20-promotion-reward-hold'
   OR expected_hold->'original_row_rewards_enabled' IS DISTINCT FROM 'true'::JSONB
   OR expected_hold->'original_promotion_required' IS DISTINCT FROM 'true'::JSONB
   OR expected_hold->'original_configuration_rewards_enabled'
      IS DISTINCT FROM 'true'::JSONB
   OR expected_hold->>'source_commit' IS DISTINCT FROM
      '5d492f1715e27c8d9b919bcc3ad69d1be5160524'
   OR expected_hold->>'scorer_image_digest' IS DISTINCT FROM
      'sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f'
   OR COALESCE(expected_hold->>'original_configuration_sha256','')
      !~ '^[0-9a-f]{64}$'
   OR active.effective_reward_epoch IS NOT NULL
   OR active.reward_basis_hash IS NOT NULL OR active.reward_basis_doc IS NOT NULL
   OR active.signing_key_doc IS NOT NULL OR active.reward_activated_at IS NOT NULL
   OR active.promotion_doc IS NOT NULL OR active.baseline_promoted_at IS NOT NULL THEN
   RAISE EXCEPTION 'existing Sep20 rerun328 authority hold differs'
    USING ERRCODE='55000';
  END IF;
  RETURN;
 END IF;

 IF active.status NOT IN(
     'stage2_scoring','stage2_judged','scored','published')
  OR active.configuration_doc->>'mode' IS DISTINCT FROM 'live'
  OR active.configuration_doc->>'round_id' IS DISTINCT FROM
     'arena-2026-09-20'
  OR active.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM
     'sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f'
  OR active.configuration_doc->>'scorer_image_reference' IS DISTINCT FROM
     '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f'
  OR active.configuration_doc->'rewards_enabled' IS DISTINCT FROM 'true'::JSONB
  OR active.rewards_enabled IS DISTINCT FROM TRUE
  OR active.promotion_required IS DISTINCT FROM TRUE
  OR active.effective_reward_epoch IS NOT NULL
  OR active.reward_basis_hash IS NOT NULL OR active.reward_basis_doc IS NOT NULL
  OR active.signing_key_doc IS NOT NULL OR active.reward_activated_at IS NOT NULL
  OR active.promotion_doc IS NOT NULL OR active.baseline_promoted_at IS NOT NULL
  OR baseline.source_ref IS DISTINCT FROM
     'arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun328-5d492f17.tar.gz'
  OR baseline.source_size_bytes IS DISTINCT FROM 854708
  OR baseline.submission_doc->>'source_sha256' IS DISTINCT FROM
     'f1f24e24e0cd736d8c9695640093f4d5f28c360e2773b9824dd5010fe5100f3e'
  OR baseline.submission_doc->>'source_commit' IS DISTINCT FROM
     '5d492f1715e27c8d9b919bcc3ad69d1be5160524' THEN
  RAISE EXCEPTION 'Sep20 rerun328 authority hold precondition differs'
   USING ERRCODE='55000';
 END IF;

 SELECT pg_catalog.count(*) INTO baseline_zero_positions FROM(
  SELECT DISTINCT ON(icp_position) icp_position,per_icp_score
  FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20'
   AND submission_id='baseline-2026-09-20' AND kind='execute'
   AND status='accepted' AND icp_position IN(1,8)
  ORDER BY icp_position,attempt DESC) latest
 WHERE per_icp_score=0;
 IF baseline_zero_positions<>2 THEN
  RAISE EXCEPTION 'Sep20 rerun328 baseline positions 1 and 8 are not sealed zeroes'
   USING ERRCODE='55000';
 END IF;

 IF active.status='published' THEN
  SELECT pg_catalog.count(*),pg_catalog.count(*) FILTER(WHERE
    CASE WHEN pg_catalog.jsonb_typeof(entry->'final_score')='number'
      THEN (entry->>'final_score')::NUMERIC=0 ELSE FALSE END)
  INTO baseline_publication_count,zero_baseline_publication_count
  FROM pg_catalog.jsonb_array_elements(CASE
   WHEN pg_catalog.jsonb_typeof(active.publication_doc->'final_ranking')='array'
   THEN active.publication_doc->'final_ranking' ELSE '[]'::JSONB END) entry
  WHERE entry->>'submission_id'='baseline-2026-09-20';
  IF baseline_publication_count<>1 OR zero_baseline_publication_count<>1 THEN
   RAISE EXCEPTION 'published Sep20 rerun328 baseline is not exactly zero'
    USING ERRCODE='55000';
  END IF;
 END IF;

 before_row:=pg_catalog.to_jsonb(active);
 original_configuration_sha256:=pg_catalog.encode(extensions.digest(
  active.configuration_doc::TEXT,'sha256'),'hex');
 expected_hold:=pg_catalog.jsonb_build_object(
  'schema_version','leadpoet.sep20.rerun328.authority_hold.v1',
  'migration','331-arena-2026-09-20-promotion-reward-hold',
  'original_configuration_sha256',original_configuration_sha256,
  'original_configuration_rewards_enabled',TRUE,
  'original_row_rewards_enabled',TRUE,
  'original_promotion_required',TRUE,
  'source_commit','5d492f1715e27c8d9b919bcc3ad69d1be5160524',
  'scorer_image_digest',
   'sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f');

 ALTER TABLE public.lab_arena_rounds
  DISABLE TRIGGER lab_arena_rounds_write_once;
 UPDATE public.lab_arena_rounds SET
  configuration_doc=configuration_doc||pg_catalog.jsonb_build_object(
   'rewards_enabled',FALSE,
   'sep20_rerun328_promotion_reward_hold',expected_hold),
  rewards_enabled=FALSE,promotion_required=FALSE,
  updated_at=pg_catalog.clock_timestamp()
 WHERE round_id='arena-2026-09-20';
 GET DIAGNOSTICS changed=ROW_COUNT;
 ALTER TABLE public.lab_arena_rounds
  ENABLE TRIGGER lab_arena_rounds_write_once;
 IF changed<>1 THEN
  RAISE EXCEPTION 'Sep20 rerun328 authority hold update missing'
   USING ERRCODE='55000';
 END IF;

 SELECT pg_catalog.to_jsonb(r) INTO after_row
 FROM public.lab_arena_rounds r WHERE round_id='arena-2026-09-20';
 IF (after_row-'configuration_doc'-'rewards_enabled'-'promotion_required'-'updated_at')
      IS DISTINCT FROM
    (before_row-'configuration_doc'-'rewards_enabled'-'promotion_required'-'updated_at')
  OR ((after_row->'configuration_doc')-'rewards_enabled'-
       'sep20_rerun328_promotion_reward_hold') IS DISTINCT FROM
     ((before_row->'configuration_doc')-'rewards_enabled')
  OR after_row->'rewards_enabled' IS DISTINCT FROM 'false'::JSONB
  OR after_row->'promotion_required' IS DISTINCT FROM 'false'::JSONB
  OR after_row->'configuration_doc'->'rewards_enabled'
     IS DISTINCT FROM 'false'::JSONB
  OR after_row->'configuration_doc'->
      'sep20_rerun328_promotion_reward_hold' IS DISTINCT FROM expected_hold THEN
  RAISE EXCEPTION 'Sep20 rerun328 authority hold preservation differs'
   USING ERRCODE='55000';
 END IF;
END $sep20_rerun328_authority_hold$;
COMMIT;
