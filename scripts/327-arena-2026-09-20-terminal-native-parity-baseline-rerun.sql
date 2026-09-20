-- Render only after the Sep20 round is terminal and the protected database
-- matches every terminal seal below. The .template suffix prevents accidental
-- application before those immutable values and the new schedule are known.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='120s';

-- The archived score rows keep their immutable global run IDs. Add a fresh
-- namespace only to the existing integrity-scoring initializer for this
-- recovery, so rerun scoring cannot collide with the archived primary keys.
DO $patch_sep20_rerun326_scoring$
DECLARE
 definition TEXT;
 definition_hash TEXT;
 anchor TEXT:=$anchor$    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;$anchor$;
 replacement TEXT:=$replacement$    IF p_round_id='arena-2026-09-20'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-20-r326archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun326';
    END IF;
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;$replacement$;
BEGIN
 definition:=pg_catalog.pg_get_functiondef(
  'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure);
 definition_hash:=pg_catalog.encode(extensions.digest(definition,'sha256'),'hex');
 IF definition_hash='5ba722ef4495108e9d40124d9727d55a67542342010e9a8e6dc430034e6b6fae' THEN
  IF (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(
      definition,anchor,'')))<>pg_catalog.length(anchor) THEN
   RAISE EXCEPTION 'Sep20 rerun326 scoring seam differs';
  END IF;
  EXECUTE pg_catalog.replace(definition,anchor,replacement);
 ELSIF definition_hash<>'30c941d0b704577219842ebe76433093c7052a21fb182ea5352422ece45274a6' THEN
  RAISE EXCEPTION 'Sep20 rerun326 scoring definition differs';
 END IF;
 IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
   'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure),
   'sha256'),'hex')<>'30c941d0b704577219842ebe76433093c7052a21fb182ea5352422ece45274a6' THEN
  RAISE EXCEPTION 'Sep20 rerun326 scoring patch differs';
 END IF;
END $patch_sep20_rerun326_scoring$;

DO $sep20_rerun326$
DECLARE
 active public.lab_arena_rounds;
 baseline public.lab_arena_submissions;
 archive_config JSONB;
 archive_participants JSONB;
 new_participants JSONB;
 expected_submission JSONB;
 miner_runs_before JSONB;
 miner_runs_after JSONB;
 miner_ledger_before JSONB;
 miner_ledger_after JSONB;
 archived_execution_judgments JSONB;
 active_count BIGINT;
 unsettled_count BIGINT;
 moved_baseline_runs BIGINT;
 moved_baseline_ledger BIGINT;
 moved_score_runs BIGINT;
 moved_score_ledger BIGINT;
 inserted_runs BIGINT:=0;
 position INTEGER;
 stage SMALLINT;
 assignment TEXT;
BEGIN
 PERFORM pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('arena-2026-09-20-rerun326',0));
 LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;

 SELECT * INTO STRICT active FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-20' FOR UPDATE;
 SELECT * INTO STRICT baseline FROM public.lab_arena_submissions
  WHERE round_id='arena-2026-09-20'
   AND submission_id='baseline-2026-09-20' FOR UPDATE;

 -- A repeated migration is a no-op only when both sides still match the
 -- prepared recovery. Later execution and scoring progress remain valid.
 IF EXISTS(SELECT 1 FROM public.lab_arena_rounds
   WHERE round_id='arena-2026-09-20-r326archive') THEN
  IF NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds archived
      WHERE archived.round_id='arena-2026-09-20-r326archive'
       AND archived.status='cancelled' AND archived.rewards_enabled IS FALSE
       AND archived.cancel_reason=
           'authorized_sep20_native_parity_baseline_archive'
       AND archived.configuration_doc->>'archived_terminal_round_sha256'=
           '7427a5dc8524a8c443d1f8b224f375f8c2d23016a28966a3f0ce88208a0af998'
       AND archived.configuration_doc->>'archived_terminal_baseline_sha256'=
           '11030f38bd24f2cd5b3f198e92a60316ecf98b84954c8314690745d77282b503'
       AND archived.configuration_doc->>'archived_terminal_runs_sha256'=
           'c14e4379ec7d29764448263aedf0804f81a9fcf5d28adc726bb1e09cc947d204'
       AND archived.configuration_doc->>'archived_terminal_ledger_sha256'=
           '3c5ea91d6efa7485d2d50bf8c3b766b0edb47aeed3d03673fcd82ecd46d81c5b'
       AND archived.configuration_doc->>'archived_bank_object_sha256'=
           '7aaa7fbb4b4ce078dd62217c4d594dafdc82161d30c7718bd4500fa659fe5ad9'
       AND pg_catalog.jsonb_array_length(
           archived.configuration_doc->'archived_execution_judgments')=
           100)
    OR baseline.source_ref IS DISTINCT FROM
       'arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun326-c26122a7.tar.gz'
    OR baseline.source_size_bytes IS DISTINCT FROM 854161
    OR baseline.submission_doc->>'source_sha256' IS DISTINCT FROM
       '61e60812b4ca4506c7c1ea7ee7a7678750e50ad5adec17ca500f44708f71b8b4'
    OR baseline.submission_doc->>'source_commit' IS DISTINCT FROM
       'c26122a7287c2e9366c2b6a897b8d56a8fab41b1'
    OR active.configuration_doc->'call_quotas' IS DISTINCT FROM
       '{"deepline":200,"openrouter":500,"scrapingdog":200}'::JSONB
    OR active.configuration_doc->'schedule' IS DISTINCT FROM
       '{"benchmark_deadline":"2026-09-20T02:42:58Z","final_scoring_close":"2026-09-20T14:42:58Z","publication_deadline":"2026-09-20T14:42:59Z","stage_1_close":"2026-09-20T05:42:59Z","stage_1_scoring_close":"2026-09-20T08:42:58Z","stage_1_start":"2026-09-20T02:42:59Z","stage_2_close":"2026-09-20T11:42:59Z","stage_2_start":"2026-09-20T08:42:59Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB
    OR active.benchmark_ref IS DISTINCT FROM
       'arena/arena-2026-09-20/benchmark.json'
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
        WHERE round_id='arena-2026-09-20-r326archive')<>8
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20-r326archive' AND kind='execute'
         AND submission_id='baseline-2026-09-20:r326archive')<>
       20
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20-r326archive' AND kind='score')<>
       88
    OR (SELECT pg_catalog.count(DISTINCT assignment_id)
        FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20'
         AND kind='execute' AND submission_id='baseline-2026-09-20'
         AND assignment_id LIKE '%:rerun326')<>20
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20' AND kind='execute'
         AND submission_id='baseline-2026-09-20'
         AND assignment_id NOT LIKE '%:rerun326')
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20' AND kind='execute'
         AND submission_id<>'baseline-2026-09-20' AND status='accepted'
         AND terminal_cause='accepted' AND output_ref IS NOT NULL)<>80 THEN
   RAISE EXCEPTION 'existing Sep20 rerun326 differs' USING ERRCODE='55000';
  END IF;
  RETURN;
 END IF;

 IF ('{"benchmark_deadline":"2026-09-20T02:42:58Z","final_scoring_close":"2026-09-20T14:42:58Z","publication_deadline":"2026-09-20T14:42:59Z","stage_1_close":"2026-09-20T05:42:59Z","stage_1_scoring_close":"2026-09-20T08:42:58Z","stage_1_start":"2026-09-20T02:42:59Z","stage_2_close":"2026-09-20T11:42:59Z","stage_2_start":"2026-09-20T08:42:59Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ
      <=pg_catalog.clock_timestamp() THEN
  RAISE EXCEPTION 'Sep20 rerun326 admission window closed' USING ERRCODE='22023';
 END IF;
 SELECT pg_catalog.count(*) INTO active_count FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-20'
   AND status IN('pending','leased','submitted');
 SELECT pg_catalog.count(*) INTO unsettled_count
 FROM public.lab_arena_submissions s
 CROSS JOIN(VALUES('execute'::TEXT),('score'::TEXT)) k(kind)
 CROSS JOIN LATERAL(SELECT public.lab_arena__successful_call_cost_state(
  s.submission_id,k.kind,NULL) state)c
 WHERE s.round_id='arena-2026-09-20'
  AND (COALESCE((c.state->>'inflight_calls')::BIGINT,-1)<>0
   OR COALESCE((c.state->>'success_unresolved_calls')::BIGINT,-1)<>0);

 IF active.status IS DISTINCT FROM 'published'
  OR active.status NOT IN('published','cancelled')
  OR active.cancel_reason IS DISTINCT FROM NULL
  OR active_count<>0 OR unsettled_count<>0
  OR active.benchmark_ref IS DISTINCT FROM
     'arena/arena-2026-09-20/benchmark.json'
  OR active.evaluation_date IS DISTINCT FROM '2026-09-20'
  OR active.icp_set_date IS DISTINCT FROM DATE '2026-09-19'
  OR pg_catalog.jsonb_array_length(active.participants)<>5
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id='arena-2026-09-20')<>8
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id='arena-2026-09-20' AND status='frozen')<>5
  OR active.configuration_doc->'call_quotas' IS DISTINCT FROM
     '{"deepline":200,"openrouter":200,"scrapingdog":200}'::JSONB
  OR active.configuration_doc->>'sourcing_cost_eligibility_policy'
     IS DISTINCT FROM 'successful_calls_per_icp_v1'
  OR (active.configuration_doc->>'execution_icp_cap_microusd')::BIGINT
     IS DISTINCT FROM 4000000
  OR (active.configuration_doc->>'cost_per_company_microusd')::BIGINT
     IS DISTINCT FROM 800000
  OR (active.configuration_doc->>'icp_wall_clock_seconds')::INTEGER
     IS DISTINCT FROM 2700
  OR (active.configuration_doc->>'lease_ttl_seconds')::INTEGER
     IS DISTINCT FROM 3600
  OR active.configuration_doc->'scoring_call_quotas' IS DISTINCT FROM
     '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB
  OR pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(active)::TEXT,
       'sha256'),'hex')<>'7427a5dc8524a8c443d1f8b224f375f8c2d23016a28966a3f0ce88208a0af998'
  OR pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(baseline)::TEXT,
       'sha256'),'hex')<>'11030f38bd24f2cd5b3f198e92a60316ecf98b84954c8314690745d77282b503'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute')<>
     100
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='score')<>
     88
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id='baseline-2026-09-20')<>
     20
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id='baseline-2026-09-20' AND status='accepted')<>
     7
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id='baseline-2026-09-20' AND status='failed')<>
     13
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs br
      WHERE br.round_id='arena-2026-09-20' AND br.kind='execute'
       AND br.submission_id='baseline-2026-09-20'
       AND br.assignment_id IS DISTINCT FROM
        'arena-2026-09-20:baseline-2026-09-20:'||br.stage::TEXT||':'||
        br.icp_position::TEXT)
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id<>'baseline-2026-09-20' AND status='accepted'
       AND terminal_cause='accepted' AND output_ref IS NOT NULL)<>80
  OR (SELECT COALESCE(max(calls),0) FROM(
       SELECT count(*) FILTER(WHERE l.provider='openrouter') calls
       FROM public.lab_arena_runs r LEFT JOIN(
        SELECT DISTINCT ON(run_id,call_identity) run_id,call_identity,provider
        FROM public.lab_arena_ledger WHERE call_identity IS NOT NULL
        ORDER BY run_id,call_identity,entry_id DESC)l USING(run_id)
       WHERE r.round_id='arena-2026-09-20' AND r.kind='execute'
        AND r.submission_id<>'baseline-2026-09-20'
       GROUP BY r.run_id) per_run)<>14 THEN
  RAISE EXCEPTION 'Sep20 rerun326 terminal preimage differs' USING ERRCODE='55000';
 END IF;

 IF pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(s)::TEXT,'sha256'),'hex'),''
    ORDER BY submission_id) FROM public.lab_arena_submissions s
    WHERE round_id='arena-2026-09-20'
     AND submission_id<>'baseline-2026-09-20'),''),'sha256'),'hex')<>
     '942d0da45f73ec33809df1a0ae935cee44073eb7a17b6d261e7e35ca77826982'
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),''
    ORDER BY run_id) FROM public.lab_arena_runs r
    WHERE round_id='arena-2026-09-20'),''),'sha256'),'hex')<>
     'c14e4379ec7d29764448263aedf0804f81a9fcf5d28adc726bb1e09cc947d204'
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(l)::TEXT,'sha256'),'hex'),''
    ORDER BY entry_id) FROM public.lab_arena_ledger l
    WHERE round_id='arena-2026-09-20'),''),'sha256'),'hex')<>
     '3c5ea91d6efa7485d2d50bf8c3b766b0edb47aeed3d03673fcd82ecd46d81c5b' THEN
  RAISE EXCEPTION 'Sep20 rerun326 terminal history differs' USING ERRCODE='55000';
 END IF;

 expected_submission:=baseline.submission_doc||pg_catalog.jsonb_build_object(
  'source_ref','arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun326-c26122a7.tar.gz',
  'source_size_bytes',854161,
  'source_sha256','61e60812b4ca4506c7c1ea7ee7a7678750e50ad5adec17ca500f44708f71b8b4',
  'source_commit','c26122a7287c2e9366c2b6a897b8d56a8fab41b1');
 SELECT pg_catalog.jsonb_agg(CASE
   WHEN item->>'submission_id'='baseline-2026-09-20' THEN
    item||pg_catalog.jsonb_build_object(
     'source_ref','arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun326-c26122a7.tar.gz',
     'source_size_bytes',854161)
   ELSE item END ORDER BY ordinal)
  INTO new_participants FROM pg_catalog.jsonb_array_elements(active.participants)
   WITH ORDINALITY entries(item,ordinal);
 SELECT pg_catalog.jsonb_agg(item||pg_catalog.jsonb_build_object(
   'submission_id',(item->>'submission_id')||':r326archive') ORDER BY ordinal)
  INTO archive_participants FROM pg_catalog.jsonb_array_elements(active.participants)
   WITH ORDINALITY entries(item,ordinal);
 SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
   'run_id',run_id,'per_icp_score',per_icp_score,
   'qualification_doc',qualification_doc,'updated_at',updated_at) ORDER BY run_id)
  INTO archived_execution_judgments FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-20' AND kind='execute';
 archive_config:=active.configuration_doc||pg_catalog.jsonb_build_object(
  'round_id','arena-2026-09-20-r326archive','mode','shadow','rewards_enabled',FALSE,
  'archived_terminal_status',active.status,
  'archived_terminal_cancel_reason',active.cancel_reason,
  'archived_terminal_round_sha256','7427a5dc8524a8c443d1f8b224f375f8c2d23016a28966a3f0ce88208a0af998',
  'archived_terminal_baseline_sha256','11030f38bd24f2cd5b3f198e92a60316ecf98b84954c8314690745d77282b503',
  'archived_terminal_miner_submissions_sha256','942d0da45f73ec33809df1a0ae935cee44073eb7a17b6d261e7e35ca77826982',
  'archived_terminal_runs_sha256','c14e4379ec7d29764448263aedf0804f81a9fcf5d28adc726bb1e09cc947d204',
  'archived_terminal_ledger_sha256','3c5ea91d6efa7485d2d50bf8c3b766b0edb47aeed3d03673fcd82ecd46d81c5b',
  'archived_rewards_enabled',active.rewards_enabled,
  'archived_effective_reward_epoch',active.effective_reward_epoch,
  'archived_reward_basis_hash',active.reward_basis_hash,
  'archived_reward_basis_doc',active.reward_basis_doc,
  'archived_signing_key_doc',active.signing_key_doc,
  'archived_reward_activated_at',active.reward_activated_at,
  'archived_king_outcome',active.king_outcome,
  'archived_king_hotkey',active.king_hotkey,
  'archived_king_start_epoch',active.king_start_epoch,
  'archived_promotion_required',active.promotion_required,
  'archived_promotion_doc',active.promotion_doc,
  'archived_baseline_promoted_at',active.baseline_promoted_at,
  'archived_execution_judgments',archived_execution_judgments,
  'archived_bank_object_sha256',
   '7aaa7fbb4b4ce078dd62217c4d594dafdc82161d30c7718bd4500fa659fe5ad9');
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)-'per_icp_score'-
   'qualification_doc'-'updated_at' ORDER BY run_id) INTO miner_runs_before
  FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-20'
   AND kind='execute' AND submission_id<>'baseline-2026-09-20';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
  INTO miner_ledger_before FROM public.lab_arena_ledger l
  WHERE round_id='arena-2026-09-20'
   AND submission_id<>'baseline-2026-09-20'
   AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs s
    WHERE s.round_id='arena-2026-09-20' AND s.kind='score' AND s.run_id=l.run_id);

 ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
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
  champion_fallback_providers)
 SELECT round_id,status,status_generation,stage_generation,configuration_doc,
  rewards_enabled,participants,benchmark_ref,evaluation_date,
  stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
  king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
  reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
  cancel_reason,published_at,created_at,updated_at,promotion_required,
  promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
  confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
  champion_funding_frozen,champion_submission_id,champion_hotkey,
  champion_fallback_providers FROM pg_catalog.jsonb_populate_record(
   NULL::public.lab_arena_rounds,pg_catalog.to_jsonb(active)||
   pg_catalog.jsonb_build_object(
    'round_id','arena-2026-09-20-r326archive','status','cancelled',
    'configuration_doc',archive_config,'participants',archive_participants,
    'rewards_enabled',FALSE,'reward_basis_hash',NULL,'reward_basis_doc',NULL,
    'signing_key_doc',NULL,'effective_reward_epoch',NULL,'reward_activated_at',NULL,
    'cancel_reason','authorized_sep20_native_parity_baseline_archive'));
 INSERT INTO public.lab_arena_submissions SELECT (pg_catalog.jsonb_populate_record(
  NULL::public.lab_arena_submissions,pg_catalog.to_jsonb(s)||
  pg_catalog.jsonb_build_object('round_id','arena-2026-09-20-r326archive',
   'submission_id',s.submission_id||':r326archive'))).*
 FROM public.lab_arena_submissions s WHERE s.round_id='arena-2026-09-20';

 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-20-r326archive',
  submission_id=submission_id||':r326archive'
 WHERE round_id='arena-2026-09-20' AND kind='score';
 GET DIAGNOSTICS moved_score_runs=ROW_COUNT;
 UPDATE public.lab_arena_ledger l SET round_id='arena-2026-09-20-r326archive',
  submission_id=submission_id||':r326archive'
 WHERE round_id='arena-2026-09-20' AND EXISTS(SELECT 1
  FROM public.lab_arena_runs s WHERE s.round_id='arena-2026-09-20-r326archive'
   AND s.kind='score' AND s.run_id=l.run_id);
 GET DIAGNOSTICS moved_score_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-20-r326archive',
  submission_id='baseline-2026-09-20:r326archive'
 WHERE round_id='arena-2026-09-20' AND kind='execute'
  AND submission_id='baseline-2026-09-20';
 GET DIAGNOSTICS moved_baseline_runs=ROW_COUNT;
 UPDATE public.lab_arena_ledger SET round_id='arena-2026-09-20-r326archive',
  submission_id='baseline-2026-09-20:r326archive'
 WHERE round_id='arena-2026-09-20'
  AND submission_id='baseline-2026-09-20';
 GET DIAGNOSTICS moved_baseline_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL
 WHERE round_id='arena-2026-09-20' AND kind='execute'
  AND submission_id<>'baseline-2026-09-20';
 UPDATE public.lab_arena_submissions SET
  source_ref='arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun326-c26122a7.tar.gz',
  source_size_bytes=854161,submission_doc=expected_submission,
  updated_at=pg_catalog.clock_timestamp()
 WHERE round_id='arena-2026-09-20' AND submission_id='baseline-2026-09-20';
 UPDATE public.lab_arena_rounds SET status='stage1',
  status_generation=status_generation+1,stage_generation=stage_generation+1,
  configuration_doc=pg_catalog.jsonb_set(pg_catalog.jsonb_set(
   configuration_doc,'{schedule}','{"benchmark_deadline":"2026-09-20T02:42:58Z","final_scoring_close":"2026-09-20T14:42:58Z","publication_deadline":"2026-09-20T14:42:59Z","stage_1_close":"2026-09-20T05:42:59Z","stage_1_scoring_close":"2026-09-20T08:42:58Z","stage_1_start":"2026-09-20T02:42:59Z","stage_2_close":"2026-09-20T11:42:59Z","stage_2_start":"2026-09-20T08:42:59Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB,TRUE),
   '{call_quotas}','{"deepline":200,"openrouter":500,"scrapingdog":200}'::JSONB,TRUE),
  participants=new_participants,stage1_scoring_plan_doc=NULL,
  stage2_scoring_plan_doc=NULL,stage3_scoring_plan_doc=NULL,finalists=NULL,
  publication_doc=NULL,published_at=NULL,cancel_reason=NULL,
  king_outcome=NULL,king_hotkey=NULL,king_start_epoch=NULL,
  effective_reward_epoch=NULL,reward_basis_hash=NULL,reward_basis_doc=NULL,
  signing_key_doc=NULL,reward_activated_at=NULL,
  promotion_required=TRUE,promotion_doc=NULL,baseline_promoted_at=NULL,
  updated_at=pg_catalog.clock_timestamp()
 WHERE round_id='arena-2026-09-20';
 FOR position IN 0..19 LOOP
  stage:=CASE WHEN position<10 THEN 1 ELSE 2 END;
  assignment:='arena-2026-09-20:baseline-2026-09-20:'||stage::TEXT||':'||
   position::TEXT||':rerun326';
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,
   miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation)
  VALUES(assignment||':1',assignment,'arena-2026-09-20','baseline-2026-09-20',
   baseline.miner_hotkey,stage,position,1,'execute','pending',
   active.stage_generation+1);
  inserted_runs:=inserted_runs+1;
 END LOOP;
 ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)-'per_icp_score'-
   'qualification_doc'-'updated_at' ORDER BY run_id) INTO miner_runs_after
  FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-20'
   AND kind='execute' AND submission_id<>'baseline-2026-09-20';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
  INTO miner_ledger_after FROM public.lab_arena_ledger l
  WHERE round_id='arena-2026-09-20'
   AND submission_id<>'baseline-2026-09-20';
 IF moved_score_runs<>88
  OR moved_score_ledger<>1320
  OR moved_baseline_runs<>20
  OR moved_baseline_ledger<>12966
  OR inserted_runs<>20 OR miner_runs_after IS DISTINCT FROM miner_runs_before
  OR miner_ledger_after IS DISTINCT FROM miner_ledger_before
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='score') THEN
  RAISE EXCEPTION 'Sep20 rerun326 preservation differs' USING ERRCODE='55000';
 END IF;
END $sep20_rerun326$;
COMMIT;
