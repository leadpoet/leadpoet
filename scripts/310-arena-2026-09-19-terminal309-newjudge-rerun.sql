-- Render only after the Sep19 recovery309 round is terminal and every seal below
-- has been captured from the protected database. This file is not an active
-- migration while it retains the .template suffix.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='120s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_sep19_rerun310_archive_valid_v1()
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public,extensions AS $archive310$
DECLARE archived public.lab_arena_rounds; derived JSONB;
BEGIN
 SELECT * INTO archived FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-19-r310archive';
 IF NOT FOUND THEN RETURN FALSE; END IF;
 derived:=archived.configuration_doc->'archived_execution_judgments';
 RETURN archived.status='cancelled'
  AND archived.rewards_enabled IS FALSE
  AND archived.cancel_reason='authorized_sep19_terminal309_newjudge_archive'
  AND archived.configuration_doc->>'archived_terminal_round_sha256'='645bd47cfcc1f775ff6d4a2362b7969f1fc7b407dcbb44e601dd7c8e8a84f794'
  AND archived.configuration_doc->>'archived_terminal_baseline_sha256'='dd1960c47667840293eb6fbca26e068cfa93836878e2acfb6f4e6d7fc10e8deb'
  AND archived.configuration_doc->>'archived_terminal_miner_submissions_sha256'='781027021d3cadc5f2c0540d9f24a67f65cb25126912ef3544867d5a70153297'
  AND archived.configuration_doc->>'archived_terminal_runs_sha256'='21ffd85b7a2557fb62ae35eee00291f52394e1f726f7cab4de64220b392afcdb'
  AND archived.configuration_doc->>'archived_terminal_ledger_sha256'='1811f08fca9b197decd2a96076257a1964c05e33c2e1b8a8f73444ffcf45f5e1'
  AND pg_catalog.encode(extensions.digest(pg_catalog.jsonb_build_object(
    'confirmation_bank_ref',archived.confirmation_bank_ref,
    'confirmation_bank_hash',archived.confirmation_bank_hash,
    'confirmation_cohort',archived.confirmation_cohort)::TEXT,'sha256'),'hex')
      ='1af19ff80c88e0fa121d57fa11ea6af14b35220017b0d11a099c70f6b60b314f'
  AND pg_catalog.encode(extensions.digest(pg_catalog.jsonb_build_object(
    'rewards_enabled',archived.configuration_doc->'archived_rewards_enabled',
    'effective_reward_epoch',archived.configuration_doc->'archived_effective_reward_epoch',
    'reward_basis_hash',archived.configuration_doc->'archived_reward_basis_hash',
    'reward_basis_doc',archived.configuration_doc->'archived_reward_basis_doc',
    'signing_key_doc',archived.configuration_doc->'archived_signing_key_doc',
    'reward_activated_at',archived.configuration_doc->'archived_reward_activated_at',
    'king_outcome',archived.configuration_doc->'archived_king_outcome',
    'king_hotkey',archived.configuration_doc->'archived_king_hotkey',
    'king_start_epoch',archived.configuration_doc->'archived_king_start_epoch',
    'promotion_required',archived.configuration_doc->'archived_promotion_required',
    'promotion_doc',archived.configuration_doc->'archived_promotion_doc',
    'baseline_promoted_at',archived.configuration_doc->'archived_baseline_promoted_at')::TEXT,
    'sha256'),'hex')='22a411e2edcd506ab991f1e65db14f92dd83bfbfeedf7c95b4d478b2d880de2b'
  AND pg_catalog.jsonb_typeof(derived)='array'
  AND pg_catalog.jsonb_array_length(derived)=102
  AND (SELECT pg_catalog.count(DISTINCT value->>'run_id')
       FROM pg_catalog.jsonb_array_elements(derived))=102
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
       WHERE round_id='arena-2026-09-19-r310archive')=5
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r310archive' AND kind='execute'
        AND submission_id='baseline-2026-09-19:r310archive')=22
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r310archive' AND kind='score')=101
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r310archive' AND kind='score'
        AND submission_id NOT LIKE '%:r310archive')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger ledger
       WHERE ledger.round_id='arena-2026-09-19-r310archive'
        AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs run
          WHERE run.round_id='arena-2026-09-19-r310archive'
           AND run.run_id=ledger.run_id));
END $archive310$;
ALTER FUNCTION public.lab_arena_sep19_rerun310_archive_valid_v1()
 OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep19_rerun310_archive_valid_v1()
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep19_rerun310_active_valid_v1()
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $active310$
DECLARE active public.lab_arena_rounds; archived public.lab_arena_rounds;
BEGIN
 SELECT * INTO active FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-19';
 SELECT * INTO archived FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-19-r310archive';
 RETURN active.round_id IS NOT NULL AND archived.round_id IS NOT NULL
  AND public.lab_arena_sep19_rerun310_archive_valid_v1() IS TRUE
  AND pg_catalog.encode(extensions.digest(pg_catalog.jsonb_build_object(
    'confirmation_bank_ref',active.confirmation_bank_ref,
    'confirmation_bank_hash',active.confirmation_bank_hash,
    'confirmation_cohort',active.confirmation_cohort)::TEXT,'sha256'),'hex')
      ='1af19ff80c88e0fa121d57fa11ea6af14b35220017b0d11a099c70f6b60b314f'
  AND pg_catalog.encode(extensions.digest(pg_catalog.jsonb_build_object(
    'rewards_enabled',active.rewards_enabled,
    'effective_reward_epoch',active.effective_reward_epoch,
    'reward_basis_hash',active.reward_basis_hash,
    'reward_basis_doc',active.reward_basis_doc,
    'signing_key_doc',active.signing_key_doc,
    'reward_activated_at',active.reward_activated_at,
    'king_outcome',active.king_outcome,
    'king_hotkey',active.king_hotkey,
    'king_start_epoch',active.king_start_epoch,
    'promotion_required',active.promotion_required,
    'promotion_doc',active.promotion_doc,
    'baseline_promoted_at',active.baseline_promoted_at)::TEXT,'sha256'),'hex')
      ='22a411e2edcd506ab991f1e65db14f92dd83bfbfeedf7c95b4d478b2d880de2b'
  AND EXISTS(SELECT 1 FROM public.lab_arena_rounds round
    WHERE round.round_id='arena-2026-09-19'
     AND round.configuration_doc->>'scorer_image_digest'='sha256:19eae396b4168866a573122fce175316596e394049b51b544f5ee4262eaf5ec8'
     AND round.configuration_doc->>'scorer_image_reference'='493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:19eae396b4168866a573122fce175316596e394049b51b544f5ee4262eaf5ec8')
  AND EXISTS(SELECT 1 FROM public.lab_arena_submissions baseline
    WHERE baseline.round_id='arena-2026-09-19'
     AND baseline.submission_id='baseline-2026-09-19'
     AND baseline.source_ref='arena/arena-2026-09-19/sources/baseline-2026-09-19-rerun310.tar.gz'
     AND baseline.source_size_bytes=688770
     AND baseline.submission_doc->>'source_sha256'='a67901a72a90965dd002a442236f44d113f81eda93f3778525c69a7dedf2cead'
     AND baseline.submission_doc->>'source_commit'='5e6d881882a2dae2be3ca783060f33e0380dd0de')
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
    WHERE round_id='arena-2026-09-19' AND kind='execute'
     AND submission_id<>'baseline-2026-09-19' AND status='accepted'
     AND terminal_cause='accepted' AND output_ref IS NOT NULL)=80
  AND (SELECT pg_catalog.count(DISTINCT assignment_id) FROM public.lab_arena_runs
    WHERE round_id='arena-2026-09-19' AND kind='execute'
     AND submission_id='baseline-2026-09-19'
     AND assignment_id LIKE '%:rerun310')=20
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs
    WHERE round_id='arena-2026-09-19' AND kind='execute'
     AND submission_id='baseline-2026-09-19'
     AND assignment_id NOT LIKE '%:rerun310')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs
    WHERE round_id='arena-2026-09-19' AND kind='score'
     AND assignment_id NOT LIKE '%:score:rerun310');
END $active310$;
ALTER FUNCTION public.lab_arena_sep19_rerun310_active_valid_v1()
 OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep19_rerun310_active_valid_v1()
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Add one exact Sep19 namespace branch to the current scorer. Do not copy a
-- historical scorer definition into this migration.
DO $patch_sep19_rerun310_scorer$
DECLARE definition TEXT; current_hash TEXT;
 assignment_seam TEXT:=$assignment$    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;$assignment$;
 guard_seam TEXT:=$guard$  FOR v_item IN$guard$;
BEGIN
 definition:=pg_catalog.pg_get_functiondef(
  'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure);
 current_hash:=pg_catalog.encode(extensions.digest(definition,'sha256'),'hex');
 IF current_hash='b81eda5ae6bb29673db0a0136d0dd4bcc26c644357ec3a895f7b918a82e3c5ca' THEN
  IF (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(
      definition,assignment_seam,'')))<>pg_catalog.length(assignment_seam)
    OR (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(
      definition,guard_seam,'')))<>pg_catalog.length(guard_seam) THEN
   RAISE EXCEPTION 'Sep19 rerun310 scorer seam differs';
  END IF;
  definition:=pg_catalog.replace(definition,guard_seam,$new_guard$  IF p_round_id='arena-2026-09-19'
     AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id='arena-2026-09-19-r310archive')
     AND public.lab_arena_sep19_rerun310_active_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_sep19_rerun310_frozen_state_invalid'
      USING ERRCODE='22023';
  END IF;
  FOR v_item IN$new_guard$);
 definition:=pg_catalog.replace(definition,assignment_seam,$new_assignment$    IF p_round_id='arena-2026-09-19'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-19-r310archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun310';
    END IF;
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;$new_assignment$);
  EXECUTE definition;
 ELSIF current_hash<>'bdc0225eff9be6e5386bf5a58780924523d01a7e416e559d9f5417821e2d6c3d' THEN
  RAISE EXCEPTION 'Sep19 rerun310 scorer definition differs';
 END IF;
 IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
   'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure),
   'sha256'),'hex')<>'bdc0225eff9be6e5386bf5a58780924523d01a7e416e559d9f5417821e2d6c3d' THEN
  RAISE EXCEPTION 'Sep19 rerun310 scorer patch differs';
 END IF;
END $patch_sep19_rerun310_scorer$;

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep19_rerun310_v1(
 p_source_size_bytes BIGINT,p_source_sha256 TEXT,p_source_commit TEXT,
 p_schedule JSONB,p_scorer_digest TEXT,p_scorer_reference TEXT)
RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,public,extensions AS $prepare310$
DECLARE
 active_round public.lab_arena_rounds; active_baseline public.lab_arena_submissions;
 expected_submission JSONB; archive_config JSONB; archive_participants JSONB;
 new_participants JSONB; derived JSONB; other_before JSONB; other_after JSONB;
 miner_runs_before JSONB; miner_runs_after JSONB;
 miner_ledger_before JSONB; miner_ledger_after JSONB;
 active_count BIGINT; unsettled BIGINT; moved_scores BIGINT;
 moved_score_ledger BIGINT; moved_baseline_runs BIGINT;
 moved_baseline_ledger BIGINT; inserted_runs BIGINT; position INTEGER;
 stage SMALLINT; assignment TEXT;
BEGIN
 PERFORM pg_catalog.set_config('lock_timeout','5s',TRUE);
 PERFORM pg_catalog.set_config('statement_timeout','120s',TRUE);
 IF p_source_size_bytes<>688770
  OR p_source_sha256<>'a67901a72a90965dd002a442236f44d113f81eda93f3778525c69a7dedf2cead'
  OR p_source_commit<>'5e6d881882a2dae2be3ca783060f33e0380dd0de'
  OR p_schedule IS DISTINCT FROM '{"benchmark_deadline":"2026-09-19T07:45:00Z","final_scoring_close":"2026-09-19T19:45:00Z","publication_deadline":"2026-09-19T19:45:01Z","stage_1_close":"2026-09-19T10:45:01Z","stage_1_scoring_close":"2026-09-19T13:45:00Z","stage_1_start":"2026-09-19T07:45:01Z","stage_2_close":"2026-09-19T16:45:01Z","stage_2_start":"2026-09-19T13:45:01Z","submission_cutoff":"2026-09-19T00:00:00Z","submission_open":"2026-09-18T00:00:00Z"}'::JSONB
  OR p_scorer_digest<>'sha256:19eae396b4168866a573122fce175316596e394049b51b544f5ee4262eaf5ec8'
  OR p_scorer_reference<>'493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:19eae396b4168866a573122fce175316596e394049b51b544f5ee4262eaf5ec8' THEN
  RAISE EXCEPTION 'Sep19 rerun310 source, judge, bank, or schedule differs'
   USING ERRCODE='22023';
 END IF;
 PERFORM pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('arena-2026-09-19-rerun310',0));
 LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
 SELECT * INTO STRICT active_round FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-19' FOR UPDATE;
 SELECT * INTO STRICT active_baseline FROM public.lab_arena_submissions
  WHERE round_id='arena-2026-09-19'
   AND submission_id='baseline-2026-09-19' FOR UPDATE;
 IF EXISTS(SELECT 1 FROM public.lab_arena_rounds
   WHERE round_id='arena-2026-09-19-r310archive') THEN
  IF public.lab_arena_sep19_rerun310_archive_valid_v1() IS NOT TRUE
   OR public.lab_arena_sep19_rerun310_active_valid_v1() IS NOT TRUE THEN
   RAISE EXCEPTION 'existing Sep19 rerun310 differs' USING ERRCODE='55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object('status','existing',
   'round_id','arena-2026-09-19','fresh_baseline_execute_count',20,
   'retained_miner_execute_count',80,'score_namespace','score:rerun310');
 END IF;
 IF (p_schedule->>'benchmark_deadline')::TIMESTAMPTZ<=pg_catalog.clock_timestamp() THEN
  RAISE EXCEPTION 'Sep19 rerun310 admission window closed' USING ERRCODE='22023';
 END IF;
 SELECT pg_catalog.count(*) INTO active_count FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-19' AND status IN('pending','leased','submitted');
 SELECT pg_catalog.count(*) INTO unsettled FROM public.lab_arena_submissions s
  CROSS JOIN(VALUES('execute'::TEXT),('score'::TEXT)) k(kind)
  CROSS JOIN LATERAL(SELECT public.lab_arena__successful_call_cost_state(
   s.submission_id,k.kind,NULL) state)c
  WHERE s.round_id='arena-2026-09-19'
   AND (COALESCE((c.state->>'inflight_calls')::BIGINT,-1)<>0
    OR COALESCE((c.state->>'success_unresolved_calls')::BIGINT,-1)<>0);
 IF active_round.status NOT IN('published','cancelled') OR active_count<>0 OR unsettled<>0
  OR active_round.configuration_doc ? 'company_quality_policy'
  OR pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(active_round)::TEXT,
       'sha256'),'hex')<>'645bd47cfcc1f775ff6d4a2362b7969f1fc7b407dcbb44e601dd7c8e8a84f794'
  OR pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(active_baseline)::TEXT,
       'sha256'),'hex')<>'dd1960c47667840293eb6fbca26e068cfa93836878e2acfb6f4e6d7fc10e8deb'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND status='accepted' AND terminal_cause='accepted' AND output_ref IS NOT NULL)<>100
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute')<>102
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='score')<>101
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='score'
       AND status NOT IN('accepted','failed'))
  OR (SELECT pg_catalog.count(DISTINCT assignment_id) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND submission_id='baseline-2026-09-19'
       AND assignment_id LIKE '%:recovery309')<>20
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND submission_id='baseline-2026-09-19')<>22
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND submission_id<>'baseline-2026-09-19')<>80 THEN
  RAISE EXCEPTION 'Sep19 rerun310 terminal preimage differs' USING ERRCODE='55000';
 END IF;
 IF pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
      pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(s)::TEXT,'sha256'),'hex'),''
      ORDER BY submission_id) FROM public.lab_arena_submissions s
      WHERE round_id='arena-2026-09-19' AND submission_id<>'baseline-2026-09-19'),''),
      'sha256'),'hex')<>'781027021d3cadc5f2c0540d9f24a67f65cb25126912ef3544867d5a70153297'
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
      pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),''
      ORDER BY run_id) FROM public.lab_arena_runs r
      WHERE round_id='arena-2026-09-19'),''),'sha256'),'hex')<>'21ffd85b7a2557fb62ae35eee00291f52394e1f726f7cab4de64220b392afcdb'
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
      pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(l)::TEXT,'sha256'),'hex'),''
      ORDER BY entry_id) FROM public.lab_arena_ledger l
      WHERE round_id='arena-2026-09-19'),''),'sha256'),'hex')<>'1811f08fca9b197decd2a96076257a1964c05e33c2e1b8a8f73444ffcf45f5e1' THEN
  RAISE EXCEPTION 'Sep19 rerun310 terminal history differs' USING ERRCODE='55000';
 END IF;
 SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
   'run_id',run_id,'per_icp_score',per_icp_score,
   'qualification_doc',qualification_doc,'updated_at',updated_at) ORDER BY run_id)
  INTO derived FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-19' AND kind='execute';
 IF pg_catalog.jsonb_array_length(derived)<>102 THEN
  RAISE EXCEPTION 'Sep19 rerun310 derived judgment count differs'; END IF;
 expected_submission:=active_baseline.submission_doc||pg_catalog.jsonb_build_object(
  'source_ref','arena/arena-2026-09-19/sources/baseline-2026-09-19-rerun310.tar.gz','source_size_bytes',688770,
  'source_sha256','a67901a72a90965dd002a442236f44d113f81eda93f3778525c69a7dedf2cead','source_commit','5e6d881882a2dae2be3ca783060f33e0380dd0de');
 SELECT pg_catalog.jsonb_agg(CASE WHEN item->>'submission_id'='baseline-2026-09-19'
   THEN item||pg_catalog.jsonb_build_object('source_ref','arena/arena-2026-09-19/sources/baseline-2026-09-19-rerun310.tar.gz',
    'source_size_bytes',688770) ELSE item END ORDER BY ordinal)
  INTO new_participants FROM pg_catalog.jsonb_array_elements(active_round.participants)
   WITH ORDINALITY entries(item,ordinal);
 archive_participants:=(SELECT pg_catalog.jsonb_agg(
   item||pg_catalog.jsonb_build_object('submission_id',(item->>'submission_id')||':r310archive')
   ORDER BY ordinal) FROM pg_catalog.jsonb_array_elements(active_round.participants)
   WITH ORDINALITY entries(item,ordinal));
 archive_config:=active_round.configuration_doc||pg_catalog.jsonb_build_object(
  'round_id','arena-2026-09-19-r310archive','mode','shadow','rewards_enabled',FALSE,
  'archived_terminal_status',active_round.status,
  'archived_terminal_cancel_reason',active_round.cancel_reason,
  'archived_rewards_enabled',active_round.rewards_enabled,
  'archived_reward_basis_hash',active_round.reward_basis_hash,
  'archived_reward_basis_doc',active_round.reward_basis_doc,
  'archived_signing_key_doc',active_round.signing_key_doc,
  'archived_effective_reward_epoch',active_round.effective_reward_epoch,
  'archived_reward_activated_at',active_round.reward_activated_at,
  'archived_king_outcome',active_round.king_outcome,
  'archived_king_hotkey',active_round.king_hotkey,
  'archived_king_start_epoch',active_round.king_start_epoch,
  'archived_promotion_required',active_round.promotion_required,
  'archived_promotion_doc',active_round.promotion_doc,
  'archived_baseline_promoted_at',active_round.baseline_promoted_at,
  'archived_terminal_round_sha256','645bd47cfcc1f775ff6d4a2362b7969f1fc7b407dcbb44e601dd7c8e8a84f794',
  'archived_terminal_baseline_sha256','dd1960c47667840293eb6fbca26e068cfa93836878e2acfb6f4e6d7fc10e8deb',
  'archived_terminal_miner_submissions_sha256','781027021d3cadc5f2c0540d9f24a67f65cb25126912ef3544867d5a70153297',
  'archived_terminal_runs_sha256','21ffd85b7a2557fb62ae35eee00291f52394e1f726f7cab4de64220b392afcdb',
  'archived_terminal_ledger_sha256','1811f08fca9b197decd2a96076257a1964c05e33c2e1b8a8f73444ffcf45f5e1',
  'archived_execution_judgments',derived);
 SELECT pg_catalog.jsonb_build_object(
  'rounds',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY round_id)
    FROM public.lab_arena_rounds r WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r310archive')),
  'submissions',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY round_id,submission_id)
    FROM public.lab_arena_submissions s WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r310archive')),
  'runs',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY run_id)
    FROM public.lab_arena_runs r WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r310archive')),
  'ledger',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
    FROM public.lab_arena_ledger l WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r310archive')),
  'weights',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(w)
    ORDER BY network,netuid,epoch) FROM public.lab_arena_accepted_weight_states w)
 ) INTO other_before;
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)-'per_icp_score'-'qualification_doc'-'updated_at'
   ORDER BY run_id) INTO miner_runs_before FROM public.lab_arena_runs r
   WHERE round_id='arena-2026-09-19' AND kind='execute'
    AND submission_id<>'baseline-2026-09-19';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
  INTO miner_ledger_before FROM public.lab_arena_ledger l
  WHERE round_id='arena-2026-09-19' AND submission_id<>'baseline-2026-09-19'
   AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs s
    WHERE s.round_id='arena-2026-09-19' AND s.kind='score' AND s.run_id=l.run_id);
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
  champion_fallback_providers
 ) SELECT round_id,status,status_generation,stage_generation,configuration_doc,
  rewards_enabled,participants,benchmark_ref,evaluation_date,
  stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
  king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
  reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
  cancel_reason,published_at,created_at,updated_at,promotion_required,
  promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
  confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
  champion_funding_frozen,champion_submission_id,champion_hotkey,
  champion_fallback_providers FROM pg_catalog.jsonb_populate_record(
  NULL::public.lab_arena_rounds,pg_catalog.to_jsonb(active_round)||pg_catalog.jsonb_build_object(
   'round_id','arena-2026-09-19-r310archive','status','cancelled',
   'configuration_doc',archive_config,'participants',archive_participants,
   'rewards_enabled',FALSE,'reward_basis_hash',NULL,'reward_basis_doc',NULL,
   'signing_key_doc',NULL,'effective_reward_epoch',NULL,'reward_activated_at',NULL,
   'cancel_reason','authorized_sep19_terminal309_newjudge_archive'));
 INSERT INTO public.lab_arena_submissions SELECT (pg_catalog.jsonb_populate_record(
  NULL::public.lab_arena_submissions,pg_catalog.to_jsonb(s)||pg_catalog.jsonb_build_object(
   'round_id','arena-2026-09-19-r310archive',
   'submission_id',s.submission_id||':r310archive'))).*
  FROM public.lab_arena_submissions s WHERE s.round_id='arena-2026-09-19';
 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-19-r310archive',
  submission_id=submission_id||':r310archive'
  WHERE round_id='arena-2026-09-19' AND kind='score';
 GET DIAGNOSTICS moved_scores=ROW_COUNT;
 UPDATE public.lab_arena_ledger l SET round_id='arena-2026-09-19-r310archive',
  submission_id=submission_id||':r310archive'
  WHERE round_id='arena-2026-09-19' AND EXISTS(SELECT 1 FROM public.lab_arena_runs s
   WHERE s.round_id='arena-2026-09-19-r310archive' AND s.kind='score' AND s.run_id=l.run_id);
 GET DIAGNOSTICS moved_score_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-19-r310archive',
  submission_id='baseline-2026-09-19:r310archive'
  WHERE round_id='arena-2026-09-19' AND kind='execute'
   AND submission_id='baseline-2026-09-19';
 GET DIAGNOSTICS moved_baseline_runs=ROW_COUNT;
 UPDATE public.lab_arena_ledger SET round_id='arena-2026-09-19-r310archive',
  submission_id='baseline-2026-09-19:r310archive'
  WHERE round_id='arena-2026-09-19' AND submission_id='baseline-2026-09-19';
 GET DIAGNOSTICS moved_baseline_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL
  WHERE round_id='arena-2026-09-19' AND kind='execute'
   AND submission_id<>'baseline-2026-09-19';
 UPDATE public.lab_arena_submissions SET source_ref='arena/arena-2026-09-19/sources/baseline-2026-09-19-rerun310.tar.gz',
  source_size_bytes=688770,submission_doc=expected_submission,
  updated_at=pg_catalog.clock_timestamp()
  WHERE round_id='arena-2026-09-19' AND submission_id='baseline-2026-09-19';
 IF NOT FOUND THEN RAISE EXCEPTION 'Sep19 rerun310 baseline update missing'; END IF;
 UPDATE public.lab_arena_rounds SET status='stage1',
  status_generation=status_generation+1,stage_generation=stage_generation+1,
  configuration_doc=pg_catalog.jsonb_set(pg_catalog.jsonb_set(pg_catalog.jsonb_set(
   configuration_doc,'{schedule}',p_schedule,TRUE),'{scorer_image_digest}',
   pg_catalog.to_jsonb(p_scorer_digest),TRUE),'{scorer_image_reference}',
   pg_catalog.to_jsonb(p_scorer_reference),TRUE),participants=new_participants,
  stage1_scoring_plan_doc=NULL,stage2_scoring_plan_doc=NULL,stage3_scoring_plan_doc=NULL,
  finalists=NULL,publication_doc=NULL,published_at=NULL,cancel_reason=NULL,
  updated_at=pg_catalog.clock_timestamp()
  WHERE round_id='arena-2026-09-19';
 IF NOT FOUND THEN RAISE EXCEPTION 'Sep19 rerun310 round update missing'; END IF;
 inserted_runs:=0;
 FOR position IN 0..19 LOOP
  stage:=CASE WHEN position<10 THEN 1 ELSE 2 END;
  assignment:='arena-2026-09-19:baseline-2026-09-19:'||stage::TEXT||':'||
   position::TEXT||':rerun310';
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,
   miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation)
  VALUES(assignment||':1',assignment,'arena-2026-09-19','baseline-2026-09-19',
   active_baseline.miner_hotkey,stage,position,1,'execute','pending',
   active_round.stage_generation+1);
  inserted_runs:=inserted_runs+1;
 END LOOP;
 ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)-'per_icp_score'-'qualification_doc'-'updated_at'
   ORDER BY run_id) INTO miner_runs_after FROM public.lab_arena_runs r
   WHERE round_id='arena-2026-09-19' AND kind='execute'
    AND submission_id<>'baseline-2026-09-19';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
  INTO miner_ledger_after FROM public.lab_arena_ledger l
  WHERE round_id='arena-2026-09-19' AND submission_id<>'baseline-2026-09-19';
 SELECT pg_catalog.jsonb_build_object(
  'rounds',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY round_id)
    FROM public.lab_arena_rounds r WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r310archive')),
  'submissions',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY round_id,submission_id)
    FROM public.lab_arena_submissions s WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r310archive')),
  'runs',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY run_id)
    FROM public.lab_arena_runs r WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r310archive')),
  'ledger',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
    FROM public.lab_arena_ledger l WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r310archive')),
  'weights',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(w)
    ORDER BY network,netuid,epoch) FROM public.lab_arena_accepted_weight_states w)
 ) INTO other_after;
 IF moved_scores<>101
  OR moved_baseline_runs<>22
  OR inserted_runs<>20 OR miner_runs_after IS DISTINCT FROM miner_runs_before
  OR miner_ledger_after IS DISTINCT FROM miner_ledger_before
  OR other_after IS DISTINCT FROM other_before
  OR public.lab_arena_sep19_rerun310_archive_valid_v1() IS NOT TRUE
  OR public.lab_arena_sep19_rerun310_active_valid_v1() IS NOT TRUE
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
    WHERE round_id='arena-2026-09-19' AND kind='score') THEN
  RAISE EXCEPTION 'Sep19 rerun310 atomic preservation differs' USING ERRCODE='55000';
 END IF;
 RETURN pg_catalog.jsonb_build_object('status','prepared',
  'round_id','arena-2026-09-19','fresh_baseline_execute_count',20,
  'retained_miner_execute_count',80,'archived_score_count',moved_scores,
  'archived_score_ledger_count',moved_score_ledger,
  'archived_baseline_ledger_count',moved_baseline_ledger,
  'score_namespace','score:rerun310');
END $prepare310$;
ALTER FUNCTION public.lab_arena_prepare_sep19_rerun310_v1(
 BIGINT,TEXT,TEXT,JSONB,TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep19_rerun310_v1(
 BIGINT,TEXT,TEXT,JSONB,TEXT,TEXT)
 FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_sep19_rerun310_v1(
 BIGINT,TEXT,TEXT,JSONB,TEXT,TEXT) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep19_rerun310_score_namespace_guard_v1()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public AS $score_guard310$
BEGIN
 IF NEW.round_id='arena-2026-09-19' AND NEW.kind='score'
  AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
   WHERE round_id='arena-2026-09-19-r310archive')
  AND NEW.assignment_id IS DISTINCT FROM NEW.round_id||':'||NEW.submission_id||':'||
   NEW.stage::TEXT||':'||NEW.icp_position::TEXT||':score:rerun310' THEN
  RAISE EXCEPTION 'Sep19 rerun310 scoring requires exact namespace'
   USING ERRCODE='55000';
 END IF;
 RETURN NEW;
END $score_guard310$;
ALTER FUNCTION public.lab_arena_sep19_rerun310_score_namespace_guard_v1()
 OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep19_rerun310_score_namespace_guard_v1()
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep19_rerun310_score_namespace_guard
 ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_sep19_rerun310_score_namespace_guard
 BEFORE INSERT ON public.lab_arena_runs FOR EACH ROW
 EXECUTE FUNCTION public.lab_arena_sep19_rerun310_score_namespace_guard_v1();

SELECT public.lab_arena_prepare_sep19_rerun310_v1(
 688770,'a67901a72a90965dd002a442236f44d113f81eda93f3778525c69a7dedf2cead','5e6d881882a2dae2be3ca783060f33e0380dd0de',
 '{"benchmark_deadline":"2026-09-19T07:45:00Z","final_scoring_close":"2026-09-19T19:45:00Z","publication_deadline":"2026-09-19T19:45:01Z","stage_1_close":"2026-09-19T10:45:01Z","stage_1_scoring_close":"2026-09-19T13:45:00Z","stage_1_start":"2026-09-19T07:45:01Z","stage_2_close":"2026-09-19T16:45:01Z","stage_2_start":"2026-09-19T13:45:01Z","submission_cutoff":"2026-09-19T00:00:00Z","submission_open":"2026-09-18T00:00:00Z"}'::JSONB,
 'sha256:19eae396b4168866a573122fce175316596e394049b51b544f5ee4262eaf5ec8','493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:19eae396b4168866a573122fce175316596e394049b51b544f5ee4262eaf5ec8');

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
